import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os
import torch
import tempfile
from datetime import datetime
from dataclasses import dataclass


from STELLE.kernels.stl_kernel import StlKernel
from STELLE.kernels.kernel_utils import _create_concepts_path, _validate_parameters
from STELLE.kernels.base_measure import BaseMeasure
from STELLE.data.dataset_loader import _load_raw_data, remove_redundant_variables
from STELLE.formula_generation.formula_manipulation import time_scaling
from STELLE.formula_generation.formula_manager import FormulaManager
from STELLE.utils import setup_environment, parse_arguments, save_results


# The hypothesis: If there's correlation between kernel_func regression performance (predicting formula robustness) and accuracy, it means:
# Positive correlation with R² → The kernel_func works well when STELLE works well
# Negative correlation with MSE → The kernel_func fails when STELLE fails
# This suggests the kernel_func choice (μ₀) is the limiting factor


class KernelRegressionAblation:
    """
    Ablation test to investigate correlation between kernel_func regression
    performance and STELLE accuracy on different datasets.
    """

    def __init__(
        self,
        datasets_dict,
        stelle_results,
        config,
        device,
        norm=True,
        exp=True,
        workers=1,
    ):
        """
        Args:
            datasets_dict: Dict[dataset_name -> {'signals': signals, 'formulae': formulae}]
            stelle_results: Dict[dataset_name -> stelle_accuracy]
        """
        self.datasets = datasets_dict
        self.stelle_results = stelle_results
        self.results = []
        self.config = config
        self.device = device
        self.pll = workers

        self.mu = BaseMeasure(device=device)
        self.kernel_args = norm, exp
        self.kernel_data = None

    def kernel_regression_on_formulae(
        self, train_formulae, test_formulae, target_type="avg_robustness"
    ):
        """
        Perform kernel_func regression to predict performance of test_formula.

        Args:
            train_signals: Training signals
            train_formulae: Training formulae
            test_formulae: Formulae to predict performance for
            target_type: 'avg_robustness', 'single_trajectory', 'satisfiability'

        Returns:
            predicted_value, actual_value, error
        """
        if self.kernel_data is None:
            # Compute kernel_func matrix between training formulae and test formula
            K_train_test, rhos_train, rhos_test, selfk_train, _ = (
                self.kernel.compute_bag_bag(
                    train_formulae, test_formulae, return_robustness=True
                )
            )
            K_train_train = self.kernel._compute_kernel(
                rhos_train, rhos_train, selfk_train, selfk_train
            )

            # Add regularization
            K_train_train += 1e-6 * np.eye(len(train_formulae))
            self.kernel_data = rhos_train, rhos_test, K_train_train, K_train_test
        else:
            rhos_train, rhos_test, K_train_train, K_train_test = self.kernel_data

        # Compute targets based on target_type
        if target_type == "avg_robustness":
            # Average robustness across all training signals
            targets = torch.mean(rhos_train, dim=[1, 2])  # must be len(train_formulae)
            actual = torch.mean(rhos_test, dim=[1, 2])

        # elif target_type == 'single_trajectory': #! not tested
        #     # Pick a random trajectory or use the first one
        #     test_signal = train_signals[0]

        #     targets = [
        #         self.kernel._compute_single_rho(f, test_signal)
        #         for f in train_formulae
        #     ]
        #     actual, _ = self.kernel._compute_single_rho(test_formula, test_signal)

        elif target_type == "satisfiability":
            # Fraction of signals satisfying the formula
            targets = torch.mean(rhos_train, dim=[1, 2])  # must be len(train_formulae)
            # Convert boolean mask to float for averaging
            actual = (rhos_test > 0).float().mean(dim=[1, 2])

        # Solve for weights: K_train_train @ weights = targets
        weights = torch.from_numpy(np.linalg.solve(K_train_train, targets))

        # Predict
        predicted = K_train_test.T.to(torch.float) @ weights.to(
            torch.float
        )  # K_train_test @ weights
        error = abs(predicted - actual)

        return predicted, actual, error

    def run_ablation_per_dataset(
        self, dataset_name, target_type="avg_robustness", n_test_formulae=10
    ):
        """
        Run kernel_func regression ablation on one dataset.
        """
        data = self.datasets[dataset_name]
        signals = data["signals"].to(torch.float32)
        formulae = data["formulae"]
        norm, exp = self.kernel_args
        self.kernel = StlKernel(
            self.mu,
            varn=signals.shape[1],
            vectorize=True,
            normalize=norm,
            exp_kernel=exp,
            signals=signals,
        )

        # Split formulae into train/test
        n_train = len(formulae) - n_test_formulae
        train_formulae = formulae[:n_train]
        test_formulae = formulae[n_train:]

        predictions, actuals, errors = self.kernel_regression_on_formulae(
            train_formulae, test_formulae, target_type
        )

        # Compute metrics
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mean_error = torch.mean(errors)
        std_error = torch.std(errors)

        return {
            "dataset": dataset_name,
            "target_type": target_type,
            "mse": mse,
            "r2": r2,
            "mean_error": mean_error,
            "std_error": std_error,
            "errors": errors,
            "predictions": predictions,
            "actuals": actuals,
            "stelle_accuracy": self.stelle_results[dataset_name],
        }

    def run_full_ablation(
        self, phis_path_og, target_types=["avg_robustness", "satisfiability"]
    ):  #
        """
        Run ablation test across all datasets and target types.
        """
        self.results = []
        for dataset_name in self.datasets.keys():
            self.kernel_data = None
            print(f"\nProcessing dataset: {dataset_name}")
            print(f"STELLE accuracy: {self.stelle_results[dataset_name]:.3f}")

            norm, exp = self.kernel_args
            nvars = self.datasets[dataset_name]["signals"].shape[1]
            formulakernel = StlKernel(
                self.mu,
                varn=nvars,
                vectorize=True,
                normalize=norm,
                exp_kernel=exp,
                samples=500,
            )

            nvars_formulae, creation_mode = _validate_parameters(
                nvars, self.config.nvars_formulae, self.config.creation_mode
            )

            phis_path = _create_concepts_path(
                phis_path_og, creation_mode, nvars_formulae, nvars, self.config.t
            )
            # Generate and scale concepts
            formula_manager = FormulaManager(
                nvars,
                formulakernel,
                self.config.pll,
                self.config.t,
                nvars_formulae,
                device=self.device,
            )

            concepts, _, _, _ = formula_manager.get_formulae(
                creation_mode,
                self.config.dim_concepts,
                phis_path,
                "concepts",
                self.config.seed,
            )
            scaled_concepts = time_scaling(
                concepts, self.datasets[dataset_name]["signals"].shape[-1]
            )

            self.datasets[dataset_name]["formulae"] = scaled_concepts

            for target_type in target_types:
                print(f"  Target type: {target_type}")
                result = self.run_ablation_per_dataset(dataset_name, target_type)
                self.results.append(result)
                print(f"    MSE: {result['mse']:.4f}, R²: {result['r2']:.4f}")

        return pd.DataFrame(self.results)

    def analyse_correlation(self):
        """
        Analyze correlation between kernel_func regression performance and STELLE accuracy.
        """
        df = pd.DataFrame(self.results)

        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)

        for target_type in df["target_type"].unique():
            subset = df[df["target_type"] == target_type]

            # Correlation between MSE and STELLE accuracy
            pearson_mse, p_mse = pearsonr(subset["mse"], subset["stelle_accuracy"])
            spearman_mse, sp_mse = spearmanr(subset["mse"], subset["stelle_accuracy"])

            # Correlation between R² and STELLE accuracy
            pearson_r2, p_r2 = pearsonr(subset["r2"], subset["stelle_accuracy"])
            spearman_r2, sp_r2 = spearmanr(subset["r2"], subset["stelle_accuracy"])

            print(f"\nTarget type: {target_type}")
            print("  MSE vs STELLE Accuracy:")
            print(f"    Pearson r={pearson_mse:.3f} (p={p_mse:.4f})")
            print(f"    Spearman ρ={spearman_mse:.3f} (p={sp_mse:.4f})")
            print("  R² vs STELLE Accuracy:")
            print(f"    Pearson r={pearson_r2:.3f} (p={p_r2:.4f})")
            print(f"    Spearman ρ={spearman_r2:.3f} (p={sp_r2:.4f})")

        return df

    def plot_results(self):
        """
        Visualize the ablation test results.
        """
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. MSE vs STELLE Accuracy
        for target_type in df["target_type"].unique():
            subset = df[df["target_type"] == target_type]
            axes[0, 0].scatter(
                subset["stelle_accuracy"],
                subset["mse"],
                label=target_type,
                alpha=0.6,
                s=100,
            )
        axes[0, 0].set_xlabel("STELLE Accuracy", fontsize=12)
        axes[0, 0].set_ylabel("Kernel Regression MSE", fontsize=12)
        axes[0, 0].set_title("MSE vs STELLE Accuracy", fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. R² vs STELLE Accuracy
        for target_type in df["target_type"].unique():
            subset = df[df["target_type"] == target_type]
            axes[0, 1].scatter(
                subset["stelle_accuracy"],
                subset["r2"],
                label=target_type,
                alpha=0.6,
                s=100,
            )
        axes[0, 1].set_xlabel("STELLE Accuracy", fontsize=12)
        axes[0, 1].set_ylabel("Kernel Regression R²", fontsize=12)
        axes[0, 1].set_title("R² vs STELLE Accuracy", fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Error distribution by dataset
        datasets = df["dataset"].unique()
        error_data = []
        labels = []
        for dataset in datasets:
            subset = df[df["dataset"] == dataset]
            for _, row in subset.iterrows():
                error_data.append(row["errors"])
                labels.extend(
                    [f"{dataset}\n{row['target_type'][:3]}"] * len(row["errors"])
                )

        axes[1, 0].boxplot(
            [row["errors"] for _, row in df.iterrows()],
            labels=[f"{row['dataset'][:8]}" for _, row in df.iterrows()],
        )
        axes[1, 0].set_xlabel("Dataset", fontsize=12)
        axes[1, 0].set_ylabel("Prediction Error", fontsize=12)
        axes[1, 0].set_title("Error Distribution by Dataset", fontsize=14)
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # 4. Mean error vs STELLE Accuracy
        for target_type in df["target_type"].unique():
            subset = df[df["target_type"] == target_type]
            axes[1, 1].scatter(
                subset["stelle_accuracy"],
                subset["mean_error"],
                label=target_type,
                alpha=0.6,
                s=100,
            )
            # Add error bars
            axes[1, 1].errorbar(
                subset["stelle_accuracy"],
                subset["mean_error"],
                yerr=subset["std_error"],
                fmt="none",
                alpha=0.3,
            )
        axes[1, 1].set_xlabel("STELLE Accuracy", fontsize=12)
        axes[1, 1].set_ylabel("Mean Prediction Error", fontsize=12)
        axes[1, 1].set_title("Mean Error vs STELLE Accuracy", fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("kernel_ablation_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        return fig


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments."""

    # Fixed parameters
    seed: int = 0
    pll: int = 8
    workers: int = 2

    # Kernel parameters
    normalize_kernel: bool = True
    exp_kernel: bool = True
    normalize_rhotau: bool = True
    exp_rhotau: bool = False

    # Concept parameters
    t: float = 0.98
    nvars_formulae: int = 1
    creation_mode: str = "all"
    dim_concepts: int = 1000
    min_total: int = 100


def main():

    args = parse_arguments()
    device = setup_environment(0)
    config = ExperimentConfig()

    base_path = "tests/results/kernel_regression/"
    model_path = "tests/results/kernel_regression/"

    if args.temp:
        base_path = tempfile.mkdtemp()
        model_path_og = tempfile.mkdtemp()
    else:
        model_path_og = (
            os.path.join(os.environ["WORK"], f"STELLE/{model_path}/checkpoints/")
            if args.demetra
            else os.path.join(model_path, "checkpoints/")
        )
    phis_path_og = (
        os.path.join(os.environ["WORK"], "STELLE/phis/") if args.demetra else "phis/"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_path, run_id)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_path_og, exist_ok=True)

    stelle_accuracies = {
        "maritime": 1.0,  # STELLE accuracy on this dataset
        "ECG200": 0.8,
        "ECG5000": 0.92,
        "EOGVerticalSignal": 0.58,
        "Epilepsy2": 0.95,
        "GunPoint": 0.92,
        "GunPointOldVersusYoung": 1.0,
        "NerveDamage": 1.0,
        "SharePriceIncrease": 0.62,
        "ArticularyWordRecognition": 0.95,
        "AtrialFibrillation": 0.36,
        "BasicMotions": 0.99,
        "Cricket": 0.96,
        "ERing": 0.84,
        "Epilepsy": 0.94,
        "EthanolConcentration": 0.32,
        "HandMovementDirection": 0.47,
        "Handwriting": 0.20,
        "Libras": 0.75,
    }

    datasets = {}
    for dataname in list(stelle_accuracies.keys()):
        X_train, _, X_test, _, _, _ = _load_raw_data(
            dataname,
        )

        x = np.concatenate((X_train, X_test))
        keep = remove_redundant_variables(x)
        x = x[:, keep, :]

        datasets[dataname] = {
            "signals": torch.tensor(x),
            "formulae": None,
        }  # le metto dopo

    ablation = KernelRegressionAblation(datasets, stelle_accuracies, config, device)
    results = ablation.run_full_ablation(
        phis_path_og, target_types=["avg_robustness", "satisfiability"]
    )

    save_results(results, results_dir)

    # Analyze correlation
    ablation.analyse_correlation()

    # 5. Plot results
    ablation.plot_results()


if __name__ == "__main__":
    main()


# Example usage:
"""
# 1. Prepare your data
datasets = {
    'dataset1': {
        'signals': [...],  # Your signal data
        'formulae': [...]  # Your STL formulae
    },
    'dataset2': {
        'signals': [...],
        'formulae': [...]
    },
    # ... more datasets where STELLE works poorly
}

stelle_accuracies = {
    'dataset1': 0.65,  # STELLE accuracy on this dataset
    'dataset2': 0.58,
    # ...
}

# 2. Define your kernel_func function
def rbf_kernel(formula1, formula2, sigma=1.0):
    # Implement your kernel_func between formulae
    # This should be the same kernel_func used in STELLE
    distance = compute_formula_distance(formula1, formula2)
    return np.exp(-distance**2 / (2 * sigma**2))

# 3. Run ablation test
ablation = KernelRegressionAblation(datasets, stelle_accuracies, rbf_kernel)
results_df = ablation.run_full_ablation(
    target_types=['avg_robustness', 'satisfiability']
)

# 4. Analyze correlation
ablation.analyse_correlation()

# 5. Plot results
ablation.plot_results()

# 6. Interpretation
# If there's strong negative correlation between MSE and STELLE accuracy:
#   -> The kernel_func doesn't capture the right structure for those datasets
#   -> Consider changing mu0 or kernel_func parameters
# If there's strong positive correlation between R² and STELLE accuracy:
#   -> The method works better when we expect it to work better
#   -> The kernel_func is appropriate for those datasets
"""
