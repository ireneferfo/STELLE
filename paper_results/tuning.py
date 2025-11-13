import os
import json
from dataclasses import dataclass, replace
from typing import Dict, Any, List

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationStep
from ax.adapter.registry import Generators

from STELLE.data.dataset_loader import get_dataset
from STELLE.kernels.kernel_utils import set_kernels_and_concepts
from STELLE.model.model_utils import train_test_model
from STELLE.utils import (
    setup_environment,
    save_results,
    parse_arguments,
    setup_paths,
    flatten_dict,
)
import warnings
from ax.exceptions.core import AxParameterWarning
from sqlalchemy.exc import SAWarning

warnings.filterwarnings("ignore", category=UserWarning, module="ax.adapter.base")
warnings.filterwarnings(
    "ignore", category=AxParameterWarning, message=".*sort_values.*"
)
warnings.filterwarnings("ignore", category=AxParameterWarning, message=".*is_ordered.*")
os.environ["SQLALCHEMY_WARN_20"] = "0"
warnings.filterwarnings(
    "ignore", category=SAWarning, message=".*TypeDecorator.*cache_ok.*"
)


N_TRIALS = 3

# TODO aggiustare paths

@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""

    # Synthetic data parameters
    n_train: int = 50
    n_test: int = 30
    nvars: int = 5
    series_length: int = 100
    num_classes: int = 3

    # Fixed parameters
    seed: int = 0
    pll: int = 8
    workers: int = 2
    samples: int = 300
    epochs: int = 3
    cf: int = 50
    patience: int = 5
    val_every_n_epochs: int = 1
    verbose: int = 1
    logging: bool = False

    # Kernel parameters - can be tuned
    normalize_kernel: bool = False
    exp_kernel: bool = False
    normalize_rhotau: bool = True
    exp_rhotau: bool = True

    # Concept parameters - can be tuned
    t: float = 1.0
    nvars_formulae: int = 1
    creation_mode: str = "all"
    dim_concepts: int = 30
    min_total: int = 100
    imp_t_l: float = 0
    imp_t_g: float = 0
    t_k: float = 0.8

    # Training parameters - can be tuned
    d: float = 0.1
    bs: int = 32
    lr: float = 1e-4
    init_eps: float = 1
    activation_str: str = "gelu"
    backprop_method: str = "ig"
    init_crel: float = 1
    h: int = 256
    n_layers: int = 1


class HyperparameterOptimizer:
    """Bayesian optimization for hyperparameter tuning using Ax."""

    def __init__(
        self,
        base_config: ExperimentConfig,
        dataset_name: str,
        paths: Dict,
        n_trials: int = 50,
        early_stopping_patience: int = 10,
        early_stopping_threshold: float = 1e-4,
        min_trials: int = 20,
    ):
        self.base_config = base_config
        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.base_path = paths["results_dir"]
        self.model_path = paths["model_path_og"]
        self.paths = paths

        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.min_trials = min_trials

        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        self.device = setup_environment(base_config.seed)
        self.trial_results: List[Dict[str, Any]] = []
        self.best_accuracy_history: List[float] = []

    def define_search_space(self) -> List[Dict[str, Any]]:
        """Define the hyperparameter search space."""
        return [
            # Training hyperparameters
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-7, 1e-2],
                "log_scale": True,
                "value_type": "float",
                "digits": 7,
            },
            {
                "name": "bs",
                "type": "choice",
                "values": [16, 32, 64, 128],
                "value_type": "int",
            },
            {
                "name": "h",
                "type": "choice",
                "values": [128, 256, 512, 1024],
                "value_type": "int",
            },
            {
                "name": "n_layers",
                "type": "range",
                "bounds": [0, 3],
                "value_type": "int",
            },
            {
                "name": "d",
                "type": "range",
                "bounds": [0.0, 0.4],
                "value_type": "float",
                "digits": 1,
            },
            {
                "name": "init_crel",
                "type": "range",
                "bounds": [1.0, 10.0],
                "value_type": "float",
                "digits": 1,
            },
            {
                "name": "init_eps",
                "type": "range",
                "bounds": [0.1, 2.0],
                "value_type": "float",
                "digits": 1,
            },
            # Activation function
            {
                "name": "activation_str",
                "type": "choice",
                "values": ["relu", "gelu"],
                "value_type": "str",
            },
        ]

    def create_ax_client(self) -> AxClient:
        """Initialize Ax client with custom generation strategy."""
        # Custom generation strategy: Sobol + qLogNEI
        gs = GenerationStrategy(
            steps=[
                # Start with Sobol for exploration
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=10,  # Initial random trials
                    model_kwargs={"seed": 0},
                ),
                # Then switch to Bayesian optimization with qLogNEI
                GenerationStep(
                    generator=Generators.BOTORCH_MODULAR,
                    num_trials=-1,  # No limit, use for remaining trials
                    model_kwargs={
                        "acquisition_class": "qLogNoisyExpectedImprovement",
                    },
                ),
            ]
        )

        ax_client = AxClient(generation_strategy=gs, verbose_logging=False)

        ax_client.create_experiment(
            name=f"{self.dataset_name}_hyperparameter_optimization",
            parameters=self.define_search_space(),
            objectives={
                "accuracy": ObjectiveProperties(minimize=False),
            },
            parameter_constraints=[],
        )

        return ax_client

    def evaluate_configuration(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single hyperparameter configuration.

        Args:
            parameters: Dictionary of hyperparameters to evaluate

        Returns:
            Dictionary containing accuracy and other metrics
        """
        # Create config with new parameters
        config = replace(self.base_config, **parameters)

        # try:
        # Load dataset
        trainloader, valloader, testloader = get_dataset(
            self.dataset_name, config, self.paths["dataset_info_path"]
        )

        # Set kernels and concepts
        kernel, _, _ = set_kernels_and_concepts(
            trainloader.dataset, self.paths["phis_path_og"], config
        )

        # Create unique model ID
        model_id = (
            f"opt_seed_{config.seed}_{config.lr}_{config.init_crel}_"
            f"{config.init_eps}_{config.h}_{config.n_layers}_{config.d}_bs{config.bs}"
        )
        print(f"Model ID: {model_id}")
        model_path_ev = os.path.join(self.model_path, f"{model_id}.pt")

        # Train and test model
        args_model = (
            kernel,
            trainloader,
            valloader,
            testloader,
            model_path_ev,
            config,
        )
        model, accuracy_results = train_test_model(args_model)

        # Compute explanations (optional)
        # args_explanations = (model_path_ev, trainloader, testloader, model, config)
        # local_metrics, global_metrics = compute_explanations(args_explanations)

        print(
            f"Trial completed: Test Accuracy = {accuracy_results.get('accuracy', 0.0)}"
        )
        return flatten_dict(accuracy_results)

        # except Exception as e:
        #     print(f"Trial failed with error: {str(e)}")
        #     # Return poor accuracy if trial fails
        #     return {
        #         'best_epoch': 0,
        #         "accuracy": 0.0,
        #         "weighted_acc": 0.0,
        #         "sensitivity": 0.0,
        #         "specificity": 0.0,
        #         "avg_valloss": 0.0,
        #         "train_time": 0.0,
        #         'test_time': 0.0
        #     }

    def check_early_stopping(self, trial_idx: int) -> bool:
        """
        Check if optimization should stop early based on convergence.

        Args:
            trial_idx: Current trial index

        Returns:
            True if should stop, False otherwise
        """
        # Don't stop before minimum trials
        if trial_idx < self.min_trials:
            return False

        # Need enough history to check
        if len(self.best_accuracy_history) < self.early_stopping_patience:
            return False

        # Get recent best accuracies
        recent_best = self.best_accuracy_history[-self.early_stopping_patience :]

        # Check if improvement is below threshold
        max_recent = max(recent_best)
        min_recent = min(recent_best)
        improvement = max_recent - min_recent

        if improvement < self.early_stopping_threshold:
            print(f"\n{'='*60}")
            print("EARLY STOPPING TRIGGERED")
            print(f"{'='*60}")
            print(
                f"No significant improvement in last {self.early_stopping_patience} trials"
            )
            print(
                f"Maximum improvement: {improvement:.6f} < threshold: {self.early_stopping_threshold}"
            )
            print(f"Stopping at trial {trial_idx + 1}/{self.n_trials}")
            return True

        return False

    def run_optimization(self):
        """Run the complete optimization loop with early stopping."""
        ax_client = self.create_ax_client()

        print(f"Starting Bayesian Optimization with up to {self.n_trials} trials")
        print(f"Dataset: {self.dataset_name}")
        print(
            f"Early stopping: patience={self.early_stopping_patience}, "
            f"threshold={self.early_stopping_threshold}, min_trials={self.min_trials}\n"
        )

        early_stopped = False

        for trial_idx in range(self.n_trials):
            print(f"\n{'='*60}")
            print(f"Trial {trial_idx + 1}/{self.n_trials}")
            print(f"{'='*60}")

            # Get next parameters to evaluate
            parameters, trial_index = ax_client.get_next_trial()

            print("Parameters to evaluate:")
            for key, value in parameters.items():
                print(f"  {key}: {value}")

            # Evaluate configuration
            results = self.evaluate_configuration(parameters)

            # Complete trial with results
            ax_client.complete_trial(trial_index=trial_index, raw_data=results)

            # Store results
            trial_result = {
                "trial_index": trial_index,
                "parameters": parameters,
                **results,
            }
            self.trial_results.append(trial_result)

            # Update best accuracy history
            _, best_values = ax_client.get_best_parameters()
            current_best = best_values[0]["accuracy"]
            self.best_accuracy_history.append(current_best)

            # Save intermediate results
            save_results(self.trial_results, self.base_path)
            print(f"Results saved to {self.base_path}")

            # Print current best
            print(f"\nCurrent best accuracy: {current_best:.4f}")

            # Check for early stopping
            if self.check_early_stopping(trial_idx):
                early_stopped = True
                break

        # Final results
        self.print_final_results(ax_client, early_stopped)

        return ax_client

    def print_final_results(self, ax_client: AxClient, early_stopped: bool = False):
        """Print and save final optimization results."""
        print("\n\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        if early_stopped:
            print("(Stopped early due to convergence)")
        print("=" * 60)

        # Get best parameters
        best_params, best_values = ax_client.get_best_parameters()

        print(f"\nTotal trials completed: {len(self.trial_results)}")
        print(f"\nBest Test Accuracy: {best_values[0]['accuracy']:.4f}")
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Save best parameters
        best_params_path = os.path.join(self.base_path, "best_parameters.json")
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "best_accuracy": best_values[0]["accuracy"],
                    "parameters": best_params,
                    "total_trials": len(self.trial_results),
                    "early_stopped": early_stopped,
                },
                f,
                indent=2,
            )

        # Save optimization summary
        summary_path = os.path.join(self.base_path, "optimization_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "early_stopped": early_stopped,
                    "total_trials": len(self.trial_results),
                    "best_accuracy": best_values[0]["accuracy"],
                    "best_accuracy_history": self.best_accuracy_history,
                    "final_improvement": (
                        self.best_accuracy_history[-1]
                        - self.best_accuracy_history[-self.early_stopping_patience]
                        if len(self.best_accuracy_history)
                        >= self.early_stopping_patience
                        else 0.0
                    ),
                },
                f,
                indent=2,
            )

        # Generate and save optimization plots
        try:
            # Save Ax client state
            ax_client.save_to_json_file(
                os.path.join(self.base_path, "ax_client_snapshot.json")
            )
            print("\nAx client saved for later analysis")
        except Exception as e:
            print(f"Could not save Ax client: {e}")


def main():
    """Main execution function."""
    args = parse_arguments()
    base_config = ExperimentConfig()
    base_path = ""
    model_path = ""
    paths = setup_paths(base_path, model_path, args, args.dataset, base_config)
    # Configure optimization
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        dataset_name=args.dataset,
        n_trials=N_TRIALS,  # Maximum number of trials
        paths=paths,
        early_stopping_patience=10,  # Stop if no improvement for 10 trials
        early_stopping_threshold=1e-4,  # Minimum improvement required (0.01%)
        min_trials=20,  # Minimum trials before early stopping can trigger
    )

    # Run optimization
    _ = optimizer.run_optimization()

    print("\nOptimization finished! Check the results directory for outputs.")


if __name__ == "__main__":
    main()
