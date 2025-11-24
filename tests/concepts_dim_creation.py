import sys
import os
from dataclasses import dataclass, replace

# ensure workspace root is on sys.path so the local `STELLE` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from STELLE.data.dataset_loader import get_dataset
from STELLE.explanations.explanation_utils import compute_explanations, load_cached_metrics, save_metrics
from STELLE.kernels.kernel_utils import set_kernels_and_concepts
from STELLE.model.model_utils import train_test_model
from STELLE.utils import (
    flatten_dict,
    setup_environment,
    save_run_settings,
    save_results,
    parse_arguments,
    setup_paths,
    merge_result_dicts,
)


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments."""

    # Synthetic data parameters
    n_train: int = 500
    n_test: int = 100
    nvars: int = 5
    series_length: int = 100
    num_classes: int = 3

    # Fixed parameters
    seed: int = 0
    pll: int = 8
    workers: int = 2
    samples: int = 5000
    epochs: int = 3000
    cf: int = 300
    patience: int = 10
    val_every_n_epochs: int = 1
    verbose: int = 100
    logging: bool = False

    # Kernel parameters
    normalize_kernel: bool = False
    exp_kernel: bool = False
    normalize_rhotau: bool = True
    exp_rhotau: bool = True

    # Concept parameters
    t: float = 1.0
    nvars_formulae: int = 1
    creation_mode: str = "one"
    dim_concepts: int = 1000
    min_total: int = 100
    imp_t_l: float = 0
    imp_t_g: float = 0
    t_k: float = 0.8
    explanation_operation: str | None = "mean"

    # Training parameters
    d: float = 0.1
    bs: int = 32
    lr: float = 1e-4
    init_eps: float = 1
    activation_str: str = "gelu"
    backprop_method: str = "ig"
    init_crel: float = 1
    h: int = 256
    n_layers: int = 1


def main():
    args = parse_arguments()
    config = ExperimentConfig()
    device = setup_environment(config.seed)
    base_path = "tests/results/concepts_dim_creation/"
    model_path = "tests/results/concepts_checkpoints/"
    paths = setup_paths(base_path, model_path, args, args.dataset, config)
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["model_path_og"], exist_ok=True)

    trainloader, valloader, testloader = get_dataset(
        args.dataset, config, paths["dataset_info_path"]
    )

    print(f"Run ID: {paths['run_id']}\n")
    save_run_settings(paths["results_dir"], **locals())

    results = []

    # from here it depends from concepts details
    for dim_concepts in [50, 100, 500, 1000, 2000, 3000, 4000, 5000]:
        print(f"\n>>>>>>>>>>>>> DIM CONCEPTS = {dim_concepts} >>>>>>>>>>>>>\n")
        for creation_mode in ["one", "all"]:
            print(f"\n>>>>>>>>>>>>> CREATION MODE = {creation_mode} >>>>>>>>>>>>>\n")
            config_i = replace(
                config, dim_concepts=dim_concepts, creation_mode=creation_mode
            )

            kernel, _, concepts_time = set_kernels_and_concepts(
                trainloader.dataset, paths["phis_path_og"], config_i
            )
            for lr in [1e-4, 1e-5, 1e-6]:
                config_i = replace(config_i, lr=lr)
                model_id = (
                    f"seed_{config_i.seed}_{config_i.lr}_{config_i.init_crel}_{config.init_eps}_{config_i.h}_"
                    f"{config_i.n_layers}_bs{config.bs}_t{config_i.t}_{len(kernel.phis)}_"
                    f"{config_i.creation_mode}_f{config_i.nvars_formulae}"
                )
                # attach to model for later reference and debugging
                print(f"Model ID: {model_id}")
                model_path_ev = os.path.join(paths["model_path_og"], f"{model_id}.pt")

                args = (kernel, trainloader, valloader, testloader, model_path_ev, config_i)
                model, accuracy_results = train_test_model(args)

                # Check for cached metrics first
                local_metrics, global_metrics = load_cached_metrics(model_path_ev, config_i)
            
                if local_metrics is None:
                    args_explanations = (
                        model_path_ev,
                        trainloader,
                        testloader,
                        model,
                        config_i,
                    )
                    local_metrics, global_metrics = compute_explanations(args_explanations, save=False)
                    save_metrics(model_path_ev, config_i, local_metrics, global_metrics)

                result_raw = merge_result_dicts(
                    [accuracy_results, local_metrics, global_metrics]
                )

                result = {
                    "dim_concepts": config_i.dim_concepts,
                    "creation_mode": config_i.creation_mode,
                    "lr": config_i.lr, 
                    "concepts_time": round(concepts_time, 3),
                    **result_raw,
                }

                result = flatten_dict(result)
                results.append(result)

                save_results(results, paths["results_dir"])


if __name__ == "__main__":
    main()
