import os
from dataclasses import dataclass, replace

from STELLE.data.dataset_loader import get_dataset
from STELLE.explanations.explanation_utils import (
    compute_explanations,
    load_cached_metrics,
    save_metrics,
)
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

# variazione t_k e imp_t_l


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
    samples: int = 500
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
    t: float = 0.98
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
    _ = setup_environment(config.seed)
    base_path = "tests/results/explanation_parameters/"
    model_path = "tests/results/explanation_checkpoints/"
    paths = setup_paths(base_path, model_path, args, args.dataset, config)
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["model_path_og"], exist_ok=True)

    trainloader, valloader, testloader, config = get_dataset(
        args.dataset, config, paths["dataset_info_path"]
    )

    print(f"Run ID: {paths['run_id']}\n")
    save_run_settings(paths["results_dir"], **locals())

    results = []

    # from here it depends from concepts details
    kernel, _, _ = set_kernels_and_concepts(
        trainloader.dataset, paths["phis_path_og"], config
    )
    for lr in [1e-5]:
        print(f"\n>>>>>>>>>>>>>>>>>>>>> lr = {lr} >>>>>>>>>>>>>>>>>>>>>\n")
        config = replace(config, lr=lr)
        model_id = (
            f"seed_{config.seed}_{config.lr}_{config.init_crel}_{config.init_eps}_{config.h}_"
            f"{config.n_layers}_bs{config.bs}"
        )

        # attach to model for later reference and debugging
        print(f"Model ID: {model_id}")
        model_path_ev = os.path.join(paths["model_path_og"], f"{model_id}.pt")

        args = (kernel, trainloader, valloader, testloader, model_path_ev, config)

        model, accuracy_results = train_test_model(args, arch_type="base")

        for t_k in [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
            print(f"\n>>>>>>>>>>>>>>>>>>>>> t_k = {t_k} >>>>>>>>>>>>>>>>>>>>>\n")
            for imp_t_l in [0, 0.001, 0.01, 0.1, 0.2, 0.3]:
                print(f"\n>>>>>>>>>>>>>>>>>>>>> imp_t_l = {imp_t_l} >>>>>>>>>>>>>>>>>>>>>\n")
                config = replace(config, t_k=t_k, imp_t_l=imp_t_l)
                expl_path = os.path.join(
                    paths["model_path_og"],
                    f"{model_id}_base_base_mean_{t_k}_{imp_t_l}.pt",
                )  # base arch_type and expl_type

                # Check for cached metrics first
                local_metrics, global_metrics = load_cached_metrics(expl_path, config)

                if local_metrics is None:
                    args_explanations = (
                        expl_path,
                        trainloader,
                        testloader,
                        model,
                        config,
                    )
                    local_metrics, global_metrics = compute_explanations(
                        args_explanations, save=False
                    )
                    save_metrics(expl_path, config, local_metrics, global_metrics)


                result_raw = merge_result_dicts(
                    [accuracy_results, local_metrics, global_metrics]
                )

                result = {
                    "t_k": t_k,
                    "imp_t_l": imp_t_l,
                    "lr": lr,
                    **result_raw,
                }

                result = flatten_dict(result)
                results.append(result)

                save_results(results, paths["results_dir"])


if __name__ == "__main__":
    main()
