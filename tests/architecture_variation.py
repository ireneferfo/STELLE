import os
from dataclasses import dataclass

from STELLE.data.dataset_loader import get_dataset
from STELLE.explanations.explanation_utils import compute_explanations
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
    normalize: bool = False
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


dim_anchors = 10


def get_anchor_items(
    kernel, train_subset, phis_path, formula_manager, dim_anchors, config
):
    import torch
    import gc
    from STELLE.kernels.kernel_utils import _validate_parameters

    """Initialize kernels and generate concepts for STL-based learning."""

    # Validate and adjust parameters
    _, creation_mode = _validate_parameters(
        config.nvars, config.nvars_formulae, config.creation_mode
    )

    stlkernel = formula_manager.stl_kernel
    concepts = kernel.phis
    rhos1 = kernel.rhos_phi
    selfk1 = kernel.selfk_phi

    base_concepts, rhos2, selfk2, base_concepts_time = formula_manager.get_formulae(
        creation_mode, dim_anchors, phis_path, "anchors", (config.seed * 2 + 10)
    )
    scaled_base_concepts = train_subset.time_scaling(base_concepts)
    kernel.phis = scaled_base_concepts
    kernel.rhos_phi = rhos2
    kernel.selfk_phi = selfk2

    concept_embeddings = stlkernel._compute_kernel(rhos1, rhos2, selfk1, selfk2)

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return base_concepts, concepts, concept_embeddings, base_concepts_time


def main():
    args = parse_arguments()
    config = ExperimentConfig()
    _ = setup_environment(config.seed)
    base_path = "tests/results/architecture_variations/"
    model_path = "tests/results/architecture_checkpoints/"
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
    kernel, formula_manager, concepts_time = set_kernels_and_concepts(
        trainloader.dataset, paths["phis_path_og"], config
    )

    for type in ["Anchor"]:  # , 'Robs', 'RobsAsHx', 'RobsAsGx', 'NoGx', 'base']:
        print(f"\n>>>>>>>>>>>>>>>>>>>>> type = {type} >>>>>>>>>>>>>>>>>>>>>\n")
        model_id = (
            f"seed_{config.seed}_{config.lr}_{config.init_crel}_{config.init_eps}_{config.h}_"
            f"{config.n_layers}_bs{config.bs}_{type}"
        )
        # attach to model for later reference and debugging
        print(f"Model ID: {model_id}")
        model_path_ev = os.path.join(paths["model_path_og"], f"{model_id}.pt")

        args = (kernel, trainloader, valloader, testloader, model_path_ev, config)
        if type == "Anchor":
            base_concepts, concepts, concept_embeddings, base_concepts_time = (
                get_anchor_items(
                    kernel,
                    trainloader.dataset,
                    paths["phis_path_og"],
                    formula_manager,
                    dim_anchors,
                    config,
                )
            )
            model, accuracy_results = train_test_model(
                args,
                type=type,
                concepts=concepts,
                concept_embeddings=concept_embeddings,
            )
        else:
            base_concepts_time = 0
            model, accuracy_results = train_test_model(args, type=type)

        args_explanations = (model_path_ev, trainloader, testloader, model, config)

        local_metrics, global_metrics = compute_explanations(args_explanations)

        result_raw = merge_result_dicts(
            [accuracy_results, local_metrics, global_metrics]
        )

        result = {
            "type": type,
            "concepts_time": concepts_time + base_concepts_time,
            **result_raw,
        }

        result = flatten_dict(result)
        results.append(result)

        save_results(results, paths["results_dir"])


if __name__ == "__main__":
    main()
