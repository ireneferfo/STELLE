import os
import math
from dataclasses import dataclass, replace
import torch

from STELLE.data.dataset_loader import get_dataset, load_UCR_from_idx, load_UEA_from_idx
from STELLE.kernels.kernel_utils import set_kernels_and_concepts
from STELLE.model.model_utils import train_test_model
from STELLE.explanations.explanation_utils import compute_explanations
from STELLE.utils import (
    setup_environment,
    save_results,
    save_run_settings,
    parse_arguments,
    setup_paths,
    flatten_dict,
)
from params import get_params

@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""

    # Synthetic data parameters
    n_train: int = 500
    n_test: int = 200
    nvars: int = 5
    series_length: int = 100
    num_classes: int = 4

    # Fixed parameters
    seed: int = 0
    pll: int = 8
    workers: int = 2
    samples: int = 3000
    epochs: int = 500
    cf: int = 10
    patience: int = 5
    val_every_n_epochs: int = 2
    verbose: int = 5
    logging: bool = False

    # Kernel parameters - can be tuned
    normalize_kernel: bool = True
    exp_kernel: bool = True
    normalize_rhotau: bool = True
    exp_rhotau: bool = False

    # Concept parameters - can be tuned
    t: float = 0.98
    nvars_formulae: int = 1
    creation_mode: str = "all"
    dim_concepts: int = 1000
    min_total: int = 1000
    
    imp_t_l: float = 0
    imp_t_g: float = 0
    t_k: float = 0.8
    explanation_operation: str | None = "mean"

    # Training parameters - can be tuned
    d: float = 0.2
    bs: int = 32
    lr: float = 1e-4
    init_eps: float = math.exp(1)
    activation_str: str = "relu"
    backprop_method: str = "ig"
    init_crel: float = 1
    h: int = 256
    n_layers: int = 1
    
    
def main():
    args = parse_arguments()
    base_config = ExperimentConfig()
    _ = setup_environment(base_config.seed)
    base_path = "paper_results/results"
    model_path = "results"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    print(f'{task_id=}')
    
    if task_id is None: 
        dataset = args.dataset
    else:
        if args.dataset in ['uni', 'univ', 'univariate']:
            dataset = load_UCR_from_idx(int(task_id)-1)
        elif args.dataset in ['multi', 'multiv', 'multivariate']:
            dataset = load_UEA_from_idx(int(task_id)-1)
                 
    paths = setup_paths(base_path, model_path, args, dataset, base_config)
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["model_path_og"], exist_ok=True)
    
    params = get_params(dataset)
    base_config = replace(base_config, **params)
    print(f"Run ID: {paths['run_id']}\n")
    save_run_settings(paths["results_dir"], base_config, **locals())
    
    results =[]
    
    for fold in range(5):
        print(f'\n >>>>>>>>>>>>>>> FOLD {fold} >>>>>>>>>>>>>>>\n')
        model_path_fold = os.path.join(paths["model_path_og"], f"fold_{fold}/")
        os.makedirs(model_path_fold, exist_ok=True)
        trainloader, valloader, testloader, base_config  = get_dataset(
            dataset, base_config, paths["dataset_info_path"], validation=False, fold = fold
        )
        kernel, _, _ = set_kernels_and_concepts(
            trainloader.dataset, paths["phis_path_og"], base_config
        ) 
        for seed in range(1):
            torch.cuda.empty_cache()
            
            config = replace(base_config, seed=seed)
            model_id = (
                    f"seed_{config.seed}_{config.lr}_{config.init_crel}_{config.init_eps}_{config.h}_"
                    f"{config.n_layers}_bs{config.bs}"
                )
            print(f"Model ID: {model_id}")
            model_path_ev = os.path.join(model_path_fold, f"{model_id}.pt")
            args = (kernel, trainloader, valloader, testloader, model_path_ev, config)

            model, accuracy_results = train_test_model(args, arch_type="base")
            accuracy_results = flatten_dict(accuracy_results)
            args_explanations = (model_path_ev, trainloader, testloader, model, config)
            local_metrics, global_metrics = {}, {}# compute_explanations(args_explanations, save = True, globals = True, locals = False, verbose = True)
            
            result = {
                    "fold": fold,
                    "seed": seed,
                    **accuracy_results,
                    **local_metrics,
                    **global_metrics
                }
            
            result = flatten_dict(result)
            results.append(result)

            save_results(results, paths["results_dir"])
            
            del model
    
if __name__ == "__main__":
    main()