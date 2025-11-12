import sys
import os
from itertools import product
from dataclasses import dataclass, replace

# ensure workspace root is on sys.path so the local `STELLE` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from STELLE.data.dataset_loader import get_dataset
from STELLE.explanations.explanation_utils import compute_explanations
from STELLE.kernels.kernel_utils import set_kernels_and_concepts
from STELLE.model.model_utils import train_test_model
from STELLE.utils import flatten_dict, setup_environment, save_run_settings, save_results, parse_arguments, setup_paths, merge_result_dicts

#! da finire

@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments."""
    # Synthetic data parameters
    n_train: int = 100
    n_test: int = 80
    n_vars: int = 2
    series_length: int = 30
    num_classes: int = 3
    
    # Fixed parameters
    seed: int = 0
    pll: int = 2
    workers: int = 0
    samples: int = 500
    epochs: int = 3
    cf: int = 50
    patience: int = 5
    val_every_n_epochs: int = 1
    verbose: int = 1
    logging: bool = False
    
    # Kernel parameters
    normalize_kernel: bool = False
    exp_kernel: bool = False
    normalize_rhotau: bool = True
    exp_rhotau: bool = True
    
    # Concept parameters
    t: float = 1.0
    n_vars_formulae: int = 1
    creation_mode: str = "one"
    dim_concepts: int = 100
    min_total: int = 100
    imp_t_l: float = 0
    imp_t_g: float = 0
    t_k: float = 0.8
    
    # Training parameters
    d: float = 0.1
    bs: int = 32
    lr: float = 1e-4
    init_eps: float = 1
    activation_str: str = "relu"
    backprop_method: str = "ig"
    init_crel: float = 1
    h: int = 256
    n_layers: int = 1


def main():
    args = parse_arguments()
    config = ExperimentConfig()
    device = setup_environment(config.seed)
    base_path = "ablation_tests/results/kernels_normexp/"
    model_path = "ablation_tests/results/kernel_checkpoints/"
    paths = setup_paths(base_path, model_path, args, args.dataset, config)
    os.makedirs(paths['results_dir'], exist_ok=True)
    os.makedirs(paths['model_path_og'], exist_ok=True)
    
    trainloader, valloader, testloader = get_dataset(args.dataset, config, paths['dataset_info_path'])
    
    print(f"Run ID: {paths['run_id']}\n")
    save_run_settings(paths['results_dir'], **locals())

    results = []
    
    for i, j, k, l in product([False, True], repeat=4):
        print(f'\n>>>>>>>>>>>>> norm/exp stl/traj = {i, j, k, l} >>>>>>>>>>>>>\n')
        
        config_i = replace(config,normalize_kernel=i, exp_kernel=j, normalize_rhotau=k, exp_rhotau=l)
        
        kernel, _ = set_kernels_and_concepts(trainloader.dataset, paths['phis_path_og'], config_i)

        model_id = (
            f"seed_{config_i.seed}_{config_i.lr}_{config_i.init_crel}_{config.init_eps}_{config_i.h}_"
            f"{config_i.n_layers}_bs{config.bs}_nstl{i}_estl{j}_ntraj{k}_etraj{l}"
        )
        # attach to model for later reference and debugging
        print(f"Model ID: {model_id}")
        model_path_ev = os.path.join(paths['model_path_og'], f"{model_id}.pt")

        args = (kernel, trainloader, valloader, testloader, model_path_ev, config_i)
        model, accuracy_results = train_test_model(args)

        args_explanations = (model_path_ev, trainloader, testloader, model, config_i)
        local_metrics, global_metrics = compute_explanations(args_explanations)
        
        result_raw = merge_result_dicts([accuracy_results, local_metrics, global_metrics])
        
        result = {
                'normalize_kernel': i,
                'exp_kernel': j,
                'normalize_rhotau':k,
                'exp_rhotau':l
                **result_raw
            }
        
        result = flatten_dict(result)
        results.append(result)
        
        save_results(results, paths['results_dir'])


if __name__ == "__main__":
    main()
