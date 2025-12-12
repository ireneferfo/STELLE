import sys
import os
import torch
import gc
import tempfile
from dataclasses import dataclass

# ensure workspace root is on sys.path so the local `STELLE` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from STELLE.utils import (
    setup_environment,
    parse_arguments,
    setup_paths,
)

from STELLE.kernels.base_measure import BaseMeasure
from STELLE.kernels.kernel_utils import _validate_parameters, _create_concepts_path, _create_stl_kernel
from STELLE.formula_generation.formula_manager import FormulaManager


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments."""

    # Fixed parameters
    seed: int = 0
    pll: int = 12
    samples: int = 3000

    # Kernel parameters
    normalize_kernel: bool = True
    exp_kernel: bool = True
    normalize_rhotau: bool = True
    exp_rhotau: bool = False

    # Concept parameters
    t: float = 0.98
    nvars_formulae: int = 1
    creation_mode: str = "all"
    dim_concepts: int = 10000
    min_total: int = 1000



def main():
    args = parse_arguments()
    config = ExperimentConfig()
    device = setup_environment(config.seed)
    base_path = tempfile.mkdtemp()
    model_path = tempfile.mkdtemp()
    paths = setup_paths(base_path, model_path, args, '', config)

    for nvars in [args.dataset]:
        nvars = int(nvars)
        print(f"\n>>>>>>>>>>>>> NVARS = {nvars} >>>>>>>>>>>>>\n")

        # Validate and adjust parameters
        nvars_formulae, creation_mode = _validate_parameters(
            nvars, config.nvars_formulae, config.creation_mode
        )

        # Setup paths
        phis_path = _create_concepts_path(
            paths["phis_path_og"], creation_mode, nvars_formulae, nvars, config.t
        )

        # Initialize components
        mu = BaseMeasure(device=device)
        stlkernel = _create_stl_kernel(
            mu, nvars, config.samples, config.normalize_kernel, config.exp_kernel
        )

        # Generate and scale concepts
        formula_manager = FormulaManager(
            nvars, stlkernel, config.pll, config.t, nvars_formulae, device=device
        )
        
        print(
            f"Getting {config.dim_concepts} concepts with {config.t=} and {config.creation_mode=}..."
        )

        _ = formula_manager.get_formulae(
            creation_mode, config.dim_concepts, phis_path, "concepts", config.seed
        )
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
    print('\n\nDONE\n')



if __name__ == "__main__":
    main()
