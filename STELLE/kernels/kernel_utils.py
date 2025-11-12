import torch
import gc
import os
import warnings

from ..utils import get_device
from .base_measure import BaseMeasure
from .stl_kernel import StlKernel
from .trajectory_kernel import TrajectoryKernel
from ..formula_generation.stl_generator import STLFormulaGenerator
from ..formula_generation.formula_manager import FormulaManager


def set_kernels_and_concepts(
    train_subset, phis_path_og, config
):
    """Initialize kernels and generate concepts for STL-based learning."""
    device = get_device()
    n_vars = train_subset.num_variables
    
    # Validate and adjust parameters
    n_vars_formulae, creation_mode = _validate_parameters(
        config.n_vars, config.n_vars_formulae, config.creation_mode
    )
    
    # Setup paths
    phis_path = _create_concepts_path(phis_path_og, creation_mode, n_vars_formulae, n_vars, config.t)
    
    print('Setting kernels...')
    # Initialize components
    mu = BaseMeasure(device=device)
    sampler = STLFormulaGenerator(max_variables=n_vars_formulae)
    stlkernel = _create_stl_kernel(mu, n_vars, config.samples, config.normalize, config.exp_kernel)
    kernel = _create_trajectory_kernel(mu, config)
    
    print(f'Getting {config.dim_concepts} concepts with {config.t=} and {config.creation_mode=}...')
    # Generate and scale concepts
    formula_manager = FormulaManager(
        n_vars, sampler, stlkernel, config.pll, config.t, n_vars_formulae, device=device
    )
    
    concepts, rhos1, selfk1, total_time = formula_manager.get_formulae(
        creation_mode, config.dim_concepts, phis_path, "concepts", config.seed
    )
    
    # Configure kernel with scaled concepts
    _configure_kernel(kernel, train_subset, concepts, rhos1, selfk1)
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    return kernel, formula_manager, total_time


def _validate_parameters(n_vars, n_vars_formulae, creation_mode):
    """Validate and adjust formula generation parameters."""
    # Check if n_vars_formulae exceeds available variables
    if n_vars < n_vars_formulae:
        warnings.warn(
            f"The dataset has {n_vars} variables. Attempting to create formulae "
            f"with more. Setting n_vars_formulae = {n_vars}."
        )
        n_vars_formulae = n_vars
    
    # Convert creation mode to numeric
    creation_mode_num = 0 if creation_mode == "all" else 1
    
    # Handle single-variable case
    if n_vars == 1:
        creation_mode_num = 0
        warnings.warn(
            "The dataset has 1 variable, thus creation_mode = 0 or 1 "
            "is the same case. Collapsing to 0."
        )
    
    return n_vars_formulae, creation_mode_num


def _create_concepts_path(phis_path_og, creation_mode, n_vars_formulae, n_vars, t):
    """Create and return the path for storing concepts."""
    phis_path = (
        f"{phis_path_og}{creation_mode}/{n_vars_formulae}_fvars/"
        f"{n_vars}_n_vars/t_{t}/"
    )
    os.makedirs(phis_path, exist_ok=True)
    return phis_path


def _create_stl_kernel(mu, n_vars, samples, normalize, exp_kernel):
    """Create and configure STL kernel."""
    return StlKernel(
        mu,
        varn=n_vars,
        samples=samples,
        vectorize=True,
        normalize=normalize,
        exp_kernel=exp_kernel,
    )


def _create_trajectory_kernel(
    mu, config
):
    """Create and configure trajectory kernel."""
    return TrajectoryKernel(
        mu,
        varn=config.n_vars,
        points=config.series_length,
        samples=config.samples,
        normalize=config.normalize,
        exp_kernel=config.exp_kernel,
        exp_rhotau=config.exp_rhotau,
        normalize_rhotau=config.normalize_rhotau,
    )


def _configure_kernel(kernel, train_subset, concepts, rhos1, selfk1):
    """Scale concepts and configure kernel with generated formulae."""
    scaled_concepts = train_subset.time_scaling(concepts)
    kernel.phis = scaled_concepts
    kernel.rhos_phi = rhos1
    kernel.selfk_phi = selfk1