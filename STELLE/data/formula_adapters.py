"""
Formula Adapters - Adapt STL formulae to different dataset characteristics.
"""

from ..formula_generation.formula_utils import from_string_to_formula
from ..formula_generation.formula_manipulation import inverse_normalize_phis, normalize_phis
from .dataset_loader import get_dataset

def rescale_formulae(
    phis,
    dataname,
    base_data_dir="/Users/ireneferfoglia/Desktop/STELLE_workspace/paper_results/stl_baselines/datasets",
):
    if isinstance(phis, str):
        phis = [from_string_to_formula(phis)]
    elif not isinstance(phis, list):
        phis = list(phis)

    train_subset, _, _ = get_dataset(
        dataname,
        None,
        None,
        loaders=False,
        validation=False,
        base_data_dir=base_data_dir,
        verbose = False
    )

    mean, std = train_subset.mean, train_subset.std

    # pt = time_scaling(phis, points, 101)
    out = inverse_normalize_phis(mean, std, phis)

    return out

def scale_formulae(
    phis,
    dataname,
    base_data_dir="/Users/ireneferfoglia/Desktop/STELLE_workspace/paper_results/stl_baselines/datasets",
):
    if isinstance(phis, str):
        phis = [from_string_to_formula(phis)]
    elif not isinstance(phis, list):
        phis = list(phis)

    train_subset, _, _ = get_dataset(
        dataname,
        None,
        None,
        loaders=False,
        validation=False,
        base_data_dir=base_data_dir,
        verbose = False
    )

    mean, std = train_subset.mean, train_subset.std

    # pt = time_scaling(phis, points, 101)
    out = normalize_phis(mean, std, phis)

    return out