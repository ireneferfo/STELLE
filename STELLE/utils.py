from dataclasses import asdict
import numpy as np
import os
import random
import torch
import pickle
import json
import csv
import argparse
import tempfile
from typing import Dict, Any, List
from datetime import datetime


def get_device():
    if not torch.backends.mps.is_available():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return "mps"


def setup_environment(seed: int) -> torch.device:
    """Initialize device and set deterministic behaviour."""
    # for absolute reproducibility avoid: AMP, benchmark=True, pin_memory=True, persistent_workers=True
    device = get_device()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy("file_system")
    set_all_possible_seeds(seed)

    print("DEVICE:", device)
    print("Assigned GPU(s):", os.environ.get("CUDA_VISIBLE_DEVICES"))

    return device

def seed_worker(worker_id):
    worker_seed = 0
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        
def merge_result_dicts(dicts: list):
    result_raw = {}
    # merge dictionaries; later ones override earlier keys on conflict
    for d in dicts:
        if isinstance(d, dict):
            result_raw.update(d)
        else:
            result_raw[f"{d=}".split("=")[0]] = d
    return result_raw


def setup_paths(
    base_path: str, model_path: str, args: argparse.Namespace, dataname: str, config
) -> Dict[str, str]:
    """Setup all required directory paths."""
    if args.temp:
        base_path = tempfile.mkdtemp()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Adjust dataset name for synthetic data
    if dataname == "synthetic":
        dataname = (
            f"synthetic_{config.n_train}-{config.n_test}_v{config.nvars}_"
            f"p{config.series_length}_c{config.num_classes}"
        )

    results_dir = os.path.join(base_path, dataname, run_id)

    # Setup phis path
    if args.tempphis:
        phis_path_og = tempfile.mkdtemp()
    else:
        phis_path_og = (
            os.path.join(os.environ["WORK"], "STELLE/phis/")
            if args.demetra
            else "phis/"
        )

    # Setup model checkpoint path
    if args.nocheckpoints or args.temp:
        model_path_og = tempfile.mkdtemp()
    else:
        model_path_og = (
            os.path.join(
                os.environ["WORK"], f"STELLE/{model_path}{dataname}/checkpoints/"
            )
            if args.demetra
            else os.path.join(model_path, dataname, "checkpoints/")
        )

    return {
        "results_dir": results_dir,
        "dataset_info_path": results_dir + "/info.txt",
        "phis_path_og": phis_path_og,
        "model_path_og": model_path_og,
        "run_id": run_id,
        "dataname": dataname,
    }


def save_results(results: List[Dict[str, Any]], results_dir: str):
    """Save experiment results to CSV file."""
    if not results:
        return
    
    # Flatten all results and normalize 'loss' to 'avg_valloss'
    flattened_results = []
    for result in results:        
        # Normalize 'loss' to 'avg_valloss'
        if 'loss' in result and 'avg_valloss' not in result:
            result['avg_valloss'] = result.pop('loss')
        elif 'loss' in result and 'avg_valloss' in result:
            result.pop('loss')
        
        flattened_results.append(result)
    
    # Collect all unique keys across all results
    # all_keys = set()
    # for result in flattened_results:
    #     all_keys.update(result.keys())
    keys = results[0].keys()
    
    print()
    print(result.keys())
    print()
    
    csv_path = os.path.join(results_dir, "results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(flattened_results)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ablation tests")
    parser.add_argument(
        "dataset", type=str, nargs="?", default="synthetic", help="Dataset name"
    )
    parser.add_argument(
        "-temp",
        action="store_true",
        default=False,
        help="Use temporary directory for results",
    )
    parser.add_argument(
        "-tempphis",
        action="store_true",
        default=False,
        help="Use temporary directory for concepts",
    )
    parser.add_argument(
        "-nocheckpoints",
        action="store_true",
        default=False,
        help="Don't save model checkpoints",
    )
    parser.add_argument(
        "-demetra", action="store_true", default=False, help="Use Demetra cluster paths"
    )

    return parser.parse_args()


def save_run_settings(results_dir, config, **kwargs):
    """Save run settings including ExperimentConfig and additional kwargs."""
    # Convert dataclass to dict
    config_dict = asdict(config)
    
    # Combine config and kwargs
    constants = {**config_dict, **kwargs}
    
    info_file_path = results_dir + "/run_info.txt"
    
    # Filter out non-serializable objects
    serializable_constants = {}
    for key, value in constants.items():
        try:
            json.dumps(value)  # Test if serializable
            serializable_constants[key] = value
        except (TypeError, ValueError):
            serializable_constants[key] = f"<{type(value).__name__} - not serializable>"
    
    with open(info_file_path, "w") as file:
        file.write(json.dumps(serializable_constants, indent=2))


def set_all_possible_seeds(s=0):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True


def load_pickle(folder, name, wholepath=None):
    # opens a pickle
    if wholepath:
        with open(wholepath, "rb") as f:  # os.path.sep = /, maybe better os.sep
            x = pickle.load(f)  # pickleload(f)
    else:
        if not name.endswith(".pickle"):
            name += ".pickle"
        with open(
            folder + os.path.sep + name, "rb"
        ) as f:  # os.path.sep = /, maybe better os.sep
            x = pickle.load(f)  # pickleload(f)
    return x


def dump_pickle(name, thing):
    if not (name.endswith(".pickle") or name.endswith(".pkl")):
        name += ".pickle"
    # saves thing as a pickle named name
    with open(name, "wb") as f:
        pickle.dump(thing, f)


def chunks(phis, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(phis), n):
        yield phis[i : i + n]


def save_file(file, filename, path="formulae_sets/"):
    os.makedirs(path, exist_ok=True)
    with open(path + filename, "wb") as f:
        pickle.dump(file, f)


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary structure."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert tuples/lists to individual columns
            for i, item in enumerate(v):
                items.append((f"{new_key}_{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def round_dict_values(d, decimals=3):
    """
    Recursively rounds all numeric values in a dictionary (including nested dicts).

    Args:
        d: Input dictionary (can contain nested dictionaries)
        decimals: Number of decimal places to round to (default: 3)

    Returns:
        New dictionary with rounded values
    """
    rounded_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively round nested dictionaries
            rounded_dict[key] = round_dict_values(value, decimals)
        elif isinstance(value, (int, float)):
            # Round numeric values
            rounded_dict[key] = round(value, decimals)
        else:
            # Preserve non-numeric values
            rounded_dict[key] = value
    return rounded_dict
