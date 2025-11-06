import numpy as np
import os
import random
import torch 
import pickle

def get_device():
    if not torch.backends.mps.is_available():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return "mps"


def set_all_possible_seeds(s=0):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True


def load_pickle(folder, name, wholepath = None):
    # opens a pickle
    if wholepath:
        with open(wholepath, "rb") as f:  # os.path.sep = /, maybe better os.sep
            x = pickle.load(f) # pickleload(f)
    else:
        if not name.endswith('.pickle'):
            name += '.pickle'
        with open(
            folder + os.path.sep + name, "rb"
            ) as f:  # os.path.sep = /, maybe better os.sep
            x = pickle.load(f)  # pickleload(f)
    return x


def dump_pickle(name, thing):
    if not (name.endswith(".pickle") or name.endswith(".pkl")):
        name += ".pickle"
    # saves thing as a pickle named name
    with open(name, 'wb') as f:
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

