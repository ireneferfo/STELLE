import os
import csv
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification
from torch.utils.data import DataLoader

from ..data.data_generation import load_data_with_difficulty
from .base_dataset import TrajectoryDataset
from .dataset_utils import remove_redundant_variables, convert_labels_to_numeric
from ..utils import seed_worker


def get_dataset(
    dataname,
    config,
    dataset_info_path,
    **kwargs
):
    """Load and prepare dataset with train/val/test splits and dataloaders."""
    print(f'Getting dataset {dataname}...')
    # Load raw data
    X_train, y_train, X_test, y_test, num_classes, diff_params = _load_raw_data(
        dataname, config, **kwargs
    )
        
    # Preprocess data
    X_train, X_test, y_train, y_test, label_map = _preprocess_data(
        X_train, y_train, X_test, y_test
    )
    
    # Create validation split
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.2, random_state=config.seed
    )
    
    # Save dataset information
    _save_dataset_info(
        dataset_info_path, dataname, X_train, X_val, X_test,
        y_train, y_val, y_test, num_classes, diff_params
    )
    
    # Create datasets and normalize
    train_subset, val_subset, test_subset = _create_normalized_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test,
        dataname, label_map, num_classes
    )
    
    # Create dataloaders
    trainloader, valloader, testloader = _create_dataloaders(
        train_subset, val_subset, test_subset, config.bs, config.workers, config.seed)
    
    return trainloader, testloader, valloader


def _load_raw_data(dataname, config = None, **kwargs):
    """Load raw data from synthetic generation or aeon datasets."""
    if "synthetic" in dataname:
        X_train, y_train ,X_test, y_test, num_classes, diff_params = load_data_with_difficulty(
            dataname, config
        )
    else:
        base_data_dir = "paper_results/stl_baselines/datasets"
        folder_path = os.path.join(base_data_dir, dataname)
        diff_params = {}
        
        if os.path.isdir(folder_path):
            print(f"Loading dataset {dataname} from: {folder_path}")
            seed = config.seed if config else 0
            # load your data here
            X_train, X_test, y_train, y_test, num_classes = _load_data_from_folder(datafolder=folder_path, seed = seed, **kwargs)
            
        else:
            X_train, y_train, metadata = load_classification(
                dataname, split="train", return_metadata=True
            )
            X_test, y_test, _ = load_classification(
                dataname, split="test", return_metadata=True
            )
            num_classes = len(metadata["class_values"])
    
    return X_train, y_train, X_test, y_test, num_classes, diff_params


def _load_data_from_folder(seed = 0, test_size = 0.2, datafolder="./data"):
    with open(
        datafolder + os.path.sep + "labels.csv", "r"
    ) as f:
        label_reader = csv.reader(f)
        labels = next(label_reader)
        labels = [int(i) for i in labels]

    num_classes = len(np.unique(labels))
    data = []
    with open(datafolder + os.path.sep + "data.csv", "r") as f:
        data_reader = csv.reader(f)
        header = next(data_reader)
        n = len(header)

        for _, row in enumerate(data_reader):
            sublists = [[] for _ in range(n)]
            for i, item in enumerate(row):
                sublists[i % n].append(float(item))
            data.append(sublists)         
            
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return x_train, x_test, y_train, y_test, num_classes


def _preprocess_data(X_train, y_train, X_test, y_test):
    """Remove redundant variables and convert labels to numeric format."""
    # Remove redundant variables
    keep = remove_redundant_variables(X_train)
    X_train = X_train[:, keep, :]
    X_test = X_test[:, keep, :]
    
    # Convert labels to numeric
    numeric_labels, label_map = convert_labels_to_numeric(y_train)
    y_train = np.asarray(numeric_labels).astype(np.int64)
    y_test = np.asarray([label_map[label] for label in y_test]).astype(np.int64)
    
    return X_train, X_test, y_train, y_test, label_map


def _save_dataset_info(
    path, dataname, X_train, X_val, X_test,
    y_train, y_val, y_test, num_classes, diff_params
):
    """Save dataset information to file."""
    with open(path, "w") as f:
        lines = [
            f"dataname: {dataname}",
            f"X_train.shape: {X_train.shape}",
            f"X_val.shape: {X_val.shape}",
            f"X_test.shape: {X_test.shape}",
            f"num_classes: {num_classes}",
            f"train_subset: {np.bincount(y_train)}",
            f"val_subset: {np.bincount(y_val)}",
            f"test_subset: {np.bincount(y_test)}",
        ]
        print()
        for line in lines:
            f.write(line + "\n")
            print(line)
        print()
        if diff_params:
            f.write("\n=== Synthetic Data Generation Parameters ===\n")
            for key, value in diff_params.items():
                f.write(f"{key}: {value}\n")


def _create_normalized_datasets(
    X_train, y_train, X_val, y_val, X_test, y_test,
    dataname, label_map, num_classes
):
    """Create TrajectoryDataset objects and apply normalization."""
    # Create datasets
    train_subset = TrajectoryDataset(
        trajectories=X_train,
        labels=y_train,
        dataname=dataname,
        label_map=label_map,
        num_classes=num_classes,
    )
    
    val_subset = TrajectoryDataset(
        trajectories=X_val,
        labels=y_val,
        dataname=dataname,
        label_map=label_map,
        num_classes=num_classes,
    )
    
    test_subset = TrajectoryDataset(
        trajectories=X_test,
        labels=y_test,
        dataname=dataname,
        label_map=label_map,
        num_classes=num_classes,
    )
    
    # Normalize using training statistics
    train_subset.normalize()
    val_subset.normalize(train_subset.mean, train_subset.std)
    test_subset.normalize(train_subset.mean, train_subset.std)
    
    return train_subset, val_subset, test_subset


def _create_dataloaders(train_subset, val_subset, test_subset, bs, workers, seed):
    """Create DataLoader objects with proper seeding."""
    g = torch.Generator()
    g.manual_seed(seed)
    
    trainloader = DataLoader(
        train_subset,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    valloader = DataLoader(
        val_subset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    testloader = DataLoader(
        test_subset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    return trainloader, valloader, testloader