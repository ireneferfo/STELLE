import torch
import numpy as np
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification
from torch.utils.data import DataLoader

from ..data.data_generation import load_data_with_difficulty
from .base_dataset import TrajectoryDataset
from .dataset_utils import remove_redundant_variables, convert_labels_to_numeric


def get_dataset(
    dataname,
    config,
    dataset_info_path,
):
    """Load and prepare dataset with train/val/test splits and dataloaders."""
    # Load raw data
    X_train, y_train, X_test, y_test, num_classes = _load_raw_data(
        dataname, config
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
        y_train, y_val, y_test, num_classes
    )
    
    # Create datasets and normalize
    train_subset, val_subset, test_subset = _create_normalized_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test,
        dataname, label_map, num_classes
    )
    
    # Create dataloaders
    trainloader, valloader, testloader = _create_dataloaders(
        train_subset, val_subset, test_subset, config.bs, config.pll, config.seed)
    
    return trainloader, testloader, valloader


def _load_raw_data(dataname, config):
    """Load raw data from synthetic generation or aeon datasets."""
    if "synthetic" in dataname:
        X_train, y_train ,X_test, y_test, num_classes = load_data_with_difficulty(
            dataname, config
        )
    else:
        X_train, y_train, metadata = load_classification(
            dataname, split="train", return_metadata=True
        )
        X_test, y_test, _ = load_classification(
            dataname, split="test", return_metadata=True
        )
        num_classes = len(metadata["class_values"])
    
    return X_train, y_train, X_test, y_test, num_classes


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
    y_train, y_val, y_test, num_classes
):
    """Save dataset information to file."""
    with open(path, "w") as f:
        f.write(f"dataname: {dataname}\n")
        f.write(f"X_train.shape: {X_train.shape}\n")
        f.write(f"X_val.shape: {X_val.shape}\n")
        f.write(f"X_test.shape: {X_test.shape}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"train_subset: {np.bincount(y_train)}\n")
        f.write(f"val_subset: {np.bincount(y_val)}\n")
        f.write(f"test_subset: {np.bincount(y_test)}\n")


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
        worker_init_fn=seed,
        generator=g,
    )
    
    valloader = DataLoader(
        val_subset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed,
        generator=g,
    )
    
    testloader = DataLoader(
        test_subset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed,
        generator=g,
    )
    
    return trainloader, valloader, testloader