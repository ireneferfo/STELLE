"""
Dataset Loader - Handles loading and preprocessing of various time series datasets.
"""

import torch
import numpy as np
import os
import csv
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification

from .base_dataset import TrajectoryDataset
from .dataset_utils import remove_redundant_variables, convert_labels_to_numeric


class DatasetLoader:
    """
    Handles loading and preprocessing of time series datasets from various sources.
    """
    
    def __init__(self, data_directory: str = "./data"):
        """
        Initialize the dataset loader.
        
        Args:
            data_directory: Root directory for dataset storage
        """
        self.data_directory = data_directory
        os.makedirs(data_directory, exist_ok=True)

    def load_from_files(
        self, 
        dataset_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load dataset from CSV files in the standard format.
        
        Args:
            dataset_name: Name of the dataset subdirectory
            
        Returns:
            Tuple of (trajectories, labels)
        """
        dataset_path = os.path.join(self.data_directory, dataset_name)
        
        # Load labels
        labels_file = os.path.join(dataset_path, "labels.csv")
        with open(labels_file, "r") as file:
            label_reader = csv.reader(file)
            labels = [int(label) for label in next(label_reader)]

        # Load data
        data_file = os.path.join(dataset_path, "data.csv")
        data = []
        with open(data_file, "r") as file:
            data_reader = csv.reader(file)
            header = next(data_reader)
            num_variables = len(header)

            for row in data_reader:
                variable_data = [[] for _ in range(num_variables)]
                for i, value in enumerate(row):
                    variable_data[i % num_variables].append(float(value))
                data.append(variable_data)
                
        return torch.tensor(data), torch.tensor(labels)

    def load_from_aeon(
        self, 
        dataset_name: str
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load dataset from AEON classification dataset repository.
        
        Args:
            dataset_name: Name of the dataset in AEON repository
            
        Returns:
            Tuple of (trajectories, labels, metadata)
        """
        trajectories, labels, metadata = load_classification(
            dataset_name, return_metadata=True
        )
        return trajectories, labels, metadata

    def create_train_val_test_split(
        self,
        trajectories: np.ndarray,
        labels: List,
        dataset_name: str,
        num_classes: Optional[int] = None,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_seed: int = 0,
        remove_redundant: bool = True,
        correlation_threshold: float = 0.9,
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
        """
        Create train/validation/test splits with optional preprocessing.
        
        Args:
            trajectories: Input trajectories
            labels: Class labels
            dataset_name: Name identifier for the dataset
            num_classes: Number of classes (auto-detected if None)
            test_size: Proportion of data for test set
            validation_size: Proportion of training data for validation
            random_seed: Random seed for reproducibility
            remove_redundant: Whether to remove highly correlated variables
            correlation_threshold: Threshold for removing correlated variables
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        # Validate input data
        if np.isnan(trajectories).any():
            raise ValueError("Dataset contains missing values (NaN).")
            
        # Convert labels to numeric format
        numeric_labels, label_mapping = convert_labels_to_numeric(labels)
        numeric_labels = np.array(numeric_labels, dtype=np.int64)
        
        # Remove redundant variables if requested
        if remove_redundant:
            variable_indices_to_keep = remove_redundant_variables(
                trajectories, correlation_threshold
            )
            trajectories = trajectories[:, variable_indices_to_keep, :]
        
        # Initial train/test split
        train_trajectories, test_trajectories, train_labels, test_labels = train_test_split(
            trajectories, numeric_labels, 
            test_size=test_size, 
            random_state=random_seed, 
            stratify=numeric_labels
        )
        
        # Create test dataset
        test_dataset = TrajectoryDataset(
            data_name=dataset_name,
            trajectories=test_trajectories,
            labels=test_labels,
            num_classes=num_classes,
            label_map=label_mapping,
        )
        
        # Further split training data for validation if requested
        if validation_size > 0:
            train_trajectories, val_trajectories, train_labels, val_labels = train_test_split(
                train_trajectories, train_labels, 
                test_size=validation_size, 
                random_state=random_seed
            )
            
            train_dataset = TrajectoryDataset(
                data_name=dataset_name,
                trajectories=train_trajectories,
                labels=train_labels,
                num_classes=num_classes,
                label_map=label_mapping,
            )
            
            val_dataset = TrajectoryDataset(
                data_name=dataset_name,
                trajectories=val_trajectories,
                labels=val_labels,
                num_classes=num_classes,
                label_map=label_mapping,
            )
            
            return train_dataset, val_dataset, test_dataset
        else:
            # No validation split
            train_dataset = TrajectoryDataset(
                data_name=dataset_name,
                trajectories=train_trajectories,
                labels=train_labels,
                num_classes=num_classes,
                label_map=label_mapping,
            )
            
            return train_dataset, test_dataset

    def load_and_preprocess_dataset(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_seed: int = 0,
        remove_redundant: bool = True,
        correlation_threshold: float = 0.9,
        normalize: bool = True,
    ) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset, int, int, int, tuple]:
        """
        Complete pipeline for loading and preprocessing a dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            test_size: Proportion of data for test set
            validation_size: Proportion of training data for validation
            random_seed: Random seed for reproducibility
            remove_redundant: Whether to remove highly correlated variables
            correlation_threshold: Threshold for removing correlated variables
            normalize: Whether to normalize the datasets
            
        Returns:
            Tuple containing:
            - train_dataset: Training dataset
            - val_dataset: Validation dataset
            - test_dataset: Test dataset
            - num_variables: Number of variables in the data
            - num_classes: Number of classes
            - num_time_points: Number of time points
            - original_shape: Original shape of the data
        """
        print(f"\nLoading dataset: {dataset_name}")
        
        # Load dataset (try AEON first, then fall back to file loading)
        try:
            trajectories, labels, metadata = self.load_from_aeon(dataset_name)
            print("Loaded from AEON repository")
        except Exception as e:
            print(f"AEON loading failed: {e}. Trying file-based loading...")
            trajectories, labels = self.load_from_files(dataset_name)
            metadata = {}
        
        original_shape = trajectories.shape
        print(f"Original data shape: {original_shape}")
        
        # Create dataset splits
        train_dataset, val_dataset, test_dataset = self.create_train_val_test_split(
            trajectories=trajectories,
            labels=labels,
            dataset_name=dataset_name,
            test_size=test_size,
            validation_size=validation_size,
            random_seed=random_seed,
            remove_redundant=remove_redundant,
            correlation_threshold=correlation_threshold,
        )
        
        # Normalize datasets using training statistics
        if normalize:
            train_dataset.normalize()
            val_dataset.normalize(train_dataset.mean, train_dataset.std)
            test_dataset.normalize(train_dataset.mean, train_dataset.std)
        
        # Extract dataset properties
        num_variables = train_dataset.num_variables
        num_classes = train_dataset.num_classes
        num_time_points = train_dataset.num_time_points
        
        print(f"Preprocessing complete:")
        print(f"  - Variables: {num_variables}")
        print(f"  - Classes: {num_classes}")
        print(f"  - Time points: {num_time_points}")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
        
        return (
            train_dataset, 
            val_dataset, 
            test_dataset, 
            num_variables, 
            num_classes, 
            num_time_points, 
            original_shape
        )

