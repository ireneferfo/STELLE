"""
Dataset Utilities - Utility functions for dataset preprocessing and manipulation.
"""

import numpy as np
from typing import List, Tuple, Dict


def remove_redundant_variables(
    trajectories: np.ndarray, 
    correlation_threshold: float = 0.9
) -> List[int]:
    """
    Identify and remove highly correlated variables from 3D time series data.
    
    Args:
        trajectories: Array of shape (samples, variables, time_points)
        correlation_threshold: Threshold for considering variables redundant
        
    Returns:
        List of variable indices to keep
    """
    _, num_variables, _ = trajectories.shape
    
    # Flatten to (variables, samples * time_points)
    flattened_trajectories = trajectories.transpose(1, 0, 2).reshape(num_variables, -1)
    
    # Center the data
    centered_data = flattened_trajectories - flattened_trajectories.mean(axis=1, keepdims=True)
    
    # Compute covariance matrix
    covariance = centered_data @ centered_data.T / (centered_data.shape[1] - 1)
    
    # Compute standard deviations
    std_devs = centered_data.std(axis=1, keepdims=True)
    
    # Compute correlation matrix
    correlation_matrix = covariance / (std_devs @ std_devs.T + 1e-8)
    
    # Identify redundant variables
    redundant_indices = set()
    
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            if abs(correlation_matrix[i, j]) > correlation_threshold:
                # Keep the variable with higher mean absolute value
                mean_abs_i = np.mean(np.abs(flattened_trajectories[i]))
                mean_abs_j = np.mean(np.abs(flattened_trajectories[j]))
                
                if mean_abs_i < mean_abs_j:
                    redundant_indices.add(i)
                else:
                    redundant_indices.add(j)
    
    # Return indices of variables to keep
    return [idx for idx in range(num_variables) if idx not in redundant_indices]


def convert_labels_to_numeric(labels: List) -> Tuple[List[int], Dict]:
    """
    Convert arbitrary labels to numeric values with consistent mapping.
    
    Args:
        labels: List of original labels (can be strings, integers, etc.)
        
    Returns:
        Tuple of (numeric_labels, label_mapping)
    """
    unique_labels = sorted(set(labels))  # Ensure consistent ordering
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_mapping[label] for label in labels]
    
    return numeric_labels, label_mapping


def convert_numeric_to_labels(
    numeric_labels: List[int], 
    label_mapping: Dict
) -> List:
    """
    Convert numeric labels back to original labels using the mapping.
    
    Args:
        numeric_labels: List of numeric labels
        label_mapping: Original mapping from labels to numbers
        
    Returns:
        List of original labels
    """
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    return [reverse_mapping[number] for number in numeric_labels]


def validate_dataset(
    trajectories: np.ndarray, 
    labels: List,
    check_missing: bool = True,
    check_infinite: bool = True,
    check_shape: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate dataset for common issues.
    
    Args:
        trajectories: Input trajectories
        labels: Class labels
        check_missing: Check for NaN values
        check_infinite: Check for infinite values
        check_shape: Check for consistent shapes
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for missing values
    if check_missing and np.isnan(trajectories).any():
        issues.append("Dataset contains missing values (NaN)")
    
    # Check for infinite values
    if check_infinite and np.isinf(trajectories).any():
        issues.append("Dataset contains infinite values")
    
    # Check shape consistency
    if check_shape:
        if len(trajectories) != len(labels):
            issues.append(f"Number of trajectories ({len(trajectories)}) doesn't match number of labels ({len(labels)})")
        
        if trajectories.ndim != 3:
            issues.append(f"Expected 3D array, got {trajectories.ndim}D")
    
    # Check label consistency
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        issues.append(f"Dataset has only {len(unique_labels)} unique class(es)")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def compute_dataset_statistics(trajectories: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute comprehensive statistics for a dataset.
    
    Args:
        trajectories: Input trajectories
        labels: Class labels
        
    Returns:
        Dictionary containing dataset statistics
    """
    num_samples, num_variables, num_time_points = trajectories.shape
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    statistics = {
        "num_samples": num_samples,
        "num_variables": num_variables,
        "num_time_points": num_time_points,
        "num_classes": len(unique_labels),
        "class_distribution": dict(zip(unique_labels, label_counts)),
        "variable_statistics": {},
    }
    
    # Compute statistics per variable
    for var_idx in range(num_variables):
        variable_data = trajectories[:, var_idx, :]
        statistics["variable_statistics"][var_idx] = {
            "mean": np.mean(variable_data),
            "std": np.std(variable_data),
            "min": np.min(variable_data),
            "max": np.max(variable_data),
            "range": np.ptp(variable_data),
        }
    
    return statistics


def print_dataset_summary(statistics: Dict) -> None:
    """
    Print a formatted summary of dataset statistics.
    
    Args:
        statistics: Statistics dictionary from compute_dataset_statistics
    """
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Samples: {statistics['num_samples']}")
    print(f"Variables: {statistics['num_variables']}")
    print(f"Time Points: {statistics['num_time_points']}")
    print(f"Classes: {statistics['num_classes']}")
    
    print("\nClass Distribution:")
    for label, count in statistics['class_distribution'].items():
        percentage = (count / statistics['num_samples']) * 100
        print(f"  Class {label}: {count} samples ({percentage:.1f}%)")
    
    print("\nVariable Statistics:")
    for var_idx, stats in statistics['variable_statistics'].items():
        print(f"  Variable {var_idx}:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print("="*50)