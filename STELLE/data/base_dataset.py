"""
Base Dataset Class - Core functionality for trajectory datasets with STL formula support.
"""

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import numpy as np

from STELLE.formula_generation.formula_manipulation import time_scaling


class TrajectoryDataset(Dataset):
    """
    Dataset class for multivariate time series data with STL formula support.
    
    Handles trajectories of shape (samples, variables, time_points) with
    support for normalization, time scaling, and formula adaptation.
    """
    
    def __init__(
        self,
        dataname: str,
        trajectories: torch.Tensor,
        labels: torch.Tensor,
        num_classes: Optional[int] = None,
        label_map: Optional[Dict] = None,
    ):
        """
        Initialize the trajectory dataset.
        
        Args:
            dataname: Name identifier for the dataset
            trajectories: Tensor of shape (samples, variables, time_points)
            labels: Tensor of class labels
            num_classes: Number of classes (auto-detected if None)
            label_map: Mapping from original labels to numeric labels
        """
        self.dataname = dataname
        self.trajectories = torch.tensor(trajectories) if not torch.is_tensor(trajectories) else trajectories
        self.labels = torch.tensor(labels) if not torch.is_tensor(labels) else labels
        self.label_map = label_map
        
        self.points = trajectories.shape[-1]
        self.num_classes = num_classes if num_classes else len(torch.unique(self.labels))
        self.one_hot_labels = torch.nn.functional.one_hot(self.labels, num_classes=self.num_classes)
        
        self.num_variables = trajectories.shape[1]
        self.num_time_points = trajectories.shape[-1]
        
        # Normalization state
        self.mean = torch.zeros(self.num_variables)
        self.std = torch.zeros(self.num_variables)
        self.is_normalized = False
        
        # Class distribution
        self.class_distribution = self._compute_class_distribution()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.trajectories.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index."""
        return self.trajectories[idx], self.one_hot_labels[idx]

    def _compute_class_distribution(self) -> torch.Tensor:
        """Compute the distribution of samples across classes."""
        return torch.bincount(self.labels)

    def get_subset(self, start_idx: int, end_idx: int) -> 'TrajectoryDataset':
        """
        Create a subset of the dataset.
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            New TrajectoryDataset containing the specified subset
        """
        return TrajectoryDataset(
            dataname=self.dataname,
            trajectories=self.trajectories[start_idx:end_idx].clone().detach(),
            labels=self.labels[start_idx:end_idx].clone().detach(),
            num_classes=self.num_classes,
        )

    def split_by_class(self) -> Dict[int, List[torch.Tensor]]:
        """
        Split trajectories by class labels.
        
        Returns:
            Dictionary mapping class indices to lists of trajectories
        """
        class_indices = torch.argmax(self.one_hot_labels, dim=1)
        return {
            class_idx: [
                trajectory for i, trajectory in enumerate(self.trajectories) 
                if class_indices[i] == class_idx
            ]
            for class_idx in range(self.num_classes)
        }

    def normalize(
        self, 
        mean: Optional[torch.Tensor] = None, 
        std: Optional[torch.Tensor] = None,
        use_local_stats: bool = False
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Normalize the dataset variables.
        
        Args:
            mean: Precomputed means for each variable (if None, computed from data)
            std: Precomputed standard deviations for each variable (if None, computed from data)
            use_local_stats: If True, return statistics without modifying dataset
            
        Returns:
            If use_local_stats is True, returns (mean, std) without modifying data
        """
        if mean is None or std is None:
            mean = torch.tensor([
                self.trajectories[:, var_idx, :].mean() 
                for var_idx in range(self.num_variables)
            ])
            std = torch.tensor([
                self.trajectories[:, var_idx, :].std() 
                for var_idx in range(self.num_variables)
            ])
        
        if use_local_stats:
            return mean, std
        
        self.mean = mean
        self.std = std
        
        # Reshape statistics to match trajectory dimensions
        mean_reshaped, std_reshaped = self._reshape_statistics()
        
        # Apply normalization
        self.trajectories = (self.trajectories - mean_reshaped) / std_reshaped
        self.is_normalized = True

    def denormalize(self, use_local_stats: bool = False) -> Optional[torch.Tensor]:
        """
        Reverse the normalization transformation.
        
        Args:
            use_local_stats: If True, return denormalized data without modifying dataset
            
        Returns:
            If use_local_stats is True, returns denormalized trajectories
        """
        mean_reshaped, std_reshaped = self._reshape_statistics()
        
        if use_local_stats:
            return (self.trajectories * std_reshaped) + mean_reshaped
        
        self.trajectories = (self.trajectories * std_reshaped) + mean_reshaped
        self.is_normalized = False

    def _reshape_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape mean and std to match trajectory dimensions.
        
        Returns:
            Tuple of (mean_reshaped, std_reshaped) with same shape as trajectories
        """
        mean_reshaped = torch.cat([
            self.mean[var_idx].repeat(self.trajectories.shape[0], self.trajectories.shape[-1]).unsqueeze(1)
            for var_idx in range(self.num_variables)
        ], dim=1)
        
        std_reshaped = torch.cat([
            self.std[var_idx].repeat(self.trajectories.shape[0], self.trajectories.shape[-1]).unsqueeze(1)
            for var_idx in range(self.num_variables)
        ], dim=1)
        
        return mean_reshaped, std_reshaped

    def time_scaling(self, phis, phi_timespan=101):
        return time_scaling(phis, self.points, phi_timespan)
    
    def plot_trajectories(
        self, 
        figsize: Tuple[int, int] = (10, 5),
        max_trajectories: Optional[int] = None
    ) -> None:
        """
        Plot the trajectories in the dataset.
        
        Args:
            figsize: Figure size as (width, height)
            max_trajectories: Maximum number of trajectories to plot (None for all)
        """
        plt.figure(figsize=figsize)
        
        # Determine which trajectories to plot
        plot_indices = range(len(self))
        if max_trajectories is not None:
            plot_indices = np.random.choice(len(self), min(max_trajectories, len(self)), replace=False)
        
        # Plot trajectories
        for idx in plot_indices:
            trajectory = self.trajectories[idx]
            class_label = self.labels[idx].item()
            color = f"C{class_label}"
            label = f"Class {class_label}"
            
            if self.num_variables == 1:
                plt.plot(trajectory[0], color=color, linewidth=0.3, label=label)
            elif self.num_variables == 2:
                plt.plot(trajectory[0], trajectory[1], color=color, linewidth=0.4, label=label)
        
        # Configure plot
        plt.grid(True)
        
        if self.num_variables == 1:
            plt.xlabel("Time Step", fontsize=14)
            plt.ylabel("Variable x₀", fontsize=14)
            plt.xticks(list(range(0, self.num_time_points, 10)), fontsize=10)
        elif self.num_variables == 2:
            plt.xlabel("Variable x₀", fontsize=14)
            plt.ylabel("Variable x₁", fontsize=14)
        
        plt.yticks(fontsize=14)
        
        # Create legend with unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        plt.legend(unique_labels.values(), unique_labels.keys(), prop={"size": 14})
        
        plt.tight_layout()
        plt.show()

        """
        Legacy plotting method with original color scheme and styling.
        """
        plt.figure(figsize=figsize)

        for i in range(len(self)):
            color = "tab:red" if self.labels[i] == 1 else "blue"
            label = "anomaly" if self.labels[i] == 1 else "regular"
            
            if self.num_variables == 1:
                plt.plot(self.trajectories[i][0], color=color, linewidth=0.3, label=label)
            elif self.num_variables == 2:
                plt.plot(
                    self.trajectories[i][0], 
                    self.trajectories[i][1], 
                    color=color, 
                    linewidth=0.2, 
                    label=label
                )

        plt.grid(True)
        
        if self.num_variables == 1:
            plt.xlabel("Time Step", fontsize=14)
            plt.ylabel("x₀", fontsize=14)
            plt.xticks(list(range(0, self.num_time_points, 10)), fontsize=14)
        elif self.num_variables == 2:
            plt.xlabel("x₀", fontsize=14)
            plt.ylabel("x₁", fontsize=14)
            plt.xticks(fontsize=14)

        plt.yticks(fontsize=14)
        
        # Legacy legend styling
        legend = plt.legend(["anomaly", "regular"], prop={"size": 14})
        legend.legend_handles[0].set_color("red")
        legend.legend_handles[0].set_linewidth(2.0)
        legend.legend_handles[1].set_color("blue")
        legend.legend_handles[1].set_linewidth(2.0)