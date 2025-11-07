"""
data_generation.py

Utilities to generate and plot synthetic multivariate time series trajectories
for classification experiments.

Functions:
- generate_synthetic_trajectories: create synthetic trajectories with controllable
    class separation, noise and drift.
- plot_trajectories: quick visualization helper to plot sample trajectories by class.
"""

from typing import Tuple

import torch
from pathlib import Path
import matplotlib.pyplot as plt


def generate_synthetic_trajectories(
    num_trajs: int,
    n_vars: int,
    points: int,
    num_classes: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic multivariate trajectories.

    Parameters
    - num_trajs: number of trajectories (samples)
    - n_vars: number of variables (channels) per trajectory
    - points: number of time points per trajectory
    - num_classes: number of distinct classes
    - seed: RNG seed for reproducibility

    Returns
    - X: tensor of shape (num_trajs, n_vars, points) with trajectory values
    - y: tensor of shape (num_trajs,) with integer class labels in [0, num_classes)
    """
    # Reproducible randomness
    torch.manual_seed(seed)

    # Preallocate tensors
    X = torch.zeros((num_trajs, n_vars, points), dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_trajs,), dtype=torch.long)

    # Uniform time axis in [0,1]
    t = torch.linspace(0.0, 1.0, points)

    for i in range(num_trajs):
        cls = int(y[i].item())

        for v in range(n_vars):
            # Base signal for this variable and class
            base = torch.zeros(points)

            # Reduce class separation and add more overlap between classes
            # class_factor controls amplitude scaling per class (smaller separation)
            class_factor = 1.0 + 0.3 * cls
            # phase_shift controls relative phase offset per class (smaller phase diff)
            phase_shift = 0.3 * cls

            # Use different functional forms depending on variable index to create variety
            if v % 4 == 0:
                # sinusoidal pattern with frequency depending on variable index
                base = class_factor * torch.sin(
                    2.0 * torch.pi * (v + 1) * t + phase_shift
                )
            elif v % 4 == 1:
                # cosinusoidal
                base = class_factor * torch.cos(
                    2.0 * torch.pi * (v + 1) * t + phase_shift
                )
            elif v % 4 == 2:
                # absolute sine to produce positive-only bursts
                base = class_factor * torch.abs(
                    torch.sin((v + 1) * torch.pi * t + phase_shift)
                )
            else:
                # localized Gaussian-like bump whose center depends on class
                base = torch.exp(-(((t - 0.2 * (cls + 1)) * (v + 1)) ** 2))

            # Small linear drift that depends on variable index and class (reduced drift)
            drift = 0.05 * v * t + 0.05 * cls

            # Additive Gaussian noise (increased noise)
            noise = 0.3 * torch.randn(points)

            # Compose final signal for this trajectory, variable
            X[i, v] = base + drift + noise

    # Convert to NumPy arrays before returning (not torch tensors)
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    return X, y


def plot_trajectories(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 3,
    samples_per_class: int = 5,
    path = None
) -> None:
    """
    Plot sample trajectories for each variable and class.

    Parameters
    - X: tensor (num_trajs, n_vars, points)
    - y: tensor (num_trajs,)
    - num_classes: number of classes to show (colors indexed by class id)
    - samples_per_class: how many examples per class to plot
    """
    _, n_vars, _ = X.shape

    # Configure figure size: one subplot per variable
    plt.figure(figsize=(15, max(2, n_vars * 2)))
    colors = plt.cm.tab10.colors  # color palette for up to 10 classes

    for var in range(n_vars):
        plt.subplot(n_vars, 1, var + 1)
        for cls in range(num_classes):
            # Indices of samples belonging to this class
            class_indices = (y == cls).nonzero(as_tuple=True)[0][:samples_per_class]
            for idx in class_indices:
                # Plot trajectory for this sample and variable
                plt.plot(
                    X[int(idx), var].cpu(),
                    label=f"class {cls}" if var == 0 else "",
                    color=colors[cls % len(colors)],
                )
        plt.title(f"Variable {var}")
        plt.xticks([])

    # Place a single legend (only uses labels from first subplot)
    plt.legend()
    plt.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
