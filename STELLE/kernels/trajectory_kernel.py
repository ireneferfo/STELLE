"""
RhoKernel - Computes similarity between trajectories and STL formulae.
"""

import torch
import gc
import warnings
from typing import Optional, List

from .stl_kernel import StlKernel


class TrajectoryKernel:
    """
    Computes kernel between trajectories and STL formulae using robustness measures.
    """

    def __init__(
        self,
        measure,
        sigma2: float = 0.44,
        samples: int = 10000,
        varn: int = 2,
        points: int = 100,
        signals: Optional[torch.Tensor] = None,
        normalize: bool = False,
        exp_kernel: bool = False,
        normalize_rhotau: bool = True,
        exp_rhotau: bool = False,
    ):
        self.traj_measure = measure
        self.sigma2 = sigma2
        self.samples = samples
        self.varn = varn
        self.points = points
        self.device = measure.device
        self.exp_kernel = exp_kernel
        self.normalize = normalize
        self.normalize_rhotau = normalize_rhotau
        self.exp_rhotau = exp_rhotau

        # Internal state
        self.phis = None
        self.rhos_phi = None
        self.selfk_phi = None
        self._buffer = None

        # Initialize STL kernel
        self.stl_kernel = StlKernel(
            measure=measure,
            varn=varn,
            points=points,
            signals=signals,
            vectorize=False,
            samples=samples,
            normalize=normalize,
            exp_kernel=exp_kernel,
        )

        # Initialize signals
        if signals is not None:
            self.signals = signals
            self.samples = len(signals)
        else:
            self.signals = measure.sample(points=points, samples=samples, varn=varn)

    def update_samples(self, samples: int) -> None:
        """Update the number of samples used for computation."""
        if samples == self.samples:
            return

        warnings.warn(
            f"RhoKernel: changed # samples from {self.samples} to {samples} to match concepts."
        )
        self.samples = samples
        self.signals = self.traj_measure.sample(
            points=self.points, samples=self.samples, varn=self.varn
        ).to(self.device)
        self.stl_kernel.update_samples(samples)

    def compute_rho_phi(
        self,
        trajectories: torch.Tensor,
        formulae: List,
        epsilon: Optional[float] = None,
        kernel_type: str = "gaussian",
        save: bool = True,
        parallel_workers: int = 0,
    ) -> torch.Tensor:
        """
        Compute kernel matrix between trajectories and STL formulae.

        Args:
            trajectories: Input trajectories tensor
            formulae: List of STL formulae
            epsilon: Scaling parameter for rho_tau
            kernel_type: Type of kernel ("gaussian", "cosine", "linear")
            save: Whether to cache computed formulae
            parallel_workers: Number of parallel_workers

        Returns:
            Kernel matrix of shape (num_trajectories, num_formulae)
        """
        if self.exp_rhotau or self.normalize_rhotau:
            if epsilon is None:
                raise ValueError(
                    "epsilon must be provided when exp_rhotau or normalize_rhotau is True"
                )

        # Compute or retrieve formula robustness
        rhos_phi, selfk_phi = self._get_formula_robustness(
            formulae, save, parallel_workers
        )

        # Compute trajectory similarities
        rhos_tau_values = self._compute_trajectory_similarities(trajectories, epsilon)

        # Compute self-kernels for trajectories
        selfk_tau = (
            torch.einsum("ijk,ijk->i", rhos_tau_values, rhos_tau_values) / self.samples
        )

        return self._compute_kernel_matrix(
            rhos_tau_values, rhos_phi, selfk_tau[:, None], selfk_phi, kernel_type
        )

    def _get_formula_robustness(
        self, formulae: List, save: bool, parallel_workers: int
    ):
        """Get robustness values for formulae, either computed or cached."""
        if self.phis is None:
            rhos_phi, selfk_phi = self.stl_kernel._compute_robustness(
                formulae, parallel_workers
            )
            if save:
                self.phis = formulae
                self.rhos_phi = rhos_phi
                self.selfk_phi = selfk_phi
            return rhos_phi, selfk_phi
        else:
            return self.rhos_phi, self.selfk_phi

    def _compute_trajectory_similarities(
        self, trajectories: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        """Compute rho_tau similarity values for all trajectories."""
        rhos_tau_values = torch.empty(
            len(trajectories), self.samples, 1, device=self.device
        )

        for i, tau in enumerate(trajectories):
            rhos_tau_values[i] = self.rho_tau_values(self.signals, tau, epsilon)

        return rhos_tau_values

    def euclidean_distance(self, xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Compute L2 distance between trajectory xi and reference trajectory tau."""
        return torch.sum(torch.abs(xi - tau).pow(2), dim=(1, 2))

    def rho_tau_values(
        self, xi: torch.Tensor, tau: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        """
        Compute rho_tau similarity function based on distance.

        Args:
            xi: Input trajectories
            tau: Reference trajectory
            epsilon: Scaling parameter

        Returns:
            Similarity values tensor
        """
        distances = self.euclidean_distance(xi, tau).to(self.device)

        if not torch.is_tensor(epsilon):
            epsilon = torch.tensor(epsilon, device=self.device)
        epsilon = epsilon.to(self.device)

        # Handle negative epsilon
        if epsilon < 0:
            epsilon = distances.max().clamp(min=1e-8)

        if self.exp_rhotau:
            if self.normalize_rhotau:  # Output in [-1,1]
                return (2 * torch.exp(-distances / epsilon) - 1).unsqueeze_(-1)
            return torch.exp(-distances / epsilon).unsqueeze_(-1)  # [0, infty)

        if self.normalize_rhotau:
            return (2 * (1 - distances / epsilon) - 1).unsqueeze_(-1)

        return distances.unsqueeze_(-1) / epsilon

    def _compute_kernel_matrix(
        self,
        rhos1: torch.Tensor,
        rhos2: torch.Tensor,
        selfk1: torch.Tensor,
        selfk2: torch.Tensor,
        kernel_type: str,
    ) -> torch.Tensor:
        """Compute kernel matrix between two sets of robustness values."""
        self._cleanup_memory()

        # Move to CPU for computation
        rhos1 = self._to_cpu_float(rhos1)
        rhos2 = self._to_cpu_float(rhos2)

        if kernel_type == "gaussian":
            kernel_matrix = self._compute_gaussian_kernel(rhos1, rhos2)
            if self.normalize:
                kernel_matrix = self._normalize_kernel(kernel_matrix, selfk1, selfk2)
            if self.exp_kernel:
                kernel_matrix = self._exponentiate_kernel(kernel_matrix)

        elif kernel_type == "cosine":
            kernel_matrix = self._compute_cosine_kernel(rhos1, rhos2)

        elif kernel_type == "linear":
            kernel_matrix = self._compute_linear_kernel(rhos1, rhos2)

        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        return kernel_matrix

    def _compute_gaussian_kernel(
        self, rhos1: torch.Tensor, rhos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gaussian kernel matrix."""
        if rhos2.ndim > 2:
            return torch.einsum("abc,dbc->ad", rhos1, rhos2) / self.samples
        else:
            rhos1 = rhos1.squeeze()
            rhos2 = rhos2.squeeze()
            return torch.matmul(rhos1, rhos2.T) / self.samples

    def _compute_cosine_kernel(
        self, rhos1: torch.Tensor, rhos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity kernel."""
        rhos1_norm = rhos1 / rhos1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        rhos2_norm = rhos2 / rhos2.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.mm(rhos1_norm, rhos2_norm.T)

    def _compute_linear_kernel(
        self, rhos1: torch.Tensor, rhos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute linear kernel."""
        return torch.mm(rhos1, rhos2.T) / self.samples

    def _normalize_kernel(
        self, kernel_matrix: torch.Tensor, selfk1: torch.Tensor, selfk2: torch.Tensor
    ) -> torch.Tensor:
        """Normalize kernel matrix."""
        if torch.cuda.is_available():
            dtype = torch.float64
        else:
            dtype = torch.float32

        normalize = torch.sqrt(
            torch.matmul(
                selfk1.to(self.device, dtype=dtype),
                selfk2.to(self.device, dtype=dtype).T,
            )
        ).to(kernel_matrix.device)

        return kernel_matrix / normalize

    def _exponentiate_kernel(self, kernel_matrix: torch.Tensor) -> torch.Tensor:
        """Apply exponential transformation to kernel matrix."""
        return torch.exp_(-(2.0 - 2 * kernel_matrix) / (2 * self.sigma2))

    def _to_cpu_float(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU with appropriate float precision."""
        if torch.cuda.is_available():
            return tensor.to("cpu", dtype=torch.float64)
        else:
            return tensor.to("cpu", dtype=torch.float32)

    def _cleanup_memory(self) -> None:
        """Clean up GPU memory and garbage collect."""
        torch.cuda.empty_cache()
        gc.collect()

    def compute_epsilon(self, dataset) -> float:
        """
        Compute epsilon as the 99th percentile of inter-class distances.

        Args:
            dataset: Dataset containing trajectories divided by class

        Returns:
            epsilon value
        """
        trajectories_by_class = dataset.divide_traj_by_class()
        num_classes = len(trajectories_by_class)

        if num_classes < 2:
            raise ValueError("Need at least 2 classes to compute epsilon")

        # Collect all pairwise distances between different classes
        distances = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                class_i = torch.stack(trajectories_by_class[i]).to(self.device)
                class_j = torch.stack(trajectories_by_class[j]).to(self.device)

                # Compute pairwise distances efficiently
                diff = class_i.unsqueeze(1) - class_j.unsqueeze(0)
                class_distances = torch.abs(diff).pow(2).sum(dim=(2, 3)).flatten()
                distances.append(class_distances)

        # Combine all inter-class distances
        all_distances = torch.cat(distances)
        self.dists = all_distances  # Store for debugging

        # Return 99th percentile of all inter-class distances
        return torch.quantile(all_distances, 0.99).item()
