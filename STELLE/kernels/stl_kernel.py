"""
StlKernel - Computes kernels between STL formulae using robustness measures.
"""

import torch
import gc
import warnings
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor


class StlKernel:
    """
    Computes kernel between STL formulae using robustness measures.
    """
    
    def __init__(
        self,
        measure,
        normalize: bool = False,
        exp_kernel: bool = False,
        sigma2: float = 0.44,
        samples: int = 10000,
        varn: int = 1,
        points: int = 100,
        boolean: bool = False,
        signals: Optional[torch.Tensor] = None,
        newstl: bool = True,
        vectorize: bool = True,
    ):
        self.traj_measure = measure
        self.normalize = normalize
        self.exp_kernel = exp_kernel
        self.sigma2 = sigma2
        self.samples = samples
        self.varn = varn
        self.points = points
        self.device = measure.device
        self.boolean = boolean
        self.newstl = newstl
        self.vectorize = vectorize
        
        # Initialize signals
        if signals is not None:
            self.signals = signals.to(self.device)
        else:
            self.signals = measure.sample(
                points=points, 
                samples=samples, 
                varn=varn
            ).to(self.device)

    def update_samples(self, samples: int) -> None:
        """Update the number of samples used for computation."""
        if samples == self.samples:
            return
            
        warnings.warn(
            f'StlKernel: changed # samples from {self.samples} to {samples} to match concepts.'
        )
        self.samples = samples
        self.signals = self.traj_measure.sample(
            points=self.points, 
            samples=self.samples, 
            varn=self.varn
        ).to(self.device)

    def compute_bag(
        self, 
        formulae: List, 
        return_robustness: bool = False, 
        workers: int = 0
    ) -> torch.Tensor:
        """
        Compute kernel matrix for a single bag of formulae.
        
        Args:
            formulae: List of STL formulae
            return_robustness: Whether to return robustness values
            workers: Number of parallel workers
            
        Returns:
            Kernel matrix or tuple with additional robustness data
        """
        rhos, selfk = self._compute_robustness_no_time(formulae, workers)
        kernel_matrix = self._compute_kernel_no_time(rhos, rhos, selfk, selfk)

        if return_robustness:
            return kernel_matrix.cpu(), rhos, selfk, None
        return kernel_matrix.cpu()

    def compute_bag_bag(
        self, 
        formulae1: List, 
        formulae2: List, 
        return_robustness: bool = False, 
        workers: int = 0
    ) -> torch.Tensor:
        """
        Compute kernel matrix between two bags of formulae.
        
        Args:
            formulae1: First list of STL formulae
            formulae2: Second list of STL formulae
            return_robustness: Whether to return robustness values
            workers: Number of parallel workers
            
        Returns:
            Kernel matrix or tuple with additional robustness data
        """
        rhos1, selfk1 = self._compute_robustness_no_time(formulae1, workers)
        rhos2, selfk2 = self._compute_robustness_no_time(formulae2, workers)
        kernel_matrix = self._compute_kernel_no_time(rhos1, rhos2, selfk1, selfk2)

        if return_robustness:
            return kernel_matrix.cpu(), rhos1, rhos2, selfk1, selfk2
        return kernel_matrix.cpu()

    def _compute_robustness_no_time(
        self, 
        formulae: List, 
        workers: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute robustness values for formulae in parallel."""
        self._cleanup_memory()
        
        if workers == 0:
            workers = 1
            
        num_signals, num_formulae = len(self.signals), len(formulae)
        
        if self.vectorize:
            shape = (num_formulae, num_signals, self.varn)
        else:
            shape = (num_formulae, num_signals)
            
        rhos = torch.zeros(shape, device="cpu")
        self_kernels = torch.zeros((num_formulae, 1), device="cpu")

        # Parallel computation
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self._compute_single_rho, 
                    formula, 
                    self.boolean, 
                    self.signals
                )
                for formula in formulae
            ]
            
            for i, future in enumerate(futures):
                rho = future.result()
                self._process_rho_result(rho, rhos, self_kernels, i, num_signals)

        self._cleanup_memory()
        return rhos, self_kernels

    def _process_rho_result(
        self, 
        rho: torch.Tensor, 
        rhos: torch.Tensor, 
        self_kernels: torch.Tensor, 
        index: int,
        num_signals: int
    ) -> None:
        """Process result of single rho computation."""
        if self.vectorize:
            self_kernels[index] = torch.einsum("ij,ij->", rho, rho) / num_signals
            rhos[index] = rho
        else:
            rho = rho.squeeze()
            self_kernels[index] = torch.dot(rho, rho) / num_signals
            rhos[index] = rho

    def _compute_single_rho(
        self, 
        formula, 
        boolean: bool, 
        signals: torch.Tensor
    ) -> torch.Tensor:
        """Compute robustness for a single formula."""
        self._cleanup_memory()
        
        if boolean:
            rho = formula.boolean(signals).float()
            rho[rho == 0.0] = -1.0
        else:
            rho = formula.quantitative(
                signals, 
                vectorize=self.vectorize, 
                normalize=self.normalize
            )
            
        return rho.to("cpu")

    def _compute_kernel_no_time(
        self, 
        rhos1: torch.Tensor, 
        rhos2: torch.Tensor, 
        selfk1: torch.Tensor, 
        selfk2: torch.Tensor
    ) -> torch.Tensor:
        """Compute kernel matrix from robustness values."""
        if self.vectorize:
            kernel_matrix = torch.einsum(
                "ijk,ljk->il", 
                rhos1.double().to(self.device), 
                rhos2.double().to(self.device)
            )
        else:
            kernel_matrix = torch.mm(
                rhos1.to(self.device), 
                rhos2.to(self.device).t()
            )

        kernel_matrix = kernel_matrix / self.samples

        if self.normalize:
            kernel_matrix = self._normalize_kernel(kernel_matrix, selfk1, selfk2)
            
        if self.exp_kernel:
            kernel_matrix = self._exponentiate_kernel(kernel_matrix, selfk1, selfk2)

        return kernel_matrix

    def _normalize_kernel(
        self, 
        kernel_matrix: torch.Tensor, 
        selfk1: torch.Tensor, 
        selfk2: torch.Tensor
    ) -> torch.Tensor:
        """Normalize kernel matrix."""
        return kernel_matrix / torch.sqrt(torch.mm(selfk1, selfk2.t()))

    def _exponentiate_kernel(
        self, 
        kernel_matrix: torch.Tensor, 
        selfk1: torch.Tensor, 
        selfk2: torch.Tensor, 
        sigma2: Optional[float] = None
    ) -> torch.Tensor:
        """Apply exponential transformation to kernel matrix."""
        sigma2 = sigma2 or self.sigma2
        
        k1, k2 = selfk1.size(0), selfk2.size(0)
        selfk = (
            selfk1.pow(2).repeat(1, k2).cpu() + 
            selfk2.pow(2).t().repeat(k1, 1).cpu()
        )
        
        return torch.exp((2 * kernel_matrix.cpu() - selfk) / (2 * sigma2))

    def _get_selfk(self, rhos):
        """
        Compute self-kernels from precomputed rhos.
        rhos: shape (k, n, varn) if vectorize else (k, n)
        Returns: self_kernels of shape (k, 1)
        """
        n = rhos.shape[1]
        if self.vectorize:
            # For each k, tensordot over n and varn
            self_kernels = torch.einsum("knv,knv->k", rhos, rhos) / n
        else:
            # For each k, dot over n
            self_kernels = torch.einsum("kn,kn->k", rhos, rhos) / n
        return self_kernels.unsqueeze(1)
    
    def _cleanup_memory(self) -> None:
        """Clean up GPU memory and garbage collect."""
        torch.cuda.empty_cache()
        gc.collect()