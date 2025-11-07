"""
Formula Manager - Handles loading, saving, and managing STL formulae with caching.
"""

import torch
import os
import gc
from time import time
from typing import List, Optional, Tuple, Dict

from ..utils import get_device
from ..kernels.stl_kernel import StlKernel
from .concept_generator import ConceptGenerator
from .stl_generator import STLFormulaGenerator
from .formula_utils import compute_permutations, get_unique_variables, get_formula_template


class FormulaManager:
    """
    Manages STL formulae with efficient caching, loading, and generation.
    
    Handles both anchor sets and concept sets with support for variable
    permutation and batched robustness computation.
    """
    
    def __init__(
        self,
        n_vars: int,
        sampler: STLFormulaGenerator,
        stl_kernel: StlKernel,
        parallel_workers: int,
        cosine_threshold: float = 1.0,
        nvars_formulae: int = 1,
        points: int = 100,
        device: Optional[str] = None,
    ):
        """
        Initialize the formula manager.
        
        Args:
            n_vars: Total number of variables in the system
            nvars_formulae: Maximum variables per formula
            points: Number of time points in signals
            device: Computation device (auto-detected if None)
        """
        self.n_vars = n_vars
        self.nvars_formulae = nvars_formulae
        self.points = points
        self.sampler = sampler
        self.stl_kernel = stl_kernel
        self.parallel_workers = parallel_workers
        self.cosine_threshold = cosine_threshold
        self.device = device or get_device()
        
        self._times_cache: Dict[str, float] = {}
        self._load_times_cache()
    
    def get_formulae(
        self,
        creation_mode: int, # 0: shared, 1: per-variable
        count: int, 
        output_directory: str,
        formulae_type: str = 'concepts',
        seed: int = 0,
        batch_size: int = 100,
        formulae_per_var: int = None,
        min_total: int = 1
        ) -> Tuple[List, torch.Tensor, torch.Tensor, float]:
        if creation_mode == 0:
            return self._get_formulae(count, formulae_type, creation_mode, output_directory, seed, batch_size)
        elif creation_mode == 1:
            if formulae_per_var == None:
                formulae_per_var = count//self.n_vars
            return self._get_formulae_per_variable(formulae_per_var, min_total, formulae_type, output_directory, seed, batch_size)
 
    def _get_formulae(
        self,
        target_count: int,
        formulae_type: str,
        creation_mode: int,
        output_directory: str,
        seed: int,
        batch_size: int,
    ) -> Tuple[List, torch.Tensor, torch.Tensor, float]:
        """
        Get formulae with efficient caching and batched generation.
        
        Args:
            target_count: Desired number of formulae
            formulae_type: Type of formulae ("anchors" or "concepts")
            creation_mode: 0 for shared formulae, 1 for per-variable formulae
            output_directory: Directory to save/load formulae
            seed: Random seed
            parallel_workers: Number of parallel workers for robustness computation
            batch_size: Batch size for generation and robustness computation
            
        Returns:
            Tuple of (formulae, robustness_vectors, self_kernels, total_time)
        """
        self._validate_inputs(formulae_type)
        
        os.makedirs(output_directory, exist_ok=True)
        
        # Try to load existing formulae
        formulae_file = os.path.join(output_directory, f"{formulae_type}_{target_count}.pickle")
        existing_data = self._try_load_existing_formulae(formulae_file)
        
        if existing_data:
            formulae, robustness, selfk = existing_data
            total_time = self._get_cached_time(formulae_type, len(formulae))
            
            if len(formulae) == target_count:
                return formulae, robustness, selfk, total_time
            elif len(formulae) > target_count:
                return self._subset_formulae(
                    formulae, robustness, selfk, target_count, formulae_file
                )
            else:
                return self._extend_formulae(
                    formulae, robustness, selfk, target_count, creation_mode,
                      output_directory, seed,
                    formulae_type,batch_size
                )
        else:
            return self._generate_new_formulae(
                target_count, creation_mode, output_directory,
                seed, formulae_type, batch_size
            )
    
    def _get_formulae_per_variable(
        self,
        formulae_per_var: int,
        min_total: int,
        formulae_type: str,
        output_directory: str,
        seed: int,
        batch_size: int,
    ) -> Tuple[List, torch.Tensor, torch.Tensor, float]:
        """Get formulae with per-variable distribution."""
        # Collapse to shared mode if only one variable
        creation_mode = 0 if self.n_vars == 1 else 1
        
        target_count = max(min_total, formulae_per_var * self.n_vars)
        
        return self._get_formulae(
            target_count=target_count,
            formulae_type=formulae_type,
            creation_mode=creation_mode,
            output_directory=output_directory,
            seed=seed,
            batch_size=batch_size,
        )
    
    def _validate_inputs(self, formulae_type: str) -> None:
        """Validate input parameters."""
        if self.cosine_threshold > 1.0:
            raise ValueError(f"self.cosine_threshold must be <= 1 (got {self.cosine_threshold})")
        if formulae_type not in {"anchors", "concepts"}:
            raise ValueError(f"formulae_type must be 'anchors' or 'concepts'")
    
    def _load_times_cache(self) -> None:
        """Load timing information from cache file."""
        # This would load from a CSV file - implementation depends on your specific needs
        pass
    
    def _get_cached_time(self, formulae_type: str, count: int) -> float:
        """Get cached generation time for formulae."""
        key = f"{formulae_type}_{count}"
        return self._times_cache.get(key, 0.0)
    
    def _try_load_existing_formulae(self, file_path: str) -> Optional[Tuple]:
        """Try to load existing formulae from file."""
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, "rb") as f:
                return torch.load(f, map_location=torch.device(self.device), weights_only=False)
        except Exception as e:
            print(f"Error loading formulae from {file_path}: {e}")
            return None
    
    def _subset_formulae(
        self,
        formulae: List,
        robustness: torch.Tensor,
        selfk: torch.Tensor,
        target_count: int,
        output_path: str,
    ) -> Tuple[List, torch.Tensor, torch.Tensor, float]:
        """Subset existing formulae to target count."""
        formulae = formulae[:target_count]
        robustness = robustness[:target_count]
        selfk = selfk[:target_count]
        
        # Save subset
        with open(output_path, "wb") as f:
            torch.save((formulae, robustness, selfk), f)
        
        print(f"Subsetted to {len(formulae)} formulae")
        return formulae, robustness, selfk, 0.0
    
    def _extend_formulae(
        self,
        existing_formulae: List,
        existing_robustness: torch.Tensor,
        existing_selfk: torch.Tensor,
        target_count: int,
        creation_mode: int,
        output_directory: str,
        seed: int,
        formulae_type: str,
        batch_size: int,
    ) -> Tuple[List, torch.Tensor, torch.Tensor, float]:
        """Extend existing formulae with new ones."""
        missing_count = target_count - len(existing_formulae)
        start_time = time()
        
        if creation_mode == 1:
            new_formulae = self._generate_with_variable_permutation(
                existing_formulae, existing_robustness, missing_count,
                 output_directory, seed, batch_size
            )
        else:
            if self.cosine_threshold == 1.0:
                new_formulae = self._generate_simple_formulae(
                    missing_count,  output_directory, seed, batch_size
                )
            else:
                new_formulae, new_robustness = self._generate_diverse_formulae(
                    existing_formulae, existing_robustness, missing_count,
                    output_directory, seed, batch_size
                )
        
        # Compute robustness for new formulae
        if 'new_robustness' not in locals():
            new_robustness, new_selfk = self._compute_batched_robustness(
                new_formulae, batch_size
            )
        else:
            new_selfk = self.stl_kernel._get_selfk(new_robustness)
        
        # Combine with existing
        combined_formulae = existing_formulae + new_formulae
        combined_robustness = torch.cat([existing_robustness.to("cpu"), new_robustness.to("cpu")], dim=0)
        combined_selfk = torch.cat([existing_selfk.to("cpu"), new_selfk.to("cpu")], dim=0)
        
        # Save results
        output_path = os.path.join(output_directory, f"{formulae_type}_{target_count}.pickle")
        with open(output_path, "wb") as f:
            torch.save((combined_formulae, combined_robustness, combined_selfk), f)
        
        total_time = time() - start_time + self._get_cached_time(formulae_type, len(existing_formulae))
        self._update_time_cache(formulae_type, target_count, total_time)
        
        print(f"Extended to {len(combined_formulae)} formulae")
        return combined_formulae, combined_robustness, combined_selfk, total_time
    
    def _generate_new_formulae(
        self,
        target_count: int,
        creation_mode: int,
        output_directory: str,
        seed: int,
        formulae_type: str,
        batch_size: int,
    ) -> Tuple[List, torch.Tensor, torch.Tensor, float]:
        """Generate completely new set of formulae."""
        # print(f"Generating {target_count} {formulae_type} from scratch")
        start_time = time()
        
        if creation_mode == 1:
            formulae = self._generate_with_variable_permutation(
                [], [], target_count,
                 output_directory, seed, batch_size
            )
        else:
            if self.cosine_threshold == 1.0:
                formulae, _ = self._generate_simple_formulae(
                    target_count, output_directory, seed, batch_size
                )
            else:
                formulae, robustness = self._generate_diverse_formulae(
                    [], [], target_count,
                    output_directory, seed, batch_size
                )
        
        # Compute robustness
        if 'robustness' not in locals():
            
            robustness, selfk = self._compute_batched_robustness(
                formulae, batch_size
            )
        else:
            selfk = self.stl_kernel._get_selfk(robustness)
        
        # Save results
        output_path = os.path.join(output_directory, f"{formulae_type}_{target_count}.pickle")
        with open(output_path, "wb") as f:
            torch.save((formulae, robustness, selfk), f)
        
        total_time = time() - start_time
        self._update_time_cache(formulae_type, target_count, total_time)
        
        return formulae, robustness, selfk, total_time
    
    def _generate_with_variable_permutation(
        self,
        existing_formulae: List,
        existing_robustness: torch.Tensor,
        missing_count: int,
        output_directory: str,
        seed: int,
        batch_size: int,
    ) -> List:
        """Generate formulae with variable permutation."""
        """
        Generate new formulae with variable permutation (creation_nvars=1 case), with n_vars_formulae variables per formula (max).
        Handles partial groups of variable permutations.
        """
        # Step 1: Group formulae by their template (variable-agnostic pattern)
        template_map = {}
        for formula in existing_formulae:
            template = get_formula_template(formula)
            if template not in template_map:
                template_map[template] = []
            template_map[template].append(formula)
        # Calculate how many complete groups we have
        remaining = missing_count
        new_formulae = []

        # Step 2: Process each template to complete its permutation group
        for template, formulae in template_map.items():
            if remaining <= 0:
                break
            # Calculate how many unique variables are in this template
            num_vars_in_template = len(get_unique_variables(str(formulae[0])))
            total_perms = self.n_vars**num_vars_in_template
            # If we haven't generated all permutations for this template
            if len(formulae) < total_perms:
                formulae_strings = [str(f) for f in formulae]
                # Generate all possible permutations
                # take formula, not template, so that var idxs relations are preserved
                all_perms = compute_permutations(formulae[:1], self.n_vars)
                # Find the missing permutations
                missing_perms = [p for p in all_perms if str(p) not in formulae_strings]

                # Add up to what we need
                add_count = min(len(missing_perms), remaining)
                added_perms = missing_perms[:add_count]
                new_formulae.extend(added_perms)
                remaining -= add_count

                # Check if we just completed this group
                if len(formulae) + len(added_perms) >= total_perms:
                    template_map[get_formula_template(all_perms[0])] = all_perms

        existing_formulae_0 = []
        existing_rhos_0 = []
        if self.cosine_threshold < 1:
            # Get the first formula of each complete group as starting set
            if template_map != {}:
                for template, formulae in template_map.items():
                    num_vars_in_template = len(get_unique_variables(template))
                    total_perms = self.n_vars**num_vars_in_template
                    # Only take formulae from complete groups
                    if len(formulae) >= total_perms:
                        formulae_strings = [str(f) for f in formulae]
                        existing_formulae_0.append(formulae[0])
                        # Find corresponding rho (assuming same order)
                        idx = formulae_strings.index(str(formulae[0]))
                        if idx < len(existing_robustness):
                            existing_rhos_0.append(
                                existing_robustness[idx][..., :self.nvars_formulae]
                            )  # Take first variable's rho

        # Step 3: If we still need more formulae, create new templates
        generator = ConceptGenerator(
            n_vars=self.n_vars,
            nvars_formulae=self.nvars_formulae,
            device=self.device,
            seed=seed
        )
        
        while remaining > 0:
            base_formula = generator.generate_concepts(
                target_dim = 1 if self.cosine_threshold == 1 else len(existing_formulae_0) + 1,
                cosine_threshold=self.cosine_threshold,  # No filtering
                output_path=output_directory,
                batch_size=batch_size,
                return_robustness=True if self.cosine_threshold < 1 else False,
                promote_simple=True,
                initial_formulae= existing_formulae_0 if existing_formulae_0 != [] else None,
                initial_robustness= torch.stack(existing_rhos_0) if len(existing_rhos_0) > 0 else None,
                seed= seed + len(existing_formulae_0) + remaining
            )

            # groups_needed = (remaining // self.n_vars) + (1 if remaining % self.n_vars else 0)
            # print(len(existing_formulae_0))
            # print(len(existing_rhos_0[0]))
            # base_formula = concept_generation(
            #     formulae_type,
            #     dim=1 if t == 1 else len(existing_formulae_0) + 1,
            #     n_vars=n_vars_formulae,  # Will be permuted across variables
            #     n_vars_formulae=n_vars_formulae,
            #     pro_simple=True,
            #     path=phis_file_path,
            #     cosine_similarity_threshold=t,
            #     checkpoints=True,
            #     device=device,
            #     sampler=sampler,
            #     seed=seed + len(existing_formulae_0) + remaining,
            #     samples=samples,
            #     normalize_flag=normalize,
            #     start_set=existing_formulae_0 if existing_formulae_0 != [] else None,
            #     start_set_rhos=(
            #         torch.stack(existing_rhos_0) if len(existing_rhos_0) > 0 else None
            #     ),
            #     get_robs=True if t < 1 else False,  # kicks in only with t < 1,
            #     batch_size = batch_size,
            # )
            base_formula, base_rho = base_formula
            if self.cosine_threshold < 1:
                base_rho = base_rho[len(existing_formulae_0) :]
                existing_rhos_0.extend(base_rho)
                base_formula = base_formula[len(existing_formulae_0) :]
                
            existing_formulae_0.extend(base_formula)
            perms = compute_permutations(base_formula, self.n_vars)
            new_formulae.extend(perms)
            remaining -= len(perms)

        return new_formulae[:missing_count]
    
    def _generate_simple_formulae(
        self,
        count: int,
        output_directory: str,
        seed: int,
        batch_size: int,
    ) -> List:
        """Generate formulae without diversity filtering."""
        generator = ConceptGenerator(
            n_vars=self.n_vars,
            nvars_formulae=self.nvars_formulae,
            device=self.device,
            seed=seed
        )
        
        return generator.generate_concepts(
            target_dim=count,
            cosine_threshold=1.0,  # No filtering
            output_path=output_directory,
            batch_size=batch_size,
            return_robustness=False,
        )
    
    def _generate_diverse_formulae(
        self,
        existing_formulae: List,
        existing_robustness: torch.Tensor,
        count: int,
        output_directory: str,
        seed: int,
        batch_size: int,
    ) -> Tuple[List, torch.Tensor]:
        """Generate formulae with diversity filtering."""
        generator = ConceptGenerator(
            n_vars=self.n_vars,
            nvars_formulae=self.nvars_formulae,
            device=self.device,
            seed=seed
        )
        
        return generator.generate_concepts(
            target_dim=len(existing_formulae) + count,
            cosine_threshold=self.cosine_threshold,
            output_path=output_directory,
            batch_size=batch_size,
            initial_formulae=existing_formulae,
            initial_robustness=existing_robustness,
            return_robustness=True,
        )
    
    def _compute_batched_robustness(
        self,
        formulae: List,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute robustness in batches to manage memory."""
        robustness_chunks = []
        selfk_chunks = []
        
        for start_idx in range(0, len(formulae), batch_size):
            gc.collect()
            torch.cuda.empty_cache()
            
            end_idx = start_idx + batch_size
            formula_batch = formulae[start_idx:end_idx]
            
            robustness_batch, selfk_batch = self.stl_kernel._compute_robustness_no_time(
                formula_batch, self.parallel_workers
            )
            
            robustness_chunks.append(robustness_batch.cpu())
            selfk_chunks.append(selfk_batch.cpu())
        
        robustness = torch.cat(robustness_chunks, dim=0).cpu()
        selfk = torch.cat(selfk_chunks, dim=0).cpu()
        
        return robustness, selfk
    
    def _update_time_cache(self, formulae_type: str, count: int, time_val: float) -> None:
        """Update the time cache with new timing information."""
        key = f"{formulae_type}_{count}"
        self._times_cache[key] = time_val
        # This would also save to a CSV file in a real implementation