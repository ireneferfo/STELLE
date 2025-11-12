"""
Concept Generator - Generates diverse STL formulae using cosine similarity filtering.
"""

import torch
import numpy as np
import os
import gc
import random
from copy import deepcopy
from typing import List, Optional, Tuple
from torch.nn.functional import normalize

from .stl_generator import STLFormulaGenerator
from ..kernels.base_measure import BaseMeasure
from ..utils import dump_pickle, load_pickle
from .formula_utils import find_n_nodes
from .formula_manipulation import filter_tautology_contradictions_serial


class ConceptGenerator:
    """
    Generates diverse STL formulae using cosine similarity-based filtering.

    Creates a set of formulae that are semantically diverse by ensuring
    their robustness vectors have low cosine similarity.
    """

    def __init__(
        self,
        nvars: int = 1,
        nvars_formulae: int = 1,
        leaf_probability: float = 0.5,
        time_bound_max_range: int = 50,
        points: int = 100,
        max_nodes: int = 5,
        device: str = "cpu",
        seed: int = 0,
    ):
        """
        Initialize the concept generator.

        Args:
            nvars: Total number of variables in the system
            nvars_formulae: Maximum variables per formula
            leaf_probability: Probability of generating a leaf node
            time_bound_max_range: Maximum time range for temporal operators
            points: Number of time points in signals
            max_nodes: Maximum number of nodes in generated formulae
            device: Computation device
            seed: Random seed for reproducibility
        """
        self.nvars = nvars
        self.nvars_formulae = nvars_formulae
        self.points = points
        self.max_nodes = max_nodes
        self.device = device

        self.sampler = STLFormulaGenerator(
            leaf_probability=leaf_probability,
            time_bound_max_range=time_bound_max_range,
            max_variables=nvars_formulae,
            max_timespan=points,
        )

        self._set_random_seeds(seed)

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # the STL generator mixes numpy and the Python `random` module
        # (see `stl_generator.py`), so seed both to ensure different
        # seeds actually change the sampled formulae.
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_concepts(
        self,
        target_dim: int,
        cosine_threshold: float = 0.9,
        promote_simple: bool = False,
        output_path: Optional[str] = None,
        enable_checkpoints: bool = False,
        batch_size: int = 1000,
        signal_samples: int = 10000,
        signals: Optional[torch.Tensor] = None,
        initial_formulae: Optional[List] = None,
        initial_robustness: Optional[torch.Tensor] = None,
        normalize_robustness: bool = False,
        return_robustness: bool = False,
        seed: int = None,
    ) -> Tuple[List, Optional[torch.Tensor]]:
        """
        Generate a diverse set of STL concepts.

        Args:
            target_dim: Target number of formulae to generate
            cosine_threshold: Maximum allowed cosine similarity between formulae
            promote_simple: Whether to prefer simpler formulae when similar
            output_path: Path to save results
            enable_checkpoints: Whether to save progress checkpoints
            batch_size: Batch size for generation
            signal_samples: Number of signal samples for robustness computation
            signals: Pre-computed signals (if None, generates new ones)
            initial_formulae: Initial set of formulae to build upon
            initial_robustness: Robustness vectors for initial formulae
            normalize_robustness: Whether to normalize robustness values
            return_robustness: Whether to return robustness vectors
            seed: override general seed set in init

        Returns:
            Tuple of (formulae_list, robustness_vectors) if return_robustness=True,
            otherwise just formulae_list
        """
        if seed is not None:
            self._set_random_seeds(seed)

        self._validate_parameters(cosine_threshold, return_robustness)

        if output_path:
            os.makedirs(output_path, exist_ok=True)

        # Initialize with existing checkpoint or initial set
        formulae, robustness_vectors = self._initialize_generation(
            output_path,
            enable_checkpoints,
            initial_formulae,
            initial_robustness,
            signals,
            signal_samples,
            normalize_robustness,
        )

        # Handle trivial case where no filtering is needed
        if cosine_threshold == 1.0:
            return self._generate_without_filtering(target_dim, output_path)

        # Main generation loop with cosine similarity filtering
        formulae, robustness_vectors = self._generate_with_filtering(
            target_dim,
            cosine_threshold,
            promote_simple,
            output_path,
            enable_checkpoints,
            batch_size,
            signal_samples,
            signals,
            formulae,
            robustness_vectors,
            normalize_robustness,
            return_robustness,
        )
        return (formulae, robustness_vectors) if return_robustness else formulae

    def _validate_parameters(
        self, cosine_threshold: float, return_robustness: bool
    ) -> None:
        """Validate input parameters."""
        if cosine_threshold == 1.0 and return_robustness:
            raise ValueError(
                "return_robustness=True is not supported when cosine_threshold == 1.0"
            )
        if cosine_threshold < 0 or cosine_threshold > 1.0:
            raise ValueError("cosine_threshold must be between 0 and 1")

    def _initialize_generation(
        self,
        output_path: Optional[str],
        enable_checkpoints: bool,
        initial_formulae: Optional[List],
        initial_robustness: Optional[torch.Tensor],
        signals: Optional[torch.Tensor],
        signal_samples: int,
        normalize_robustness: bool,
    ) -> Tuple[List, torch.Tensor]:
        """Initialize the generation process with existing data or new formulae."""
        checkpoint_path = (
            os.path.join(output_path, "concepts_checkpoint.pickle")
            if output_path
            else None
        )

        # Try to load from checkpoint
        if enable_checkpoints and checkpoint_path and os.path.exists(checkpoint_path):
            formulae = load_pickle(output_path, "concepts_checkpoint.pickle")
            print(f"Loaded {len(formulae)} formulae from checkpoint")
            return formulae, None

        # Use initial set if provided
        if initial_formulae:
            formulae = deepcopy(initial_formulae)
            robustness_vectors = (
                initial_robustness.detach().clone().to(self.device)
                if initial_robustness is not None
                else None
            )
        else:
            # Start with a random formula
            formulae = [self.sampler.sample_formula(self.nvars)]
            robustness_vectors = None

        # Compute robustness vectors if not provided
        if robustness_vectors is None:
            signals = self._get_signals(signals, signal_samples)
            robustness_vectors = self._compute_robustness_vectors(
                formulae, signals, normalize_robustness
            )

        return formulae, robustness_vectors

    def _get_signals(
        self, signals: Optional[torch.Tensor], signal_samples: int
    ) -> torch.Tensor:
        """Get or generate signals for robustness computation."""
        if signals is not None:
            return signals

        measure = BaseMeasure(device=self.device)
        with torch.no_grad():
            return measure.sample(
                samples=signal_samples, varn=self.nvars, points=self.points
            ).to(self.device)

    def _compute_robustness_vectors(
        self, formulae: List, signals: torch.Tensor, normalize: bool
    ) -> torch.Tensor:
        """Compute robustness vectors for a list of formulae."""
        robustness_vectors = []
        with torch.no_grad():
            for phi in formulae:
                robustness = phi.quantitative(
                    signals, normalize=normalize, vectorize=True
                )
                robustness_vectors.append(robustness.to(self.device))
        return torch.stack(robustness_vectors)

    def _generate_without_filtering(
        self,
        target_dim: int,
        output_path: Optional[str],
    ) -> Tuple[List, Optional[torch.Tensor]]:
        """Generate formulae without cosine similarity filtering."""
        current_formulae = []
        while len(current_formulae) < target_dim:
            batch_size = min(1000, target_dim - len(current_formulae))
            batch = self.sampler.sample_formula_bag(batch_size, self.nvars)

            # Filter by node count and remove tautologies/contradictions
            batch = [phi for phi in batch if find_n_nodes(phi) <= self.max_nodes]
            batch = filter_tautology_contradictions_serial(
                batch,
                points=self.points,
                max_nvars=self.nvars,
                device=self.device,
                samples=1000,
            )
            current_formulae.extend(batch)

        current_formulae = current_formulae[:target_dim]
        if output_path:
            filename = os.path.join(output_path, "concepts")
            dump_pickle(filename, current_formulae)

        return current_formulae, None

    def _generate_with_filtering(
        self,
        target_dim: int,
        cosine_threshold: float,
        promote_simple: bool,
        output_path: Optional[str],
        enable_checkpoints: bool,
        batch_size: int,
        signal_samples: int,
        signals: Optional[torch.Tensor],
        current_formulae: List,
        current_robustness: torch.Tensor,
        normalize_robustness: bool,
        return_robustness: bool,
    ) -> Tuple[List, torch.Tensor]:
        """Generate formulae with cosine similarity filtering."""
        while len(current_formulae) < target_dim:
            remaining = target_dim - len(current_formulae)
            batch = min(batch_size, remaining)

            # Generate candidate formulae
            candidates = self._generate_candidate_batch(batch)
            if not candidates:
                continue

            # Compute robustness for candidates
            signals = self._get_signals(signals, signal_samples)
            candidate_robustness = self._compute_robustness_vectors(
                candidates, signals, normalize_robustness
            )

            # Filter candidates by cosine similarity
            keep_indices = self._filter_by_cosine_similarity(
                candidate_robustness,
                current_robustness,
                cosine_threshold,
                promote_simple,
                current_formulae,
            )

            # Add filtered candidates
            self._add_filtered_candidates(
                candidates,
                candidate_robustness,
                keep_indices,
                current_formulae,
                current_robustness,
            )

            # Clean up memory
            self._cleanup_memory(candidate_robustness)

            # Save checkpoint if enabled
            if enable_checkpoints and output_path:
                checkpoint_path = os.path.join(
                    output_path, "concepts_checkpoint.pickle"
                )
                dump_pickle(checkpoint_path, current_formulae[:target_dim])
                print(f"Checkpoint: {len(current_formulae)} formulae")

        # Trim to target dimension and save final results
        final_formulae = current_formulae[:target_dim]
        final_robustness = current_robustness[:target_dim]

        if output_path:
            filename = os.path.join(output_path, "concepts")
            dump_pickle(filename, final_formulae)

        return final_formulae, final_robustness if return_robustness else None

    def _generate_candidate_batch(self, batch_size: int) -> List:
        """Generate a batch of candidate formulae."""
        candidates = self.sampler.sample_formula_bag(batch_size, self.nvars)

        # Filter by node count
        candidates = [phi for phi in candidates if find_n_nodes(phi) <= self.max_nodes]

        # Remove tautologies and contradictions
        return filter_tautology_contradictions_serial(
            candidates,
            points=self.points,
            max_nvars=self.nvars,
            device=self.device,
            samples=1000,
        )

    def _filter_by_cosine_similarity(
        self,
        candidate_robustness: torch.Tensor,
        current_robustness: torch.Tensor,
        threshold: float,
        promote_simple: bool,
        current_formulae: List,
    ) -> List[int]:
        """Filter candidates based on cosine similarity with current set."""
        # Compute cosine similarity matrix
        similarity_matrix = self._compute_cosine_similarity(
            candidate_robustness, current_robustness
        )

        # Find candidates that are too similar to existing formulae
        is_similar = similarity_matrix > threshold
        similar_indices = is_similar.any(dim=1).cpu().numpy()

        # Determine which candidates to keep
        keep_indices = []
        for i, is_sim in enumerate(similar_indices):
            if not is_sim:
                keep_indices.append(i)
            elif promote_simple:
                # If promoting simple formulae, keep candidate if it's simpler than similar ones
                similar_formula_indices = torch.where(similarity_matrix[i] > threshold)[
                    0
                ]
                similar_formulae = [
                    current_formulae[j] for j in similar_formula_indices.tolist()
                ]
                simplest_existing = min(similar_formulae, key=find_n_nodes)

                if find_n_nodes(
                    self.sampler.sample_formula_bag(1, self.nvars)[i]
                ) < find_n_nodes(simplest_existing):
                    keep_indices.append(i)

        return keep_indices

    def _compute_cosine_similarity(
        self, candidate_robustness: torch.Tensor, current_robustness: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between candidate and current robustness vectors."""
        if len(candidate_robustness.shape) > 2:
            # Flatten last two dimensions for vectorized robustness
            cand_flat = normalize(
                candidate_robustness.reshape(candidate_robustness.shape[0], -1), dim=1
            )
            curr_flat = normalize(
                current_robustness.reshape(current_robustness.shape[0], -1), dim=1
            )
            if self.device == "mps":
                return cand_flat @ curr_flat.t()
            return cand_flat.double() @ curr_flat.t().double()
        else:
            return torch.tril(
                normalize(candidate_robustness) @ normalize(current_robustness).t(),
                diagonal=-1,
            )

    def _add_filtered_candidates(
        self,
        candidates: List,
        candidate_robustness: torch.Tensor,
        keep_indices: List[int],
        current_formulae: List,
        current_robustness: torch.Tensor,
    ) -> None:
        """Add filtered candidates to the current set."""
        keep_formulae = [candidates[i] for i in keep_indices]
        keep_robustness = candidate_robustness[keep_indices]

        current_formulae.extend(keep_formulae)
        current_robustness = torch.cat([current_robustness, keep_robustness], dim=0)

    def _cleanup_memory(self, *tensors) -> None:
        """Clean up GPU memory and garbage collect."""
        for tensor in tensors:
            del tensor
        gc.collect()
        torch.cuda.empty_cache()
