import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .base_model import BaseConceptModel, BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """Configuration for model initialization."""
    pass # nothing to add


class ConceptBasedModel(BaseConceptModel):
    """
    Neural network model with concept-based explanations.

    This model uses temporal embeddings and concept_relevance mechanisms to classify
    trajectories while providing interpretable concept-based explanations.
    """

    def _init_variant_parameters(self, config):
        pass


    def _compute_temporal_embeddings(
        self, x: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        """Compute temporal embeddings for input trajectories."""
        if trainingmode:
            # Use cached embeddings during training
            temb = torch.stack(
                [self.temb_dict[tuple(map(tuple, traj.tolist()))] for traj in x]
            ).to(self.device)
        else:
            # Compute embeddings on the fly during inference
            epsilon = self.epsilon if self.epsilon else F.softplus(self.log_eps)
            with torch.no_grad():
                temb = (
                    self.kernel.compute_rho_phi(
                        x,
                        self.concepts,
                        epsilon,
                        save=True,
                        parallel_workers=self.parallel_workers,
                    )
                    .to(self.device, non_blocking=True)
                    .float()
                )

        return temb
    
    def _compute_G_phis_matrix(
        self, x: torch.Tensor, concept_relevance: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        """Compute G_phis matrix based on configuration."""
        
        if trainingmode:
            return torch.stack(
                [
                    self._compute_G_phis_from_robs(
                        self.robs_dict.get(tuple(map(tuple, traj.tolist())))
                    )
                    for traj in x
                ]
            ).to(self.device, non_blocking=True)
        else:
            with torch.no_grad():
                robs = self._compute_selective_robustness(x, concept_relevance)
                return self._compute_G_phis_from_robs(robs).to(self.device)