import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from .base_model import BaseConceptModel, BaseModelConfig

#! da testare tutte le explanations globali

class ConceptBasedModel_RobsAsGx(BaseConceptModel):

    def _compute_G_phis_matrix(
        self, x: torch.Tensor, concept_relevance: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        """Compute G_phis using only robustness values."""
        if trainingmode:
            G_phis = torch.stack(
                [self.robs_dict.get(tuple(map(tuple, traj.tolist()))) for traj in x]
            ).to(self.device, non_blocking=True)
        else:
            with torch.no_grad():
                G_phis = self._compute_selective_robustness(x, concept_relevance)

        return G_phis.unsqueeze(-1).expand(-1, -1, self.num_classes)


class ConceptBasedModel_NoGx(BaseConceptModel):
    
    def _compute_G_phis_matrix(
        self, x: torch.Tensor, concept_relevance: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        """Compute identity G_phis matrix."""
        G_phis = torch.eye(len(self.concepts), self.num_classes, device=self.device)
        return G_phis.unsqueeze(0).expand(x.size(0), -1, -1)


class ConceptBasedModel_RobsAsHx(BaseConceptModel):
    
    def _compute_temporal_embeddings(
        self, x: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        if trainingmode:
            x_robs = torch.stack(
                [self.robs_dict.get(tuple(map(tuple, traj.tolist()))) for traj in x]
            ).to(self.device, non_blocking=True)
        else:
            with torch.no_grad():
                x_robs = torch.zeros(x.size(0), len(self.concepts), device=self.device)
                for idx, phi in enumerate(self.concepts):
                    # Only compute if any sample in batch has concept_relevance > threshold for this concept
                    x_robs[:, idx] = (
                        phi.quantitative(
                            x,
                            evaluate_at_all_times=False,
                            vectorize=False,
                            normalize=self.normalize,
                        )
                        .squeeze()
                        .to(self.device, non_blocking=True)
                    )
                    
        return x_robs


class ConceptBasedModel_Robs(BaseConceptModel):
    
    def forward(
        self, x: torch.Tensor, trainingmode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = x.to(self.device)

        # Compute temporal embeddings (variant-specific)
        robs = self._compute_temporal_embeddings(x, trainingmode)

        # Apply dropout
        robs_dropped = self.dropout(robs)

        # G_phis as identity
        G_phis = torch.eye(len(self.concepts), self.num_classes, device=self.device)
        G_phis = G_phis.unsqueeze(0).expand(x.size(0), -1, -1)

        # Compute concept_relevance-weighted features
        crelG_raw = self._compute_weighted_features(robs_dropped, G_phis)

        # Apply output activation and classify
        crelG = self.output_activation(crelG_raw).to(self.device)
        class_scores = self.classifier(crelG.float().to(self.device))
        
        if trainingmode:
            return class_scores 
        return class_scores, robs, crelG_raw, G_phis
     
    def _compute_temporal_embeddings(
        self, x: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
        if trainingmode:
            x_robs = torch.stack(
                [self.robs_dict.get(tuple(map(tuple, traj.tolist()))) for traj in x]
            ).to(self.device, non_blocking=True)
        else:
            with torch.no_grad():
                x_robs = torch.zeros(x.size(0), len(self.concepts), device=self.device)
                for idx, phi in enumerate(self.concepts):
                    # Only compute if any sample in batch has concept_relevance > threshold for this concept
                    x_robs[:, idx] = (
                        phi.quantitative(
                            x,
                            evaluate_at_all_times=False,
                            vectorize=False,
                            normalize=self.normalize,
                        )
                        .squeeze()
                        .to(self.device, non_blocking=True)
                    )
                    
        return x_robs


@dataclass
class AnchorModelConfig(BaseModelConfig):
    """Configuration for model initialization."""
    base_concepts: Any = None
    concept_embeddings: Any = None


class ConceptBasedModel_Anchor(BaseConceptModel):
    """
    Ablation model combining temporal embeddings and G_phi.

    This model uses temporal embeddings and concept_relevance mechanisms to classify
    trajectories while providing interpretable concept-based explanations.
    """

    def __init__(self, config: AnchorModelConfig):
        super().__init__(config)
        
        self.base_concepts = config.base_concepts
        self.concept_embeddings = config.concept_embeddings.clone().detach().to(self.device)

    def forward(
        self, x: torch.Tensor, trainingmode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input trajectories
            trainingmode: Whether in training mode (uses cached embeddings)

        Returns:
            Tuple of (class_scores, concept_relevance, temb, crelG_raw, G_phis)
        """
        x = x.to(self.device)

        # Compute temporal embeddings
        temb = self._compute_temporal_embeddings(x, trainingmode)
        # Compute product of trajectory embedding and concepts embedding
        if self.device == 'mps':
            concept_product = torch.matmul(temb, self.concept_embeddings.t())
        else:
            concept_product = torch.matmul(temb.double(), self.concept_embeddings.t().double())
        concept_relevance = self._compute_concept_relevance(concept_product)

        # Apply dropout
        concept_relevance_dropped = self.dropout(concept_relevance)

        # Compute G_phis based on robustness values
        G_phis = self._compute_G_phis_matrix(x, concept_relevance, trainingmode)

        # Compute concept_relevance-weighted features
        crelG_raw = self._compute_weighted_features(concept_relevance_dropped, G_phis)

        # Apply output activation and classify
        crelG = self.output_activation(crelG_raw).to(self.device)
        class_scores = self.classifier(crelG.float().to(self.device))

        # Return the class scores, concept_relevance weights, temporal embeddings, and concept_relevance-weighted G_phi values
        if trainingmode:
            return class_scores
        return class_scores, concept_relevance, crelG_raw, G_phis

    def _compute_temporal_embeddings(
        self, x: torch.Tensor, trainingmode: bool
    ) -> torch.Tensor:
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
                        self.base_concepts,
                        epsilon,
                        save=True,
                        parallel_workers=self.parallel_workers,
                    )
                    .to(self.device, non_blocking=True)
                    .float()
                )

        return temb

    def _update_temporal_embeddings(
        self, trainloader, last_epsilon: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Update temporal embeddings if epsilon changed."""
        epsilon = self.epsilon if self.epsilon else F.softplus(self.log_eps)

        if last_epsilon is None or not torch.allclose(last_epsilon, epsilon):
            self.eval()
            temb_dict = {}

            for batch in trainloader:
                trajectories = batch[0].to(self.device)
                temb_batch = (
                    self.kernel.compute_rho_phi(
                        trajectories,
                        self.base_concepts,
                        epsilon,
                        save=True,
                        parallel_workers=self.parallel_workers,
                    )
                    .to(self.device, non_blocking=True)
                    .to(torch.float)
                )

                for i, traj in enumerate(trajectories):
                    temb_dict[tuple(map(tuple, traj.cpu().tolist()))] = temb_batch[i]

            self.temb_dict = temb_dict

            if self.logging:
                temb_std = torch.stack(list(temb_dict.values())).std()
                print(f"Computed temporal embeddings (std: {temb_std.item():.6f})")

            return epsilon

        return last_epsilon     


# TODO andranno identità di shape giuste
# TODO matching di shapes non sarà banale
class ConceptBasedModel_AltExplanations(BaseConceptModel):

    def get_explanations(
        self,
        x: torch.Tensor,
        trajbyclass,
        layer,
        type: str,
        y_true: Optional[torch.Tensor] = None,
        getmatrix: bool = False,
        k: Optional[int] = None,
        t_k: float = 0.9,
        method: str = "ig",
        filter_onlycorrect: bool = False,
        op: str = "mean",
        norm: bool = False,
    ):
        """
        Generate concept-based explanations for predictions.

        Args:
            x: Input trajectories
            trajbyclass: Trajectory examples by class
            layer: Layer to explain
            type: way of extracting the explanation. "crel" (concept_relevance only), "lw" (backprop weights only), "crelGx" (crel x Gx), "Gxlw" (Gx x backprop), "crellw" (crel x backprop)
            y_true: True labels (optional)
            getmatrix: Whether to return attribution matrix
            k: Number of top concepts (None for adaptive)
            t_k: Cumulative score threshold
            method: Attribution method ('ig', 'deeplift', 'og', 'random', 'identity')
            filter_onlycorrect: Only explain correct predictions
            op: Comparison operation ('mean', 'max', or None)
            norm: Whether to normalize attribution scores

        Returns:
            Top concept indices, attribution weights, and explanations
        """
        self.eval()

        # Get model predictions
        with torch.no_grad():
            x = x.to(self.device)
            class_scores, concept_relevance, crelGs_raw, G_phis = self.forward(x, trainingmode=False)
            
        y_pred = class_scores.argmax(dim=1).cpu()
        
        if type == 'crelGx':
            term1 = self.output_activation(crelGs_raw) if norm else crelGs_raw
        elif type in ['crel', 'crellw']:
            term1 = self.output_activation(concept_relevance) if norm else concept_relevance
        else:
            term1 = G_phis
            
        # Filter for correct predictions if requested
        if filter_onlycorrect and y_true is not None:
            x, y_pred, term1, class_scores = self._filter_correct_predictions(
                x, y_true, y_pred, term1, class_scores
            )
            
        # Compute attributions
        x_requires_grad = x.requires_grad_()
        targets = y_pred if y_true is None else y_pred

        if type in['lw', 'Gxlw', 'crellw']:
            term2 = self._compute_attributions(
                x_requires_grad, targets, layer, method
            )
        else:
            term2 = 1
            
        # Compute final attribution matrix
        final_matrix = self._compute_final_attributions(
            term1, term2, y_pred
        )
        # Get discriminative scores
        discriminative_scores = self._compute_discriminative_scores(
            final_matrix, targets, op
        )

        if getmatrix:
            grouped_matrix = self._group_matrix_by_class(final_matrix, targets)
            return grouped_matrix, y_pred

        # Generate explanations
        explanations = self._generate_explanations(
            x, y_true, y_pred, discriminative_scores, trajbyclass, k, t_k
        )

        return explanations
