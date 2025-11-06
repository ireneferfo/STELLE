"""
Local Explanation - Single trajectory explanations using STL formulae.
"""

import torch
import warnings
from typing import List, Optional, Dict

from .base_explanations import ExplanationBase, ExplanationResult
from ..formula_generation.stl import Not
from ECATS.stl_utils import (
    rescale_var_thresholds,
    conjunction,
    simplify,
    evaluate_and_simplify,
    contains_boolean,
    inverse_normalize_phis,
)
from .explanation_metrics import readability


class LocalExplanation(ExplanationBase):
    """
    Generates local explanations for individual trajectories using STL formulae.
    
    Finds formulae that separate a target trajectory from trajectories of other classes.
    """
    
    def __init__(
        self,
        trajectory: torch.Tensor,
        true_label: Optional[int],
        predicted_label: int,
        candidate_formulae: List,
        trajectories_by_class: Dict[int, List[torch.Tensor]],
        normalize_robustness: bool = False,
    ):
        """
        Initialize local explanation generator.
        
        Args:
            trajectory: Target trajectory to explain
            true_label: True class label (if known)
            predicted_label: Predicted class label
            candidate_formulae: Candidate STL formulae for explanation
            trajectories_by_class: Training trajectories grouped by class
            normalize_robustness: Whether to normalize robustness values
        """
        super().__init__(normalize_robustness)
        
        self.trajectory = trajectory
        self.true_label = true_label
        self.predicted_label = predicted_label
        self.candidate_formulae = candidate_formulae
        self.trajectories_by_class = trajectories_by_class
        
        self.num_classes = len(trajectories_by_class)
        self.target_class = true_label if true_label is not None else predicted_label
        self.is_misclassified = (
            true_label is not None and true_label != predicted_label
        )
        
        # Explanation results
        self._processed_formulae = None
        self.explanation_result = None
        self._robustness_cache = None
        
        # Precompute robustness values
        self._precompute_robustness()
    
    def _precompute_robustness(self) -> None:
        """Precompute robustness values for candidate formulae."""
        if not self.candidate_formulae:
            self._robustness_cache = torch.tensor([], device=self.device)
            return
        
        trajectory_batch = self.trajectory.unsqueeze(0)
        self._robustness_cache = torch.stack([
            self.compute_robustness(formula, trajectory_batch)
            for formula in self.candidate_formulae
        ]).squeeze()
    
    def generate_explanation(
        self,
        improvement_threshold: float = 0.01,
        enable_postprocessing: bool = True
    ) -> Optional[ExplanationResult]:
        """
        Generate explanation for the target trajectory.
        
        Args:
            improvement_threshold: Minimum improvement required to add new formula
            enable_postprocessing: Whether to apply postprocessing and simplification
            
        Returns:
            Explanation result or None if no explanation found
        """
        if self.explanation_result is not None:
            return self.explanation_result
        
        if not self.candidate_formulae:
            self._set_empty_explanation()
            return None
        
        self.cleanup_memory()
        
        # Process candidate formulae
        processed_formulae = self._postprocess_formulae(self.candidate_formulae)
        explanation_formula = self._build_explanation_formula(
            processed_formulae, improvement_threshold
        )
        
        if explanation_formula is None:
            self._set_empty_explanation()
            return None
        
        # Finalize explanation
        if enable_postprocessing:
            explanation_result = self._finalize_explanation(explanation_formula)
        else:
            explanation_result = self._create_explanation_result(explanation_formula)
        
        self.explanation_result = explanation_result
        return explanation_result
    
    def _postprocess_formulae(self, formulae: List) -> List:
        """
        Post-process formulae to maximize separation.
        
        Args:
            formulae: Formulae to process
            
        Returns:
            Post-processed formulae
        """
        if not formulae:
            return []
        
        trajectory_batch = self.trajectory.unsqueeze(0)
        processed_formulae = []
        
        for formula in formulae:
            # Compute target robustness
            target_robustness = self.compute_robustness(formula, trajectory_batch)
            
            # Compute opponent robustness values
            opponent_robustness = []
            for class_label, class_trajectories in self.trajectories_by_class.items():
                if class_label == self.target_class:
                    continue
                for traj in class_trajectories:
                    robustness = self.compute_robustness(formula, traj.unsqueeze(0))
                    opponent_robustness.append(robustness)
            
            if opponent_robustness:
                opponent_robustness = torch.cat(opponent_robustness)
            else:
                opponent_robustness = torch.tensor([], device=self.device)
            
            # Determine if negation improves separation
            less_than_count = (opponent_robustness < target_robustness).sum().item()
            greater_than_count = (opponent_robustness > target_robustness).sum().item()
            
            if greater_than_count > less_than_count:
                formula = Not(formula)
                target_robustness = -target_robustness
                opponent_robustness = -opponent_robustness
            
            # Find optimal threshold translation
            if opponent_robustness.numel() > 0:
                valid_opponents = opponent_robustness[opponent_robustness < target_robustness]
                if valid_opponents.numel() > 0:
                    closest_opponent = torch.max(valid_opponents).item()
                    translation = (target_robustness.item() + closest_opponent) / 2
                else:
                    translation = target_robustness.item()
            else:
                translation = target_robustness.item()
            
            # Apply translation
            translated_formula = rescale_var_thresholds(formula, -translation)
            final_robustness = self.compute_robustness(translated_formula, trajectory_batch)
            
            # Ensure positive evaluation
            if final_robustness.item() < 0:
                translated_formula = Not(translated_formula)
            
            processed_formulae.append(simplify(translated_formula))
        
        return processed_formulae
    
    def _build_explanation_formula(
        self,
        processed_formulae: List,
        improvement_threshold: float
    ):
        """
        Build explanation formula by sequentially adding formulae.
        
        Args:
            processed_formulae: Post-processed formulae
            improvement_threshold: Minimum improvement threshold
            
        Returns:
            Combined explanation formula or None
        """
        if not processed_formulae:
            return None
        
        selected_formulae = []
        current_formula = None
        previous_score = 0.0
        trajectory_batch = self.trajectory.unsqueeze(0)
        
        for formula in processed_formulae:
            # Add formula to current combination
            if current_formula is None:
                current_formula = formula
            else:
                current_formula = conjunction(selected_formulae + [formula])
            
            if current_formula is None:
                continue
            
            # Compute robustness for current combination
            target_robustness = self.compute_robustness(current_formula, trajectory_batch)
            
            # Compute opponent robustness
            opponent_robustness = self._compute_opponent_robustness(current_formula)
            
            # Calculate current separation score
            current_score = self.calculate_separation_score(
                target_robustness, opponent_robustness
            )
            
            # Check stopping conditions
            if self.is_perfect_separation(target_robustness, opponent_robustness):
                selected_formulae.append(formula)
                break
            
            improvement = current_score - previous_score
            if improvement < improvement_threshold and selected_formulae:
                # Insufficient improvement, keep previous combination
                current_formula = conjunction(selected_formulae)
                break
            
            selected_formulae.append(formula)
            previous_score = current_score
        
        return conjunction(selected_formulae) if selected_formulae else None
    
    def _compute_opponent_robustness(self, formula) -> torch.Tensor:
        """
        Compute robustness values for opponent classes.
        
        Args:
            formula: Formula to evaluate
            
        Returns:
            Robustness values for all opponent trajectories
        """
        opponent_robustness = []
        
        for class_label, class_trajectories in self.trajectories_by_class.items():
            if class_label == self.target_class:
                continue
            
            if class_trajectories:
                class_tensor = torch.stack(class_trajectories)
                robustness = self.compute_robustness(formula, class_tensor)
                opponent_robustness.append(robustness.cpu())  # Move to CPU to save GPU memory
        
        return torch.cat(opponent_robustness) if opponent_robustness else torch.tensor([])
    
    def _finalize_explanation(self, explanation_formula) -> ExplanationResult:
        """
        Finalize explanation with simplification and metrics.
        
        Args:
            explanation_formula: Raw explanation formula
            
        Returns:
            Finalized explanation result
        """
        # Prepare data for simplification
        all_trajectories = []
        for class_label, class_trajectories in self.trajectories_by_class.items():
            if class_label == self.target_class:
                all_trajectories.append(self.trajectory.unsqueeze(0))
            else:
                all_trajectories.extend([traj.unsqueeze(0) for traj in class_trajectories])
        
        if all_trajectories:
            all_trajectories_tensor = torch.cat(all_trajectories)
            simplified_formula = evaluate_and_simplify(
                explanation_formula, all_trajectories_tensor, "and"
            )
        else:
            simplified_formula = explanation_formula
        
        # Verify simplification preserves separation
        final_formula = self._verify_simplified_formula(
            explanation_formula, simplified_formula
        )
        
        return self._create_explanation_result(final_formula)
    
    def _verify_simplified_formula(self, original_formula, simplified_formula):
        """
        Verify that simplified formula preserves separation properties.
        
        Args:
            original_formula: Original explanation formula
            simplified_formula: Simplified formula
            
        Returns:
            Verified formula (simplified if valid, original otherwise)
        """
        if str(original_formula) == str(simplified_formula):
            return original_formula
        
        if contains_boolean(simplified_formula):
            return original_formula
        
        try:
            # Check if simplification preserves robustness signs
            original_target = self.compute_robustness(original_formula, self.trajectory.unsqueeze(0))
            simplified_target = self.compute_robustness(simplified_formula, self.trajectory.unsqueeze(0))
            
            original_opponents = self._compute_opponent_robustness(original_formula)
            simplified_opponents = self._compute_opponent_robustness(simplified_formula)
            
            # Check if signs are preserved
            original_signs = torch.sign(torch.cat([original_target, original_opponents]))
            simplified_signs = torch.sign(torch.cat([simplified_target, simplified_opponents]))
            
            if torch.all(original_signs == simplified_signs):
                return simplified_formula
            else:
                return original_formula
                
        except Exception as e:
            warnings.warn(f"Simplification verification failed: {e}")
            return original_formula
    
    def _create_explanation_result(self, explanation_formula) -> ExplanationResult:
        """
        Create explanation result with metrics.
        
        Args:
            explanation_formula: Final explanation formula
            
        Returns:
            Explanation result with computed metrics
        """
        target_robustness = self.compute_robustness(
            explanation_formula, self.trajectory.unsqueeze(0)
        )
        opponent_robustness = self._compute_opponent_robustness(explanation_formula)
        
        separation_percentage = self._calculate_separation_percentage(
            target_robustness, opponent_robustness
        )
        readability_score = readability(explanation_formula, "and")
        
        return ExplanationResult(
            formula=explanation_formula,
            target_robustness=target_robustness,
            opponent_robustness=opponent_robustness,
            readability_score=readability_score,
            separation_percentage=separation_percentage
        )
    
    def _calculate_separation_percentage(
        self,
        target_robustness: torch.Tensor,
        opponent_robustness: torch.Tensor
    ) -> float:
        """
        Calculate percentage of successful separation.
        
        Args:
            target_robustness: Robustness for target
            opponent_robustness: Robustness for opponents
            
        Returns:
            Separation percentage (0-100)
        """
        if opponent_robustness.numel() == 0:
            return 0.0
        
        target_sign = torch.sign(target_robustness).item()
        opponent_signs = torch.sign(opponent_robustness)
        opposite_sign_count = (opponent_signs != target_sign).sum().item()
        total_opponents = len(opponent_robustness)
        
        return (opposite_sign_count / total_opponents) * 100 if total_opponents > 0 else 0.0
    
    def _set_empty_explanation(self) -> None:
        """Set empty explanation result."""
        self.explanation_result = ExplanationResult(
            formula=None,
            target_robustness=torch.tensor(0.0),
            opponent_robustness=torch.tensor([]),
            readability_score=None,
            separation_percentage=0.0
        )
    
    def inverse_normalize_formula(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Inverse normalize formula thresholds.
        
        Args:
            mean: Mean values for denormalization
            std: Standard deviation values for denormalization
            
        Returns:
            Denormalized formula
        """
        if self.explanation_result and self.explanation_result.formula:
            formulae = inverse_normalize_phis(mean, std, [self.explanation_result.formula])
            return formulae[0] if formulae else None
        return None