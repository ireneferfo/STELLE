"""
Base Explanation Classes - Core functionality for STL-based explanations.
"""

import torch
from typing import List, Optional, Dict


class ExplanationBase:
    """
    Base class for STL-based explanations with common functionality.
    
    Provides shared methods for robustness computation, normalization,
    and basic explanation properties.
    """
    
    def __init__(self, normalize_robustness: bool = False):
        """
        Initialize base explanation class.
        
        Args:
            normalize_robustness: Whether to normalize robustness values
        """
        self.normalize_robustness = normalize_robustness
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_robustness(
        self, 
        formula, 
        trajectories: torch.Tensor,
        vectorize: bool = False
    ) -> torch.Tensor:
        """
        Compute robustness values for a formula on given trajectories.
        
        Args:
            formula: STL formula to evaluate
            trajectories: Input trajectories tensor
            vectorize: Whether to return vectorized robustness
            
        Returns:
            Robustness values tensor
        """
        return formula.quantitative(
            trajectories,
            evaluate_at_all_times=False,
            vectorize=vectorize,
            normalize=self.normalize_robustness,
        )
    
    def compute_batch_robustness(
        self,
        formulae: List,
        trajectories: torch.Tensor,
        vectorize: bool = False
    ) -> torch.Tensor:
        """
        Compute robustness for multiple formulae in batch.
        
        Args:
            formulae: List of STL formulae
            trajectories: Input trajectories tensor
            vectorize: Whether to return vectorized robustness
            
        Returns:
            Robustness matrix of shape (num_formulae, num_trajectories)
        """
        robustness_matrix = torch.stack([
            self.compute_robustness(formula, trajectories, vectorize)
            for formula in formulae
        ])
        return robustness_matrix
    
    def calculate_separation_score(
        self,
        target_robustness: torch.Tensor,
        opponent_robustness: torch.Tensor
    ) -> float:
        """
        Calculate separation score between target and opponent robustness values.
        
        Args:
            target_robustness: Robustness values for target class
            opponent_robustness: Robustness values for opponent classes
            
        Returns:
            Separation score (higher is better)
        """
        if not torch.is_tensor(target_robustness):
            target_robustness = torch.tensor(target_robustness, device=self.device)
        if not torch.is_tensor(opponent_robustness):
            opponent_robustness = torch.tensor(opponent_robustness, device=self.device)
        
        # Flatten and remove invalid values
        target_flat = target_robustness.flatten()
        opponent_flat = opponent_robustness.flatten()
        opponent_flat = opponent_flat[torch.isfinite(opponent_flat)]
        
        if opponent_flat.numel() == 0:
            return 0.0
        
        # Count targets that are clearly separated from opponents
        max_opponent = torch.max(opponent_flat)
        min_opponent = torch.min(opponent_flat)
        
        clearly_separated = (
            (target_flat > max_opponent).sum() + 
            (target_flat < min_opponent).sum()
        ).item()
        
        return float(clearly_separated)
    
    def is_perfect_separation(
        self,
        target_robustness: torch.Tensor,
        opponent_robustness: torch.Tensor
    ) -> bool:
        """
        Check if perfect separation between classes is achieved.
        
        Args:
            target_robustness: Robustness value for target
            opponent_robustness: Robustness values for opponents
            
        Returns:
            True if perfect separation is achieved
        """
        if opponent_robustness.numel() == 0:
            return False
        
        target_value = target_robustness.item() if target_robustness.numel() == 1 else target_robustness
        min_opponent = torch.min(opponent_robustness).item()
        max_opponent = torch.max(opponent_robustness).item()
        
        return (
            min_opponent == max_opponent or
            not (min_opponent <= target_value <= max_opponent)
        )
    
    def cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ExplanationResult:
    """
    Container for explanation results with metrics.
    """
    
    def __init__(
        self,
        formula,
        target_robustness: torch.Tensor,
        opponent_robustness: torch.Tensor,
        readability_score: Optional[float] = None,
        separation_percentage: Optional[float] = None
    ):
        """
        Initialize explanation result.
        
        Args:
            formula: The explanation formula
            target_robustness: Robustness for target class
            opponent_robustness: Robustness for opponent classes
            readability_score: Readability metric for the formula
            separation_percentage: Percentage of successful separation
        """
        self.formula = formula
        self.target_robustness = target_robustness
        self.opponent_robustness = opponent_robustness
        self.readability_score = readability_score
        self.separation_percentage = separation_percentage
    
    @property
    def formula_string(self) -> str:
        """Get string representation of the formula."""
        return str(self.formula)
    
    def get_separation_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive separation metrics.
        
        Returns:
            Dictionary of separation metrics
        """
        if self.opponent_robustness.numel() == 0:
            return {
                "separation_percentage": 0.0,
                "target_mean": self.target_robustness.mean().item(),
                "opponent_mean": 0.0,
                "separation_gap": 0.0
            }
        
        target_positive = (self.target_robustness >= 0).float().mean().item()
        opponent_negative = (self.opponent_robustness < 0).float().mean().item()
        
        separation_gap = (
            self.target_robustness.mean() - self.opponent_robustness.mean()
        ).item()
        
        return {
            "separation_percentage": self.separation_percentage or 0.0,
            "target_positive_rate": target_positive,
            "opponent_negative_rate": opponent_negative,
            "separation_gap": separation_gap
        }