"""
Formula Adapters - Adapt STL formulae to different dataset characteristics.
"""

import math
from typing import List

from ..formula_generation.formula_utils import from_string_to_formula


class FormulaTimeScaler:
    """
    Adapts STL formulae time intervals to match dataset characteristics.
    """
    
    def __init__(self, target_time_points: int, reference_time_points: int = 101):
        """
        Initialize the time scaler.
        
        Args:
            target_time_points: Number of time points in the target dataset
            reference_time_points: Number of time points the formulae were designed for
        """
        self.target_time_points = target_time_points
        self.reference_time_points = reference_time_points
        self.scaling_factor = target_time_points / reference_time_points

    def scale_formulae(self, formulae: List) -> List:
        """
        Scale time intervals in STL formulae to match target dataset.
        
        Args:
            formulae: List of STL formulae to scale
            
        Returns:
            List of formulae with scaled time intervals
        """
        scaled_formulae = []
        
        for formula in formulae:
            scaled_formula = self._scale_single_formula(formula)
            scaled_formulae.append(scaled_formula)
            
        return scaled_formulae

    def _scale_single_formula(self, formula) -> object:
        """
        Scale time intervals for a single formula.
        
        Args:
            formula: STL formula to scale
            
        Returns:
            Formula with scaled time intervals
        """
        formula_string = str(formula)
        
        # Find temporal interval boundaries
        interval_starts = [i for i in range(len(formula_string)) if formula_string.startswith("[", i)]
        interval_middles = [i for i in range(len(formula_string)) if formula_string.startswith(",", i)]
        interval_ends = [i for i in range(len(formula_string)) if formula_string.startswith("]", i)]
        
        if not interval_starts:
            return formula  # No temporal intervals to scale
            
        # Start building the new formula string
        start_idx = interval_starts[0]
        formula_parts = [formula_string[:start_idx]]
        new_intervals = []
        
        # Process each temporal interval
        for i, (start, middle, end) in enumerate(zip(
            interval_starts, interval_middles, interval_ends
        )):
            # Check if interval is right-unbounded
            is_right_unbounded = formula_string[end - 1] == 'f'
            
            # Extract interval bounds
            left_bound = float(formula_string[start + 1:middle])
            right_bound = -1.0 if is_right_unbounded else float(formula_string[middle + 1:end])
            
            # Scale the interval
            scaled_left = math.floor(left_bound * self.scaling_factor)
            
            if is_right_unbounded:
                scaled_right = "inf"
            else:
                interval_width = right_bound - left_bound
                scaled_width = max(math.floor(interval_width * self.scaling_factor), 1)
                scaled_right = min(scaled_left + scaled_width, self.target_time_points)
                scaled_right = str(scaled_right)
            
            # Create new interval string
            new_interval = f"[{scaled_left},{scaled_right}]"
            new_intervals.append(new_interval)
            
            # Extract the text after this interval
            next_start = interval_starts[i + 1] if i < len(interval_starts) - 1 else None
            formula_parts.append(formula_string[end + 1:next_start])
        
        # Reconstruct the formula string
        scaled_formula_string = ""
        for i in range(len(new_intervals)):
            scaled_formula_string += formula_parts[i] + new_intervals[i]
        scaled_formula_string += formula_parts[-1]
        
        # Convert back to formula object
        return from_string_to_formula(scaled_formula_string)


class FormulaVariableAdapter:
    """
    Adapts STL formulae to work with different variable configurations.
    """
    
    @staticmethod
    def create_variable_permutations(
        base_formulae: List, 
        num_variables: int,
        variables_per_formula: int = 1
    ) -> List:
        """
        Create permutations of formulae across different variables.
        
        Args:
            base_formulae: Base formulae to permute
            num_variables: Total number of available variables
            variables_per_formula: Number of variables each formula uses
            
        Returns:
            List of permuted formulae
        """
        # This would implement the variable permutation logic
        # For now, return base formulae (implementation depends on specific needs)
        return base_formulae

    @staticmethod
    def adapt_formula_thresholds(
        formulae: List,
        dataset_statistics: dict,
        normalization_factor: float = 1.0
    ) -> List:
        """
        Adapt formula thresholds based on dataset statistics.
        
        Args:
            formulae: Formulae to adapt
            dataset_statistics: Statistics about the dataset variables
            normalization_factor: Factor to adjust threshold scaling
            
        Returns:
            List of formulae with adapted thresholds
        """
        # This would implement threshold adaptation based on dataset characteristics
        # For now, return original formulae
        return formulae