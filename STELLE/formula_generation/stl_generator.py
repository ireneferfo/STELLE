"""
STL Formula Generator - Generates random STL formulae via sampling.
"""

import numpy as np
from typing import List, Optional, Tuple
import random

from .stl import Node, Atom, Not, And, Or, Globally, Eventually, Until


class STLFormulaGenerator:
    """
    Generator for random Signal Temporal Logic (STL) formulae.
    
    Creates formulae by recursively sampling from different node types
    with configurable probability distributions.
    """
    
    def __init__(
        self,
        leaf_probability: float = 0.5,
        node_probabilities: Optional[List[float]] = None,
        threshold_mean: float = 0.0,
        threshold_std: float = 1.0,
        unbound_probability: float = 0.2,
        right_unbound_probability: float = 0.2,
        max_variables: int = 1,
        time_bound_max_range: float = 50,
        adaptive_unbound_temporal_ops: bool = True,
        max_timespan: int = 100,
    ):
        """
        Initialize the STL formula generator.
        
        Args:
            leaf_probability: Probability of generating a leaf node (always 0 for root)
            node_probabilities: Probability distribution for node types:
                ["not", "and", "or", "always", "eventually", "until"]
            threshold_mean: Mean for normal distribution of atomic thresholds
            threshold_std: Standard deviation for normal distribution of atomic thresholds
            unbound_probability: Probability of unbounded temporal operators [0,∞]
            right_unbound_probability: Probability of right-unbound temporal operators
            max_variables: Maximum number of distinct variables in a formula
            time_bound_max_range: Maximum time span for temporal operators
            adaptive_unbound_temporal_ops: If True, unbounded operators computed from 
                current point to signal end; otherwise only at time zero
            max_timespan: Maximum time depth of a formula
        """
        if node_probabilities is None:
            node_probabilities = [0.166, 0.166, 0.166, 0.17, 0.166, 0.166]
            
        self.leaf_probability = leaf_probability
        self.node_probabilities = node_probabilities
        self.threshold_mean = threshold_mean
        self.threshold_std = threshold_std
        self.unbound_probability = unbound_probability
        self.right_unbound_probability = right_unbound_probability
        self.time_bound_max_range = time_bound_max_range
        self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops
        self.max_timespan = max_timespan
        self.max_variables = max_variables
        
        self.node_types = ["not", "and", "or", "always", "eventually", "until"]
        self._validate_probabilities()

    def _validate_probabilities(self) -> None:
        """Validate probability distributions."""
        if len(self.node_probabilities) != len(self.node_types):
            raise ValueError(
                f"node_probabilities must have length {len(self.node_types)}"
            )
        if not np.isclose(sum(self.node_probabilities), 1.0):
            raise ValueError("node_probabilities must sum to 1.0")
        if self.leaf_probability < 0 or self.leaf_probability > 1:
            raise ValueError("leaf_probability must be between 0 and 1")

    def sample_formula(self, num_variables: int) -> Node:
        """
        Sample a random STL formula.
        
        Args:
            num_variables: Number of variables available for the formula
            
        Returns:
            Random STL formula as a Node object
        """
        self._used_variables = []  # Track variables used in current formula
        return self._sample_internal_node(num_variables)

    def sample_formula_bag(
        self, 
        bag_size: int, 
        num_variables: int
    ) -> List[Node]:
        """
        Sample a bag of STL formulae.
        
        Args:
            bag_size: Number of formulae to sample
            num_variables: Number of variables available for formulae
            
        Returns:
            List of random STL formulae
        """
        
        return [self.sample_formula(num_variables) for _ in range(bag_size)]

    def _sample_internal_node(self, num_variables: int) -> Node:
        """
        Recursively sample an internal node (non-leaf).
        
        Args:
            num_variables: Number of variables available
            
        Returns:
            STL formula node
        """
        node_type = np.random.choice(self.node_types, p=self.node_probabilities)
        
        while True:
            node = self._create_node_by_type(node_type, num_variables)
            if node is not None and node.time_depth() < self.max_timespan:
                return node

    def _create_node_by_type(
        self, 
        node_type: str, 
        num_variables: int
    ) -> Optional[Node]:
        """Create a node of the specified type."""
        if node_type == "not":
            child = self._sample_node(num_variables)
            return Not(child)
            
        elif node_type == "and":
            left_child = self._sample_node(num_variables)
            right_child = self._sample_node(num_variables)
            return And(left_child, right_child)
            
        elif node_type == "or":
            left_child = self._sample_node(num_variables)
            right_child = self._sample_node(num_variables)
            return Or(left_child, right_child)
            
        elif node_type == "always":
            child = self._sample_node(num_variables)
            temporal_params = self._sample_temporal_parameters()
            return Globally(child, *temporal_params, self.adaptive_unbound_temporal_ops)
            
        elif node_type == "eventually":
            child = self._sample_node(num_variables)
            temporal_params = self._sample_temporal_parameters()
            return Eventually(child, *temporal_params, self.adaptive_unbound_temporal_ops)
            
        elif node_type == "until":
            left_child = self._sample_node(num_variables)
            right_child = self._sample_node(num_variables)
            temporal_params = self._sample_temporal_parameters()
            return Until(left_child, right_child, *temporal_params)
            
        return None

    def _sample_node(self, num_variables: int) -> Node:
        """
        Sample either a leaf node or internal node based on probability.
        
        Args:
            num_variables: Number of variables available
            
        Returns:
            STL formula node (atomic or compound)
        """
        if np.random.rand() < self.leaf_probability:
            return self._sample_atomic_formula(num_variables)
        else:
            return self._sample_internal_node(num_variables)

    def _sample_temporal_parameters(self) -> Tuple[bool, bool, int, int]:
        """
        Sample temporal bounds for temporal operators.
        
        Returns:
            Tuple of (unbound, right_unbound, left_bound, right_bound)
        """
        if np.random.rand() < self.unbound_probability:
            return True, False, 0, 0
        elif np.random.rand() < self.right_unbound_probability:
            left_bound = np.random.randint(self.max_timespan - self.time_bound_max_range)
            return False, True, left_bound, 1
        else:
            left_bound = np.random.randint(0, self.max_timespan - self.time_bound_max_range - 1)
            right_bound = np.random.randint(
                left_bound + 1, left_bound + self.time_bound_max_range
            )
            return False, False, left_bound, right_bound

    def _sample_atomic_formula(self, num_variables: int) -> Atom:
        """
        Sample an atomic formula (variable comparison with threshold).
        
        Args:
            num_variables: Number of variables available
            
        Returns:
            Atomic STL formula
        """
        # Choose variable: prefer unused ones if we haven't reached max_variables
        if len(self._used_variables) < self.max_variables:
            variable = np.random.randint(num_variables)
            self._used_variables.append(variable)
        else:
            variable = random.choice(self._used_variables)
            
        # Randomly choose comparison direction
        less_than_equal = np.random.rand() > 0.5
        
        # Sample threshold from normal distribution
        threshold = np.random.normal(self.threshold_mean, self.threshold_std)
        
        return Atom(variable, threshold, less_than_equal)