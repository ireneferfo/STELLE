"""
Class Explanation - Class-level explanations from local explanations.
"""

import torch
import re
import itertools
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    LpStatus,
    PULP_CBC_CMD,
)

from .base_explanations import ExplanationBase, ExplanationResult
from .local_explanation import LocalExplanation
from ..formula_generation.stl import Not, Boolean
from ..formula_generation.formula_utils import find_n_nodes
from ..formula_generation.formula_manipulation import (
    rescale_var_thresholds,
    disjunction,
    simplify,
    evaluate_and_simplify,
    rhos_disjunction,
)
from .explanation_metrics import readability, _division_perc


class ClassExplanation(ExplanationBase):
    """
    Generates class-level explanations from local explanations.

    Combines multiple local explanations into a unified class explanation
    using set cover optimization.
    """

    def __init__(
        self,
        local_explanations: List[LocalExplanation],
        is_training: bool = True,
        normalize_robustness: bool = False,
    ):
        """
        Initialize class explanation generator.

        Args:
            local_explanations: List of local explanations for the class
            is_training: Whether explanations are from training data
            normalize_robustness: Whether to normalize robustness values
        """
        super().__init__(normalize_robustness)

        self.local_explanations = local_explanations
        self.is_training = is_training

        if local_explanations:
            self.trajectories_by_class = local_explanations[0].trajectories_by_class
            self.num_classes = len(self.trajectories_by_class)
        else:
            self.trajectories_by_class = {}
            self.num_classes = 0

        # Precomputed data
        self._precompute_class_data()

        # Explanation results
        self.class_explanations = {}
        self._processed_formulae_by_class = {}
        self.explanation_readability_pre = {}
        self.explanation_readability_post = {}
        self.local_separation_percentages = self._compute_local_separation_percentages()
        self.global_separation_percentages = None

    def _precompute_class_data(self) -> None:
        """Precompute class-specific data from local explanations."""
        self.trajectories_by_predicted_class = {}
        self.local_explanations_by_class = {}
        self.formulae_by_class = {}

        for class_label in range(self.num_classes):
            class_explanations = [
                exp
                for exp in self.local_explanations
                if exp.target_class == class_label
            ]

            # Count None explanations
            none_count = sum(
                1 for exp in class_explanations if exp.explanation_result is None
            )
            total_count = len(class_explanations)
            if total_count > 0:
                print(
                    f"Class {class_label}: {none_count} invalid explanations out of {total_count}"
                )

            self.local_explanations_by_class[class_label] = class_explanations
            self.formulae_by_class[class_label] = [
                exp.explanation_result.formula
                for exp in class_explanations
                if exp.explanation_result and exp.explanation_result.formula is not None
            ]

            # Store trajectories for this class
            if class_explanations:
                self.trajectories_by_predicted_class[class_label] = torch.stack(
                    [exp.trajectory for exp in class_explanations]
                )
            else:
                self.trajectories_by_predicted_class[class_label] = None

    def _compute_local_separation_percentages(self) -> Dict[int, float]:
        """
        Compute average separation percentages for local explanations by class.

        Returns:
            Dictionary mapping class labels to average separation percentages
        """
        separation_percentages = {}

        for class_label, explanations in self.local_explanations_by_class.items():
            percentages = [
                exp.explanation_result.separation_percentage
                for exp in explanations
                if exp.explanation_result
                and exp.explanation_result.separation_percentage is not None
            ]
            valid_percentages = [p for p in percentages if p is not None]
            separation_percentages[class_label] = (
                np.mean(valid_percentages) if valid_percentages else None
            )

        return separation_percentages

    def generate_class_explanations(
        self, improvement_threshold: float = 0.01, max_workers: Optional[int] = None
    ) -> Dict[int, ExplanationResult]:
        """
        Generate class-level explanations for all classes.

        Args:
            improvement_threshold: Minimum improvement threshold
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary mapping class labels to explanation results
        """
        if self.class_explanations:
            return self.class_explanations

        improvement_threshold = float(improvement_threshold)

        # Process classes in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                class_label: executor.submit(
                    self._process_single_class, class_label, improvement_threshold
                )
                for class_label in range(self.num_classes)
            }

            for class_label, future in futures.items():
                explanation_result = future.result()
                if explanation_result:
                    self.class_explanations[class_label] = explanation_result

        self.global_separation_percentages = (
            self._compute_global_separation_percentages()
        )
        return self.class_explanations

    def _process_single_class(
        self, class_label: int, improvement_threshold: float
    ) -> Optional[ExplanationResult]:
        """
        Process explanation for a single class.

        Args:
            class_label: Class to process
            improvement_threshold: Improvement threshold

        Returns:
            Explanation result for the class
        """
        # Filter and preprocess formulae
        filtered_formulae = self._filter_redundant_formulae(
            self.formulae_by_class[class_label]
        )
        if not filtered_formulae:
            print(f"Warning: No valid formulae for class {class_label}")
            return None

        processed_formulae = self._postprocess_class_formulae(
            filtered_formulae, class_label
        )

        # Build separation matrix
        separation_matrix = self._build_separation_matrix(
            class_label, processed_formulae
        )
        if separation_matrix is None:
            print(f"Warning: No separation matrix for class {class_label}")
            return None

        # Find optimal formula set
        selected_indices = self._find_optimal_formula_set(
            separation_matrix, processed_formulae, class_label, improvement_threshold
        )

        if not selected_indices:
            return None

        # Build class explanation
        explanation_formula = disjunction(
            [processed_formulae[i] for i in selected_indices]
        )

        # Store pre-simplification readability
        self.explanation_readability_pre[class_label] = readability(
            explanation_formula, "or"
        )

        if isinstance(explanation_formula, Boolean):
            self.explanation_readability_post[class_label] = readability(
                explanation_formula, "or"
            )
            return self._create_class_explanation_result(
                explanation_formula, class_label
            )

        # Postprocess and simplify
        final_formula = self._finalize_class_explanation(
            explanation_formula, class_label
        )
        self.explanation_readability_post[class_label] = readability(
            final_formula, "or"
        )

        return self._create_class_explanation_result(final_formula, class_label)

    def _filter_redundant_formulae(self, formulae: List) -> List:
        """
        Filter out redundant formulae based on structure.

        Args:
            formulae: Formulae to filter

        Returns:
            Filtered formulae list
        """
        formula_groups = defaultdict(list)

        for formula in formulae:
            # Normalize formula string by replacing thresholds
            formula_str = str(formula)
            normalized_str = re.sub(r"[-+]?\d*\.\d+|\d+", "{THRESHOLD}", formula_str)
            formula_groups[normalized_str].append(formula)

        filtered_formulae = []

        for normalized_str, group in formula_groups.items():
            if len(group) == 1:
                filtered_formulae.append(group[0])
            else:
                # Select representative formula based on threshold strictness
                if ">=" in normalized_str:
                    representative = max(
                        group,
                        key=lambda f: float(
                            re.search(r"([-+]?\d*\.\d+|\d+)", str(f)).group()
                        ),
                    )
                elif "<=" in normalized_str:
                    representative = min(
                        group,
                        key=lambda f: float(
                            re.search(r"([-+]?\d*\.\d+|\d+)", str(f)).group()
                        ),
                    )
                else:
                    representative = group[0]
                filtered_formulae.append(representative)

        return filtered_formulae

    def _postprocess_class_formulae(self, formulae: List, class_label: int) -> List:
        """
        Post-process class formulae for optimal separation.

        Args:
            formulae: Formulae to process
            class_label: Target class label

        Returns:
            Post-processed formulae
        """
        if not formulae:
            return []

        class_trajectories = self._get_class_trajectories(class_label)
        opponent_trajectories = self._get_opponent_trajectories(class_label)

        if class_trajectories is None or opponent_trajectories is None:
            return formulae

        # Batch compute robustness values
        with torch.no_grad():
            class_robustness = torch.stack(
                [
                    self.compute_robustness(formula, class_trajectories)
                    for formula in formulae
                ]
            )
            opponent_robustness = torch.stack(
                [
                    self.compute_robustness(formula, opponent_trajectories)
                    for formula in formulae
                ]
            )

        processed_formulae = []

        for i, formula in enumerate(formulae):
            # Find optimal threshold for separation
            combined_robustness = torch.cat(
                [class_robustness[i], opponent_robustness[i]]
            )
            labels = torch.cat(
                [
                    torch.ones_like(class_robustness[i]),
                    torch.zeros_like(opponent_robustness[i]),
                ]
            )

            optimal_threshold, separation_score = self._find_optimal_threshold(
                combined_robustness, labels
            )

            # Apply threshold adjustment
            if separation_score < 0:
                formula = Not(formula)
                optimal_threshold *= -1

            adjusted_formula = rescale_var_thresholds(formula, -optimal_threshold)
            processed_formulae.append(simplify(adjusted_formula))

        return processed_formulae

    def _find_optimal_threshold(
        self, robustness_values: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for binary classification.

        Args:
            robustness_values: Robustness values
            labels: Binary labels (1 for target, 0 for opponent)

        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        robustness_flat = robustness_values.flatten()
        labels_flat = labels.flatten()

        # Sort values
        sorted_robustness, sort_indices = torch.sort(robustness_flat)
        sorted_labels = labels_flat[sort_indices]

        # Compute cumulative sums for efficient scoring
        cumulative_positives = torch.cumsum(sorted_labels, 0)
        total_positives = cumulative_positives[-1]
        cumulative_negatives = (
            torch.arange(1, len(robustness_flat) + 1) - cumulative_positives
        )

        # Compute scores for all possible thresholds
        scores = cumulative_negatives + (total_positives - cumulative_positives)
        best_index = torch.argmax(scores).item()

        # Interpolate threshold
        if best_index == len(robustness_flat) - 1:
            optimal_threshold = sorted_robustness[-1].item()
        else:
            optimal_threshold = (
                sorted_robustness[best_index] + sorted_robustness[best_index + 1]
            ).item() / 2

        return optimal_threshold, scores[best_index].item()

    def _build_separation_matrix(
        self, class_label: int, formulae: List
    ) -> Optional[torch.Tensor]:
        """
        Build binary separation matrix for the class.

        Args:
            class_label: Target class label
            formulae: Processed formulae

        Returns:
            Binary separation matrix or None
        """
        class_trajectories = self._get_class_trajectories(class_label)
        opponent_trajectories = self._get_opponent_trajectories(class_label)

        if (
            class_trajectories is None
            or len(class_trajectories) == 0
            or opponent_trajectories is None
            or len(opponent_trajectories) == 0
        ):
            return None

        if not formulae:
            return None

        # Precompute robustness values
        with torch.no_grad():
            class_robustness = torch.stack(
                [
                    self.compute_robustness(formula, class_trajectories)
                    for formula in formulae
                ]
            )
            opponent_robustness = torch.stack(
                [
                    self.compute_robustness(formula, opponent_trajectories)
                    for formula in formulae
                ]
            )

        # Compute min/max for opponents
        min_opponent = opponent_robustness.min(dim=1).values
        max_opponent = opponent_robustness.max(dim=1).values

        # Check separation for each trajectory-formula pair
        separated = (class_robustness < min_opponent.unsqueeze(1)) | (
            class_robustness > max_opponent.unsqueeze(1)
        )

        return separated.int().T

    def _find_optimal_formula_set(
        self,
        separation_matrix: torch.Tensor,
        formulae: List,
        class_label: int,
        improvement_threshold: float,
    ) -> List[int]:
        """
        Find optimal set of formulae using ILP and pruning.

        Args:
            separation_matrix: Binary separation matrix
            formulae: Candidate formulae
            class_label: Target class label
            improvement_threshold: Improvement threshold for pruning

        Returns:
            List of selected formula indices
        """
        num_trajectories, num_formulae = separation_matrix.shape

        # Handle simple cases
        if num_formulae == 1:
            return [0]

        # Solve ILP for minimum node cover
        selected_indices = self._solve_set_cover_ilp(separation_matrix, formulae)

        if not selected_indices:
            return self._fallback_formula_selection(
                formulae, class_label, improvement_threshold
            )

        # Apply improvement-based pruning
        if improvement_threshold > 0:
            selected_indices = self._prune_formula_set(
                selected_indices, formulae, class_label, improvement_threshold
            )

        return selected_indices

    def _solve_set_cover_ilp(
        self, separation_matrix: torch.Tensor, formulae: List
    ) -> List[int]:
        """
        Solve set cover problem using integer linear programming.

        Args:
            separation_matrix: Binary separation matrix
            formulae: Candidate formulae

        Returns:
            List of selected formula indices
        """
        num_trajectories, num_formulae = separation_matrix.shape

        # Create ILP problem
        problem = LpProblem("MinimumNodeSetCover", LpMinimize)

        # Create binary variables for each formula
        formula_vars = [LpVariable(f"x_{j}", cat=LpBinary) for j in range(num_formulae)]

        # Objective: minimize total node count (complexity)
        node_costs = [find_n_nodes(formula) for formula in formulae]
        problem += lpSum(formula_vars[j] * node_costs[j] for j in range(num_formulae))

        # Constraints: each trajectory must be covered by at least one formula
        for i in range(num_trajectories):
            problem += (
                lpSum(
                    formula_vars[j] * int(separation_matrix[i][j])
                    for j in range(num_formulae)
                )
                >= 1
            )

        # Solve the problem
        status = problem.solve(PULP_CBC_CMD(msg=0))

        # Extract solution
        if LpStatus[status] == "Optimal":
            selected_indices = [
                j for j in range(num_formulae) if formula_vars[j].value() == 1.0
            ]
            return selected_indices
        else:
            return []

    def _prune_formula_set(
        self,
        selected_indices: List[int],
        formulae: List,
        class_label: int,
        improvement_threshold: float,
    ) -> List[int]:
        """
        Prune formula set based on improvement threshold.

        Args:
            selected_indices: Initially selected formula indices
            formulae: All candidate formulae
            class_label: Target class label
            improvement_threshold: Minimum improvement threshold

        Returns:
            Pruned list of formula indices
        """
        if len(selected_indices) <= 1:
            return selected_indices

        class_trajectories = self._get_class_trajectories(class_label)

        # Compute full coverage
        full_formula = disjunction([formulae[i] for i in selected_indices])
        full_output = self.compute_robustness(full_formula, class_trajectories)
        full_coverage = (full_output >= 0).sum().item()

        minimal_set = set(selected_indices)

        # Try removing each formula
        for formula_idx in selected_indices:
            subset_indices = [idx for idx in minimal_set if idx != formula_idx]
            if not subset_indices:
                continue

            subset_formula = disjunction([formulae[i] for i in subset_indices])
            subset_output = self.compute_robustness(subset_formula, class_trajectories)
            subset_coverage = (subset_output >= 0).sum().item()

            # Check if coverage drop is acceptable
            coverage_drop = (full_coverage - subset_coverage) / class_trajectories.size(
                0
            )
            if coverage_drop <= improvement_threshold:
                minimal_set.discard(formula_idx)

        return list(minimal_set)

    def _fallback_formula_selection(
        self, formulae: List, class_label: int, improvement_threshold: float
    ) -> List[int]:
        """
        Fallback method when ILP fails to find solution.

        Args:
            formulae: Candidate formulae
            class_label: Target class label
            improvement_threshold: Improvement threshold

        Returns:
            Selected formula indices
        """
        class_trajectories = self._get_class_trajectories(class_label)

        # Precompute robustness for all formulae
        precomputed_robustness = [
            self.compute_robustness(formula, class_trajectories) for formula in formulae
        ]

        # Compute target values (disjunction of all formulae)
        target_robustness = rhos_disjunction(
            precomputed_robustness, range(len(formulae))
        )

        if improvement_threshold == 0:
            return self._find_exact_cover(
                formulae, precomputed_robustness, target_robustness
            )
        else:
            return self._find_approximate_cover(
                formulae,
                precomputed_robustness,
                target_robustness,
                improvement_threshold,
            )

    def _find_exact_cover(
        self,
        formulae: List,
        precomputed_robustness: List[torch.Tensor],
        target_robustness: torch.Tensor,
    ) -> List[int]:
        """
        Find exact cover using brute-force search.

        Args:
            formulae: Candidate formulae
            precomputed_robustness: Precomputed robustness values
            target_robustness: Target robustness values

        Returns:
            Selected formula indices
        """
        target_mask = target_robustness >= 0
        if target_mask.sum() == 0:
            return []

        num_formulae = len(formulae)

        # Try increasingly larger subsets
        for subset_size in range(1, num_formulae + 1):
            for subset_indices in itertools.combinations(
                range(num_formulae), subset_size
            ):
                subset_output = rhos_disjunction(precomputed_robustness, subset_indices)

                # Check if subset covers all positive targets
                if (subset_output[target_mask] >= 0).all():
                    return list(subset_indices)

        # Fallback: return all formulae
        return list(range(num_formulae))

    def _find_approximate_cover(
        self,
        formulae: List,
        precomputed_robustness: List[torch.Tensor],
        target_robustness: torch.Tensor,
        improvement_threshold: float,
        max_score_drop: float = 0.01,
    ) -> List[int]:
        """
        Find approximate cover balancing quality and simplicity.

        Args:
            formulae: Candidate formulae
            precomputed_robustness: Precomputed robustness values
            target_robustness: Target robustness values
            improvement_threshold: Improvement threshold
            max_score_drop: Maximum allowed score drop

        Returns:
            Selected formula indices
        """
        target_mask = target_robustness >= 0
        opponent_mask = ~target_mask

        if target_mask.sum() == 0:
            return list(range(len(formulae)))

        best_subset = None
        best_score = -1.0
        best_cost = float("inf")

        # Try subsets of increasing size
        for subset_size in range(1, len(formulae) + 1):
            found_improvement = False

            for subset_indices in itertools.combinations(
                range(len(formulae)), subset_size
            ):
                subset_output = rhos_disjunction(precomputed_robustness, subset_indices)

                # Compute coverage scores
                target_coverage = (
                    (subset_output[target_mask] >= 0).float().mean().item()
                )
                if opponent_mask.sum() > 0:
                    opponent_coverage = (
                        (subset_output[opponent_mask] < 0).float().mean().item()
                    )
                else:
                    opponent_coverage = 0.0

                total_score = target_coverage + opponent_coverage
                total_cost = sum(find_n_nodes(formulae[i]) for i in subset_indices)

                # Check for strict improvement
                if total_score > best_score + improvement_threshold:
                    best_score = total_score
                    best_cost = total_cost
                    best_subset = subset_indices
                    found_improvement = True

                # Accept slightly worse solution if much cheaper
                elif (
                    total_score >= best_score - max_score_drop
                    and total_cost < best_cost
                ):
                    best_score = total_score
                    best_cost = total_cost
                    best_subset = subset_indices
                    found_improvement = True

            # Stop if no improvement found at this size
            if best_subset is not None and not found_improvement:
                break

        return (
            list(best_subset) if best_subset is not None else list(range(len(formulae)))
        )

    def _finalize_class_explanation(self, explanation_formula, class_label: int):
        """
        Finalize class explanation with simplification.

        Args:
            explanation_formula: Raw explanation formula
            class_label: Target class label

        Returns:
            Finalized explanation formula
        """
        # Prepare data for simplification
        class_trajectories = self._get_class_trajectories(class_label)
        opponent_trajectories = self._get_opponent_trajectories(class_label)

        if class_trajectories is None or opponent_trajectories is None:
            return explanation_formula

        all_trajectories = torch.cat([class_trajectories, opponent_trajectories])

        # Simplify the formula
        simplified_formula = evaluate_and_simplify(
            explanation_formula, all_trajectories, "or"
        )

        return simplified_formula

    def _create_class_explanation_result(
        self, formula, class_label: int
    ) -> ExplanationResult:
        """
        Create explanation result for class-level explanation.

        Args:
            formula: Explanation formula
            class_label: Target class label

        Returns:
            Explanation result with metrics
        """
        class_trajectories = self._get_class_trajectories(class_label)
        opponent_trajectories = self._get_opponent_trajectories(class_label)

        if class_trajectories is None or opponent_trajectories is None:
            return ExplanationResult(
                formula=formula,
                target_robustness=torch.tensor(0.0),
                opponent_robustness=torch.tensor([]),
                readability_score=None,
                separation_percentage=0.0,
            )

        # Compute robustness values
        target_robustness = self.compute_robustness(formula, class_trajectories)
        opponent_robustness = self.compute_robustness(formula, opponent_trajectories)

        # Compute separation percentage
        separation_percentage = self._calculate_class_separation_percentage(
            target_robustness, opponent_robustness
        )

        return ExplanationResult(
            formula=formula,
            target_robustness=target_robustness,
            opponent_robustness=opponent_robustness,
            readability_score=readability(formula, "or"),
            separation_percentage=separation_percentage,
        )

    def _calculate_class_separation_percentage(
        self, target_robustness: torch.Tensor, opponent_robustness: torch.Tensor
    ) -> float:
        """
        Calculate separation percentage for class-level explanation.

        Args:
            target_robustness: Robustness for class trajectories
            opponent_robustness: Robustness for opponent trajectories

        Returns:
            Separation percentage
        """
        if opponent_robustness.numel() == 0:
            return 0.0

        target_positive = (target_robustness >= 0).float().mean().item()
        opponent_negative = (opponent_robustness < 0).float().mean().item()

        # Combined separation score
        separation_score = (target_positive + opponent_negative) / 2 * 100

        return separation_score

    def _get_class_trajectories(self, class_label: int) -> Optional[torch.Tensor]:
        """
        Get all trajectories for a specific class.

        Args:
            class_label: Class label

        Returns:
            Tensor of class trajectories or None
        """
        # Get test trajectories from explanations
        test_trajectories = self.trajectories_by_predicted_class.get(class_label)

        if self.is_training:
            return test_trajectories

        # For test mode, include training trajectories
        training_trajectories = torch.stack(self.trajectories_by_class[class_label])

        if test_trajectories is not None:
            return torch.cat([training_trajectories, test_trajectories])
        else:
            return training_trajectories

    def _get_opponent_trajectories(self, class_label: int) -> Optional[torch.Tensor]:
        """
        Get all opponent trajectories for a specific class.

        Args:
            class_label: Target class label

        Returns:
            Tensor of opponent trajectories or None
        """
        opponent_trajectories = []

        for opponent_label, trajectories in self.trajectories_by_class.items():
            if opponent_label != class_label and trajectories:
                opponent_trajectories.extend(trajectories)

        return torch.stack(opponent_trajectories) if opponent_trajectories else None

    def _compute_global_separation_percentages(self) -> Tuple[Dict[int, float], float]:
        """
        Compute global separation percentages for class explanations.

        Returns:
            Tuple of (per_class_percentages, global_percentage)
        """
        total_mismatch = 0
        total_points = 0
        per_class_percentages = {}

        for class_label, explanation_result in self.class_explanations.items():
            if explanation_result is None:
                per_class_percentages[class_label] = 0.0
                continue

            class_trajectories = self._get_class_trajectories(class_label)
            if class_trajectories is None:
                per_class_percentages[class_label] = 0.0
                continue

            # Use the existing division percentage calculation
            class_percentage, total_points, total_mismatch = _division_perc(
                self,
                class_trajectories,
                class_label,
                explanation_result.formula,
                per_class_percentages,
                total_points,
                total_mismatch,
            )
            per_class_percentages[class_label] = class_percentage

        global_percentage = (
            (1 - (total_mismatch / total_points)) * 100 if total_points > 0 else 0.0
        )

        return per_class_percentages, global_percentage

    def compute_class_explanation_for_new_point(
        self,
        new_local_explanation: LocalExplanation,
        improvement_threshold: float = 0.01,
    ):
        """
        Compute class-level explanation including a new test point.

        Args:
            new_local_explanation: New local explanation to include
            improvement_threshold: Improvement threshold

        Returns:
            Tuple of (explanation_formula, separation_percentage)
        """
        class_label = new_local_explanation.target_class

        # Create augmented local explanations list
        augmented_explanations = self.local_explanations_by_class[class_label] + [
            new_local_explanation
        ]

        # Extract formulae
        formulae = [
            exp.explanation_result.formula
            for exp in augmented_explanations
            if exp.explanation_result and exp.explanation_result.formula is not None
        ]

        if not formulae:
            return None, None

        # Process formulae
        filtered_formulae = self._filter_redundant_formulae(formulae)
        processed_formulae = self._postprocess_class_formulae(
            filtered_formulae, class_label
        )

        # Build separation matrix
        separation_matrix = self._build_separation_matrix(
            class_label, processed_formulae
        )
        if separation_matrix is None:
            return None, None

        # Find optimal formula set
        selected_indices = self._find_optimal_formula_set(
            separation_matrix, processed_formulae, class_label, improvement_threshold
        )

        if not selected_indices:
            return None, None

        # Build explanation
        explanation_formula = disjunction(
            [processed_formulae[i] for i in selected_indices]
        )

        # Simplify with augmented data
        class_trajectories = self._get_class_trajectories(class_label)
        new_trajectory = new_local_explanation.trajectory.unsqueeze(0)
        augmented_trajectories = torch.cat([class_trajectories, new_trajectory])

        all_trajectories = torch.cat(
            [
                torch.stack(trajs) if cls != class_label else augmented_trajectories
                for cls, trajs in self.trajectories_by_class.items()
            ]
        )

        simplified_formula = evaluate_and_simplify(
            explanation_formula, all_trajectories, "or"
        )

        # Compute separation percentage
        target_robustness = self.compute_robustness(
            simplified_formula, augmented_trajectories
        )
        opponent_trajectories = self._get_opponent_trajectories(class_label)
        opponent_robustness = self.compute_robustness(
            simplified_formula, opponent_trajectories
        )

        separation_percentage = self._calculate_class_separation_percentage(
            target_robustness, opponent_robustness
        )

        return simplified_formula, separation_percentage

    def plot_class_robustness(
        self,
        class_label: int,
        formulae: Optional[List] = None,
        figsize: Tuple[int, int] = (10, 6),
        colormap: str = "tab20",
        highlight_target: bool = True,
    ) -> plt.Figure:
        """
        Plot robustness distribution for class formulae.

        Args:
            class_label: Target class label
            formulae: Formulae to plot (uses class explanations if None)
            figsize: Figure size
            colormap: Color map
            highlight_target: Whether to highlight target class

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        if formulae is None:
            if (
                class_label in self.class_explanations
                and self.class_explanations[class_label]
            ):
                formulae = [self.class_explanations[class_label].formula]
            else:
                formulae = self.formulae_by_class.get(class_label, [])

        if not formulae:
            print(f"No formulae available for class {class_label}")
            return plt.figure(figsize=figsize)

        fig, ax = plt.subplots(figsize=figsize)

        cmap = plt.get_cmap(colormap)
        default_colors = [cmap(i) for i in range(self.num_classes)]
        gray_color = (0.7, 0.7, 0.7)

        colors = [
            "red" if (highlight_target and i == class_label) else gray_color
            for i in range(self.num_classes)
        ]
        jitter = (
            np.zeros(self.num_classes)
            if highlight_target
            else np.linspace(-0.1, 0.1, self.num_classes)
        )

        class_trajectories = self._get_class_trajectories(class_label)

        for formula_idx, formula in enumerate(formulae):
            # Compute robustness for target class
            target_robustness = self.compute_robustness(formula, class_trajectories)

            # Compute robustness for all classes
            robustness_by_class = {}
            for other_class in range(self.num_classes):
                if other_class == class_label:
                    robustness_by_class[other_class] = target_robustness
                else:
                    other_trajectories = torch.stack(
                        self.trajectories_by_class[other_class]
                    )
                    robustness_by_class[other_class] = self.compute_robustness(
                        formula, other_trajectories
                    )

            # Plot each class
            for class_idx in range(self.num_classes):
                class_robustness = robustness_by_class[class_idx]
                ax.scatter(
                    class_robustness.tolist(),
                    [formula_idx + jitter[class_idx]] * len(class_robustness),
                    color=colors[class_idx],
                    marker="o",
                    alpha=0.7,
                    s=60 if highlight_target and class_idx == class_label else 40,
                    zorder=(
                        1000
                        if highlight_target and class_idx == class_label
                        else class_idx + 5
                    ),
                )

        # Create legend
        if highlight_target:
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=8,
                    label="Target Class",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=gray_color,
                    markersize=8,
                    label="Other Classes",
                ),
            ]
        else:
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=default_colors[i],
                    markersize=8,
                    label=f"Class {i}",
                )
                for i in range(self.num_classes)
            ]

        ax.legend(handles=legend_elements, loc="best")
        ax.set_ylim(-1, len(formulae))
        ax.set_xlabel("Robustness Value", fontsize=12)
        ax.set_ylabel("Formula Index", fontsize=12)
        ax.set_yticks(range(len(formulae)))
        ax.set_yticklabels([f"F{i+1}" for i in range(len(formulae))])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)

        plt.tight_layout()
        return fig


def get_training_explanations(
    model,
    trainloader,
    explanation_layer,
    backprop_method,
    explanation_operation = 'mean',
    imp_t_l=0.01,
    imp_t_g=0.01,
    t_k=0.95,
    k=None,
    pll = 1
):
    training_local_explanations = model.get_explanations(
            x=trainloader.dataset.trajectories,
            y_true=trainloader.dataset.labels,
            trajbyclass=trainloader.dataset.split_by_class(),
            layer=explanation_layer,
            t_k=t_k,
            method=backprop_method,
            k=k,
            op = explanation_operation
            )
    if len(training_local_explanations) > 0:
        for e in training_local_explanations:
            # without final postprocessing, ie simplifications and such (faster)
            e.generate_explanation(
                improvement_threshold=imp_t_l, enable_postprocessing=False
            )
        class_explanations = ClassExplanation(
            training_local_explanations, is_training=True
        )
        class_explanations.generate_class_explanations(improvement_threshold=imp_t_g, max_workers=pll)
        return class_explanations
    return None
