"""
Explanation Utilities - Utility functions for explanation generation and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from time import time
import sys
import os
import pickle

# ensure workspace root is on sys.path so the local `STELLE` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from STELLE.explanations.global_explanation import get_training_explanations
from STELLE.explanations.explanation_metrics import (
    get_local_metrics,
    get_global_metrics,
)


def compute_explanations(args, globals = False, save=True, **kwargs):
    (model_path_ev, trainloader, testloader, model, config) = args
    device = model.device
    explanation_layer = model.output_activation.to(device)
    trajbyclass = trainloader.dataset.split_by_class()
    local_explanations_true_pred = []
    expl_type = kwargs.get('expl_type', None)
    
    for i in ["true", "pred"]:
        explpath = model_path_ev[:-3] + f"_local_explanations_{i}.pickle"
        compute = True
        if os.path.exists(explpath):
            try:
                with open(explpath, "rb") as f:
                    local_explanations, local_explanations_time = pickle.load(f)
                compute = hasattr(local_explanations[0], 'explanation_result') # recompute if old version
                if not compute: 
                    local_explanations_true_pred.append(local_explanations)
                    print(f'Loaded local explanations ({i}) from {explpath}.')
                else:
                    print(f'Old version of local explanations ({i}) loaded from {explpath}. Recomputing.')
            except Exception as e:
                print(f"Failed to load existing local explanations ({i}) ({e}).")
        
        if compute:
            print(f'Getting local explanations ({i})...')
            start_time = time()
            if expl_type not in [None, 'base']: # ablation tests
                local_explanations = get_alternative_explanations(
                    model = model,
                    x=testloader.dataset.trajectories,
                    y_true=testloader.dataset.labels if i == "true" else None,
                    trajbyclass=trajbyclass,
                    layer=explanation_layer,
                    t_k=config.t_k,
                    method = kwargs.get('method', 'ig'),
                    op = kwargs.get('explanation_operation', 'mean'),
                    expl_type = expl_type
                )
            else:
                local_explanations = model.get_explanations(
                    x=testloader.dataset.trajectories,
                    y_true=testloader.dataset.labels if i == "true" else None,
                    trajbyclass=trajbyclass,
                    layer=explanation_layer,
                    t_k=config.t_k,
                    method = kwargs.get('method', 'ig'),
                    op = kwargs.get('explanation_operation', 'mean')
                )
            for e in local_explanations:
                e.generate_explanation(
                    improvement_threshold=config.imp_t_l, enable_postprocessing=True
                )
            local_explanations_time = time() - start_time

            local_explanations_true_pred.append(local_explanations)
            if save:
                with open(explpath, "wb") as f:
                    pickle.dump((local_explanations, local_explanations_time), f)
                print(f"Saved local explanations ({i}) to {explpath}")

    try: 
        local_metrics = get_local_metrics(local_explanations_true_pred, testloader)
    except Exception as e:
        print(e)
        compute = True

    # global
    if globals:
        print('Getting global explanations...')
        globpath = model_path_ev[:-3] + "_global_explanations.pickle"
        compute = True
        if os.path.exists(globpath):
            try:
                with open(globpath, "rb") as f:
                    global_explanations, global_explanations_time = pickle.load(f)
                compute = False
                print(f'Loaded global explanations from {globpath}')
            except Exception as e:
                print(f"Failed to load existing global explanations ({e}).")
        
        if compute:
            start_time = time()

            global_explanations = get_training_explanations(
                model, trainloader, 
                explanation_layer, kwargs.get('method', 'ig'), 
                config.imp_t_l, config.imp_t_g, config.t_k, 
                explanation_operation = kwargs.get('explanation_operation', 'mean')
            )
            global_explanations_time = time() - start_time
            if save:
                with open(globpath, "wb") as f:
                    pickle.dump((global_explanations, global_explanations_time), f)
                print(f"Saved global explanations to {globpath}")

        global_metrics = get_global_metrics(global_explanations)
    else:
        global_metrics = {}
    del model
    return local_metrics, global_metrics



def load_cached_metrics(model_path_ev, config):
    """
    Load cached metrics if they exist for this configuration.
    Returns:
        tuple: (local_metrics, global_metrics) if found, or (None, None) if not found
    """
    metrics_path = model_path_ev[:-3] + "_metrics.pickle"
    if not os.path.exists(metrics_path):
        return None, None
    
    # Parameters to ignore when comparing configs
    IGNORED_PARAMS = {'verbose', 'logging', 'workers', 'pll', 'epochs'}
    
    try:
        with open(metrics_path, "rb") as f:
            cached_data = pickle.load(f)
        
        cached_config_str = cached_data.get('config_hash', '')
        current_config_str = str(config)
        
        # Parse config strings to extract key-value pairs
        def parse_config_str(config_str):
            """Extract key=value pairs from config string"""
            import re
            pattern = r'(\w+)=([\w\.\-]+)'
            matches = re.findall(pattern, config_str)
            return dict(matches)
        
        cached_config_dict = parse_config_str(cached_config_str)
        current_config_dict = parse_config_str(current_config_str)
        
        # Remove ignored parameters
        for param in IGNORED_PARAMS:
            cached_config_dict.pop(param, None)
            current_config_dict.pop(param, None)
        
        # Compare configs (ignoring specified params)
        if cached_config_dict == current_config_dict:
            print(f'Loaded cached metrics from {metrics_path}')
            return cached_data['local_metrics'], cached_data['global_metrics']
        
        # Mismatch detected - find differences
        print('Config mismatch for cached metrics. Recomputing.')
        all_keys = set(cached_config_dict.keys()) | set(current_config_dict.keys())
        mismatches = []
        
        for key in sorted(all_keys):
            cached_val = cached_config_dict.get(key, '<MISSING>')
            current_val = current_config_dict.get(key, '<MISSING>')
            if cached_val != current_val:
                mismatches.append(f"  {key}: {cached_val} -> {current_val}")
        
        if mismatches:
            print("Mismatched config keys:")
            print("\n".join(mismatches))
        
        return None, None
        
    except Exception as e:
        print(f"Failed to load cached metrics ({e}). Recomputing.")
        return None, None
    
    
    
def save_metrics(model_path_ev, config, local_metrics, global_metrics):
    """
    Save computed metrics to cache file.
    """
    metrics_path = model_path_ev[:-3] + "_metrics.pickle"
    
    try:
        cached_data = {
            'config_hash': str(config),
            'local_metrics': local_metrics,
            'global_metrics': global_metrics
        }
        with open(metrics_path, "wb") as f:
            pickle.dump(cached_data, f)
        print(f"Saved metrics to {metrics_path}")
    except Exception as e:
        print(f"Failed to save metrics ({e}).")
        
        
# TODO matching di shapes

def get_alternative_explanations(
    model,
    x: torch.Tensor,
    trajbyclass,
    layer,
    expl_type: str,
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
        model: trained model
        x: Input trajectories
        trajbyclass: Trajectory examples by class
        layer: Layer to explain
        arch_type: way of extracting the explanation. "crel" (concept_relevance only), "lw" (backprop weights only), "crelGx" (crel x Gx), "Gxlw" (Gx x backprop), "crellw" (crel x backprop)
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
    model.eval()

    # Get model predictions
    with torch.no_grad():
        x = x.to(model.device)
        class_scores, concept_relevance, crelGs_raw, G_phis = model.forward(
            x, trainingmode=False
        )

    y_pred = class_scores.argmax(dim=1).cpu()

    if expl_type == "crel_gx": 
        term1 = model.output_activation(crelGs_raw) if norm else crelGs_raw # (batch, phis * classes)
    elif expl_type in ["crel", "crel_lw"]:
        term1 = (
            model.output_activation(concept_relevance) if norm else concept_relevance
        ) # (batch, phis)
    else: 
        term1 = crelGs_raw # (batch, phis, classes)

    # Filter for correct predictions if requested
    if filter_onlycorrect and y_true is not None:
        x, y_pred, term1, class_scores = model._filter_correct_predictions(
            x, y_true, y_pred, term1, class_scores
        )
    if term1.shape != crelGs_raw.shape:
        # Calculate how many times we need to repeat
        repeat_factor = crelGs_raw.shape[1] // term1.shape[1]
        print(f'{crelGs_raw.shape=}, {term1.shape=}, {repeat_factor=}')
        term1 = term1.repeat(1, repeat_factor)
    # Compute attributions
    x_requires_grad = x.requires_grad_()
    targets = y_pred if y_true is None else y_pred

    if expl_type in ["lw", "gx_lw", "crel_lw"]:
        term2 = model._compute_attributions(x_requires_grad, targets, layer, method) # (batch, phis * classes)
    else:
        term2 = torch.ones(*crelGs_raw.shape) # (batch, phis * classes)

    # Compute final attribution matrix
    final_matrix = model._compute_final_attributions(term1, term2, y_pred)
    # Get discriminative scores
    discriminative_scores = model._compute_discriminative_scores(
        final_matrix, targets, op
    )

    if getmatrix:
        grouped_matrix = model._group_matrix_by_class(final_matrix, targets)
        return grouped_matrix, y_pred

    # Generate explanations
    explanations = model._generate_explanations(
        x, y_true, y_pred, discriminative_scores, trajbyclass, k, t_k
    )

    return explanations



class ExplanationVisualizer:
    """
    Visualization utilities for STL explanations.
    """
    
    @staticmethod
    def plot_robustness_distribution(
        explanation_results: List,
        class_labels: List[int],
        figsize: Tuple[int, int] = (12, 6),
        colormap: str = "tab10"
    ) -> plt.Figure:
        """
        Plot robustness distribution for explanations across classes.
        
        Args:
            explanation_results: List of explanation results
            class_labels: Class labels for each explanation
            figsize: Figure size
            colormap: Color map for classes
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract robustness values
        target_robustness = []
        opponent_robustness = []
        colors = []
        
        cmap = plt.get_cmap(colormap)
        unique_classes = sorted(set(class_labels))
        
        for result, class_label in zip(explanation_results, class_labels):
            if result is None:
                continue
            
            target_robustness.append(result.target_robustness.item())
            if result.opponent_robustness.numel() > 0:
                opponent_robustness.extend(result.opponent_robustness.tolist())
                colors.extend([cmap(class_label)] * len(result.opponent_robustness))
        
        # Plot target vs opponent robustness
        if target_robustness and opponent_robustness:
            ax1.scatter(
                target_robustness, 
                opponent_robustness[:len(target_robustness)],
                c=colors[:len(target_robustness)],
                alpha=0.6
            )
            ax1.set_xlabel('Target Robustness')
            ax1.set_ylabel('Opponent Robustness')
            ax1.set_title('Target vs Opponent Robustness')
            ax1.grid(True, alpha=0.3)
        
        # Plot robustness distribution by class
        robustness_by_class = defaultdict(list)
        for result, class_label in zip(explanation_results, class_labels):
            if result is not None:
                robustness_by_class[class_label].append(result.target_robustness.item())
        
        positions = []
        values = []
        labels = []
        for i, class_label in enumerate(unique_classes):
            if class_label in robustness_by_class:
                positions.extend([i] * len(robustness_by_class[class_label]))
                values.extend(robustness_by_class[class_label])
                labels.append(f'Class {class_label}')
        
        if positions and values:
            ax2.scatter(values, positions, c=[cmap(p) for p in positions], alpha=0.6)
            ax2.set_yticks(range(len(unique_classes)))
            ax2.set_yticklabels(labels)
            ax2.set_xlabel('Robustness Value')
            ax2.set_title('Robustness Distribution by Class')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_formula_robustness_comparison(
        formulae: List,
        trajectories: torch.Tensor,
        labels: torch.Tensor,
        highlight_class: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6),
        colormap: str = "tab20"
    ) -> plt.Figure:
        """
        Plot robustness values for multiple formulae across trajectories.
        
        Args:
            formulae: Formulae to evaluate
            trajectories: Input trajectories
            labels: Class labels
            highlight_class: Specific class to highlight
            figsize: Figure size
            colormap: Color map
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        cmap = plt.get_cmap(colormap)
        num_classes = len(torch.unique(labels))
        default_colors = [cmap(i) for i in range(num_classes)]
        
        # Use gray for non-highlighted classes if highlighting
        if highlight_class is not None:
            colors = [
                "red" if label == highlight_class else (0.7, 0.7, 0.7)
                for label in labels
            ]
            jitter = np.zeros(num_classes)
        else:
            colors = [default_colors[label] for label in labels]
            jitter = np.linspace(-0.1, 0.1, num_classes)
        
        # Compute robustness for all formulae
        for formula_idx, formula in enumerate(formulae):
            robustness_values = []
            trajectory_classes = []
            
            for i, trajectory in enumerate(trajectories):
                robustness = formula.quantitative(
                    trajectory.unsqueeze(0),
                    evaluate_at_all_times=False,
                    vectorize=False,
                    normalize=False,
                ).item()
                robustness_values.append(robustness)
                trajectory_classes.append(labels[i].item())
            
            # Plot with jitter for visibility
            for class_label in range(num_classes):
                class_indices = [i for i, c in enumerate(trajectory_classes) if c == class_label]
                if class_indices:
                    class_robustness = [robustness_values[i] for i in class_indices]
                    ax.scatter(
                        class_robustness,
                        [formula_idx + jitter[class_label]] * len(class_robustness),
                        color=colors[class_indices[0]],
                        marker='o',
                        alpha=0.7,
                        s=60 if highlight_class and class_label == highlight_class else 40,
                    )
        
        # Create legend
        if highlight_class is not None:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='red', markersize=8, label='Target Class'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=(0.7, 0.7, 0.7), markersize=8, label='Other Classes'),
            ]
        else:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=default_colors[i], markersize=8, label=f'Class {i}')
                for i in range(num_classes)
            ]
        
        ax.legend(handles=legend_elements, loc='best')
        ax.set_ylim(-1, len(formulae))
        ax.set_xlabel('Robustness Value', fontsize=12)
        ax.set_ylabel('Formula Index', fontsize=12)
        ax.set_yticks(range(len(formulae)))
        ax.set_yticklabels([f'F{i+1}' for i in range(len(formulae))])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig


class ExplanationAnalyzer:
    """
    Analysis utilities for STL explanations.
    """
    
    @staticmethod
    def analyze_formula_patterns(
        formulae: List,
        max_patterns: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze common patterns in explanation formulae.
        
        Args:
            formulae: List of STL formulae
            max_patterns: Maximum number of patterns to return
            
        Returns:
            Dictionary with pattern analysis results
        """
        from collections import Counter
        import re
        
        # Extract formula templates by replacing thresholds and variables
        templates = []
        formula_complexity = []
        
        for formula in formulae:
            if formula is None:
                continue
            
            formula_str = str(formula)
            
            # Replace thresholds with placeholder
            template = re.sub(r"[-+]?\d*\.\d+|\d+", "{T}", formula_str)
            # Replace variable indices
            template = re.sub(r"x_\d+", "x_{V}", template)
            
            templates.append(template)
            formula_complexity.append(ExplanationAnalyzer._compute_formula_complexity(formula))
        
        # Count pattern frequency
        pattern_counts = Counter(templates)
        
        # Analyze temporal operators
        temporal_operators = {
            'always': sum(1 for t in templates if 'always' in t),
            'eventually': sum(1 for t in templates if 'eventually' in t),
            'until': sum(1 for t in templates if 'until' in t),
            'globally': sum(1 for t in templates if 'globally' in t),
        }
        
        # Analyze logical structure
        logical_operators = {
            'and': sum(1 for t in templates if 'and' in t),
            'or': sum(1 for t in templates if 'or' in t),
            'not': sum(1 for t in templates if 'not' in t),
        }
        
        return {
            'total_formulae': len(formulae),
            'unique_patterns': len(pattern_counts),
            'most_common_patterns': pattern_counts.most_common(max_patterns),
            'temporal_operator_frequency': temporal_operators,
            'logical_operator_frequency': logical_operators,
            'complexity_stats': {
                'mean': np.mean(formula_complexity) if formula_complexity else 0,
                'std': np.std(formula_complexity) if formula_complexity else 0,
                'min': min(formula_complexity) if formula_complexity else 0,
                'max': max(formula_complexity) if formula_complexity else 0,
            }
        }
    
    @staticmethod
    def _compute_formula_complexity(formula) -> int:
        """
        Compute complexity of a formula (number of nodes).
        
        Args:
            formula: STL formula
            
        Returns:
            Complexity score (number of nodes)
        """
        # This is a simplified implementation
        # A more complete implementation would traverse the formula tree
        formula_str = str(formula)
        return formula_str.count('(') + formula_str.count(')')
    
    @staticmethod
    def compare_explanation_sets(
        set_a: List,
        set_b: List,
        trajectories: torch.Tensor,
        metric: str = 'separation'
    ) -> Dict[str, float]:
        """
        Compare two sets of explanations using various metrics.
        
        Args:
            set_a: First set of explanations
            set_b: Second set of explanations
            trajectories: Trajectories for evaluation
            metric: Comparison metric ('separation', 'robustness', 'both')
            
        Returns:
            Dictionary of comparison results
        """
        if len(set_a) != len(set_b):
            raise ValueError("Explanation sets must have the same length")
        
        comparison_results = {
            'set_a_valid': sum(1 for exp in set_a if exp is not None),
            'set_b_valid': sum(1 for exp in set_b if exp is not None),
            'agreement_count': 0,
        }
        
        separation_differences = []
        robustness_correlations = []
        
        for exp_a, exp_b in zip(set_a, set_b):
            if exp_a is None or exp_b is None:
                continue
            
            # Agreement in separation direction
            if (exp_a.target_robustness.item() >= 0) == (exp_b.target_robustness.item() >= 0):
                comparison_results['agreement_count'] += 1
            
            # Separation percentage difference
            if (exp_a.separation_percentage is not None and 
                exp_b.separation_percentage is not None):
                separation_differences.append(
                    abs(exp_a.separation_percentage - exp_b.separation_percentage)
                )
            
            # Robustness correlation
            if (exp_a.opponent_robustness.numel() > 0 and 
                exp_b.opponent_robustness.numel() > 0):
                try:
                    correlation = torch.corrcoef(torch.stack([
                        exp_a.opponent_robustness.flatten(),
                        exp_b.opponent_robustness.flatten()
                    ]))[0, 1].item()
                    if not np.isnan(correlation):
                        robustness_correlations.append(correlation)
                except Exception as e:
                    print(e)
                    pass
        
        # Compute aggregate metrics
        total_comparisons = comparison_results['set_a_valid']
        if total_comparisons > 0:
            comparison_results['agreement_rate'] = (
                comparison_results['agreement_count'] / total_comparisons * 100
            )
        
        if separation_differences:
            comparison_results.update({
                'mean_separation_difference': np.mean(separation_differences),
                'max_separation_difference': np.max(separation_differences),
            })
        
        if robustness_correlations:
            comparison_results['mean_robustness_correlation'] = np.mean(robustness_correlations)
        
        return comparison_results