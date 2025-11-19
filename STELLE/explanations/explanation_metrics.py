import torch
from collections import defaultdict
import numpy as np

from ..formula_generation.stl import Boolean
from ..formula_generation.formula_utils import find_n_nodes, get_num_vars

from STELLE.utils import flatten_dict, round_dict_values


def division_percentage_local(explanations, y_true=None, incorrect=False):
    # if y is passed, filter by only correctly classified targets
    # if incorrect, return the metric wrt misclassified trajs
    if y_true is None:
        return _division_percentage_local_unfiltered(explanations)

    y_expl = [e.target_class for e in explanations]
    percs = [e.explanation_result_post.separation_percentage for e in explanations]

    if incorrect:
        # Filter explanations where y_pred != y
        filtered_percs = [
            (perc, y_pred)
            for perc, y_pred, true_y in zip(percs, y_expl, y_true)
            if y_pred != true_y and perc is not None
        ]
    else:
        # Filter explanations where y_pred == y
        filtered_percs = [
            (perc, y_pred)
            for perc, y_pred, true_y in zip(percs, y_expl, y_true)
            if y_pred == true_y and perc is not None
        ]

    if not filtered_percs:
        # Return default values if no valid data
        default_stats = {"mean": 0.0, "med": 0.0, "var": 0.0, "std": 0.0}
        return {0: default_stats}, default_stats

    # Group by predicted class
    class_percs = defaultdict(list)
    for perc, y_pred in filtered_percs:
        if perc is not None:
            class_percs[y_pred].append(perc)

    # Compute mean, median, and variance by class
    stats_by_class = {}
    for cls, perc_list in sorted(class_percs.items()):
        # Filter out any remaining None values and ensure we have numbers
        clean_percs = [p for p in perc_list if p is not None]
        if clean_percs:
            stats_by_class[cls] = {
                "mean": np.mean(clean_percs),
                "med": np.median(clean_percs),
                "var": np.var(clean_percs),
                "std": np.std(clean_percs),
            }
        else:
            stats_by_class[cls] = {
                "mean": 0.0,
                "med": 0.0,
                "var": 0.0,
                "std": 0.0,
            }
    # stats_by_class = {
    # cls: {
    #         "mean": np.mean(percs) if percs else 0.0,
    #         "med": np.median(percs) if percs else 0.0,
    #         "var": np.var(percs) if percs else 0.0,
    #         "std": np.std(percs) if percs else 0.0,
    #     }
    #     for cls, percs in sorted(class_percs.items())
    # }

    # Compute total mean, median, and variance
    total_percs = [perc for perc, _ in filtered_percs if perc is not None]
    if total_percs:
        total_stats = {
            "mean": np.mean(total_percs),
            "med": np.median(total_percs),
            "var": np.var(total_percs),
            "std": np.std(total_percs),
        }
    else:
        total_stats = {
            "mean": 0.0,
            "med": 0.0,
            "var": 0.0,
            "std": 0.0,
        }

    return round_dict_values(stats_by_class), round_dict_values(total_stats)


def _division_percentage_local_unfiltered(explanations):

    # higher is better
    # explanations: list of Explanation
    percs = [e.explanation_result_post.separation_percentage for e in explanations]
    y_preds = [e.target_class for e in explanations]

    # Count None values for debugging
    none_count = sum(1 for perc in percs if perc is None)
    if none_count > 0:
        print(
            f"Warning: {none_count} explanations out of {len(percs)} have None division_percentage"
        )

    # Filter out None values
    valid_data = [
        (perc, y_pred) for perc, y_pred in zip(percs, y_preds) if perc is not None
    ]

    if not valid_data:
        print("Warning: No valid division_percentage values found")
        return {}, {"mean": 0.0, "med": 0.0, "var": 0.0, "std": 0.0}

    valid_percs, valid_y_preds = zip(*valid_data)

    # Rest of the function remains the same...
    class_percs = defaultdict(list)
    for perc, y_pred in zip(valid_percs, valid_y_preds):
        class_percs[y_pred].append(perc)

    stats_by_class = {
        cls: {
            "mean": np.mean(perc),
            "med": np.median(perc),
            "var": np.var(perc),
            "std": np.std(perc),
        }
        for cls, perc in sorted(class_percs.items())
    }

    total_stats = {
        "mean": np.mean(valid_percs),
        "med": np.median(valid_percs),
        "var": np.var(valid_percs),
        "std": np.std(valid_percs),
    }

    return round_dict_values(stats_by_class), round_dict_values(total_stats)


def division_percentage_global(classexplanation, y=None):
    if y is None:
        # Use the function from the class
        return classexplanation.division_perc()

    total_mismatch = 0
    total_points = 0
    division_perc_byclass = {}

    # if y is passed, filter target points by only correctly classified
    for c, phi in classexplanation.explanations.items():
        target_trajectories = [
            e.x
            for i, e in enumerate(classexplanation.localexplanations)
            if e.target_class == c and e.target_class == y[i]
        ]
        target_trajectories = (
            torch.stack(target_trajectories, dim=0)
            if len(target_trajectories) != 0
            else target_trajectories
        )
        (division_perc_byclass, total_points, total_mismatch) = _division_perc(
            classexplanation,
            target_trajectories,
            c,
            phi,
            division_perc_byclass,
            total_points,
            total_mismatch,
        )

    global_division = (
        ((1 - (total_mismatch / total_points)) * 100)
        if total_points > 0
        else float("nan")
    )

    return round_dict_values(division_perc_byclass), round(global_division, 3)


# ? ha senso contare nvars? non cerco attivamente di ottimizzarlo
def readability(formula, str_match):
    """
    Computes readability metrics for a single STL formula.

    Returns:
        (nodes, nvars, n_subformulae)
    """
    nodes = find_n_nodes(formula)
    nvars = get_num_vars(formula)
    nphis = str(formula).count(f" {str_match} ") + 1
    return nodes, nvars, nphis


def readability_local(explanations, y=None):
    try:
        # collect raw metrics then filter out None and any entries containing NaNs
        raw_pre = [e.explanation_result_pre.readability_score for e in explanations]
        raw_post = [e.explanation_result_post.readability_score for e in explanations]

        def _valid(m):
            if m is None:
                return False
            try:
                return not any(np.isnan(x) for x in m)
            except Exception:
                return False

        pre_metrics = [m for m in raw_pre if _valid(m)]
        post_metrics = [m for m in raw_post if _valid(m)]

        if pre_metrics is None:
            pre_nodes = pre_nvars = pre_nphis = [np.nan] * (
                len(pre_metrics) if pre_metrics else 0
            )
        else:
            pre_nodes, pre_nvars, pre_nphis = zip(
                *[x if x is not None else (np.nan, np.nan, np.nan) for x in pre_metrics]
            )

        if post_metrics is None:
            post_nodes = post_nvars = post_nphis = [np.nan] * (
                len(post_metrics) if post_metrics else 0
            )
        else:
            post_nodes, post_nvars, post_nphis = zip(
                *[
                    x if x is not None else (np.nan, np.nan, np.nan)
                    for x in post_metrics
                ]
            )

        def round_tuple(t):
            return tuple(round(float(x), 2) if not np.isnan(x) else np.nan for x in t)

        def safe_mean(arr):
            """Compute mean, returning nan for empty arrays or all-NaN arrays (avoids runtime warnings)."""
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                return np.nan
            if np.all(np.isnan(arr)):
                return np.nan
            return float(np.nanmean(arr))
        
        def safe_std(arr):
            """Compute std, returning nan for empty arrays or all-NaN arrays (avoids runtime warnings)."""
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                return np.nan
            if np.all(np.isnan(arr)):
                return np.nan
            return float(np.nanstd(arr))

        if y is None:
            return {
                "overall": {
                    "pre": {
                        "mean": round_tuple(
                            (safe_mean(pre_nodes), safe_mean(pre_nvars), safe_mean(pre_nphis))
                        ),
                        "std": round_tuple(
                            (safe_std(pre_nodes), safe_std(pre_nvars), safe_std(pre_nphis))
                        ),
                    },
                    "post": {
                        "mean": round_tuple(
                            (
                                safe_mean(post_nodes),
                                safe_mean(post_nvars),
                                safe_mean(post_nphis),
                            )
                        ),
                        "std": round_tuple(
                            (safe_std(post_nodes), safe_std(post_nvars), safe_std(post_nphis))
                        ),
                    },
                }
            }

        y_preds = [e.target_class for e in explanations]
        correct_indices = [i for i, (yp, yt) in enumerate(zip(y_preds, y)) if yp == yt]
        incorrect_indices = [i for i in range(len(y)) if i not in correct_indices]

        def stats(indices, nodes, nvars, nphis):
            if len(indices) > 0:
                return {
                    "mean": round_tuple(
                        (
                            safe_mean([nodes[i] for i in indices]),
                            safe_mean([nvars[i] for i in indices]),
                            safe_mean([nphis[i] for i in indices]),
                        )
                    ),
                    "std": round_tuple(
                        (
                            safe_std([nodes[i] for i in indices]),
                            safe_std([nvars[i] for i in indices]),
                            safe_std([nphis[i] for i in indices]),
                        )
                    ),
                }
            else:
                return {
                    "mean": (np.nan, np.nan, np.nan),
                    "std": (np.nan, np.nan, np.nan),
                }

        return {
            "overall": {
                "pre": {
                    "mean": round_tuple(
                        (safe_mean(pre_nodes), safe_mean(pre_nvars), safe_mean(pre_nphis))
                    ),
                    "std": round_tuple(
                        (safe_std(pre_nodes), safe_std(pre_nvars), safe_std(pre_nphis))
                    ),
                },
                "post": {
                    "mean": round_tuple(
                        (safe_mean(post_nodes), safe_mean(post_nvars), safe_mean(post_nphis))
                    ),
                    "std": round_tuple(
                        (safe_std(post_nodes), safe_std(post_nvars), safe_std(post_nphis))
                    ),
                },
            },
            "correct": {
                "pre": stats(correct_indices, pre_nodes, pre_nvars, pre_nphis),
                "post": stats(correct_indices, post_nodes, post_nvars, post_nphis),
            },
            "incorrect": {
                "pre": stats(incorrect_indices, pre_nodes, pre_nvars, pre_nphis),
                "post": stats(incorrect_indices, post_nodes, post_nvars, post_nphis),
            },
        }
    except Exception as e:
        # Return a valid structure with all NaN values instead of None
        nan_stats = {
            "mean": (np.nan, np.nan, np.nan),
            "std": (np.nan, np.nan, np.nan),
        }
        
        result = {
            "overall": {
                "pre": nan_stats.copy(),
                "post": nan_stats.copy(),
            }
        }
        
        if y is not None:
            result["correct"] = {
                "pre": nan_stats.copy(),
                "post": nan_stats.copy(),
            }
            result["incorrect"] = {
                "pre": nan_stats.copy(),
                "post": nan_stats.copy(),
            }
        print(e)
        return result
    
def readability_global(class_expl):
    try:
        pre_metrics = list(class_expl.explanations_readability_pre.values())
        post_metrics = list(class_expl.explanations_readability_post.values())
        
        def _valid(m):
            if m is None:
                return False
            try:
                return not any(np.isnan(x) for x in m)
            except Exception:
                return False
        
        def to_arrays(metrics):
            clean = [m for m in metrics if _valid(m)]
            if not clean:
                return np.array([]), np.array([]), np.array([])
            nodes, nvars, nphis = zip(*clean)
            return (
                np.array(nodes, dtype=float),
                np.array(nvars, dtype=float),
                np.array(nphis, dtype=float),
            )
        
        pre_nodes, pre_nvars, pre_nphis = to_arrays(pre_metrics)
        post_nodes, post_nvars, post_nphis = to_arrays(post_metrics)
        
        def safe_round(x):
            try:
                if np.isnan(x):
                    return float("nan")
                return round(float(x), 2)
            except Exception:
                return float("nan")
        
        def round_tuple(t):
            return tuple(safe_round(x) for x in t)
        
        def safe_mean(arr):
            """Compute mean, returning nan for empty arrays"""
            return np.nan if len(arr) == 0 else np.nanmean(arr)
        
        def safe_std(arr):
            """Compute std, returning nan for empty arrays"""
            return np.nan if len(arr) == 0 else np.nanstd(arr)
        
        return {
            "pre": {
                "mean": round_tuple(
                    (safe_mean(pre_nodes), safe_mean(pre_nvars), safe_mean(pre_nphis))
                ),
                "std": round_tuple(
                    (safe_std(pre_nodes), safe_std(pre_nvars), safe_std(pre_nphis))
                ),
            },
            "post": {
                "mean": round_tuple(
                    (safe_mean(post_nodes), safe_mean(post_nvars), safe_mean(post_nphis))
                ),
                "std": round_tuple(
                    (safe_std(post_nodes), safe_std(post_nvars), safe_std(post_nphis))
                ),
            },
        }
    except Exception as e:
        # Return a valid structure with all NaN values instead of None
        nan_stats = {
            "mean": (float("nan"), float("nan"), float("nan")),
            "std": (float("nan"), float("nan"), float("nan")),
        }
        print(e)
        return {
            "pre": nan_stats.copy(),
            "post": nan_stats.copy(),
        }

def _division_perc(
    classexplanation,
    target_trajectories,
    c,
    phi,
    class_scores,
    total_trajectories,
    total_mismatch,
):
    """Evaluates how well a class explanation (φ) separates trajectories of class c from other classes.

    Computes:
    1. The percentage of trajectories correctly "divided" by φ (i.e., satisfying expected robustness sign)
    2. Updates global mismatch counters for overall evaluation

    Args:
        classexplanation: Object containing explanations and normalization settings.
            Must have:
            - normalize (bool): Whether to normalize robustness values
            - _get_opponent_trajectories(c): Method returning trajectories not in class c

        target_trajectories (torch.Tensor): Trajectories belonging to class c.
            Tensor of shape (n_trajectories, *traj_dims).

        c (int): Class index being evaluated.

        phi (STLFormula/Boolean/None): Explanation formula for class c.
            - If Boolean or None, treats as degenerate case (no valid explanation).

        class_scores (dict): Dictionary to store per-class division percentages.
            Modified in-place with results for class c.

        total_trajectories (int): Running total of trajectories evaluated across all classes.
            Updated in-place.

        total_mismatch (int): Running total of trajectories failing explanation checks.
            Updated in-place.

    Returns:
        tuple: Updated (class_scores, total_trajectories, total_mismatch) where:
            - class_scores[c] contains:
                * float: Percentage (0-100) of correct divisions for class c
                * nan: If no trajectories exist for class c
                * 0: If φ is Boolean (degenerate explanation)
            - total_trajectories: Incremented by trajectories processed in this call
            - total_mismatch: Incremented by mismatches found in this call

    Notes:
        1. A "mismatch" occurs when:
            - Target trajectory (class c) does NOT satisfy φ (FN)
            - Opponent trajectory (¬class c) DOES satisfy φ (FP)
        2. The division percentage for class c is calculated as:
            100 * (1 - (FN + FP) / total_trajectories) (i.e. (TN+TP)/tot)
        3. Handles edge cases:
            - Empty target trajectories → returns nan for class c
            - Boolean φ → treats all opponent trajectories as mismatches
    """
    normalize = classexplanation.normalize  # for quantitative
    opp_trajs = classexplanation._get_opponent_trajectories(c)
    if target_trajectories is None or len(target_trajectories) == 0:
        class_scores[c] = float("nan")

        # Count all points (even if phi is invalid)
        total_trajectories += len(opp_trajs)

        #! no! classe non c'è, non va contata
        # # all target points have the same sign, but so do the opposite. The opposite are wrong.
        # if isinstance(phi, Boolean):
        #     total_mismatch += len(opp_trajs)

        return class_scores, total_trajectories, total_mismatch

    # credo che None non venga mai in realtà
    if phi is None or isinstance(phi, Boolean):
        class_scores[c] = 0 if isinstance(phi, Boolean) else float("nan")
        # Get the target and opposite points for this class
        # Count all points (even if phi is invalid)
        current_total = len(target_trajectories) + len(opp_trajs)
        total_trajectories += current_total

        # all target points have the same sign, but so do the opposite. The opposite are wrong.
        if isinstance(phi, Boolean):
            total_mismatch += len(opp_trajs)

        return class_scores, total_trajectories, total_mismatch

    # Compute target robustness
    target_rho = phi.quantitative(
        target_trajectories,
        evaluate_at_all_times=False,
        vectorize=False,
        normalize=normalize,
    )
    expected_sign = torch.sign(
        target_rho.mean()
    ).item()  # Expected polarity for class c

    # Count target mismatches (wrong sign)
    fn = (torch.sign(target_rho) != expected_sign).sum().item()
    target_total = len(target_rho)

    # Get opposite-class trajectories
    opp_rho = phi.quantitative(
        opp_trajs, evaluate_at_all_times=False, vectorize=False, normalize=normalize
    )

    # Count opposite mismatches (same sign as target = BAD)
    fp = (torch.sign(opp_rho) == expected_sign).sum().item()
    opp_total = len(opp_rho)

    # Update totals
    class_errors = fn + fp
    current_total = target_total + opp_total

    total_mismatch += class_errors
    total_trajectories += current_total

    # Per-class goodness of division percentage
    # i.e. Total correct / total trajectories
    class_scores[c] = (
        ((1 - (class_errors / current_total)) * 100)
        if current_total > 0
        else float("nan")
    )
    return class_scores, total_trajectories, total_mismatch


def evaluate_class_explanations(
    model, class_explanations, test_subset, filter=False, incorrect=False
):
    x = test_subset.trajectories.cpu()
    y = model.predict(x).cpu()
    y_true = test_subset.y.cpu()

    if filter:
        tokeep = y != y_true if incorrect else y == y_true
        x = x[tokeep]
        y = y[tokeep]
        y_true = y_true[tokeep]

    # Group trajectories by predicted class
    traj_by_class = {}
    for c in range(model.num_classes):
        class_mask = y == c
        traj_by_class[c] = x[class_mask] if torch.any(class_mask) else torch.tensor([])

    total_mismatch = 0
    total_points = 0
    division_perc_byclass = {}

    if class_explanations is None:
        for c in range(model.num_classes):
            division_perc_byclass[c] = float("nan")
        return division_perc_byclass, float("nan")

    for c, phi in class_explanations.explanations.items():
        target_trajectories = traj_by_class.get(c, [])

        # Skip if no explanation for this class
        if phi is None:
            division_perc_byclass[c] = float("nan")
            continue

        (division_perc_byclass, total_points, total_mismatch) = _division_perc(
            class_explanations,
            target_trajectories,
            c,
            phi,
            division_perc_byclass,
            total_points,
            total_mismatch,
        )

    global_division = (
        ((1 - (total_mismatch / total_points)) * 100)
        if total_points > 0
        else float("nan")
    )

    return round_dict_values(division_perc_byclass), round(global_division, 3)


def evaluate_class_explanations_f1(
    model, class_explanations, test_subset, filter=False, incorrect=False
):
    x = test_subset.trajectories.cpu()
    y = model.predict(x).cpu()
    y_true = test_subset.y.cpu()

    if filter:
        tokeep = y != y_true if incorrect else y == y_true
        x = x[tokeep]
        y = y[tokeep]
        y_true = y_true[tokeep]

    # Group test trajectories by predicted class
    test_traj_by_class = {}
    for c in range(model.num_classes):
        class_mask = y == c
        test_traj_by_class[c] = (
            x[class_mask] if torch.any(class_mask) else torch.tensor([])
        )

    # Initialize results with cumulative metrics
    results = {
        "precision": {},
        "recall": {},  # = sensitivity
        "specificity": {},
        "f1": {},
        "cumulative": {
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
        },
    }

    if class_explanations is None:
        return {
            "precision": {},
            "recall": {},
            "specificity": {},
            "f1": {},
            "cumulative": {
                "precision": float("nan"),
                "recall": float("nan"),
                "specificity": float("nan"),
                "f1": float("nan"),
            },
        }
    # Initialize cumulative counters
    cum_tp, cum_fp, cum_fn, cum_tn = 0, 0, 0, 0

    for c, phi in class_explanations.explanations.items():
        target_trajectories = test_traj_by_class.get(c, [])
        opp_trajs = class_explanations._get_opponent_trajectories(c)

        if phi is None or isinstance(phi, Boolean):
            results["precision"][
                c
            ] = 0.0  # if isinstance(phi, Boolean) else float("nan")
            results["recall"][c] = 0.0  # if isinstance(phi, Boolean) else float("nan")
            results["specificity"][
                c
            ] = 0.0  # if isinstance(phi, Boolean) else float("nan")
            results["f1"][c] = 0.0  # if isinstance(phi, Boolean) else float("nan")
            continue

        # Safety check: Skip if no target trajectories
        if len(target_trajectories) == 0:
            results["precision"][c] = float("nan")
            results["recall"][c] = float("nan")
            results["specificity"][c] = float("nan")
            results["f1"][c] = float("nan")
            continue

        # Compute robustness for target and opponent trajectories
        target_rho = phi.quantitative(
            target_trajectories, normalize=class_explanations.normalize
        )
        opp_rho = phi.quantitative(opp_trajs, normalize=class_explanations.normalize)

        # Class-specific metrics
        tp = (target_rho >= 0).sum().item()
        fp = (opp_rho >= 0).sum().item()
        fn = (target_rho < 0).sum().item()
        tn = (opp_rho < 0).sum().item()

        # Update cumulative counters
        cum_tp += tp
        cum_fp += fp
        cum_fn += fn
        cum_tn += tn

        # Calculate class-wise metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Store class results
        results["precision"][c] = round(precision * 100, 3)
        results["recall"][c] = round(recall * 100, 3)
        results["specificity"][c] = round(specificity * 100, 3)
        results["f1"][c] = round(f1 * 100, 3)

    # Calculate cumulative (micro-averaged) metrics
    cum_precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
    cum_recall = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0.0
    cum_specificity = cum_tn / (cum_tn + cum_fp) if (cum_tn + cum_fp) > 0 else 0.0
    cum_f1 = (
        2 * (cum_precision * cum_recall) / (cum_precision + cum_recall)
        if (cum_precision + cum_recall) > 0
        else 0.0
    )

    results["cumulative"]["precision"] = round(cum_precision * 100, 3)
    results["cumulative"]["recall"] = round(cum_recall * 100, 3)
    results["cumulative"]["specificity"] = round(cum_specificity * 100, 3)
    results["cumulative"]["f1"] = round(cum_f1 * 100, 3)

    return results


def get_local_metrics(explanations, testloader):
    local_explanations_true, local_explanations_pred = explanations

    # div perc true - correct = div perc pred - correct
    local_explanations_correct = [
        e for e in local_explanations_true
        if not e.is_misclassified
    ]
    _, local_division_correct = division_percentage_local(
        local_explanations_correct  # filtered by correct
    )
    readability_loc_correct = readability_local(
        local_explanations_correct
    )

    # div perc pred - incorrect -- spiegazioni fatte per la pred, traiettorie misclassificate
    local_explanations_pred_incorrect = [
        ep
        for ep, et in zip(local_explanations_pred, local_explanations_true)
        if et.is_misclassified
    ]
    _, local_division_pred_incorrect = division_percentage_local(
        local_explanations_pred_incorrect  # filtered by misclassified
    )
    
    readability_loc_pred_incorrect = readability_local(
        local_explanations_pred_incorrect
    )

    # div perc true - incorrect -- spiegazioni fatte per la true, traiettorie misclassificate
    local_explanations_true_incorrect = [
        et for et in local_explanations_true if et.is_misclassified
    ]
    _, local_division_true_incorrect = division_percentage_local(
        local_explanations_true_incorrect  # filtered by misclassified
    )
    readability_loc_true_incorrect = readability_local(
        local_explanations_true_incorrect
    )

    # div perc true - all, div perc pred - all
    _, local_division_true = division_percentage_local(
        local_explanations_true
    )  # unfiltered
    _, local_division_pred = division_percentage_local(
        local_explanations_pred
    )  # unfiltered

    local_metrics = {
        "division_correct": local_division_correct,
        "readability_correct": readability_loc_correct,
        "division_pred_incorrect": local_division_pred_incorrect,
        "readability_pred_incorrect": readability_loc_pred_incorrect,
        "division_true_incorrect": local_division_true_incorrect,
        "readability_true_incorrect": readability_loc_true_incorrect,
        "division_true_all": local_division_true,
        "division_pred_all": local_division_pred,
    }
    
    del (
        local_explanations_true,
        local_explanations_pred,
        local_explanations_pred_incorrect,
        local_explanations_true_incorrect,
    )

    return flatten_dict(local_metrics)


def get_global_metrics(class_explanations, model, testloader):
    # only correct
    _, global_division_correct = evaluate_class_explanations(
        model, class_explanations, testloader.dataset, filter=True
    )

    # only incorrect
    _, global_division_incorrect = evaluate_class_explanations(
        model, class_explanations, testloader.dataset, filter=True, incorrect=True
    )

    # all predicted, unfiltered
    _, global_division_unf = evaluate_class_explanations(
        model, class_explanations, testloader.dataset, filter=False
    )

    readability_glob = readability_global(class_explanations)

    f1results = evaluate_class_explanations_f1(
        model, class_explanations, testloader.dataset  # , filter=True
    )

    global_metrics = {
        "division_correct": global_division_correct,
        "division_incorrect": global_division_incorrect,
        "division_unfiltered": global_division_unf,
        "readability": readability_glob,
        "f1": f1results["cumulative"],
    }

    del class_explanations

    return flatten_dict(global_metrics)
