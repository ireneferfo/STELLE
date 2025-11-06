from collections import Counter
import numpy as np

from STELLE.utils import round_dict_values

def weighted_accuracy(labels, preds):
    label_counts = Counter(labels)
    total = len(labels)
    weighted_acc = 0
    for cls in label_counts:
        idx = [i for i, y in enumerate(labels) if y == cls]
        cls_correct = sum(preds[i] == labels[i] for i in idx)
        weighted_acc += (label_counts[cls] / total) * (cls_correct / label_counts[cls])
    return weighted_acc * 100


def sensitivity_specificity(labels, preds, classes=None):
    """
    Computes sensitivity (recall, true positive rate) and specificity (true negative rate)
    for binary classification.

    Args:
        labels: list or array of true labels (0 or 1)
        preds: list or array of predicted labels (0 or 1)

    Returns:
        sensitivity, specificity
    """
    labels = np.array(labels)
    preds = np.array(preds)

    if classes is None:
        classes = np.unique(labels)

    per_class_sensitivity = {}
    per_class_specificity = {}

    # Initialize cumulative counters
    cum_tp, cum_tn, cum_fp, cum_fn = 0, 0, 0, 0

    for cls in range(classes):
        # One-vs-rest for each class
        tp = np.sum((labels == cls) & (preds == cls))
        tn = np.sum((labels != cls) & (preds != cls))
        fp = np.sum((labels != cls) & (preds == cls))
        fn = np.sum((labels == cls) & (preds != cls))

        # Update cumulative counts
        cum_tp += tp
        cum_tn += tn
        cum_fp += fp
        cum_fn += fn

        # Per-class metrics
        per_class_sensitivity[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_specificity[cls] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Cumulative (micro-averaged) metrics
    global_sensitivity = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0.0
    global_specificity = cum_tn / (cum_tn + cum_fp) if (cum_tn + cum_fp) > 0 else 0.0

    return round_dict_values(
        {"cumulative": global_sensitivity, "per_class": per_class_sensitivity}
    ), round_dict_values(
        {
            "cumulative": global_specificity,
            "per_class": per_class_specificity,
        }
    )