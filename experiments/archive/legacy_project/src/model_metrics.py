"""Consistent three-class metrics shared by training and final evaluation."""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def metrics(y_true, prediction):
    """Use the same three-class metrics for validation and held-out test data."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1, 2], zero_division=0
    )
    result = {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": f1_score(y_true, prediction, average="weighted"),
        "high_precision": precision[2],
        "high_recall": recall[2],
        "high_f1": f1[2],
        "missed_high": int(((y_true == 2) & (prediction != 2)).sum()),
        "false_high_alerts": int(((y_true != 2) & (prediction == 2)).sum()),
    }
    for label in range(3):
        result[f"class_{label}_precision"] = precision[label]
        result[f"class_{label}_recall"] = recall[label]
        result[f"class_{label}_f1"] = f1[label]
        result[f"class_{label}_support"] = int(support[label])
    return result
