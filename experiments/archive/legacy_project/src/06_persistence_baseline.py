"""Compare current-height persistence with the frozen 24-hour ML test results."""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from feature_pipeline import (
    DATA_PATH,
    EXPERIMENT_DIR,
    classify_heights,
    prepare_features,
)
from model_metrics import metrics

CLASS_NAMES = ["Normal", "Elevated", "High"]
ML_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


def plot_results(matrix, comparison):
    """Save the persistence confusion matrix and five-method Macro-F1 plot."""
    figures = EXPERIMENT_DIR / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("24-hour persistence: true vs predicted")
    fig.tight_layout()
    fig.savefig(figures / "persistence_confusion_matrix.png", dpi=160)
    plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    comparison.set_index("family")["macro_f1"].plot.bar(
        ax=ax, color=["darkorange"] + ["steelblue"] * 4, rot=15
    )
    ax.set(
        title="24-hour exploratory test: persistence versus ML",
        ylabel="Macro-F1",
        xlabel="Method",
        ylim=(0, 1),
    )
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3)
    fig.tight_layout()
    fig.savefig(figures / "persistence_vs_ml.png", dpi=160)
    plt.show()
    plt.close(fig)


def main():
    """Use only current VHM0 to predict the unchanged 24-hour future labels."""
    metadata = json.loads((EXPERIMENT_DIR / "model_metadata.json").read_text())
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Source CSV differs from the frozen model dataset.")
    _, _, _, _, X_test, y_test = prepare_features()
    recorded = pd.read_csv(
        EXPERIMENT_DIR / "test_predictions.csv",
        index_col="timestamp",
        float_precision="round_trip",
    )
    recorded.index = pd.DatetimeIndex(pd.to_datetime(recorded.index, utc=True)).as_unit(
        "ns"
    )
    pd.testing.assert_series_equal(recorded["target_class"], y_test, check_exact=True)
    pd.testing.assert_series_equal(
        recorded["current_VHM0"], X_test["VHM0"], check_names=False, check_exact=True
    )
    ml = pd.read_csv(EXPERIMENT_DIR / "test_comparison.csv")
    if len(ml) != 4 or set(ml.family) != set(ML_NAMES):
        raise ValueError("Expected four frozen ML test results.")
    prediction = classify_heights(X_test["VHM0"])
    if not prediction.index.equals(y_test.index):
        raise ValueError("Persistence/test label alignment changed.")
    result = metrics(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1, 2])
    transition = pd.DataFrame(matrix.T, index=CLASS_NAMES, columns=CLASS_NAMES)
    transition.index.name = "current_class"
    transition.columns.name = "future_max_class"
    match = float(prediction.eq(y_test).mean())
    np.testing.assert_allclose(match, result["accuracy"])
    onset = X_test["VHM0"].le(3.0) & y_test.eq(2)
    baseline = {
        "family": "Persistence",
        **result,
        "confusion_matrix": json.dumps(matrix.tolist()),
        "test_rows": len(y_test),
        "test_start": str(y_test.index[0]),
        "test_end": str(y_test.index[-1]),
        "category_match_proportion": match,
        "actual_high_already_high": int((X_test["VHM0"].gt(3.0) & y_test.eq(2)).sum()),
        "new_high_transitions": int(onset.sum()),
        "new_high_transitions_missed": int(onset.sum()),
    }
    columns = [
        "family",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "high_precision",
        "high_recall",
        "high_f1",
        "missed_high",
        "false_high_alerts",
    ]
    ml = ml.set_index("family").loc[ML_NAMES].reset_index()
    comparison = pd.concat(
        [pd.DataFrame([baseline])[columns], ml[columns]], ignore_index=True
    )
    for metric in ["macro_f1", "high_recall", "missed_high", "false_high_alerts"]:
        comparison[metric + "_difference_from_persistence"] = (
            comparison[metric] - result[metric]
        )
    comparison["macro_f1_relative_improvement_percent"] = (
        comparison["macro_f1_difference_from_persistence"] / result["macro_f1"] * 100
        if result["macro_f1"]
        else np.nan
    )
    pd.DataFrame([baseline]).to_csv(
        EXPERIMENT_DIR / "persistence_baseline.csv", index=False
    )
    comparison.to_csv(EXPERIMENT_DIR / "persistence_comparison.csv", index=False)
    transition.to_csv(EXPERIMENT_DIR / "transition_matrix.csv")
    print("Persistence metrics:\n", pd.Series(result))
    print("Current-to-future transitions (rows=current, columns=future):\n", transition)
    print("Category agreement:", match, "New High origins:", int(onset.sum()))
    print("Five-method comparison:\n", comparison.to_string(index=False))
    print(
        "Differences are ML minus persistence; relative Macro-F1 differences are percentages."
    )
    plot_results(matrix, comparison)


if __name__ == "__main__":
    main()
