"""Diagnose six-hour category persistence without fitting or rerunning ML models."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from feature_pipeline import (
    build_labels,
    load_source,
    prepare_features,
    split_labels,
)
from model_metrics import metrics

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "udupi_wave_wind_2015_2024.csv"
MODEL_DIR = ROOT_DIR / "models"
CLASS_NAMES = ["Normal", "Elevated", "High"]
ML_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


def predict_persistence(current_height):
    """Classify only the observed height at t; no future values are inputs."""
    if current_height.isna().any() or not np.isfinite(current_height).all():
        raise ValueError("Current heights must be complete and finite.")
    prediction = pd.Series(0, index=current_height.index, dtype="int64")
    prediction.loc[current_height.between(2.5, 3.0)] = 1
    prediction.loc[current_height.gt(3.0)] = 2
    return prediction.rename("persistence_class")


def analyze_transitions(current_class, y_test):
    """Report current-to-future categories, rather than independent storm events."""
    # This orientation is the transpose of a true-versus-predicted confusion matrix.
    transition = pd.crosstab(current_class, y_test).reindex(
        index=[0, 1, 2], columns=[0, 1, 2], fill_value=0
    )
    transition.index = pd.Index(CLASS_NAMES, name="Current class")
    transition.columns = pd.Index(CLASS_NAMES, name="Future maximum class")
    print("\nCurrent-to-future transitions (counts):\n", transition)
    percentages = transition.div(transition.sum(axis=1), axis=0).mul(100)
    print("\nTransitions as percentages of each current class:\n", percentages)
    print("Normal -> Elevated:", transition.loc["Normal", "Elevated"])
    print("Normal -> High:", transition.loc["Normal", "High"])
    print("Elevated -> High:", transition.loc["Elevated", "High"])
    print("High -> High:", transition.loc["High", "High"])
    return transition


def plot_results(matrix, comparison):
    """Save and display the same class order and visual style used in Phase 5."""
    figure_dir = MODEL_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Persistence: held-out confusion matrix")
    fig.tight_layout()
    fig.savefig(figure_dir / "persistence_confusion_matrix.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["darkorange"] + ["steelblue"] * 4
    comparison.set_index("family")["macro_f1"].plot.bar(ax=ax, color=colors, rot=15)
    ax.set(
        title="Six-hour held-out Macro-F1: persistence versus ML",
        xlabel="Method",
        ylabel="Macro-F1",
        ylim=(0, 1),
    )
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "persistence_vs_ml.png", dpi=160)
    plt.show()
    plt.close(fig)


def main():
    """Evaluate one deterministic baseline on the exact Phase 5 test partition."""
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text())
    ml_results = pd.read_csv(MODEL_DIR / "test_comparison.csv")
    if len(ml_results) != 4 or set(ml_results["family"]) != set(ML_NAMES):
        raise ValueError("Expected exactly the four existing ML test results.")
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("The CSV differs from the Phase 5 source.")

    source = load_source(DATA_PATH)
    # Phase 5 uses this exact shared call. No features or true labels are redefined.
    _, _, _, _, X_test, y_test = prepare_features(source=source)
    if X_test.shape != (5845, 46) or not X_test.index.equals(y_test.index):
        raise ValueError("The Phase 5 test shape or timestamp alignment changed.")
    if X_test.columns.tolist() != metadata["feature_names"]:
        raise ValueError("The Phase 5 feature order changed.")
    reference_labels = split_labels(build_labels(source))["Testing"]["target_class"]
    pd.testing.assert_series_equal(y_test, reference_labels, check_exact=True)
    current_height = X_test["VHM0"]
    pd.testing.assert_series_equal(
        current_height, source.loc[y_test.index, "VHM0"], check_exact=True
    )
    support = y_test.value_counts().reindex([0, 1, 2], fill_value=0)
    for row in ml_results.itertuples():
        if row.test_rows != len(y_test):
            raise ValueError("Saved ML test row count differs.")
        if (pd.Timestamp(row.test_start), pd.Timestamp(row.test_end)) != (
            y_test.index[0],
            y_test.index[-1],
        ):
            raise ValueError("Saved ML test period differs.")
        matrix = np.array(json.loads(row.confusion_matrix))
        np.testing.assert_array_equal(matrix.sum(axis=1), support.to_numpy())
        if row.model_sha256 != metadata["models"][row.family]["sha256"]:
            raise ValueError("Saved ML results and model metadata disagree.")

    prediction = predict_persistence(current_height)
    if (
        not prediction.index.equals(y_test.index)
        or not prediction.isin([0, 1, 2]).all()
    ):
        raise ValueError("Persistence prediction alignment or classes are invalid.")
    result = metrics(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1, 2])
    category_match = prediction.eq(y_test).mean()
    np.testing.assert_allclose(category_match, result["accuracy"], rtol=0, atol=1e-12)
    transition = analyze_transitions(prediction, y_test)
    np.testing.assert_array_equal(transition.to_numpy(), matrix.T)
    actual_high = y_test.eq(2)
    already_high = int((actual_high & prediction.eq(2)).sum())
    new_high = actual_high & prediction.ne(2)
    new_high_missed = int((new_high & prediction.ne(2)).sum())
    print("\nTest rows and UTC range:", len(y_test), y_test.index[0], y_test.index[-1])
    print("Current category equals future maximum category:", f"{category_match:.6%}")
    print("Actual High observations already High at t:", already_high)
    print(
        "Actual High observations with current height < 3.0 m:",
        int((actual_high & current_height.lt(3.0)).sum()),
    )
    print(
        "Actual High observations with current height == 3.0 m:",
        int((actual_high & current_height.eq(3.0)).sum()),
    )
    print("New High transitions (current <= 3.0 m):", int(new_high.sum()))
    print("New High transitions missed by persistence:", new_high_missed)
    if new_high_missed != result["missed_high"]:
        raise ValueError("High-transition counts disagree with classification metrics.")
    print(
        classification_report(
            y_test,
            prediction,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            digits=6,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):\n", matrix)

    baseline = {
        "family": "Persistence",
        "strategy": "current VHM0 category",
        **result,
        "confusion_matrix": json.dumps(matrix.tolist()),
        "test_rows": len(y_test),
        "test_start": str(y_test.index[0]),
        "test_end": str(y_test.index[-1]),
        "category_match_proportion": category_match,
        "current_to_future_transition_matrix": json.dumps(
            transition.to_numpy().tolist()
        ),
        "actual_high_already_high": already_high,
        "new_high_transitions": int(new_high.sum()),
        "new_high_transitions_missed": new_high_missed,
    }
    ml_results = ml_results.set_index("family").loc[ML_NAMES].reset_index()
    columns = [
        "family",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "high_precision",
        "high_recall",
        "missed_high",
        "false_high_alerts",
    ]
    comparison = pd.concat(
        [pd.DataFrame([baseline])[columns], ml_results[columns]], ignore_index=True
    )
    comparison["macro_f1_absolute_improvement_over_persistence"] = (
        comparison["macro_f1"] - result["macro_f1"]
    )
    relative_improvement = comparison["macro_f1_absolute_improvement_over_persistence"]
    comparison["macro_f1_relative_improvement_percent"] = (
        relative_improvement / result["macro_f1"] * 100
        if result["macro_f1"]
        else np.nan
    )
    print("\nFive-method test comparison:\n", comparison.to_string(index=False))
    print(
        "Absolute improvement is a score difference; relative improvement divides by persistence Macro-F1."
    )
    print("Category transitions are correlated observations, not independent storms.")
    pd.DataFrame([baseline]).to_csv(MODEL_DIR / "persistence_baseline.csv", index=False)
    comparison.to_csv(MODEL_DIR / "persistence_comparison.csv", index=False)
    plot_results(matrix, comparison)
    print("Diagnostic complete: no model fitting or new ML predictions performed.")


if __name__ == "__main__":
    main()
