"""Evaluate four frozen models on held-out 2023-2024 data without retraining."""

import hashlib
import json
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from threadpoolctl import threadpool_limits

from feature_pipeline import (
    DATA_PATH,
    EXPERIMENT_DIR,
    HORIZON_HOURS,
    HORIZON_STEPS,
    prepare_features,
)
from model_metrics import metrics

RESULT_DIR = EXPERIMENT_DIR
CLASS_NAMES = ["Normal", "Elevated", "High"]


def check_compatibility(metadata, X_test, y_test):
    """Reject incompatible data, feature order or dependency versions."""
    expected = metadata["test_partition"]
    if X_test.shape != (expected["rows"], 46) or not X_test.index.equals(y_test.index):
        raise ValueError("Held-out partition shape/alignment changed.")
    if (
        metadata["horizon_hours"] != HORIZON_HOURS
        or metadata["horizon_steps"] != HORIZON_STEPS
    ):
        raise ValueError("Saved forecast horizon differs from the active target.")
    if (str(X_test.index[0]), str(X_test.index[-1])) != (
        expected["start"],
        expected["end"],
    ):
        raise ValueError("Test dates differ from the saved partition.")
    if X_test.columns.tolist() != metadata["feature_names"]:
        raise ValueError("Test feature names or ordering differ from training.")
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("The source CSV differs from the training dataset.")
    refit_end = pd.Timestamp(metadata["refit_end"])
    if refit_end + pd.Timedelta(hours=HORIZON_HOURS) >= X_test.index[0]:
        raise ValueError("Saved refit label horizons overlap testing.")
    if metadata["test_used_for_selection"]:
        raise ValueError("This study requires selection without test data.")
    for package, expected in metadata["versions"].items():
        if version(package) != expected:
            raise ValueError(f"{package} version differs from saved metadata.")


def plot_test_results(comparison, matrices):
    """Display and save four descriptive test comparisons, without selection."""
    figure_dir = RESULT_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    scores = comparison.set_index("family")
    fig, ax = plt.subplots(figsize=(9, 4))
    scores["macro_f1"].plot.bar(ax=ax, color="steelblue", rot=15)
    ax.set(
        title="24-hour exploratory test Macro-F1",
        xlabel="Algorithm",
        ylabel="Macro-F1",
        ylim=(0, 1),
    )
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "test_macro_f1.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    scores[["high_precision", "high_recall"]].rename(
        columns={"high_precision": "High precision", "high_recall": "High recall"}
    ).plot.bar(ax=ax, rot=15)
    ax.set(
        title="Held-out High-class precision and recall",
        xlabel="Algorithm",
        ylabel="Score",
        ylim=(0, 1),
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.set_title("Held-out High-class precision and recall", pad=42)
    fig.tight_layout()
    fig.savefig(figure_dir / "high_precision_recall.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    scores[["missed_high", "false_high_alerts"]].rename(
        columns={
            "missed_high": "Missed High observations",
            "false_high_alerts": "False High alerts",
        }
    ).plot.bar(ax=ax, rot=15, color=["darkorange", "steelblue"])
    ax.set(
        title="Held-out High-class error counts",
        xlabel="Algorithm",
        ylabel="Observations",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "missed_false_high.png", dpi=160)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, matrix) in zip(axes.flat, matrices.items()):
        display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name)
    fig.suptitle("Held-out confusion matrices: rows=true, columns=predicted")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrices.png", dpi=160)
    plt.show()
    plt.close(fig)


def main():
    """Predict once with each frozen model and report independent test results."""
    metadata_path = RESULT_DIR / "model_metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    _, _, _, _, X_test, y_test = prepare_features(DATA_PATH)
    check_compatibility(metadata, X_test, y_test)
    print("Held-out test:", X_test.shape, X_test.index[0], "to", X_test.index[-1])
    print("Test class counts:\n", y_test.value_counts().sort_index())

    # Check all model artifacts before loading or predicting with any of them.
    hashes = {}
    for name, info in metadata["models"].items():
        path = RESULT_DIR / info["file"]
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[name] != info["sha256"]:
            raise ValueError(f"Saved model hash mismatch: {name}")
    lr = joblib.load(RESULT_DIR / "models/logistic_regression.joblib")
    dt = joblib.load(RESULT_DIR / "models/decision_tree.joblib")
    rf = joblib.load(RESULT_DIR / "models/random_forest.joblib")
    xgb = joblib.load(RESULT_DIR / "models/xgboost.joblib")
    models = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
        "XGBoost": xgb,
    }
    for name, model in models.items():
        if list(model.feature_names_in_) != metadata["feature_names"]:
            raise ValueError(f"Saved feature order mismatch: {name}")
        np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    with threadpool_limits(limits=4):
        lr_pred = lr.predict(X_test)
        dt_pred = dt.predict(X_test)
        rf_pred = rf.predict(X_test)
        xgb_pred = xgb.predict(X_test)
    predictions = {
        "Logistic Regression": lr_pred,
        "Decision Tree": dt_pred,
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
    }
    rows = []
    differences = []
    matrices = {}
    for name, prediction in predictions.items():
        result = metrics(y_test, prediction)
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1, 2])
        matrices[name] = matrix
        rows.append(
            {
                "family": name,
                "strategy": metadata["models"][name]["strategy"],
                **result,
                "confusion_matrix": json.dumps(matrix.tolist()),
                "test_rows": len(X_test),
                "test_start": str(X_test.index[0]),
                "test_end": str(X_test.index[-1]),
                "model_sha256": hashes[name],
            }
        )
        validation = metadata["models"][name]["validation_metrics_before_refit"]
        differences.append(
            {
                "family": name,
                "strategy": metadata["models"][name]["strategy"],
                "validation_macro_f1": validation["macro_f1"],
                "test_macro_f1": result["macro_f1"],
                "macro_f1_change_test_minus_validation": result["macro_f1"]
                - validation["macro_f1"],
                "validation_high_recall": validation["high_recall"],
                "test_high_recall": result["high_recall"],
                "high_recall_change_test_minus_validation": result["high_recall"]
                - validation["high_recall"],
            }
        )
        print("\n", name)
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
        print("Confusion matrix:\n", matrix)
    # Reporting order only; test scores never alter the four frozen selections.
    comparison = pd.DataFrame(rows).sort_values(
        "macro_f1", ascending=False, kind="stable"
    )
    differences = pd.DataFrame(differences)
    for name, info in metadata["models"].items():
        path = RESULT_DIR / info["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise RuntimeError("A model file changed during evaluation.")
    if metadata_path.read_bytes() != metadata_bytes:
        raise RuntimeError("Model metadata changed during evaluation.")
    prediction_table = pd.DataFrame(
        {
            "current_VHM0": X_test["VHM0"],
            "target_class": y_test,
            **predictions,
        },
        index=X_test.index,
    )
    prediction_table.to_csv(
        RESULT_DIR / "test_predictions.csv", index_label="timestamp"
    )
    comparison.to_csv(RESULT_DIR / "test_comparison.csv", index=False)
    differences.to_csv(RESULT_DIR / "validation_test_comparison.csv", index=False)
    print("\nFinal held-out comparison:\n", comparison.to_string(index=False))
    print("\nValidation versus test:\n", differences.to_string(index=False))
    plot_test_results(comparison, matrices)
    print("Phase 5 completed without refitting or modifying saved models.")


if __name__ == "__main__":
    main()
