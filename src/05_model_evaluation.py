"""Evaluate saved 12h regressors without fitting; export metrics and figures."""

import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    precision_recall_fscore_support,
    confusion_matrix,
)

features = import_module("03_feature_engineering")
ROOT = features.preprocessing.ROOT
MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}
PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "joblib",
    "matplotlib",
    "threadpoolctl",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_record(model):
    """Record all constructor defaults, including scaler settings and NaN sentinel."""
    if hasattr(model, "named_steps"):
        return {
            name: parameter_record(step) for name, step in model.named_steps.items()
        }
    return {
        key: ("NaN" if isinstance(value, float) and np.isnan(value) else value)
        for key, value in model.get_params(deep=False).items()
    }


def regression_metrics(actual, predicted):
    return {
        "sample_count": len(actual),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
        "mean_bias": float(np.mean(np.asarray(predicted) - np.asarray(actual))),
        "median_absolute_error": median_absolute_error(actual, predicted),
    }


def binary_metrics(actual, predicted):
    actual = (np.asarray(actual) >= 2.5).astype(int)
    predicted = (np.asarray(predicted) >= 2.5).astype(int)
    p, r, f, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    values = {
        "sample_count": len(actual),
        "normal_count": int(support[0]),
        "elevated_count": int(support[1]),
        "elevated_percent": 100 * float(support[1]) / len(actual),
        "accuracy": float((actual == predicted).mean()),
        "balanced_accuracy": float(r[support > 0].mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(np.average(f, weights=support)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    for i, name in enumerate(["normal", "elevated"]):
        values.update(
            {
                f"{name}_precision": float(p[i]),
                f"{name}_recall": float(r[i]),
                f"{name}_f1": float(f[i]),
            }
        )
    return values


def save_figures(predictions, regression, classification):
    directory = ROOT / "results/figures"
    directory.mkdir(parents=True, exist_ok=True)
    names = list(MODEL_FILES)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["mae", "rmse", "r2"]):
        ax.bar(names, regression[metric])
        ax.set_title(f"12h {metric.upper()}")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(directory / "regression_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    classification[["elevated_precision", "elevated_recall", "elevated_f1"]].plot.bar(
        ax=ax
    )
    ax.set(
        title="12h Elevated detection (height >= 2.5 m)", ylim=(0, 1), xlabel="Model"
    )
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(directory / "classification_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.flat, names):
        row = classification.loc[name]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues")
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                color="white" if count > matrix.max() / 2 else "black",
            )
        ax.set(
            title=name,
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Elevated"],
            yticklabels=["Normal", "Elevated"],
            xlabel="Predicted",
            ylabel="Actual",
        )
    fig.suptitle("12h confusion matrices")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=180)
    plt.close(fig)
    sample = predictions.loc["2023-07-01":"2023-07-14"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        ax.plot(sample.index, sample.actual_VHM0, label="Actual t+12h")
        ax.plot(sample.index, sample[name], label="Predicted t+12h", alpha=0.8)
        ax.axhline(2.5, color="grey", linestyle="--")
        ax.set(title=name, ylabel="Height (m)", xlabel="Forecast origin (UTC)")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.flat, names):
        ax.hist(predictions[name] - predictions.actual_VHM0, bins=60)
        ax.axvline(0, color="black", linestyle="--")
        ax.set(title=name, xlabel="Predicted - actual (m)", ylabel="Count")
    fig.tight_layout()
    fig.savefig(directory / "residuals.png", dpi=180)
    plt.close(fig)


def main():
    _, _, _, _, X, y, partitions = features.prepare_features()
    metadata = json.loads((ROOT / "models/model_metadata.json").read_text())
    if (
        metadata["horizon_hours"] != 12
        or metadata["feature_names"] != X.columns.tolist()
        or metadata["source_sha256"] != sha(features.preprocessing.DATA_PATH)
        or metadata["partitions"] != partitions
    ):
        raise ValueError(
            "Saved model provenance differs from the current dataset or configuration."
        )
    if metadata["versions"] != {name: version(name) for name in PACKAGES}:
        raise ValueError(
            "Package versions differ from the recorded training environment."
        )
    predictions = pd.DataFrame({"actual_VHM0": y}, index=X.index)
    predictions["target_timestamp"] = X.index + pd.Timedelta(hours=12)
    regression, classification = [], []
    for name, filename in MODEL_FILES.items():
        path = ROOT / "models" / filename
        if sha(path) != metadata["models"][name]["sha256"]:
            raise ValueError(f"Model checksum changed: {name}")
        model = joblib.load(path)
        if parameter_record(model) != metadata["models"][name]["parameters"]:
            raise ValueError(f"Model parameters changed: {name}")
        predicted = np.asarray(model.predict(X), dtype=np.float64)
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite predictions")
        predictions[name] = predicted
        regression.append({"model": name, **regression_metrics(y, predicted)})
        classification.append({"model": name, **binary_metrics(y, predicted)})
    regression = pd.DataFrame(regression).set_index("model")
    classification = pd.DataFrame(classification).set_index("model")
    metrics = ROOT / "results/metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    output = ROOT / "results/predictions"
    output.mkdir(parents=True, exist_ok=True)
    regression.to_csv(metrics / "regression_metrics.csv")
    classification.to_csv(metrics / "classification_metrics.csv")
    classification[["tn", "fp", "fn", "tp"]].to_csv(metrics / "confusion_matrices.csv")
    predictions["actual_class"] = (y >= 2.5).astype(int)
    for name in MODEL_FILES:
        predictions[f"{name}_class"] = (predictions[name] >= 2.5).astype(int)
    predictions.to_csv(output / "test_predictions.csv", index_label="timestamp")
    save_figures(predictions, regression, classification)
    print("12-hour test regression metrics:\n", regression.to_string())
    print(
        "12-hour regression-derived classification metrics:\n",
        classification.to_string(),
    )


if __name__ == "__main__":
    main()
