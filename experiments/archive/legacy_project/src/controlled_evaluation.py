"""Score the 32 frozen controlled models; no fitting or feature selection."""

import hashlib
import json
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from controlled_pipeline import OUTPUT_DIR, prepare_controlled
from feature_pipeline import DATA_PATH
from regression_evaluation import save_figure
from regression_pipeline import (
    METHODS,
    category_errors,
    persistence_improvements,
    regression_metrics,
)


def data_hash(series):
    """Fingerprint origin-aligned numeric values for baseline/sample identity checks."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(series, index=True).values.tobytes()
    ).hexdigest()


def plot_configuration(predictions, horizon, feature_set):
    """Use a fixed calendar interval and all residuals, without choosing good cases."""
    directory = OUTPUT_DIR / "figures" / f"{horizon}_hour" / feature_set
    directory.mkdir(parents=True, exist_ok=True)
    year = predictions.index[0].year
    interval = predictions.loc[f"{year}-07-01" : f"{year}-07-14"]
    if interval.empty:
        interval = predictions.loc[: predictions.index[0] + pd.Timedelta(days=14)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, frame, label in [
        (axes[0], predictions, "Entire test period"),
        (axes[1], interval, "July 1-14 of first test year; fixed rule"),
    ]:
        ax.plot(
            frame.index, frame.actual_VHM0, color="black", linewidth=1.5, label="Actual"
        )
        for name in METHODS:
            ax.plot(frame.index, frame[name], linewidth=0.8, label=name, alpha=0.8)
        ax.set(title=label, ylabel="Significant height (m)")
        ax.legend(ncol=3, fontsize=8)
    fig.suptitle(f"{horizon}h {feature_set}: x-axis is forecast-origin UTC")
    save_figure(fig, directory, "time_series.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    residuals = predictions[METHODS].sub(predictions.actual_VHM0, axis=0)
    bins = np.linspace(residuals.to_numpy().min(), residuals.to_numpy().max(), 65)
    for name in METHODS:
        ax.hist(residuals[name], bins=bins, histtype="step", density=True, label=name)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        title=f"{horizon}h {feature_set}: full test residual distributions",
        xlabel="Prediction minus actual (m)",
        ylabel="Density",
    )
    ax.legend(fontsize=8)
    save_figure(fig, directory, "residual_distributions.png")


def evaluate_configuration(horizon, feature_set):
    """Load four fitted models and evaluate on exactly the prepared test origins."""
    directory = OUTPUT_DIR / f"{horizon}_hour" / feature_set
    metadata_path = directory / "model_metadata.json"
    original_metadata = metadata_path.read_bytes()
    metadata = json.loads(original_metadata)
    _, _, _, _, X, y, summary = prepare_controlled(horizon, feature_set)
    if summary != {key: metadata[key] for key in summary}:
        raise ValueError("Saved controlled dataset/feature definition changed.")
    if X.columns.tolist() != metadata["feature_names"]:
        raise ValueError("Inference feature order changed.")
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Source dataset changed.")
    for package, expected in metadata["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Saved model dependency mismatch: {package}")
    for name, info in metadata["models"].items():
        if (
            hashlib.sha256((directory / info["file"]).read_bytes()).hexdigest()
            != info["sha256"]
        ):
            raise ValueError(f"Saved model changed: {name}")
    linear = joblib.load(directory / "models/linear_regression.joblib")
    tree = joblib.load(directory / "models/decision_tree.joblib")
    forest = joblib.load(directory / "models/random_forest.joblib")
    xgb = joblib.load(directory / "models/xgboost.joblib")
    for model in [linear, tree, forest, xgb]:
        if list(model.feature_names_in_) != X.columns.tolist():
            raise ValueError("Model input column order changed.")
    with threadpool_limits(limits=4):
        linear_pred = linear.predict(X)
        tree_pred = tree.predict(X)
        forest_pred = forest.predict(X)
        xgb_pred = xgb.predict(X)
    predictions = pd.DataFrame(
        {
            "actual_VHM0": y,
            "current_VHM0": X.VHM0,
            "target_timestamp": X.index + pd.Timedelta(hours=horizon),
            "Persistence": X.VHM0.to_numpy(),
            "Linear Regression": linear_pred,
            "Decision Tree": tree_pred,
            "Random Forest": forest_pred,
            "XGBoost": xgb_pred,
        },
        index=X.index,
    )
    predictions[METHODS] = predictions[METHODS].astype("float64")
    baseline_path = OUTPUT_DIR / f"{horizon}_hour/persistence_results.csv"
    target_fingerprint, current_fingerprint = data_hash(y), data_hash(X.VHM0)
    if feature_set == "full":
        baseline = {
            "family": "Persistence",
            **regression_metrics(y, predictions.Persistence),
            "target_values_sha256": target_fingerprint,
            "current_values_sha256": current_fingerprint,
        }
        pd.DataFrame([baseline]).to_csv(baseline_path, index=False)
    else:
        baseline = (
            pd.read_csv(baseline_path, float_precision="round_trip").iloc[0].to_dict()
        )
        if (
            baseline["target_values_sha256"] != target_fingerprint
            or baseline["current_values_sha256"] != current_fingerprint
        ):
            raise ValueError("Full and compact baseline origins/targets differ.")
    rows = [
        {key: value for key, value in baseline.items() if not key.endswith("sha256")}
    ]
    for name in METHODS[1:]:
        rows.append({"family": name, **regression_metrics(y, predictions[name])})
    results = persistence_improvements(pd.DataFrame(rows)).sort_values(
        "rmse", kind="stable"
    )
    conditional = []
    for name in METHODS:
        conditional.extend(
            {"family": name, **row} for row in category_errors(y, predictions[name])
        )
    validation = pd.read_csv(directory / "validation_results.csv")
    paired = validation[["family", "mae", "rmse", "r2"]].merge(
        results[["family", "mae", "rmse", "r2"]],
        on="family",
        suffixes=("_validation", "_test"),
    )
    paired["rmse_change_test_minus_validation"] = (
        paired.rmse_test - paired.rmse_validation
    )
    results.to_csv(directory / "test_results.csv", index=False)
    pd.DataFrame(conditional).to_csv(directory / "category_errors.csv", index=False)
    paired.to_csv(directory / "validation_test_comparison.csv", index=False)
    predictions.to_csv(directory / "test_predictions.csv", index_label="timestamp")
    if metadata_path.read_bytes() != original_metadata:
        raise RuntimeError("Model metadata changed during evaluation.")
    for name, info in metadata["models"].items():
        if (
            hashlib.sha256((directory / info["file"]).read_bytes()).hexdigest()
            != info["sha256"]
        ):
            raise RuntimeError(f"Model changed during evaluation: {name}")
    print(
        "TEST",
        horizon,
        feature_set,
        results[["family", "mae", "rmse", "r2", "high_rmse"]].to_string(index=False),
        flush=True,
    )
    plot_configuration(predictions, horizon, feature_set)


def main():
    # Require the entire training stage before exposing any controlled test scores.
    for horizon in [6, 24, 48, 72]:
        for feature_set in ["full", "compact"]:
            if not (
                OUTPUT_DIR / f"{horizon}_hour" / feature_set / "model_metadata.json"
            ).exists():
                raise FileNotFoundError(
                    "Complete all eight training configurations first."
                )
    for horizon in [6, 24, 48, 72]:
        for feature_set in ["full", "compact"]:
            evaluate_configuration(horizon, feature_set)


if __name__ == "__main__":
    main()
