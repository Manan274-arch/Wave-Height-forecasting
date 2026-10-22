"""Evaluate frozen point-forecast regressors and persistence without refitting."""

import hashlib
import json
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from feature_pipeline import DATA_PATH
from regression_pipeline import (
    METHODS,
    REGRESSION_DIR,
    category_errors,
    parse_horizon,
    persistence_improvements,
    prepare_regression,
    regression_metrics,
)


def save_figure(fig, directory, name):
    """Save report-ready PNGs and retain normal interactive plotting."""
    fig.tight_layout()
    fig.savefig(directory / name, dpi=160)
    plt.show()
    plt.close(fig)


def plot_results(predictions, validation, test, importance, directory, horizon):
    """Use every method and a predeclared calendar interval, without cherry-picking."""
    directory.mkdir(exist_ok=True)
    truth = predictions["actual_VHM0"]
    year = truth.index[0].year
    interval = predictions.loc[f"{year}-07-01" : f"{year}-07-14"]
    if interval.empty:
        interval = predictions.loc[: truth.index[0] + pd.Timedelta(days=14)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, frame, title in [
        (axes[0], predictions, "Complete test period"),
        (axes[1], interval, "July 1-14 of first test year (fixed calendar rule)"),
    ]:
        ax.plot(
            frame.index, frame.actual_VHM0, color="black", linewidth=1.5, label="Actual"
        )
        for name in METHODS:
            ax.plot(frame.index, frame[name], linewidth=0.8, alpha=0.8, label=name)
        ax.set(title=title, ylabel="Significant height (m)")
        ax.axhline(3.0, color="gray", linestyle=":", linewidth=0.7)
        ax.legend(ncol=3, fontsize=8)
    fig.suptitle(f"{horizon}h-ahead point forecasts; x-axis is prediction-origin UTC")
    save_figure(fig, directory, "time_series.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, name in zip(axes.flat, METHODS):
        lower = min(truth.min(), predictions[name].min())
        upper = max(truth.max(), predictions[name].max())
        ax.scatter(truth, predictions[name], s=5, alpha=0.2)
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set(title=name, xlabel="Actual future VHM0 (m)", ylabel="Predicted VHM0 (m)")
    axes.flat[-1].axis("off")
    fig.suptitle(f"{horizon}h-ahead actual versus predicted: all test observations")
    save_figure(fig, directory, "actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    residuals = predictions[METHODS].sub(truth, axis=0)
    edges = np.linspace(residuals.to_numpy().min(), residuals.to_numpy().max(), 65)
    for name in METHODS:
        ax.hist(residuals[name], bins=edges, density=True, histtype="step", label=name)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        title=f"{horizon}h signed residual distributions",
        xlabel="Prediction minus actual (m)",
        ylabel="Density",
    )
    ax.legend(fontsize=8)
    save_figure(fig, directory, "residual_distributions.png")

    scores = test.set_index("family")
    for error in ["mae", "rmse"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        scores[error].plot.bar(ax=ax, rot=15, color="steelblue")
        ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
        ax.set_ylim(0, scores[error].max() * 1.2)
        ax.set(
            title=f"{horizon}h test {error.upper()}: lower is better",
            ylabel=f"{error.upper()} (m)",
            xlabel="Method",
        )
        save_figure(fig, directory, error + "_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    paired = pd.DataFrame(
        {
            "Validation": validation.set_index("family").rmse,
            "Test": scores.rmse,
        }
    ).reindex(METHODS)
    paired.plot.bar(ax=ax, rot=15)
    ax.set(
        title=f"{horizon}h validation versus test", ylabel="RMSE (m)", xlabel="Method"
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.set_title(f"{horizon}h validation versus test", pad=42)
    save_figure(fig, directory, "validation_vs_test_rmse.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    scores.loc[scores.index != "Persistence", "rmse_improvement_percent"].plot.bar(
        ax=ax, rot=15, color="steelblue"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"{horizon}h RMSE improvement over persistence",
        ylabel="Improvement (%)",
        xlabel="ML method",
    )
    save_figure(fig, directory, "persistence_vs_ml_rmse.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    scores[["high_mae", "high_rmse"]].rename(
        columns={"high_mae": "MAE", "high_rmse": "RMSE"}
    ).plot.bar(ax=ax, rot=15)
    ax.set(
        title=f"{horizon}h errors where actual future VHM0 > 3.0 m (n={int(test.high_count.iloc[0])})",
        ylabel="Error (m)",
        xlabel="Method",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.set_title(
        f"{horizon}h errors where actual future VHM0 > 3.0 m (n={int(test.high_count.iloc[0])})",
        pad=42,
    )
    save_figure(fig, directory, "high_wave_errors.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, name in zip(axes, ["Decision Tree", "Random Forest", "XGBoost"]):
        importance[name].nlargest(12).sort_values().plot.barh(ax=ax)
        measure = (
            "normalized gain" if name == "XGBoost" else "normalized impurity reduction"
        )
        ax.set(title=name, xlabel=measure)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        f"{horizon}h fitted-model importance: correlated features; not causal effects"
    )
    save_figure(fig, directory, "feature_importance.png")


def main():
    """Load one horizon's four saved regressors, then report five frozen methods."""
    horizon = parse_horizon()
    directory = REGRESSION_DIR / f"{horizon}_hour"
    metadata_path = directory / "model_metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    _, _, _, _, X_test, y_test, summary = prepare_regression(horizon)
    if summary != {key: metadata[key] for key in summary}:
        raise ValueError("Regression dataset summary differs from training metadata.")
    if X_test.columns.tolist() != metadata["feature_names"]:
        raise ValueError("Regression inference feature order changed.")
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Original data changed after training.")
    for package, expected in metadata["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Dependency version mismatch: {package}")
    for name, info in metadata["models"].items():
        if (
            hashlib.sha256((directory / info["file"]).read_bytes()).hexdigest()
            != info["sha256"]
        ):
            raise ValueError(f"Frozen model changed: {name}")
    linear = joblib.load(directory / "models/linear_regression.joblib")
    tree = joblib.load(directory / "models/decision_tree.joblib")
    forest = joblib.load(directory / "models/random_forest.joblib")
    xgb = joblib.load(directory / "models/xgboost.joblib")
    models = {
        "Linear Regression": linear,
        "Decision Tree": tree,
        "Random Forest": forest,
        "XGBoost": xgb,
    }
    for name, model in models.items():
        if list(model.feature_names_in_) != X_test.columns.tolist():
            raise ValueError(f"Model feature order mismatch: {name}")
    with threadpool_limits(limits=4):
        linear_pred = linear.predict(X_test)
        tree_pred = tree.predict(X_test)
        forest_pred = forest.predict(X_test)
        xgb_pred = xgb.predict(X_test)
    predictions = pd.DataFrame(
        {
            "target_timestamp": X_test.index + pd.Timedelta(hours=horizon),
            "actual_VHM0": y_test,
            "current_VHM0": X_test.VHM0,
            "Persistence": X_test.VHM0.to_numpy(),
            "Linear Regression": linear_pred,
            "Decision Tree": tree_pred,
            "Random Forest": forest_pred,
            "XGBoost": xgb_pred,
        },
        index=X_test.index,
    )
    # Preserve float32 XGBoost values exactly when serializing decimal predictions.
    # Promotion changes representation only, not any predicted value.
    predictions[METHODS] = predictions[METHODS].astype("float64")
    rows = []
    conditional = []
    for name in METHODS:
        rows.append({"family": name, **regression_metrics(y_test, predictions[name])})
        conditional.extend(
            {"family": name, **row}
            for row in category_errors(y_test, predictions[name])
        )
    results = persistence_improvements(pd.DataFrame(rows)).sort_values(
        "rmse", kind="stable"
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
    importance = pd.DataFrame(
        {
            "Decision Tree": tree.feature_importances_,
            "Random Forest": forest.feature_importances_,
            "XGBoost": xgb.feature_importances_,
        },
        index=X_test.columns,
    )
    for name, info in metadata["models"].items():
        if (
            hashlib.sha256((directory / info["file"]).read_bytes()).hexdigest()
            != info["sha256"]
        ):
            raise RuntimeError(f"A frozen model changed during evaluation: {name}")
    if metadata_path.read_bytes() != metadata_bytes:
        raise RuntimeError("Model metadata changed during evaluation.")
    results.to_csv(directory / "test_results.csv", index=False)
    results[
        [
            "family",
            "mae",
            "rmse",
            "mae_absolute_improvement",
            "mae_improvement_percent",
            "rmse_absolute_improvement",
            "rmse_improvement_percent",
        ]
    ].to_csv(directory / "persistence_comparison.csv", index=False)
    paired.to_csv(directory / "validation_test_comparison.csv", index=False)
    pd.DataFrame(conditional).to_csv(directory / "category_errors.csv", index=False)
    predictions.to_csv(directory / "test_predictions.csv", index_label="timestamp")
    importance.to_csv(directory / "feature_importance.csv", index_label="feature")
    print(f"{horizon}h test results:\n", results.to_string(index=False))
    print(
        "Actual-future-category diagnostics:\n",
        pd.DataFrame(conditional).to_string(index=False),
    )
    plot_results(
        predictions, validation, results, importance, directory / "figures", horizon
    )
    print(
        "Evaluation completed with no fitting, clipping, tuning or model modification."
    )


if __name__ == "__main__":
    main()
