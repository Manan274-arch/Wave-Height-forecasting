"""Fixed-configuration 12h regression, paired 6h comparisons and block uncertainty."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.base import clone, is_regressor
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
OLD = ROOT / "experiments/controlled_regression/6_hour/compact"
sys.path.insert(0, str(ROOT / "src"))
from controlled_pipeline import prepare_controlled  # noqa: E402
from feature_pipeline import DATA_PATH, load_source, make_features  # noqa: E402
from regression_pipeline import regression_metrics  # noqa: E402

NAMES = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
COLORS = dict(zip(NAMES, ["#2878B5", "#E1812C", "#339966", "#C74343"]))
BLOCK_LENGTHS = [56, 112]
REPLICATES = 2000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def binary_metrics(truth, prediction):
    """Use unrounded heights, fixed class order and safe zero divisions."""
    actual = (np.asarray(truth) >= 2.5).astype(int)
    predicted = (np.asarray(prediction) >= 2.5).astype(int)
    p, r, f, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "sample_count": len(actual),
        "normal_count": int(support[0]),
        "elevated_count": int(support[1]),
        "elevated_percent": 100 * float(support[1]) / len(actual),
        "accuracy": float((actual == predicted).mean()),
        "balanced_accuracy": float(r[support > 0].mean()),
        "normal_precision": p[0],
        "normal_recall": r[0],
        "normal_f1": f[0],
        "elevated_precision": p[1],
        "elevated_recall": r[1],
        "elevated_f1": f[1],
        "macro_f1": float(f.mean()),
        "weighted_f1": float(np.average(f, weights=support)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def prepare_12h():
    """Mirror original raw-boundary exclusions; reuse unchanged feature definitions."""
    source = load_source(DATA_PATH)
    names = json.loads((OLD / "model_metadata.json").read_text())["feature_names"]
    features = make_features(source).loc[:, names]
    assert len(names) == 18
    # Timestamp lookup independently verifies the shift on the actual regular grid.
    future_times = source.index + pd.Timedelta(hours=12)
    target = pd.Series(
        source.VHM0.reindex(future_times).to_numpy(),
        index=source.index,
        name="future_VHM0",
    )
    pd.testing.assert_series_equal(target, source.VHM0.shift(-4).rename("future_VHM0"))
    missing_history = features.isna().any(axis=1)
    assert features.index[missing_history].equals(source.index[:8])
    validation_position = int(np.floor(0.7 * len(source)))
    test_position = validation_position + int(np.floor(0.1 * len(source)))
    nominal = {
        "Training": source.index[:validation_position],
        "Validation": source.index[validation_position:test_position],
        "Testing": source.index[test_position:],
    }
    next_start = {
        "Training": source.index[validation_position],
        "Validation": source.index[test_position],
    }
    objects, summary = [], {}
    for name, origins in nominal.items():
        complete = origins[target.loc[origins].notna().to_numpy()]
        if name in next_start:
            eligible = complete[complete + pd.Timedelta(hours=12) < next_start[name]]
        else:
            eligible = complete
        purged = len(complete) - len(eligible)
        assert purged == (4 if name != "Testing" else 0)
        history_count = int(missing_history.loc[eligible].sum())
        eligible = eligible[~missing_history.loc[eligible].to_numpy()]
        X, y = features.loc[eligible].copy(), target.loc[eligible].copy()
        assert (
            X.index.equals(y.index)
            and np.isfinite(X.to_numpy()).all()
            and np.isfinite(y).all()
        )
        objects.extend([X, y])
        summary[name] = {
            "nominal_rows": len(origins),
            "rows": len(X),
            "start": str(X.index[0]),
            "end": str(X.index[-1]),
            "history_excluded": history_count,
            "purged": purged,
            "unavailable_targets": len(origins) - len(complete),
        }
    assert [len(objects[i]) for i in [0, 2, 4]] == [20444, 2918, 5842]
    # Prove column values exactly equal the existing compact features at shared origins.
    existing = prepare_controlled(6, "compact")
    for position in [0, 2, 4]:
        pd.testing.assert_frame_equal(
            objects[position],
            existing[position].loc[objects[position].index],
            check_exact=True,
        )
    return (*objects, summary)


def initialize():
    """Freeze provenance and analysis choices before any new test predictions."""
    manifest = {}
    for base, directories, filenames in os.walk(ROOT):
        directories[:] = [
            d
            for d in directories
            if d
            not in {
                ".git",
                ".venv",
                "__pycache__",
                ".ruff_cache",
                ".pytest_cache",
                ".ipynb_checkpoints",
            }
            and Path(base, d) != OUTPUT
        ]
        for filename in filenames:
            path = Path(base, filename)
            manifest[str(path.relative_to(ROOT))] = sha(path)
    write_json(OUTPUT / "original_files_sha256.json", manifest)
    old = json.loads((OLD / "model_metadata.json").read_text())
    write_json(
        OUTPUT / "design.json",
        {
            "horizon_hours": 12,
            "forecast_steps": 4,
            "feature_names": old["feature_names"],
            "model_source": str(OLD.relative_to(ROOT)),
            "hyperparameter_search": False,
            "binary_threshold_m": 2.5,
            "raw_boundaries": [20456, 23378],
            "bootstrap_block_lengths_observations": BLOCK_LENGTHS,
            "bootstrap_replicates": REPLICATES,
            "bootstrap_seed": 42,
            "representative_interval": "July 1-14 of first test forecast-origin year, chosen before prediction.",
            "source_sha256": sha(DATA_PATH),
            "original_models": old["models"],
            "versions": {
                key: version(key)
                for key in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "xgboost",
                    "joblib",
                    "matplotlib",
                    "threadpoolctl",
                ]
            },
        },
    )
    source = load_source(DATA_PATH)
    changed = source.copy()
    changed.iloc[15001:] *= 1.1
    pd.testing.assert_frame_equal(
        make_features(source).iloc[:15001], make_features(changed).iloc[:15001]
    )


def train():
    """Fixed original configurations: training validation, then train+val refits."""
    if (OUTPUT / "model_metadata.json").exists():
        raise FileExistsError("Refusing to overwrite completed 12-hour models.")
    initialize()
    X, y, Xv, yv, Xt, _, partitions = prepare_12h()
    source_metadata = json.loads((OLD / "model_metadata.json").read_text())
    predictions = pd.DataFrame({"actual_VHM0": yv}, index=Xv.index)
    validation, metadata = (
        [],
        {
            "horizon_hours": 12,
            "feature_names": X.columns.tolist(),
            "partitions": partitions,
            "source_sha256": sha(DATA_PATH),
            "refit_rows": len(X) + len(Xv),
            "selection": "No new tuning, model selection or early stopping; clone original six-hour compact estimators.",
            "models": {},
        },
    )
    (OUTPUT / "models").mkdir(exist_ok=True)
    combined_X, combined_y = pd.concat([X, Xv]), pd.concat([y, yv])
    assert combined_X.index[-1] + pd.Timedelta(hours=12) < Xt.index[0]
    for name in NAMES:
        info = source_metadata["models"][name]
        path = OLD / info["file"]
        assert sha(path) == info["sha256"]
        original = joblib.load(path)
        assert (
            is_regressor(original)
            and list(original.feature_names_in_) == X.columns.tolist()
        )
        model = clone(original)
        model.fit(X, y)
        prediction = model.predict(Xv).astype("float64")
        predictions[name] = prediction
        validation.append({"model": name, **regression_metrics(yv, prediction)})
        final = clone(original)
        final.fit(combined_X, combined_y)
        destination = OUTPUT / info["file"]
        joblib.dump(final, destination)
        loaded = joblib.load(destination)
        np.testing.assert_allclose(
            final.predict(X.head(32)), loaded.predict(X.head(32)), rtol=0, atol=1e-12
        )
        metadata["models"][name] = {
            "file": info["file"],
            "sha256": sha(destination),
            "selected_parameters": info["selected_parameters"],
            "estimator_parameters": info["estimator_parameters"],
            "original_model_sha256": sha(path),
        }
        print(
            "FROZEN",
            name,
            "training rows",
            len(X),
            "refit rows",
            len(combined_X),
            flush=True,
        )
    predictions.to_csv(OUTPUT / "validation_predictions.csv", index_label="timestamp")
    pd.DataFrame(validation).to_csv(OUTPUT / "validation_metrics.csv", index=False)
    write_json(OUTPUT / "model_metadata.json", metadata)


def evaluate():
    """Evaluate four frozen models; independently recalculate the original baseline."""
    _, _, _, _, X, y, _ = prepare_12h()
    metadata = json.loads((OUTPUT / "model_metadata.json").read_text())
    design = json.loads((OUTPUT / "design.json").read_text())
    assert sha(DATA_PATH) == metadata["source_sha256"]
    for package, expected in design["versions"].items():
        assert version(package) == expected
    predictions = pd.DataFrame(
        {"actual_VHM0": y, "target_timestamp": X.index + pd.Timedelta(hours=12)},
        index=X.index,
    )
    for name, info in metadata["models"].items():
        path = OUTPUT / info["file"]
        assert sha(path) == info["sha256"]
        model = joblib.load(path)
        assert list(model.feature_names_in_) == X.columns.tolist()
        predictions[name] = model.predict(X).astype("float64")
    predictions.to_csv(OUTPUT / "test_predictions.csv", index_label="timestamp")
    previous = pd.read_csv(
        OLD / "test_predictions.csv",
        index_col="timestamp",
        float_precision="round_trip",
    )
    previous.index = pd.DatetimeIndex(pd.to_datetime(previous.index, utc=True)).as_unit(
        "ns"
    )
    _, _, _, _, old_X, old_y, _ = prepare_controlled(6, "compact")
    pd.testing.assert_index_equal(previous.index, old_X.index)
    np.testing.assert_array_equal(previous.actual_VHM0, old_y)
    frames = {6: previous, 12: predictions}
    regression, classification, confusion = [], [], []
    saved_old = pd.read_csv(OLD / "test_results.csv").set_index("family")
    for horizon, frame in frames.items():
        for name in NAMES:
            numeric = regression_metrics(frame.actual_VHM0, frame[name])
            if horizon == 6:
                for metric, value in numeric.items():
                    np.testing.assert_allclose(
                        saved_old.loc[name, metric], value, rtol=1e-12, atol=1e-12
                    )
            regression.append({"horizon_hours": horizon, "model": name, **numeric})
            classification.append(
                {
                    "horizon_hours": horizon,
                    "model": name,
                    **binary_metrics(frame.actual_VHM0, frame[name]),
                }
            )
            if horizon == 12:
                matrix = confusion_matrix(
                    frame.actual_VHM0 >= 2.5, frame[name] >= 2.5, labels=[False, True]
                )
                confusion.extend(
                    {
                        "model": name,
                        "actual": i,
                        "predicted": j,
                        "count": int(matrix[i, j]),
                    }
                    for i in [0, 1]
                    for j in [0, 1]
                )
    regression, classification = pd.DataFrame(regression), pd.DataFrame(classification)
    regression.to_csv(OUTPUT / "regression_comparison.csv", index=False)
    classification.to_csv(OUTPUT / "classification_comparison.csv", index=False)
    regression.loc[regression.horizon_hours.eq(12)].to_csv(
        OUTPUT / "regression_metrics.csv", index=False
    )
    classification.loc[classification.horizon_hours.eq(12)].to_csv(
        OUTPUT / "classification_metrics.csv", index=False
    )
    pd.DataFrame(confusion).to_csv(OUTPUT / "confusion_matrices.csv", index=False)
    differences = []
    for name in NAMES:
        row = {"model": name}
        for frame, metrics in [
            (regression, ["mae", "rmse", "r2"]),
            (classification, ["elevated_precision", "elevated_recall", "elevated_f1"]),
        ]:
            values = frame.loc[frame.model.eq(name)].set_index("horizon_hours")
            for metric in metrics:
                row.update(
                    {
                        "six_hour_" + metric: values.loc[6, metric],
                        "twelve_hour_" + metric: values.loc[12, metric],
                        "change_12h_minus_6h_" + metric: values.loc[12, metric]
                        - values.loc[6, metric],
                    }
                )
        differences.append(row)
    pd.DataFrame(differences).to_csv(OUTPUT / "horizon_differences.csv", index=False)
    separation, ranks = [], []
    for horizon in [6, 12]:
        numeric = regression.loc[regression.horizon_hours.eq(horizon)].set_index(
            "model"
        )
        categorical = classification.loc[
            classification.horizon_hours.eq(horizon)
        ].set_index("model")
        row = {
            "horizon_hours": horizon,
            "best_rmse": numeric.rmse.idxmin(),
            "best_mae": numeric.mae.idxmin(),
            "best_elevated_f1": categorical.elevated_f1.idxmax(),
            "best_elevated_precision": categorical.elevated_precision.idxmax(),
            "best_elevated_recall": ", ".join(
                categorical.index[
                    categorical.elevated_recall.eq(categorical.elevated_recall.max())
                ]
            ),
            "best_worst_rmse_gap": numeric.rmse.max() - numeric.rmse.min(),
        }
        for name in ["Linear Regression", "Random Forest"]:
            gap = numeric.loc[name, "rmse"] - numeric.loc["XGBoost", "rmse"]
            row[name + "_minus_xgboost_rmse"] = gap
            row["xgboost_advantage_percent_over_" + name] = (
                100 * gap / numeric.loc[name, "rmse"]
            )
        separation.append(row)
        for metric, ascending in [("rmse", True), ("mae", True), ("r2", False)]:
            for name, rank in (
                numeric[metric].rank(ascending=ascending, method="min").items()
            ):
                ranks.append(
                    {
                        "horizon_hours": horizon,
                        "metric": metric,
                        "model": name,
                        "rank": int(rank),
                        "score": numeric.loc[name, metric],
                    }
                )
    pd.DataFrame(separation).to_csv(OUTPUT / "model_separation.csv", index=False)
    pd.DataFrame(ranks).to_csv(OUTPUT / "rankings.csv", index=False)
    common = previous.index.intersection(predictions.index)
    common_rows = []
    for horizon, frame in frames.items():
        selected = frame.loc[common]
        for name in NAMES:
            common_rows.append(
                {
                    "horizon_hours": horizon,
                    "model": name,
                    "start": str(common[0]),
                    "end": str(common[-1]),
                    **regression_metrics(selected.actual_VHM0, selected[name]),
                    **binary_metrics(selected.actual_VHM0, selected[name]),
                }
            )
    pd.DataFrame(common_rows).to_csv(
        OUTPUT / "common_origin_comparison.csv", index=False
    )
    bootstrap(frames, common)
    plots(regression, classification, predictions)
    print(
        "12H REGRESSION\n",
        regression.loc[regression.horizon_hours.eq(12)].to_string(index=False),
        flush=True,
    )
    print(
        "12H CLASSIFICATION\n",
        classification.loc[classification.horizon_hours.eq(12)].to_string(index=False),
        flush=True,
    )


def bootstrap(frames, common):
    """Ordinary noncircular moving blocks, paired across models and horizons."""
    n = len(common)
    errors = {
        h: frame.loc[common, NAMES]
        .sub(frame.loc[common, "actual_VHM0"], axis=0)
        .to_numpy()
        ** 2
        for h, frame in frames.items()
    }
    rng = np.random.default_rng(42)
    rows = []
    for length in BLOCK_LENGTHS:
        samples = {h: [] for h in frames}
        blocks = int(np.ceil(n / length))
        for _ in range(REPLICATES):
            starts = rng.integers(0, n - length + 1, size=blocks)
            indexes = (starts[:, None] + np.arange(length)).ravel()[:n]
            for h, squared in errors.items():
                rmses = np.sqrt(squared[indexes].mean(axis=0))
                samples[h].append([rmses[3] - rmses[0], rmses[3] - rmses[2]])
        for h in frames:
            for column, reference in enumerate(["Linear Regression", "Random Forest"]):
                deltas = np.asarray(samples[h])[:, column]
                point = np.sqrt(errors[h].mean(axis=0))
                lower, upper = np.quantile(deltas, [0.025, 0.975])
                rows.append(
                    {
                        "horizon_hours": h,
                        "reference_model": reference,
                        "common_origins": n,
                        "block_observations": length,
                        "block_days": length / 8,
                        "replicates": REPLICATES,
                        "xgboost_minus_reference_rmse": point[3]
                        - point[0 if column == 0 else 2],
                        "percentile_95_lower": lower,
                        "percentile_95_upper": upper,
                        "interval_includes_zero": bool(lower <= 0 <= upper),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT / "moving_block_bootstrap.csv", index=False)


def save_figure(fig, name):
    (OUTPUT / "figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figures" / name, dpi=220, facecolor="white")
    plt.close(fig)


def plots(regression, classification, predictions):
    """Seven readable figures, with zero-based metric axes and fixed calendar panel."""
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    x = np.arange(4)
    for frame, metric, ylabel, filename in [
        (regression, "rmse", "Test RMSE (m)", "rmse_6h_vs_12h.png"),
        (regression, "mae", "Test MAE (m)", "mae_6h_vs_12h.png"),
        (regression, "r2", "Test R-squared", "r2_6h_vs_12h.png"),
        (classification, "elevated_f1", "Elevated F1", "elevated_f1_6h_vs_12h.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for h, offset, alpha, hatch in [(6, -0.2, 0.55, "//"), (12, 0.2, 1, "")]:
            values = (
                frame.loc[frame.horizon_hours.eq(h)]
                .set_index("model")
                .loc[NAMES, metric]
            )
            bars = ax.bar(
                x + offset,
                values,
                width=0.36,
                color=[COLORS[name] for name in NAMES],
                alpha=alpha,
                hatch=hatch,
            )
            ax.bar_label(
                bars, labels=[f"{value:.6f}" for value in values], fontsize=8, padding=3
            )
        upper = 1.08 if metric in ["r2", "elevated_f1"] else frame[metric].max() * 1.18
        ax.set(
            xticks=x,
            xticklabels=NAMES,
            ylabel=ylabel,
            ylim=(0, upper),
            title="Native eligible test origins: original fixed configurations",
        )
        ax.legend(
            handles=[
                Patch(facecolor="gray", alpha=0.55, hatch="//", label="6 hours"),
                Patch(facecolor="gray", label="12 hours"),
            ]
        )
        save_figure(fig, filename)
    twelve = classification.loc[classification.horizon_hours.eq(12)].set_index("model")
    fig, ax = plt.subplots(figsize=(11, 5))
    for metric, offset, alpha, hatch in [
        ("elevated_precision", -0.2, 0.55, "//"),
        ("elevated_recall", 0.2, 1, ""),
    ]:
        bars = ax.bar(
            x + offset,
            twelve.loc[NAMES, metric],
            width=0.36,
            color=[COLORS[name] for name in NAMES],
            alpha=alpha,
            hatch=hatch,
        )
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.set(
        xticks=x,
        xticklabels=NAMES,
        ylabel="Score",
        ylim=(0, 1.08),
        title="12h Elevated detection at fixed 2.5 m threshold",
    )
    ax.legend(
        handles=[
            Patch(facecolor="gray", alpha=0.55, hatch="//", label="Precision"),
            Patch(facecolor="gray", label="Recall"),
        ]
    )
    save_figure(fig, "precision_recall_12h.png")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    maximum = int(twelve.tn.max())
    for ax, name in zip(axes.flat, NAMES):
        row = twelve.loc[name]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                color="white" if count > maximum / 2 else "black",
            )
        ax.set(
            title=name,
            xlabel="Predicted",
            ylabel="Actual",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Elevated"],
            yticklabels=["Normal", "Elevated"],
        )
    fig.suptitle("12h derived binary classification: actual rows, predicted columns")
    save_figure(fig, "confusion_matrices_12h.png")
    year = predictions.index[0].year
    interval = predictions.loc[f"{year}-07-01" : f"{year}-07-14"]
    assert len(interval) > 0
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, NAMES):
        ax.plot(
            interval.index,
            interval.actual_VHM0,
            color="black",
            label="Actual VHM0(t+12h)",
            linewidth=1.4,
        )
        ax.plot(
            interval.index,
            interval[name],
            color=COLORS[name],
            label="Predicted",
            linewidth=1.1,
        )
        ax.set(
            title=name,
            ylabel="Significant wave height (m)",
            ylim=(0, interval[NAMES + ["actual_VHM0"]].max().max() * 1.1),
        )
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"12h forecasts: July 1-14, {year}; x-axis is forecast-origin UTC")
    save_figure(fig, "actual_predicted_12h.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run", "train", "evaluate"], default="run")
    stage = parser.parse_args().stage
    with threadpool_limits(limits=4):
        if stage in ["run", "train"]:
            train()
        if stage in ["run", "evaluate"]:
            evaluate()


if __name__ == "__main__":
    main()
