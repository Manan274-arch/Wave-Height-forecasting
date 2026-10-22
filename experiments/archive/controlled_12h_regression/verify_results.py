"""Independent frozen-model, numeric, temporal and paired-bootstrap checks."""

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import run_experiment as experiment

OUTPUT = Path(__file__).resolve().parent


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12, equal_nan=True)


def regression(truth, prediction):
    residual = np.asarray(prediction) - np.asarray(truth)
    high = np.asarray(truth) > 3
    return {
        "sample_count": len(truth),
        "mae": np.abs(residual).mean(),
        "rmse": np.sqrt(np.mean(residual**2)),
        "r2": 1 - np.sum(residual**2) / np.sum((truth - np.mean(truth)) ** 2),
        "mean_signed_error": residual.mean(),
        "median_absolute_error": np.median(np.abs(residual)),
        "maximum_absolute_error": np.abs(residual).max(),
        "high_count": high.sum(),
        "high_mae": np.abs(residual[high]).mean(),
        "high_rmse": np.sqrt(np.mean(residual[high] ** 2)),
        "negative_prediction_count": (np.asarray(prediction) < 0).sum(),
    }


def classification(truth, prediction):
    actual, predicted = np.asarray(truth) >= 2.5, np.asarray(prediction) >= 2.5
    tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
    tn, fn = int((~actual & ~predicted).sum()), int((actual & ~predicted).sum())

    def divide(a, b):
        return a / b if b else 0.0

    p, r = divide(tp, tp + fp), divide(tp, tp + fn)
    normal_p, normal_r = divide(tn, tn + fn), divide(tn, tn + fp)
    f1, normal_f1 = divide(2 * tp, 2 * tp + fp + fn), divide(2 * tn, 2 * tn + fp + fn)
    return {
        "sample_count": len(actual),
        "normal_count": tn + fp,
        "elevated_count": tp + fn,
        "elevated_percent": 100 * (tp + fn) / len(actual),
        "accuracy": (tp + tn) / len(actual),
        "balanced_accuracy": (r + normal_r) / 2,
        "normal_precision": normal_p,
        "normal_recall": normal_r,
        "normal_f1": normal_f1,
        "elevated_precision": p,
        "elevated_recall": r,
        "elevated_f1": f1,
        "macro_f1": (f1 + normal_f1) / 2,
        "weighted_f1": ((tp + fn) * f1 + (tn + fp) * normal_f1) / len(actual),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def read_predictions(path):
    frame = pd.read_csv(path, index_col="timestamp", float_precision="round_trip")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).as_unit("ns")
    return frame


def main():
    manifest = json.loads((OUTPUT / "original_files_sha256.json").read_text())
    for relative, expected in manifest.items():
        assert experiment.sha(experiment.ROOT / relative) == expected, relative
    X, y, Xv, yv, Xt, yt, partitions = experiment.prepare_12h()
    metadata = json.loads((OUTPUT / "model_metadata.json").read_text())
    original = json.loads((experiment.OLD / "model_metadata.json").read_text())
    assert partitions == metadata["partitions"] and metadata["refit_rows"] == 23362
    source = experiment.load_source(experiment.DATA_PATH)
    assert X.index[-1] + pd.Timedelta(hours=12) < Xv.index[0]
    assert Xv.index[-1] + pd.Timedelta(hours=12) < Xt.index[0]
    for xx, yy in [(X, y), (Xv, yv), (Xt, yt)]:
        np.testing.assert_array_equal(
            yy, source.loc[xx.index + pd.Timedelta(hours=12), "VHM0"]
        )
    frames = {
        6: read_predictions(experiment.OLD / "test_predictions.csv"),
        12: read_predictions(OUTPUT / "test_predictions.csv"),
    }
    numeric = pd.read_csv(OUTPUT / "regression_comparison.csv").set_index(
        ["horizon_hours", "model"]
    )
    categorical = pd.read_csv(OUTPUT / "classification_comparison.csv").set_index(
        ["horizon_hours", "model"]
    )
    matrices = pd.read_csv(OUTPUT / "confusion_matrices.csv")
    validation_predictions = read_predictions(OUTPUT / "validation_predictions.csv")
    pd.testing.assert_index_equal(validation_predictions.index, Xv.index)
    np.testing.assert_array_equal(validation_predictions.actual_VHM0, yv)
    validation_metrics = pd.read_csv(OUTPUT / "validation_metrics.csv").set_index(
        "model"
    )
    for horizon, frame in frames.items():
        np.testing.assert_array_equal(
            frame.actual_VHM0,
            source.loc[frame.index + pd.Timedelta(hours=horizon), "VHM0"],
        )
        assert (
            pd.to_datetime(frame.target_timestamp, utc=True).array
            == frame.index + pd.Timedelta(hours=horizon)
        ).all()
        for name in experiment.NAMES:
            for metric, value in regression(
                frame.actual_VHM0.to_numpy(), frame[name].to_numpy()
            ).items():
                close(numeric.loc[(horizon, name), metric], value)
            for metric, value in classification(frame.actual_VHM0, frame[name]).items():
                close(categorical.loc[(horizon, name), metric], value)
    for name, info in metadata["models"].items():
        assert (
            info["selected_parameters"]
            == original["models"][name]["selected_parameters"]
        )
        path = OUTPUT / info["file"]
        assert experiment.sha(path) == info["sha256"]
        new, old = joblib.load(path), joblib.load(experiment.OLD / info["file"])
        assert (
            list(new.feature_names_in_)
            == list(old.feature_names_in_)
            == X.columns.tolist()
        )
        if isinstance(new, Pipeline):
            for step in old.named_steps:
                assert (
                    old.named_steps[step].get_params()
                    == new.named_steps[step].get_params()
                )
            assert new.named_steps["scale"].n_samples_seen_ == 23362
        else:
            for key, value in old.get_params().items():
                other = new.get_params()[key]
                if isinstance(value, float) and np.isnan(value):
                    assert isinstance(other, float) and np.isnan(other), key
                else:
                    assert other == value, key
        close(new.predict(Xt), frames[12][name])
        for metric, value in regression(
            yv.to_numpy(), validation_predictions[name].to_numpy()
        ).items():
            close(validation_metrics.loc[name, metric], value)
        row = categorical.loc[(12, name)]
        saved = matrices.loc[matrices.model.eq(name)].pivot(
            index="actual", columns="predicted", values="count"
        )
        np.testing.assert_array_equal(saved, [[row.tn, row.fp], [row.fn, row.tp]])
        assert saved.to_numpy().sum() == 5842
    common = frames[6].index.intersection(frames[12].index)
    assert len(common) == 5842
    common_metrics = pd.read_csv(OUTPUT / "common_origin_comparison.csv").set_index(
        ["horizon_hours", "model"]
    )
    for horizon, frame in frames.items():
        for name in experiment.NAMES:
            selected = frame.loc[common]
            for metric, value in {
                **regression(
                    selected.actual_VHM0.to_numpy(), selected[name].to_numpy()
                ),
                **classification(selected.actual_VHM0, selected[name]),
            }.items():
                close(common_metrics.loc[(horizon, name), metric], value)
    # Independent paired MBB implementation uses sampled-origin multiplicities.
    bootstrap = pd.read_csv(OUTPUT / "moving_block_bootstrap.csv")
    squared = {
        h: frame.loc[common, experiment.NAMES]
        .sub(frame.loc[common, "actual_VHM0"], axis=0)
        .to_numpy()
        ** 2
        for h, frame in frames.items()
    }
    rng = np.random.default_rng(42)
    for length in experiment.BLOCK_LENGTHS:
        deltas = {h: [] for h in frames}
        for _ in range(experiment.REPLICATES):
            starts = rng.integers(
                0, len(common) - length + 1, size=int(np.ceil(len(common) / length))
            )
            sampled = np.concatenate(
                [np.arange(start, start + length) for start in starts]
            )[: len(common)]
            weights = np.bincount(sampled, minlength=len(common)) / len(common)
            for h in frames:
                rmse = np.sqrt(weights @ squared[h])
                deltas[h].append([rmse[3] - rmse[0], rmse[3] - rmse[2]])
        for h in frames:
            for index, name in enumerate(["Linear Regression", "Random Forest"]):
                row = bootstrap.loc[
                    bootstrap.horizon_hours.eq(h)
                    & bootstrap.block_observations.eq(length)
                    & bootstrap.reference_model.eq(name)
                ].iloc[0]
                close(
                    [row.percentile_95_lower, row.percentile_95_upper],
                    np.quantile(np.asarray(deltas[h])[:, index], [0.025, 0.975]),
                )
    differences = pd.read_csv(OUTPUT / "horizon_differences.csv").set_index("model")
    for name in experiment.NAMES:
        for frame, metrics in [
            (numeric, ["mae", "rmse", "r2"]),
            (categorical, ["elevated_precision", "elevated_recall", "elevated_f1"]),
        ]:
            for metric in metrics:
                close(
                    differences.loc[name, "change_12h_minus_6h_" + metric],
                    frame.loc[(12, name), metric] - frame.loc[(6, name), metric],
                )
    tree = ast.parse((OUTPUT / "run_experiment.py").read_text())
    evaluator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("fit")
        for node in ast.walk(evaluator)
    )
    assert len(list((OUTPUT / "models").glob("*.joblib"))) == 4
    assert len(list((OUTPUT / "figures").glob("*.png"))) == 7
    report = {
        "status": "passed",
        "existing_files_byte_identical": len(manifest),
        "original_csv_sha256": experiment.sha(experiment.DATA_PATH),
        "fixed_original_six_hour_configurations_verified": True,
        "new_hyperparameter_search": False,
        "horizon_hours": 12,
        "features": 18,
        "training_rows": len(X),
        "validation_rows": len(Xv),
        "test_rows": len(Xt),
        "refit_rows": 23362,
        "common_test_origins": len(common),
        "models_reloaded": 4,
        "figures": 7,
        "independent_validation_test_classification_comparison_metrics": True,
        "paired_bootstrap_independently_reproduced": True,
        "evaluation_contains_no_fit_calls": True,
    }
    experiment.write_json(OUTPUT / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
