"""Independent numeric, model, preservation and chronological stopping audit."""

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_experiment as experiment

OUTPUT = Path(__file__).resolve().parent


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12, equal_nan=True)


def independent_regression(truth, prediction):
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


def independent_binary(truth, prediction):
    actual, predicted = np.asarray(truth) >= 2.5, np.asarray(prediction) >= 2.5
    tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
    tn, fn = int((~actual & ~predicted).sum()), int((actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    normal_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    return {
        "accuracy": (tp + tn) / len(actual),
        "balanced_accuracy": (recall + tn / (tn + fp)) / 2,
        "elevated_precision": precision,
        "elevated_recall": recall,
        "elevated_f1": f1,
        "macro_f1": (f1 + normal_f1) / 2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    manifest = json.loads((OUTPUT / "original_files_sha256.json").read_text())
    for relative, expected in manifest.items():
        assert experiment.sha(experiment.ROOT / relative) == expected, relative
    design = json.loads((OUTPUT / "design.json").read_text())
    metadata = json.loads((OUTPUT / "model_metadata.json").read_text())
    X, y, Xv, yv, Xt, yt, summary = experiment.data()
    assert summary == metadata["summary"]
    assert X.columns.tolist() == metadata["feature_names"] == design["features"]
    assert len(X.columns) == 18 and metadata["refit_rows"] == 23366
    assert X.index[-1] + pd.Timedelta(hours=6) < Xv.index[0]
    assert Xv.index[-1] + pd.Timedelta(hours=6) < Xt.index[0]
    previous = pd.read_csv(
        experiment.OLD / "test_predictions.csv", float_precision="round_trip"
    )
    regression = pd.read_csv(OUTPUT / "test_regression_metrics.csv").set_index("model")
    comparison = pd.read_csv(OUTPUT / "regression_comparison.csv").set_index("model")
    classes = pd.read_csv(OUTPUT / "derived_classification_metrics.csv").set_index(
        "model"
    )
    class_comparison = pd.read_csv(OUTPUT / "classification_comparison.csv").set_index(
        "model"
    )
    validation = pd.read_csv(OUTPUT / "validation_metrics.csv").set_index("model")
    for filename, xx, truth in [
        ("test_predictions.csv", Xt, yt),
        ("validation_predictions.csv", Xv, yv),
    ]:
        frame = pd.read_csv(
            OUTPUT / filename, index_col="timestamp", float_precision="round_trip"
        )
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).as_unit("ns"),
            xx.index,
        )
        np.testing.assert_array_equal(frame.actual_VHM0, truth)
        if filename.startswith("test"):
            assert (
                pd.to_datetime(frame.target_timestamp, utc=True).array
                == xx.index + pd.Timedelta(hours=6)
            ).all()
        for name, info in metadata["models"].items():
            calculated = independent_regression(
                truth.to_numpy(), frame[name].to_numpy()
            )
            saved = regression if filename.startswith("test") else validation
            for metric, value in calculated.items():
                close(saved.loc[name, metric], value)
            if filename.startswith("test"):
                old = independent_regression(
                    truth.to_numpy(), previous[name].to_numpy()
                )
                for metric in ["mae", "rmse", "r2"]:
                    close(comparison.loc[name, "old_" + metric], old[metric])
                    close(comparison.loc[name, "tuned_" + metric], calculated[metric])
                close(
                    comparison.loc[name, "rmse_improvement_percent"],
                    100 * (old["rmse"] - calculated["rmse"]) / old["rmse"],
                )
                current_binary = independent_binary(
                    truth.to_numpy(), frame[name].to_numpy()
                )
                old_binary = independent_binary(
                    truth.to_numpy(), previous[name].to_numpy()
                )
                for metric, value in current_binary.items():
                    close(classes.loc[name, metric], value)
                    close(
                        class_comparison.loc[name, "old_" + metric], old_binary[metric]
                    )
                    close(class_comparison.loc[name, "tuned_" + metric], value)
                    close(
                        class_comparison.loc[name, "change_" + metric],
                        value - old_binary[metric],
                    )
                path = OUTPUT / info["file"]
                assert experiment.sha(path) == info["sha256"]
                model = joblib.load(path)
                assert list(model.feature_names_in_) == X.columns.tolist()
                with threadpool_limits(limits=4):
                    close(model.predict(Xt), frame[name])
                if name == "Linear Regression":
                    assert model.named_steps["regressor"].fit_intercept is True
                    assert model.named_steps["scale"].n_samples_seen_ == 23366
                if name == "XGBoost":
                    assert model.get_params()["early_stopping_rounds"] is None
                    assert (
                        model.get_booster().num_boosted_rounds()
                        == info["final_parameters"]["n_estimators"]
                    )
    candidates = json.loads((OUTPUT / "candidate_results.json").read_text())
    cv = pd.read_csv(OUTPUT / "cv_results.csv", float_precision="round_trip")
    for name, info in metadata["models"].items():
        records = [item for item in candidates if item["model"] == name]
        best = min(
            records,
            key=lambda item: (
                item["mean_cv_rmse"],
                item["std_cv_rmse"],
                item["candidate"],
            ),
        )
        assert (
            best["candidate"] == info["candidate"]
            and best["parameters"] == info["parameters"]
        )
        for candidate in records:
            rows = cv.loc[cv.model.eq(name) & cv.candidate.eq(candidate["candidate"])]
            assert len(rows) == 3
            close(rows.fold_rmse.mean(), candidate["mean_cv_rmse"])
            close(rows.fold_rmse.std(ddof=1), candidate["std_cv_rmse"])
            assert (
                pd.to_datetime(rows.training_end, utc=True) + pd.Timedelta(hours=6)
                < pd.to_datetime(rows.scoring_start, utc=True)
            ).all()
    xgb = cv.loc[cv.model.eq("XGBoost")]
    assert len(xgb) == 240
    assert (
        len(xgb.loc[xgb.stage.eq("A")]) == 180 and len(xgb.loc[xgb.stage.eq("B")]) == 60
    )
    assert (
        pd.to_datetime(xgb.core_end, utc=True) + pd.Timedelta(hours=6)
        < pd.to_datetime(xgb.early_stopping_start, utc=True)
    ).all()
    assert (
        pd.to_datetime(xgb.early_stopping_end, utc=True) + pd.Timedelta(hours=6)
        < pd.to_datetime(xgb.scoring_start, utc=True)
    ).all()
    assert xgb.best_rounds.eq(xgb.best_iteration + 1).all()
    assert xgb.fitted_rounds.ge(xgb.best_rounds).all()
    for row in xgb.itertuples():
        assert row.fitted_rounds <= json.loads(row.parameters)["n_estimators"]
    gaps = pd.read_csv(OUTPUT / "xgboost_comparison.csv")
    for row in gaps.itertuples():
        close(row.reference_minus_xgboost_rmse, row.reference_rmse - row.xgboost_rmse)
        close(
            row.xgboost_advantage_percent,
            100 * (row.reference_rmse - row.xgboost_rmse) / row.reference_rmse,
        )
    stopping = pd.read_csv(OUTPUT / "early_stopping_summary.csv")
    assert (
        pd.to_datetime(stopping.fit_end, utc=True) + pd.Timedelta(hours=6)
        < pd.to_datetime(stopping.stopping_start, utc=True)
    ).all()
    final = stopping.loc[
        stopping.purpose.eq("original_validation_final_round_selection")
    ].iloc[0]
    assert (
        int(final.best_rounds)
        == metadata["models"]["XGBoost"]["final_parameters"]["n_estimators"]
    )
    source = ast.parse((OUTPUT / "run_experiment.py").read_text())
    evaluator = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("fit")
        for node in ast.walk(evaluator)
    )
    assert len(list((OUTPUT / "models").glob("*.joblib"))) == 4
    report = {
        "status": "passed",
        "original_files_byte_identical": len(manifest),
        "source_sha256": experiment.sha(experiment.DATA_PATH),
        "models_reloaded": 4,
        "feature_count": 18,
        "horizon_hours": 6,
        "train_rows": len(X),
        "validation_rows": len(Xv),
        "test_rows": len(Xt),
        "refit_rows": 23366,
        "cv_fold_scores": len(cv),
        "xgboost_outer_scoring_folds": len(xgb),
        "nested_stopping_and_outer_scoring_separation": True,
        "test_and_validation_metrics_independently_recalculated": True,
        "classification_and_comparison_metrics_independently_recalculated": True,
        "evaluation_contains_no_fit_calls": True,
        "final_xgboost_rounds": int(final.best_rounds),
    }
    experiment.write_json(OUTPUT / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
