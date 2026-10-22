"""Independently audit controlled artifacts, numeric errors and preserved files."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from controlled_pipeline import OUTPUT_DIR, prepare_controlled
from controlled_training import candidate_grids
from feature_pipeline import DATA_PATH, load_source, make_features
from regression_pipeline import METHODS, prepare_regression

ROOT = Path(__file__).resolve().parent.parent


def close(actual, expected):
    """Allow only roundoff from decimal serialization and summation."""
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12, equal_nan=True)


def numeric_errors(truth, prediction):
    """Calculate errors directly in NumPy, independently of scoring helpers."""
    residual = np.asarray(prediction) - np.asarray(truth)
    high = np.asarray(truth) > 3
    return {
        "sample_count": len(residual),
        "mae": np.abs(residual).mean(),
        "rmse": np.sqrt(np.mean(residual**2)),
        "r2": 1 - np.sum(residual**2) / np.sum((truth - np.mean(truth)) ** 2),
        "mean_signed_error": residual.mean(),
        "median_absolute_error": np.median(np.abs(residual)),
        "maximum_absolute_error": np.abs(residual).max(),
        "high_count": high.sum(),
        "high_mae": np.abs(residual[high]).mean() if high.any() else np.nan,
        "high_rmse": np.sqrt(np.mean(residual[high] ** 2)) if high.any() else np.nan,
        "negative_prediction_count": (np.asarray(prediction) < 0).sum(),
    }


def verify_preservation():
    """Check every protected artifact and the sole permitted two-line source edit."""
    manifest = json.loads(
        (OUTPUT_DIR / "provenance/previous_files_sha256.json").read_text()
    )
    for relative, expected in manifest.items():
        path = ROOT / relative
        assert path.is_file(), relative
        if path == ROOT / "src/regression_pipeline.py":
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    original = (OUTPUT_DIR / "provenance/regression_pipeline_before.py").read_text()
    current = (ROOT / "src/regression_pipeline.py").read_text()
    expected = original.replace(
        "if horizon_hours not in (6, 24):",
        "if horizon_hours not in (6, 24, 48, 72):",
    ).replace(
        "Only six-hour and 24-hour regression are implemented.",
        "Supported regression horizons are 6, 24, 48 and 72 hours.",
    )
    assert current == expected, list(
        difflib.unified_diff(expected.splitlines(), current.splitlines())
    )
    fingerprints = json.loads(
        (OUTPUT_DIR / "provenance/previous_dataset_fingerprints.json").read_text()
    )
    for horizon in [6, 24]:
        for obj, saved in zip(
            prepare_regression(horizon)[:6], fingerprints[str(horizon)]
        ):
            assert list(obj.shape) == saved["shape"]
            assert (
                hashlib.sha256(
                    pd.util.hash_pandas_object(obj, index=True).values.tobytes()
                ).hexdigest()
                == saved["hash"]
            )
            assert str(obj.dtypes) == saved["dtypes"]
            assert (
                obj.columns.tolist() if isinstance(obj, pd.DataFrame) else obj.name
            ) == saved["columns"]
    return len(manifest) - 1


def verify_configuration(horizon, feature_set, source):
    """Audit alignment, CV, model reloads, tuning and every stored test metric."""
    directory = OUTPUT_DIR / f"{horizon}_hour" / feature_set
    metadata = json.loads((directory / "model_metadata.json").read_text())
    data = prepare_controlled(horizon, feature_set)
    X_train, _, _, _, X_test, y_test, summary = data
    assert metadata["feature_names"] == X_train.columns.tolist()
    assert metadata["cv"]["gap"] == horizon // 3
    for position in [0, 2, 4]:
        X, y = data[position : position + 2]
        expected = source.loc[X.index + pd.Timedelta(hours=horizon), "VHM0"].to_numpy()
        np.testing.assert_array_equal(y.to_numpy(), expected)
    for train, validation in TimeSeriesSplit(n_splits=3, gap=horizon // 3).split(
        X_train
    ):
        assert (
            X_train.index[train[-1]] + pd.Timedelta(hours=horizon)
            < X_train.index[validation[0]]
        )
    assert data[0].index[-1] + pd.Timedelta(hours=horizon) < pd.Timestamp(
        summary["nominal_validation_start"]
    )
    assert data[2].index[-1] + pd.Timedelta(hours=horizon) < pd.Timestamp(
        summary["nominal_test_start"]
    )
    predictions = pd.read_csv(
        directory / "test_predictions.csv",
        index_col="timestamp",
        float_precision="round_trip",
    )
    predictions.index = pd.DatetimeIndex(
        pd.to_datetime(predictions.index, utc=True)
    ).as_unit("ns")
    pd.testing.assert_index_equal(X_test.index, predictions.index)
    close(predictions.actual_VHM0, y_test)
    close(predictions.Persistence, X_test.VHM0)
    assert (
        pd.to_datetime(predictions.target_timestamp, utc=True).array
        == predictions.index + pd.Timedelta(hours=horizon)
    ).all()
    results = pd.read_csv(
        directory / "test_results.csv", float_precision="round_trip"
    ).set_index("family")
    categories = pd.read_csv(
        directory / "category_errors.csv", float_precision="round_trip"
    )
    tuning = pd.read_csv(directory / "tuning_results.csv", float_precision="round_trip")
    baseline_rmse = numeric_errors(y_test, predictions.Persistence)["rmse"]
    baseline_mae = numeric_errors(y_test, predictions.Persistence)["mae"]
    validation = pd.read_csv(directory / "validation_results.csv").set_index("family")
    # Check pre-refit validation records without fitting again after test exposure.
    for name, info in metadata["models"].items():
        for metric, expected in info["validation_metrics_before_refit"].items():
            close(validation.loc[name, metric], expected)
    for metric, expected in numeric_errors(data[3], data[2].VHM0).items():
        close(validation.loc["Persistence", metric], expected)
    for name in METHODS:
        calculated = numeric_errors(y_test.to_numpy(), predictions[name].to_numpy())
        for metric, value in calculated.items():
            close(results.loc[name, metric], value)
        for metric, baseline in [("rmse", baseline_rmse), ("mae", baseline_mae)]:
            close(
                results.loc[name, metric + "_improvement_percent"],
                100 * (baseline - calculated[metric]) / baseline,
            )
        for category, mask in [
            ("Normal", y_test.to_numpy() < 2.5),
            ("Elevated", (y_test.to_numpy() >= 2.5) & (y_test.to_numpy() <= 3)),
            ("High", y_test.to_numpy() > 3),
        ]:
            saved = categories.loc[
                categories.family.eq(name) & categories.category.eq(category)
            ].iloc[0]
            residual = predictions[name].to_numpy()[mask] - y_test.to_numpy()[mask]
            under = residual < 0
            values = {
                "sample_count": len(residual),
                "mae": np.abs(residual).mean(),
                "rmse": np.sqrt(np.mean(residual**2)),
                "mean_signed_error": residual.mean(),
                "underprediction_frequency": under.mean(),
                "mean_underprediction_magnitude": -residual[under].mean()
                if under.any()
                else np.nan,
                "underprediction_count": under.sum(),
            }
            for metric, value in values.items():
                close(saved[metric], value)
        if name == "Persistence":
            continue
        info = metadata["models"][name]
        model_path = directory / info["file"]
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == info["sha256"]
        model = joblib.load(model_path)
        assert list(model.feature_names_in_) == X_test.columns.tolist()
        with threadpool_limits(limits=4):
            close(model.predict(X_test), predictions[name])
        candidates = tuning.loc[tuning.family.eq(name)]
        ranking = []
        for candidate, rows in candidates.groupby("candidate"):
            assert len(rows) == 3
            mean, std = rows.fold_rmse.mean(), rows.fold_rmse.std(ddof=1)
            close(rows.mean_cv_rmse, mean)
            close(rows.std_cv_rmse, std)
            ranking.append((mean, std, candidate))
        best = min(ranking)[2]
        chosen = candidates.loc[candidates.candidate.eq(best)]
        assert chosen.selected.all()
        assert not candidates.loc[candidates.candidate.ne(best), "selected"].any()
        assert json.loads(chosen.parameters.iloc[0]) == info["selected_parameters"]
        close(chosen.mean_cv_rmse.iloc[0], info["mean_cv_rmse"])
    return predictions, results


def verify_comparisons():
    """Recalculate consolidated and common-origin scores from stored predictions."""
    master = pd.read_csv(OUTPUT_DIR / "master_results.csv")
    common = pd.read_csv(OUTPUT_DIR / "common_origin_results.csv")
    all_predictions = {}
    shared = None
    for horizon in [6, 24, 48, 72]:
        for feature_set in ["full", "compact"]:
            directory = OUTPUT_DIR / f"{horizon}_hour" / feature_set
            frame = pd.read_csv(
                directory / "test_predictions.csv",
                index_col="timestamp",
                float_precision="round_trip",
            )
            all_predictions[horizon, feature_set] = frame
            shared = frame.index if shared is None else shared.intersection(frame.index)
            results = pd.read_csv(directory / "test_results.csv").set_index("family")
            validation = pd.read_csv(directory / "validation_results.csv").set_index(
                "family"
            )
            metadata = json.loads((directory / "model_metadata.json").read_text())
            selected = master.loc[
                master.horizon_hours.eq(horizon) & master.feature_set.eq(feature_set)
            ].set_index("model")
            assert len(selected) == 4
            for name in METHODS[1:]:
                for metric in ["mae", "rmse", "r2"]:
                    close(
                        selected.loc[name, "test_" + metric], results.loc[name, metric]
                    )
                    close(
                        selected.loc[name, "validation_" + metric],
                        validation.loc[name, metric],
                    )
                close(
                    selected.loc[name, "high_wave_rmse"], results.loc[name, "high_rmse"]
                )
                close(
                    selected.loc[name, "rmse_improvement_pct"],
                    results.loc[name, "rmse_improvement_percent"],
                )
                assert (
                    json.loads(selected.loc[name, "selected_parameters"])
                    == metadata["models"][name]["selected_parameters"]
                )
                close(
                    selected.loc[name, "cv_rmse"],
                    metadata["models"][name]["mean_cv_rmse"],
                )
    assert len(shared) == 5822 and len(common) == 40
    for (horizon, feature_set), frame in all_predictions.items():
        frame = frame.loc[shared]
        saved = common.loc[
            common.horizon_hours.eq(horizon) & common.feature_set.eq(feature_set)
        ].set_index("family")
        for method in METHODS:
            for metric, expected in numeric_errors(
                frame.actual_VHM0.to_numpy(), frame[method].to_numpy()
            ).items():
                close(saved.loc[method, metric], expected)
    differences = pd.read_csv(OUTPUT_DIR / "full_vs_compact.csv")
    for row in differences.itertuples():
        close(
            row.rmse_change_compact_minus_full,
            row.test_rmse_compact - row.test_rmse_full,
        )
        close(
            row.rmse_change_percent_of_full,
            100 * (row.test_rmse_compact - row.test_rmse_full) / row.test_rmse_full,
        )
    paired = pd.read_csv(OUTPUT_DIR / "xgboost_vs_linear.csv")
    for row in paired.itertuples():
        close(
            row.rmse_reduction_linear_minus_xgb,
            row.test_rmse_linear - row.test_rmse_xgb,
        )
        close(
            row.xgb_improvement_percent_over_linear,
            100 * (row.test_rmse_linear - row.test_rmse_xgb) / row.test_rmse_linear,
        )


def main():
    protected = verify_preservation()
    source = load_source(DATA_PATH)
    assert source.shape == (29224, 19)
    features = make_features(source)
    # Alter only the future: all predictors through the cutoff must stay identical.
    changed = source.copy()
    cutoff = 15000
    changed.iloc[cutoff + 1 :, :] = changed.iloc[cutoff + 1 :, :] * 1.1
    pd.testing.assert_frame_equal(
        features.iloc[: cutoff + 1], make_features(changed).iloc[: cutoff + 1]
    )
    frozen = json.loads((OUTPUT_DIR / "compact_features.json").read_text())
    grids = candidate_grids()
    assert json.loads((OUTPUT_DIR / "candidate_grids.json").read_text()) == {
        "Linear Regression": [{}],
        "Decision Tree": grids[0],
        "Random Forest": grids[1],
        "XGBoost": grids[2],
    }
    full_vs_compact = []
    previous_reproduced = []
    for horizon in [6, 24, 48, 72]:
        full_data = prepare_controlled(horizon, "full")
        compact_data = prepare_controlled(horizon, "compact")
        for position in [0, 2, 4]:
            pd.testing.assert_frame_equal(
                full_data[position][frozen["features"]], compact_data[position]
            )
            pd.testing.assert_series_equal(
                full_data[position + 1], compact_data[position + 1]
            )
        predictions, results = verify_configuration(horizon, "full", source)
        compact_predictions, compact_results = verify_configuration(
            horizon, "compact", source
        )
        pd.testing.assert_frame_equal(
            predictions[
                ["actual_VHM0", "current_VHM0", "Persistence", "target_timestamp"]
            ],
            compact_predictions[
                ["actual_VHM0", "current_VHM0", "Persistence", "target_timestamp"]
            ],
        )
        pd.testing.assert_series_equal(
            results.loc["Persistence"], compact_results.loc["Persistence"]
        )
        full_vs_compact.append(horizon)
        if horizon in [6, 24]:
            old = pd.read_csv(
                ROOT / f"experiments/regression/{horizon}_hour/test_results.csv"
            ).set_index("family")
            for method in METHODS:
                for metric in ["mae", "rmse", "r2", "high_mae", "high_rmse"]:
                    close(old.loc[method, metric], results.loc[method, metric])
            previous_reproduced.append(horizon)
    models = list(OUTPUT_DIR.glob("*_hour/*/models/*.joblib"))
    baselines = list(OUTPUT_DIR.glob("*_hour/persistence_results.csv"))
    figures = list((OUTPUT_DIR / "figures").rglob("*.png"))
    assert len(models) == 32 and len(baselines) == 4 and len(figures) == 22
    master = pd.read_csv(OUTPUT_DIR / "master_results.csv")
    assert len(master) == 32
    common = pd.read_csv(OUTPUT_DIR / "common_origin_results.csv")
    assert len(common) == 40 and common.common_test_origins.eq(5822).all()
    verify_comparisons()
    for filename in ["controlled_evaluation.py", "controlled_comparison.py"]:
        tree = ast.parse((ROOT / "src" / filename).read_text())
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ["fit", "fit_transform", "fit_predict"]
            for node in ast.walk(tree)
        )
    report = {
        "status": "passed",
        "previous_files_byte_identical": protected,
        "shared_source_change": "Only horizon guard and error message extended from 6/24 to 6/24/48/72; old CLI unchanged.",
        "previous_6h_24h_data_fingerprints_identical": True,
        "previous_full_test_metrics_reproduced": previous_reproduced,
        "full_compact_origins_targets_identical": full_vs_compact,
        "exact_targets_and_cv_gaps_verified": True,
        "future_perturbation_causality_check": "passed",
        "models_reloaded_and_predictions_verified": len(models),
        "unique_persistence_baselines": len(baselines),
        "figures": len(figures),
        "master_rows": len(master),
        "common_test_origins": 5822,
        "test_and_category_metrics_independently_recalculated": True,
        "common_origin_metrics_independently_recalculated": True,
        "master_and_comparison_tables_verified": True,
        "validation_verification_scope": "Persistence independently recalculated; ML validation metrics agree with pre-refit metadata. Training-only validation predictions were not retained, so those metrics cannot be independently reconstructed without post-test fitting, which was not performed.",
        "evaluation_and_comparison_contain_no_fit_calls": True,
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    }
    (OUTPUT_DIR / "provenance/final_verification.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
