"""Exact-future regression data and metrics; classification code stays unchanged."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from feature_pipeline import DATA_PATH, load_source, make_features

REGRESSION_DIR = Path(__file__).resolve().parent.parent / "experiments/regression"
METHODS = [
    "Persistence",
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]


def parse_horizon():
    """Use one explicit horizon per training/evaluation invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=[6, 24], required=True)
    return parser.parse_args().horizon


def prepare_regression(horizon_hours):
    """Return six aligned X/y objects and a split summary on common raw boundaries."""
    if horizon_hours not in (6, 24, 48, 72):
        raise ValueError("Supported regression horizons are 6, 24, 48 and 72 hours.")
    steps = horizon_hours // 3
    source = load_source(DATA_PATH)
    features = make_features(source)
    if len(features.columns) != 46 or any(
        name.startswith(("target", "future")) for name in features
    ):
        raise ValueError("Regression must use the original 46 causal predictors.")
    times = source.index.to_series()
    target_time = times.shift(-steps)
    aligned = target_time.eq(times + pd.Timedelta(hours=horizon_hours))
    invalid_alignment = target_time.notna() & ~aligned
    if invalid_alignment.any():
        raise ValueError(f"Invalid target timestamps: {invalid_alignment.sum()}")
    target = source["VHM0"].shift(-steps).rename("future_VHM0")
    valid_target = aligned & target.notna()
    missing_history = features.isna().any(axis=1)
    if not features.index[missing_history].equals(source.index[:8]):
        raise ValueError("Historical feature requirements changed.")

    # Fix boundaries before target exclusions so both horizons share the same periods.
    validation_position = int(np.floor(0.70 * len(source)))
    test_position = validation_position + int(np.floor(0.10 * len(source)))
    validation_start = source.index[validation_position]
    test_start = source.index[test_position]
    nominal = {
        "Training": source.index[:validation_position],
        "Validation": source.index[validation_position:test_position],
        "Testing": source.index[test_position:],
    }
    next_start = {"Training": validation_start, "Validation": test_start}
    objects = []
    partitions = {}
    horizon = pd.Timedelta(hours=horizon_hours)
    for name, origins in nominal.items():
        valid = valid_target.loc[origins]
        complete = origins[valid.to_numpy()]
        if name in next_start:
            keep = complete + horizon < next_start[name]
            eligible = complete[keep]
            purged = int((~keep).sum())
            if purged != steps or not (eligible + horizon < next_start[name]).all():
                raise ValueError("Incorrect horizon-aware boundary purge.")
        else:
            eligible = complete
            purged = 0
        history_excluded = int(missing_history.loc[eligible].sum())
        eligible = eligible[~missing_history.loc[eligible].to_numpy()]
        X = features.loc[eligible].copy()
        y = target.loc[eligible].copy()
        if not X.index.equals(y.index) or not np.isfinite(X.to_numpy()).all():
            raise ValueError("Invalid regression inputs or label alignment.")
        if (
            not np.isfinite(y).all()
            or not target_time.loc[eligible].eq(eligible + horizon).all()
        ):
            raise ValueError(
                "Regression targets are not at the exact future timestamp."
            )
        objects.extend([X, y])
        partitions[name] = {
            "nominal_rows": len(origins),
            "rows": len(eligible),
            "start": str(eligible[0]),
            "end": str(eligible[-1]),
            "incomplete_targets_excluded": int((~valid).sum()),
            "boundary_purged": purged,
            "history_excluded": history_excluded,
        }
    summary = {
        "horizon_hours": horizon_hours,
        "forecast_steps": steps,
        "target": f"VHM0(t+{horizon_hours}h), exactly; not a future maximum",
        "raw_rows": len(source),
        "feature_count": len(features.columns),
        "invalid_target_alignment_count": int(invalid_alignment.sum()),
        "incomplete_future_targets": int((~valid_target).sum()),
        "boundary_rule": "70/10/20 of raw timestamps before horizon exclusions",
        "nominal_validation_start": str(validation_start),
        "nominal_test_start": str(test_start),
        "partitions": partitions,
    }
    return (*objects, summary)


def regression_metrics(truth, prediction):
    """Use unmodified continuous predictions; errors are in metres."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape or finiteness is invalid.")
    residual = prediction - truth
    high = truth > 3.0
    return {
        "sample_count": len(truth),
        "mae": mean_absolute_error(truth, prediction),
        "rmse": root_mean_squared_error(truth, prediction),
        "r2": r2_score(truth, prediction),
        "mean_signed_error": float(residual.mean()),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "maximum_absolute_error": float(np.abs(residual).max()),
        "high_count": int(high.sum()),
        "high_mae": mean_absolute_error(truth[high], prediction[high])
        if high.any()
        else np.nan,
        "high_rmse": root_mean_squared_error(truth[high], prediction[high])
        if high.any()
        else np.nan,
        "negative_prediction_count": int((prediction < 0).sum()),
    }


def persistence_improvements(results):
    """Positive improvement means lower error than current-height persistence."""
    results = results.copy()
    baseline = results.loc[results.family.eq("Persistence")].iloc[0]
    for error in ["mae", "rmse"]:
        improvement = baseline[error] - results[error]
        results[error + "_absolute_improvement"] = improvement
        results[error + "_improvement_percent"] = (
            improvement / baseline[error] * 100 if baseline[error] != 0 else np.nan
        )
    return results


def category_errors(truth, prediction):
    """Condition diagnostic errors on actual future height, never training labels."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    rows = []
    for category, mask in [
        ("Normal", truth < 2.5),
        ("Elevated", (truth >= 2.5) & (truth <= 3.0)),
        ("High", truth > 3.0),
    ]:
        residual = prediction[mask] - truth[mask]
        under = residual < 0
        rows.append(
            {
                "category": category,
                "sample_count": int(mask.sum()),
                "mae": float(np.abs(residual).mean()) if mask.any() else np.nan,
                "rmse": float(np.sqrt(np.mean(residual**2))) if mask.any() else np.nan,
                "mean_signed_error": float(residual.mean()) if mask.any() else np.nan,
                "underprediction_frequency": float(under.mean())
                if mask.any()
                else np.nan,
                "mean_underprediction_magnitude": float(-residual[under].mean())
                if under.any()
                else np.nan,
                "underprediction_count": int(under.sum()),
            }
        )
    return rows


def main():
    """Inspect one regression dataset without training or writing data files."""
    *_, summary = prepare_regression(parse_horizon())
    print(pd.DataFrame(summary["partitions"]).T.to_string())
    print({key: value for key, value in summary.items() if key != "partitions"})


if __name__ == "__main__":
    main()
