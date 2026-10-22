"""Freeze a training-only compact subset and prepare controlled regression data."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from feature_pipeline import DATA_PATH, load_source, make_features
from regression_pipeline import prepare_regression

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "experiments/controlled_regression"
)
HORIZONS = {"6_hour": 2, "24_hour": 8, "48_hour": 16, "72_hour": 24}
FEATURE_ROLES = {
    "VHM0": "Current total sea-state height; legitimate persistence information.",
    "VHM0_SW1": "One dominant-swell height; retained for physical partition information despite high correlation with total height.",
    "VHM0_WW": "Wind-sea height distinguishes local forcing from swell.",
    "VTM02": "Current mean wave period; representative spectral period.",
    "VTPK": "Peak period complements the mean-period measure.",
    "eastward_wind": "Signed eastward forcing and wind orientation.",
    "northward_wind": "Signed northward forcing and wind orientation.",
    "wind_speed": "Current forcing magnitude; nonlinear magnitude is useful even with signed components.",
    "VMDR_sin": "Total wave direction without a discontinuity at north.",
    "VMDR_cos": "Paired circular encoding needed to represent wave direction.",
    "VHM0_lag_6h": "Representative recent sea-state history.",
    "VHM0_lag_24h": "Longer daily sea-state history without all intermediate lags.",
    "VHM0_change_3h": "Recent trend; also encodes the omitted three-hour lag.",
    "VHM0_roll_mean_12h": "One trailing mean summarizing recent sea-state level.",
    "VHM0_roll_std_12h": "Trailing variability rather than another height-level summary.",
    "wind_speed_roll_mean_12h": "One trailing wind-forcing memory measure.",
    "month_sin": "Seasonal phase representing monsoon-related annual variation.",
    "month_cos": "Companion seasonal encoding avoids a year-end discontinuity.",
}


def selection_reason(name):
    """Explain omitted physical groups without ranking them on predictive scores."""
    if name == "VHM0_SW2":
        return "Limit swell-height representation to the dominant partition; retain total and wind-sea heights."
    if name.startswith("VMDR"):
        return "Replace multiple partition-specific directions with total direction and signed wind components."
    if name.startswith(("VTM", "VTP")):
        return "Use VTM02 and VTPK as representative mean/peak periods instead of many period variants and a period lag."
    if "lag" in name:
        return "Keep total-height 6h/24h lags; 3h height history is encoded by current height and its 3h change."
    if "change" in name:
        return "Keep one height trend; avoid multiple overlapping wave/wind changes and exact linear redundancies."
    if "roll" in name:
        return "Keep one height mean/variability pair and one wind mean; omit overlapping means, maxima and windows."
    return (
        "Use representative physical variables within the fixed compact feature budget."
    )


def freeze_features():
    """Use only the training origins eligible for every horizon; never read targets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = load_source(DATA_PATH)
    features = make_features(source)
    end = int(np.floor(0.70 * len(source))) - max(HORIZONS.values())
    training = features.iloc[8:end]
    correlations = training.corr()
    selected = [name for name in features if name in FEATURE_ROLES]
    if len(selected) != 18:
        raise ValueError("The prespecified physical subset must contain 18 features.")
    path = OUTPUT_DIR / "compact_features.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen["features"] != selected:
            raise ValueError("Refusing to replace the frozen compact feature set.")
        print("Using existing frozen compact features.")
        return
    rows = []
    for name in features:
        others = correlations.loc[name, selected].drop(name, errors="ignore").abs()
        closest = others.idxmax()
        rows.append(
            {
                "feature": name,
                "selected": name in selected,
                "physical_role_or_removal_reason": FEATURE_ROLES.get(
                    name, selection_reason(name)
                ),
                "most_correlated_other_selected": closest,
                "absolute_training_correlation": others[closest],
                "training_start": str(training.index[0]),
                "training_end": str(training.index[-1]),
                "selection_used_target": False,
            }
        )
    information = {
        "features": selected,
        "feature_count": len(selected),
        "selection": "Physical roles plus unsupervised shared-training redundancy diagnostics; no predictive target or validation/test score used.",
        "training_rows": len(training),
        "training_start": str(training.index[0]),
        "training_end": str(training.index[-1]),
        "frozen_before_controlled_test": True,
        "roles": FEATURE_ROLES,
        "full_standardized_rank": int(
            np.linalg.matrix_rank((training - training.mean()) / training.std())
        ),
        "compact_standardized_rank": int(
            np.linalg.matrix_rank(
                (training[selected] - training[selected].mean())
                / training[selected].std()
            )
        ),
    }
    path.write_text(json.dumps(information, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "feature_selection_report.csv", index=False)
    correlations.to_csv(
        OUTPUT_DIR / "training_feature_correlations.csv", index_label="feature"
    )
    print("Frozen compact features:", selected)
    print(
        "Full/compact training rank:",
        information["full_standardized_rank"],
        information["compact_standardized_rank"],
    )


def prepare_controlled(horizon, feature_set):
    """Subset columns only; both feature sets retain identical X/y origin indexes."""
    if horizon not in [6, 24, 48, 72] or feature_set not in ["full", "compact"]:
        raise ValueError("Unknown controlled configuration.")
    frozen_path = OUTPUT_DIR / "compact_features.json"
    frozen = json.loads(frozen_path.read_text())
    data = list(prepare_regression(horizon))
    full_names = data[0].columns.tolist()
    names = full_names if feature_set == "full" else frozen["features"]
    if not set(names).issubset(full_names):
        raise ValueError(
            "Compact features are not a subset of the original predictors."
        )
    for position in [0, 2, 4]:
        data[position] = data[position].loc[:, names].copy()
    data[-1] = {
        **data[-1],
        "feature_set": feature_set,
        "feature_count": len(names),
        "compact_feature_file_sha256": hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest(),
        "controlled_study_status": "Exploratory extension; fixed features/grids; no confirmatory independence claimed.",
    }
    return tuple(data)


def smoke_test():
    """Small training-only end-to-end check; serialize in memory, save no ML artifact."""
    report = []
    for horizon in [6, 24, 48, 72]:
        full = prepare_controlled(horizon, "full")
        compact = prepare_controlled(horizon, "compact")
        for position in [0, 2, 4]:
            pd.testing.assert_index_equal(full[position].index, compact[position].index)
            pd.testing.assert_series_equal(full[position + 1], compact[position + 1])
        X, y = compact[:2]
        X, y = X.iloc[:2000], y.iloc[:2000]
        for fold, (train, validation) in enumerate(
            TimeSeriesSplit(n_splits=3, gap=horizon // 3).split(X), start=1
        ):
            assert (
                X.index[train[-1]] + pd.Timedelta(hours=horizon)
                < X.index[validation[0]]
            )
            model = DecisionTreeRegressor(max_depth=3, random_state=42)
            model.fit(X.iloc[train], y.iloc[train])
            prediction = model.predict(X.iloc[validation])
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            np.testing.assert_array_equal(
                joblib.load(buffer).predict(X.iloc[validation]), prediction
            )
            report.append(
                {
                    "horizon": horizon,
                    "fold": fold,
                    "smoke_only_rmse": root_mean_squared_error(
                        y.iloc[validation], prediction
                    ),
                }
            )
    (OUTPUT_DIR / "provenance").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "provenance/smoke_test.json").write_text(json.dumps(report, indent=2))
    print(
        "Smoke test passed: targets, subsets, gaps, memory reload and controlled output paths."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-features", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    if options.select_features:
        freeze_features()
    if options.smoke:
        smoke_test()


if __name__ == "__main__":
    main()
