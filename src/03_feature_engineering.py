"""Construct exactly 18 causal predictors, the exact 12h target and purged splits."""

from importlib import import_module
import numpy as np
import pandas as pd

preprocessing = import_module("02_preprocessing")
COMPACT_FEATURES = [
    "VHM0",
    "VHM0_SW1",
    "VHM0_WW",
    "VTM02",
    "VTPK",
    "eastward_wind",
    "northward_wind",
    "wind_speed",
    "VMDR_sin",
    "VMDR_cos",
    "VHM0_lag_6h",
    "VHM0_lag_24h",
    "VHM0_change_3h",
    "VHM0_roll_mean_12h",
    "VHM0_roll_std_12h",
    "wind_speed_roll_mean_12h",
    "month_sin",
    "month_cos",
]


def make_features(source):
    """Rolling windows include t and three past rows; std uses ddof=1."""
    features = source[COMPACT_FEATURES[:8]].copy()
    # Keep CSV wind_speed exactly; preprocessing independently validates hypot(u, v).
    radians = np.deg2rad(source.VMDR)
    features["VMDR_sin"] = np.sin(radians)
    features["VMDR_cos"] = np.cos(radians)
    features["VHM0_lag_6h"] = source.VHM0.shift(2)
    features["VHM0_lag_24h"] = source.VHM0.shift(8)
    features["VHM0_change_3h"] = source.VHM0 - source.VHM0.shift(1)
    window = source.VHM0.rolling(4, min_periods=4, center=False)
    features["VHM0_roll_mean_12h"] = window.mean()
    features["VHM0_roll_std_12h"] = window.std(ddof=1)
    features["wind_speed_roll_mean_12h"] = source.wind_speed.rolling(
        4, min_periods=4, center=False
    ).mean()
    features["month_sin"] = np.sin(2 * np.pi * source.index.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * source.index.month / 12)
    return features.loc[:, COMPACT_FEATURES]


def prepare_features():
    """Purge target overlap at raw boundaries; do not impute warm-up or tail rows."""
    source = preprocessing.load_source(preprocessing.DATA_PATH)
    features = make_features(source)
    target = pd.Series(
        source.VHM0.reindex(source.index + pd.Timedelta(hours=12)).to_numpy(),
        index=source.index,
        name="future_VHM0",
    )
    pd.testing.assert_series_equal(
        target, source.VHM0.shift(-4).rename("future_VHM0"), check_exact=True
    )
    incomplete = features.isna().any(axis=1)
    assert features.index[incomplete].equals(source.index[:8])
    validation, testing = preprocessing.split_boundaries(source)
    nominal = {
        "Training": source.index[:validation],
        "Validation": source.index[validation:testing],
        "Testing": source.index[testing:],
    }
    next_start = {
        "Training": source.index[validation],
        "Validation": source.index[testing],
    }
    objects, summary = [], {}
    for name, origins in nominal.items():
        complete = origins[target.loc[origins].notna().to_numpy()]
        eligible = (
            complete[complete + pd.Timedelta(hours=12) < next_start[name]]
            if name in next_start
            else complete
        )
        purged = len(complete) - len(eligible)
        history = int(incomplete.loc[eligible].sum())
        eligible = eligible[~incomplete.loc[eligible].to_numpy()]
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
            "history_excluded": history,
            "purged": purged,
            "unavailable_targets": len(origins) - len(complete),
        }
    assert [len(objects[i]) for i in (0, 2, 4)] == [20444, 2918, 5842]
    return (*objects, summary)


def check_causality():
    source = preprocessing.load_source(preprocessing.DATA_PATH)
    changed = source.copy()
    changed.iloc[15001:] *= 1.1
    pd.testing.assert_frame_equal(
        make_features(source).iloc[:15001],
        make_features(changed).iloc[:15001],
        check_exact=True,
    )


if __name__ == "__main__":
    check_causality()
    *objects, summary = prepare_features()
    print("Feature order:", COMPACT_FEATURES)
    print(objects[0].head().to_string())
    print(summary)
    print("Exact timestamp target alignment and causal-feature checks passed.")
