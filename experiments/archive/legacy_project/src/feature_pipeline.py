"""Deterministic Phase 3 features and fixed chronological classification partitions.

The active target is the maximum over the next eight observations (24 hours).
No fitted transforms, balancing, model training or file output is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data/udupi_wave_wind_2015_2024.csv"
EXPERIMENT_DIR = ROOT_DIR / "experiments/24_hour"
MODEL_DIR = EXPERIMENT_DIR / "models"
HORIZON_STEPS = 8
HORIZON_HOURS = 3 * HORIZON_STEPS


current_features = {
    "VHM0": ("m", "Total significant wave height"),
    "VHM0_SW1": ("m", "Primary swell significant height"),
    "VHM0_SW2": ("m", "Secondary swell significant height"),
    "VHM0_WW": ("m", "Wind-wave significant height"),
    "VTM01_SW1": ("s", "Primary swell mean period, moments (0, 1)"),
    "VTM01_SW2": ("s", "Secondary swell mean period, moments (0, 1)"),
    "VTM01_WW": ("s", "Wind-wave mean period, moments (0, 1)"),
    "VTM02": ("s", "Total mean period, moments (0, 2)"),
    "VTM10": ("s", "Total period, moments (-1, 0)"),
    "VTPK": ("s", "Peak wave period"),
    "eastward_wind": ("m/s", "Interpolated stress-equivalent 10 m eastward wind"),
    "northward_wind": ("m/s", "Interpolated stress-equivalent 10 m northward wind"),
    "wind_speed": ("m/s", "Magnitude of interpolated wind components"),
}

directions = {
    "VMDR": "Total wave from-direction",
    "VMDR_SW1": "Primary swell from-direction",
    "VMDR_SW2": "Secondary swell from-direction",
    "VMDR_WW": "Wind-wave from-direction",
}

lag_hours = {
    "VHM0": (3, 6, 12, 24),
    "wind_speed": (3, 6, 12),
    "VHM0_SW1": (6,),
    "VTM02": (3,),
}

change_hours = {"VHM0": (3, 6, 12), "wind_speed": (3, 6)}

rolling_specs = [
    ("VHM0", "mean", 6),
    ("VHM0", "mean", 12),
    ("VHM0", "mean", 24),
    ("VHM0", "max", 12),
    ("VHM0", "max", 24),
    ("VHM0", "std", 12),
    ("wind_speed", "mean", 6),
    ("wind_speed", "mean", 12),
    ("wind_speed", "max", 12),
]


def make_features(measurements):
    """Build the 46 ordered predictors using current and past measurements."""
    result = measurements[list(current_features)].copy()
    for column in directions:
        radians = np.deg2rad(measurements[column])
        result[f"{column}_sin"] = np.sin(radians)
        result[f"{column}_cos"] = np.cos(radians)
    for column, hours_list in lag_hours.items():
        for hours in hours_list:
            result[f"{column}_lag_{hours}h"] = measurements[column].shift(hours // 3)
    for column, hours_list in change_hours.items():
        for hours in hours_list:
            result[f"{column}_change_{hours}h"] = measurements[column] - measurements[
                column
            ].shift(hours // 3)
    for column, statistic, hours in rolling_specs:
        window = measurements[column].rolling(
            window=hours // 3, min_periods=hours // 3, center=False
        )
        if statistic == "std":
            values = window.std(ddof=1)
        elif statistic == "mean":
            values = window.mean()
        else:
            values = window.max()
        result[f"{column}_roll_{statistic}_{hours}h"] = values
    month = measurements.index.month
    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)
    return result


def load_source(data_path):
    """Read, sort and validate the unchanged three-hourly UTC observations."""
    source = pd.read_csv(data_path, float_precision="round_trip")
    expected_columns = [
        "timestamp",
        "latitude",
        "longitude",
        "VHM0",
        "VHM0_SW1",
        "VHM0_SW2",
        "VHM0_WW",
        "VMDR",
        "VMDR_SW1",
        "VMDR_SW2",
        "VMDR_WW",
        "VTM01_SW1",
        "VTM01_SW2",
        "VTM01_WW",
        "VTM02",
        "VTM10",
        "VTPK",
        "eastward_wind",
        "northward_wind",
        "wind_speed",
    ]
    if set(source.columns) != set(expected_columns) or source.shape != (29224, 20):
        raise ValueError("Source schema or size differs from completed Phase 2.")
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="raise")
    if source["timestamp"].isna().any() or source["timestamp"].duplicated().any():
        raise ValueError("Missing or duplicate timestamps prevent reliable labels.")
    source = source.sort_values("timestamp").set_index("timestamp")
    source.index = source.index.as_unit("ns")
    expected_index = pd.date_range(
        "2015-01-01", "2024-12-31T21:00", freq="3h", tz="UTC"
    ).as_unit("ns")
    if not source.index.equals(expected_index):
        raise ValueError("The regular three-hourly UTC grid is incomplete or changed.")
    if not all(pd.api.types.is_numeric_dtype(source[c]) for c in source):
        raise ValueError("Source measurements must be numeric.")
    if source.isna().any().any() or not np.isfinite(source.to_numpy()).all():
        raise ValueError(
            "Missing/nonfinite source values require investigation; no filling is performed."
        )
    height_period_columns = [c for c in source if c.startswith(("VHM", "VTM", "VTP"))]
    direction_columns = [c for c in source if c.startswith("VMDR")]
    if source[height_period_columns].lt(0).any().any() or any(
        not source[c].between(0, 360).all() for c in direction_columns
    ):
        raise ValueError(
            "Invalid heights, periods or directions require investigation."
        )
    if len(source[["latitude", "longitude"]].drop_duplicates()) != 1 or not np.allclose(
        source[["latitude", "longitude"]].iloc[0],
        [13.399999618530273, 74.5999984741211],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Unexpected reference coordinates.")
    if source[["eastward_wind", "northward_wind"]].abs().gt(
        100
    ).any().any() or not np.allclose(
        source["wind_speed"],
        np.hypot(source["eastward_wind"], source["northward_wind"]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Wind validity or speed arithmetic differs from Phase 2.")
    return source


def classify_heights(heights):
    """Apply fixed research thresholds to complete, finite significant heights."""
    if heights.isna().any() or not np.isfinite(heights.to_numpy()).all():
        raise ValueError("Cannot classify missing or nonfinite heights.")
    values = np.select([heights.lt(2.5), heights.le(3.0)], [0, 1], default=2)
    return pd.Series(values, index=heights.index, dtype="int64")


def build_labels(source, horizon_steps=HORIZON_STEPS):
    """Classify the maximum across every exact future three-hourly observation."""
    if not isinstance(horizon_steps, int) or horizon_steps < 1:
        raise ValueError("The horizon must be a positive number of observations.")
    times = source.index.to_series()
    complete_horizon = pd.Series(True, index=source.index)
    for step in range(1, horizon_steps + 1):
        expected = times + pd.Timedelta(hours=3 * step)
        exact = times.shift(-step).eq(expected)
        if not exact.iloc[:-step].all():
            raise ValueError("Missing or irregular timestamps in future windows.")
        complete_horizon &= exact
    future_heights = pd.concat(
        [source["VHM0"].shift(-step) for step in range(1, horizon_steps + 1)],
        axis=1,
    )
    future_max = future_heights.max(axis=1, skipna=False)
    complete_horizon &= future_max.notna()
    if not complete_horizon.iloc[:-horizon_steps].all():
        raise ValueError("Unexpected incomplete horizons within the source data.")
    if complete_horizon.iloc[-horizon_steps:].any():
        raise ValueError("Final incomplete horizons must remain unclassified.")
    future_max = future_max.loc[complete_horizon]
    target_class = classify_heights(future_max)
    labeled = pd.DataFrame(
        {
            "future_max_wave_height": future_max,
            "target_class": target_class,
        }
    )
    if len(labeled) != len(source) - horizon_steps:
        raise ValueError("Incomplete-horizon exclusion count is incorrect.")
    return labeled


def split_labels(labeled, horizon_steps=HORIZON_STEPS):
    """Split complete labels 70/10/20 and purge their entire future horizons."""
    validation_position = int(np.floor(0.70 * len(labeled)))
    test_position = validation_position + int(np.floor(0.10 * len(labeled)))
    nominal = {
        "Training": labeled.iloc[:validation_position],
        "Validation": labeled.iloc[validation_position:test_position],
        "Testing": labeled.iloc[test_position:],
    }
    next_starts = {
        "Training": nominal["Validation"].index[0],
        "Validation": nominal["Testing"].index[0],
    }
    partitions = {}
    horizon = pd.Timedelta(hours=3 * horizon_steps)
    for name, part in nominal.items():
        if name in next_starts:
            keep = part.index + horizon < next_starts[name]
            retained = part.loc[keep]
            if len(part) - len(retained) != horizon_steps:
                raise ValueError("Unexpected horizon-aware boundary purge.")
            if not (retained.index + horizon < next_starts[name]).all():
                raise ValueError("A label crosses the next partition boundary.")
            partitions[name] = retained.copy()
        else:
            partitions[name] = part.copy()
    return partitions


def prepare_features(csv_path=None, *, source=None, horizon_steps=HORIZON_STEPS):
    """Return X_train, y_train, X_val, y_val, X_test, y_test in Phase 3 order."""
    if source is None:
        source = load_source(DATA_PATH if csv_path is None else csv_path)
    labeled = build_labels(source, horizon_steps)
    partitions = split_labels(labeled, horizon_steps)
    features = make_features(source)
    if len(features.columns) != 46 or any(
        c.startswith(("future_", "target_")) for c in features
    ):
        raise ValueError(
            "Feature schema differs from Phase 3 or contains future inputs."
        )
    if set(features.columns).intersection(
        {"timestamp", "latitude", "longitude", "month", "year"}
    ):
        raise ValueError("Metadata entered model inputs.")
    incomplete = features.isna().any(axis=1)
    if not features.index[incomplete].equals(source.index[:8]):
        raise ValueError("Unexpected missing historical features.")
    objects = []
    for name, part in partitions.items():
        retained = part.loc[~incomplete.reindex(part.index)]
        if len(part) - len(retained) != (8 if name == "Training" else 0):
            raise ValueError("Historical-context exclusions differ from Phase 3.")
        X = features.loc[retained.index].copy()
        y = retained["target_class"].astype("int64").copy()
        if (
            not X.index.equals(y.index)
            or not np.isfinite(X.to_numpy()).all()
            or not y.isin([0, 1, 2]).all()
        ):
            raise ValueError("Invalid model inputs or labels.")
        objects.extend([X, y])
    for position, part in enumerate(partitions.values()):
        expected_rows = len(part) - (8 if position == 0 else 0)
        if objects[2 * position].shape != (expected_rows, 46):
            raise ValueError("Prepared partition size or feature count is incorrect.")
    return tuple(objects)
