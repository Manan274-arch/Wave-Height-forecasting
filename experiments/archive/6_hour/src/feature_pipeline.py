"""Deterministic Phase 3 features and fixed chronological classification partitions.

No fitted transforms, balancing, model training or file output is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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


def build_labels(source):
    """Classify the maximum of exact +3h and +6h significant heights."""
    times = source.index.to_series()
    exact_3h = times.shift(-1).eq(times + pd.Timedelta(hours=3))
    exact_6h = times.shift(-2).eq(times + pd.Timedelta(hours=6))
    if not exact_3h.iloc[:-1].all() or not exact_6h.iloc[:-2].all():
        raise ValueError("Shifted future rows are not exactly +3h/+6h.")
    future_3h = source["VHM0"].shift(-1)
    future_6h = source["VHM0"].shift(-2)
    future_max_wave_height = pd.concat([future_3h, future_6h], axis=1).max(
        axis=1, skipna=False
    )
    complete_horizon = exact_3h & exact_6h & future_max_wave_height.notna()
    if not complete_horizon.iloc[:-2].all() or complete_horizon.iloc[-2:].any():
        raise ValueError("Unexpected incomplete horizons beyond the final two rows.")
    # Incomplete horizons are excluded rather than assigned a misleading class.
    future_max = future_max_wave_height.loc[complete_horizon]
    target_class = np.select(
        [future_max.lt(2.5), future_max.le(3.0)], [0, 1], default=2
    ).astype("int64")
    labeled = pd.DataFrame(
        {
            "future_max_wave_height": future_max,
            "target_class": target_class,
        }
    )
    if len(labeled) != 29222:
        raise ValueError("Label count differs from the verified dataset.")
    return labeled


def split_labels(labeled):
    """Apply fixed 70/10/20 chronology and purge overlapping label horizons."""
    n_labeled = len(labeled)
    validation_position = int(np.floor(0.70 * n_labeled))
    test_position = validation_position + int(np.floor(0.10 * n_labeled))
    nominal_partitions = {
        "Training": labeled.iloc[:validation_position],
        "Validation": labeled.iloc[validation_position:test_position],
        "Testing": labeled.iloc[test_position:],
    }
    next_starts = {
        "Training": nominal_partitions["Validation"].index[0],
        "Validation": nominal_partitions["Testing"].index[0],
    }
    partitions = {}
    for name, part in nominal_partitions.items():
        if name in next_starts:
            # Targets must end strictly before the next nominal partition.
            keep = part.index + pd.Timedelta(hours=6) < next_starts[name]
        else:
            keep = np.ones(len(part), dtype=bool)
        partitions[name] = part.loc[keep].copy()
    phase2_expected = {
        "Training": (20453, "2015-01-01T00:00", "2021-12-31T12:00"),
        "Validation": (2920, "2021-12-31T21:00", "2022-12-31T18:00"),
        "Testing": (5845, "2023-01-01T03:00", "2024-12-31T15:00"),
    }
    for name, part in partitions.items():
        count, start, end = phase2_expected[name]
        expected_start = pd.Timestamp(start, tz="UTC")
        expected_end = pd.Timestamp(end, tz="UTC")
        if (len(part), part.index[0], part.index[-1]) != (
            count,
            expected_start,
            expected_end,
        ):
            raise ValueError(f"{name} differs from the verified chronology.")
    excluded = len(labeled) - sum(len(part) for part in partitions.values())
    if excluded != 4:
        raise ValueError("Expected four label-boundary exclusions.")
    return partitions


def prepare_features(csv_path=None, *, source=None):
    """Return X_train, y_train, X_val, y_val, X_test, y_test in Phase 3 order."""
    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parent.parent
            / "data/udupi_wave_wind_2015_2024.csv"
        )
    if source is None:
        source = load_source(csv_path)
    labeled = build_labels(source)
    partitions = split_labels(labeled)
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
    expected = ((20445, 46), (20445,), (2920, 46), (2920,), (5845, 46), (5845,))
    if tuple(obj.shape for obj in objects) != expected:
        raise ValueError("Prepared shapes differ from established Phase 3 outputs.")
    return tuple(objects)
