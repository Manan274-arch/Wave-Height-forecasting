"""Validate the immutable offshore CSV and define raw chronological boundaries."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/udupi_wave_wind_2015_2024.csv"
RESULTS = ROOT / "results"


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


def split_boundaries(source):
    """Use the original floor-based 70/10/20 boundaries before exclusions."""
    validation = int(np.floor(0.7 * len(source)))
    testing = validation + int(np.floor(0.1 * len(source)))
    return validation, testing


if __name__ == "__main__":
    source = load_source(DATA_PATH)
    print(
        f"Validated {len(source):,} rows; {source.index.min()} through {source.index.max()}"
    )
    print("Three-hourly UTC grid; no missing values, duplicates or imputation.")
    print("Raw validation/test positions:", split_boundaries(source))
