"""Inspect the shared 46 causal inputs without changing features or partitions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from feature_pipeline import (
    change_hours,
    current_features,
    directions,
    lag_hours,
    load_source,
    make_features,
    prepare_features,
    rolling_specs,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data/udupi_wave_wind_2015_2024.csv"


def describe_features(features):
    """Explain each predictor in the same order used by every model."""
    # 2. Explain every input in its actual column order.
    descriptions = {
        c: f"Current {meaning}; {unit}"
        for c, (unit, meaning) in current_features.items()
    }
    for column, meaning in directions.items():
        for encoding in ("sin", "cos"):
            descriptions[f"{column}_{encoding}"] = (
                f"Circular {encoding} of {meaning}; unitless"
            )
    for column, hours_list in lag_hours.items():
        for hours in hours_list:
            descriptions[f"{column}_lag_{hours}h"] = f"{column} at t-{hours}h"
    for column, hours_list in change_hours.items():
        for hours in hours_list:
            descriptions[f"{column}_change_{hours}h"] = (
                f"Current {column} minus t-{hours}h"
            )
    for column, statistic, hours in rolling_specs:
        descriptions[f"{column}_roll_{statistic}_{hours}h"] = (
            f"{statistic}: {hours // 3} observations from t-{hours - 3}h through t"
        )
    descriptions["month_sin"] = "sin(2*pi*UTC month/12)"
    descriptions["month_cos"] = "cos(2*pi*UTC month/12)"
    if list(descriptions) != features.columns.tolist():
        raise ValueError("Feature descriptions and input column ordering disagree.")
    print(
        "\nOrdered input dictionary:\n",
        pd.Series(descriptions, name="description").to_string(),
    )
    print("Total inputs:", len(features.columns))
    print("\nFeature preview:\n", features.dropna().head().to_string())


def check_causality(source, features):
    """Changing future observations must leave past predictors unchanged."""
    # 3. Check causality across the 2021/2022 year boundary.
    cutoff = pd.Timestamp("2022-01-01", tz="UTC")
    changed = source.copy()
    changed.loc[changed.index > cutoff, list(current_features) + list(directions)] *= (
        1.25
    )
    pd.testing.assert_frame_equal(
        features.loc[:cutoff], make_features(changed).loc[:cutoff]
    )
    forbidden = {
        "future_max_wave_height",
        "future_3h",
        "future_6h",
        "target_class",
        "latitude",
        "longitude",
        "timestamp",
        "year",
        "month",
    }
    if forbidden.intersection(features.columns) or any(
        column.startswith(("future_", "target_")) for column in features
    ):
        raise ValueError("Future-derived or metadata input detected.")
    missing = features.isna().any(axis=1)
    if not features.index[missing].equals(source.index[:8]):
        raise ValueError("Unexpected missing historical context.")
    print("First complete historical context:", features.index[~missing][0])
    print("Historical-context exclusions: Training=8, Validation=0, Testing=0")
    print("Leakage checks passed; no fitted transforms or class balancing.")


def plot_training_features(X_train):
    """Plot correlations and a fixed historical example, using training only."""
    # 5. Training-only plots; correlation does not determine feature importance.
    columns = [
        "VHM0",
        "VHM0_SW1",
        "VHM0_WW",
        "VTM02",
        "wind_speed",
        "eastward_wind",
        "VHM0_lag_6h",
        "VHM0_lag_24h",
        "VHM0_change_6h",
        "VHM0_roll_mean_12h",
        "VHM0_roll_std_12h",
        "wind_speed_roll_mean_12h",
        "month_sin",
        "month_cos",
    ]
    corr = X_train[columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    ax.set_title("Training feature correlations: descriptive only")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.show()
    window = X_train.loc["2020-06-01":"2020-06-14"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    window[["VHM0", "VHM0_lag_6h", "VHM0_roll_mean_12h"]].plot(ax=axes[0])
    window[["VHM0_change_3h", "VHM0_change_12h"]].plot(ax=axes[1])
    axes[0].set_ylabel("Height (m)")
    axes[1].set_ylabel("Height difference (m)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    plt.show()


def main():
    """Inspect deterministic features and their chronological partitions."""
    source = load_source(DATA_PATH)
    features = make_features(source)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(source=source)
    describe_features(features)
    check_causality(source, features)
    # 4. Preserve Phase 3 X/y shapes, alignment and diagnostic class counts.
    for name, X, y in [
        ("Training", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Testing", X_test, y_test),
    ]:
        print(f"\n{name}: X={X.shape}, y={y.shape}, {X.index.min()} to {X.index.max()}")
        counts = y.value_counts().reindex([0, 1, 2], fill_value=0)
        print(pd.DataFrame({"count": counts, "percent": counts / len(y) * 100}))

    plot_training_features(X_train)


if __name__ == "__main__":
    main()
