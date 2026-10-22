"""Read-only historical EDA. Run: python src/01_eda.py.

Full-period summaries are descriptive; they must not drive test-based model tuning.
"""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data/udupi_wave_wind_2015_2024.csv"


def inspect_dataset(df):
    """Summarize schema, missing values and timestamp continuity."""
    print(f"Loaded: {DATA_PATH}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.head())

    # 2. Columns, data types and missing values
    print("Columns:", df.columns.tolist())
    column_summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean().mul(100),
        }
    )
    print(column_summary)

    # 3. Timestamp coverage and three-hour continuity
    timestamps = df["timestamp"]
    ordered_times = timestamps.dropna().sort_values()
    expected_times = pd.date_range(
        "2015-01-01 00:00", "2024-12-31 21:00", freq="3h", tz="UTC"
    )
    actual_times = pd.DatetimeIndex(ordered_times)
    missing_times = expected_times.difference(actual_times)
    unexpected_times = actual_times.difference(expected_times)
    intervals = ordered_times.diff().dropna()

    print("Timestamp range:", timestamps.min(), "to", timestamps.max())
    print("Timezone:", timestamps.dt.tz)
    print("Original order is chronological:", timestamps.is_monotonic_increasing)
    print(
        "Inferred frequency:",
        pd.infer_freq(actual_times)
        if len(actual_times) >= 3
        else "Insufficient timestamps",
    )
    print(
        "Duplicate timestamps beyond the first occurrence:",
        timestamps.duplicated().sum(),
    )
    print("Missing expected timestamps:", len(missing_times))
    print("Unexpected/off-grid timestamps:", len(unexpected_times))
    print(
        "Intervals different from 3 hours:", intervals.ne(pd.Timedelta(hours=3)).sum()
    )
    print(intervals.value_counts().rename_axis("interval").to_frame("count"))
    if len(missing_times):
        print(pd.DataFrame({"missing_timestamp": missing_times}))
    if len(unexpected_times):
        print(pd.DataFrame({"unexpected_timestamp": unexpected_times}))
    if timestamps.duplicated(keep=False).any():
        print(df.loc[timestamps.duplicated(keep=False)])


def analyze_wave_heights(df):
    """Describe observed height ranges and display their distribution."""
    # 4. Numerical descriptive statistics
    numerical = df.select_dtypes(include="number")
    statistics = numerical.describe(percentiles=[0.50, 0.90, 0.95, 0.99]).T
    statistics = statistics.rename(
        columns={"50%": "median", "90%": "p90", "95%": "p95", "99%": "p99"}
    )
    statistics["missing_count"] = numerical.isna().sum()
    statistics["infinite_count"] = np.isinf(numerical).sum()
    print(statistics)

    # 5. Significant wave-height distribution and provisional ranges
    wave = df["VHM0"]
    valid_wave = wave.dropna()
    if valid_wave.empty:
        raise ValueError("VHM0 has no valid observations to summarize.")
    print(
        valid_wave.describe(percentiles=[0.50, 0.90, 0.95, 0.99]).to_frame("VHM0 (m)")
    )

    range_counts = pd.Series(
        {
            "Below 2.5 m": valid_wave.lt(2.5).sum(),
            "2.5-3.0 m inclusive": valid_wave.between(2.5, 3.0, inclusive="both").sum(),
            "Above 3.0 m": valid_wave.gt(3.0).sum(),
        },
        name="count",
    )
    range_summary = range_counts.to_frame()
    range_summary["percent_of_nonmissing_heights"] = range_counts.div(
        len(valid_wave)
    ).mul(100)
    print(range_summary)
    print("Missing VHM0 measurements:", wave.isna().sum())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(valid_wave, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(2.5, color="darkorange", linestyle="--", label="2.5 m")
    ax.axvline(3.0, color="firebrick", linestyle="--", label="3.0 m")
    ax.set(
        title="Distribution of significant wave height (2015-2024)",
        xlabel="VHM0 (m)",
        ylabel="Observation count",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


def analyze_seasonality(df):
    """Summarize monthly/yearly heights and plot seasonal patterns."""
    wave = df["VHM0"]
    # 6. Monthly and yearly patterns
    monthly_groups = wave.groupby(df["timestamp"].dt.month)
    monthly = monthly_groups.agg(["count", "min", "max", "mean", "median", "std"])
    monthly["percent_above_2.5m"] = monthly_groups.apply(
        lambda values: values.dropna().gt(2.5).mean() * 100
    )
    monthly["percent_above_3.0m"] = monthly_groups.apply(
        lambda values: values.dropna().gt(3.0).mean() * 100
    )
    monthly.index.name = "month"
    print(monthly)
    print(
        wave.groupby(df["timestamp"].dt.year).agg(
            ["count", "min", "max", "mean", "median", "std"]
        )
    )
    print("Months ranked by proportion above 2.5 m:")
    print(
        monthly[["count", "percent_above_2.5m", "percent_above_3.0m"]].sort_values(
            "percent_above_2.5m", ascending=False
        )
    )

    month_numbers = list(range(1, 13))
    month_labels = [calendar.month_abbr[month] for month in month_numbers]
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_samples = [
        wave.loc[df["timestamp"].dt.month.eq(month)].dropna().to_numpy()
        for month in month_numbers
    ]
    ax.boxplot(monthly_samples, showfliers=False)
    ax.set_xticks(month_numbers, month_labels)
    ax.set(
        title="Monthly significant wave-height distribution",
        xlabel="Month",
        ylabel="VHM0 (m)",
    )
    ax.text(
        0.01,
        0.98,
        "Outliers hidden for readability; data remain unchanged",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly[["percent_above_2.5m", "percent_above_3.0m"]].rename(
        columns={
            "percent_above_2.5m": "VHM0 > 2.5 m",
            "percent_above_3.0m": "VHM0 > 3.0 m",
        }
    ).plot.bar(ax=ax, color=["darkorange", "firebrick"])
    ax.set_xticklabels(
        [calendar.month_abbr[int(month)] for month in monthly.index], rotation=0
    )
    ax.set(
        title="Monthly proportion of observations exceeding provisional thresholds",
        xlabel="Month",
        ylabel="Observations (%)",
    )
    fig.tight_layout()
    plt.show()


def explore_wind(df):
    """Display descriptive wave/wind relationships."""
    # Wave/wind relationship: descriptive only, not model-feature selection.
    print("\nCurrent wave/wind correlation:")
    print(df[["VHM0", "wind_speed", "eastward_wind", "northward_wind"]].corr())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["wind_speed"], df["VHM0"], s=5, alpha=0.12)
    ax.set(
        xlabel="Interpolated wind speed (m/s)",
        ylabel="Significant wave height (m)",
        title="Historical wave/wind relationship",
    )
    fig.tight_layout()
    plt.show()


def main():
    """Run read-only EDA in a fixed, easy-to-follow order."""
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    df = pd.read_csv(DATA_PATH, float_precision="round_trip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    inspect_dataset(df)
    analyze_wave_heights(df)
    analyze_seasonality(df)
    explore_wind(df)
    print("High waves are concentrated in June-August, especially July.")
    print("Full-period EDA is descriptive; it does not select models.")


if __name__ == "__main__":
    main()
