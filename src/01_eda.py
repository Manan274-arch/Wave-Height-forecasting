"""Describe the immutable historical dataset; save descriptive EDA figures."""

from importlib import import_module
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

preprocessing = import_module("02_preprocessing")


def main():
    source = preprocessing.load_source(preprocessing.DATA_PATH)
    print("CSV shape:", (len(source), len(source.columns) + 1))
    print("Columns:", ["timestamp", *source.columns])
    print("Dtypes (timestamp parsed as UTC index):\n", source.dtypes.to_string())
    print("Missing values:\n", source.isna().sum().to_string())
    print("UTC coverage:", source.index.min(), source.index.max())
    print("Intervals:\n", source.index.to_series().diff().value_counts().to_string())
    print("Duplicate timestamps:", source.index.duplicated().sum())
    print("Numerical description:\n", source.describe().T.to_string())
    elevated = source.VHM0.ge(2.5)
    print(
        f"Elevated >=2.5 m: {elevated.sum():,}/{len(source):,} ({100 * elevated.mean():.3f}%)"
    )
    monthly = source.groupby(source.index.month).VHM0.agg(
        ["count", "mean", "median", "max"]
    )
    monthly["elevated_percent"] = elevated.groupby(source.index.month).mean() * 100
    print("Monthly statistics:\n", monthly.to_string())
    print(
        "Yearly statistics:\n",
        source.groupby(source.index.year)
        .VHM0.agg(["count", "mean", "median", "max"])
        .to_string(),
    )
    output = preprocessing.RESULTS / "figures/eda"
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(source.VHM0, bins=60)
    ax.axvline(2.5, color="red", linestyle="--", label="Research threshold: 2.5 m")
    ax.set(
        xlabel="Significant wave height (m)",
        ylabel="Count",
        title="Historical offshore wave-height distribution",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "wave_height_distribution.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(
        [
            source.loc[source.index.month == month, "VHM0"].to_numpy()
            for month in range(1, 13)
        ],
        showfliers=False,
    )
    axes[0].set(
        xlabel="Month", ylabel="Height (m)", title="Monthly wave-height distribution"
    )
    axes[1].bar(monthly.index, monthly.elevated_percent)
    axes[1].set(
        xlabel="Month",
        ylabel="Elevated observations (%)",
        title="Seasonal Elevated frequency",
        xticks=np.arange(1, 13),
    )
    fig.tight_layout()
    fig.savefig(output / "seasonal_patterns.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(source.wind_speed, source.VHM0, s=2, alpha=0.15)
    axes[0].set(
        xlabel="Wind speed (m/s)",
        ylabel="Wave height (m)",
        title="Wind and offshore waves",
    )
    columns = ["VHM0", "VHM0_SW1", "VHM0_WW", "wind_speed"]
    image = axes[1].imshow(source[columns].corr(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=columns,
        yticklabels=columns,
        title="Pearson correlation",
    )
    axes[1].tick_params(axis="x", rotation=30)
    fig.colorbar(image, ax=axes[1])
    fig.tight_layout()
    fig.savefig(output / "wave_wind_relationships.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
