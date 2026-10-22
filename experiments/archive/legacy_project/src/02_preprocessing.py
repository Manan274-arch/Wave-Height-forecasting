"""Validate observations and reproduce the finalized labels and Phase 2 split."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from feature_pipeline import (
    HORIZON_HOURS,
    HORIZON_STEPS,
    build_labels,
    load_source,
    split_labels,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data/udupi_wave_wind_2015_2024.csv"


def main():
    """Inspect shared validation, labels and purged partitions."""
    # 1. Load, sort and validate the complete three-hourly UTC series.
    source = load_source(DATA_PATH)
    print("Source shape, including timestamp:", (len(source), len(source.columns) + 1))
    print("UTC range:", source.index.min(), "to", source.index.max())
    print("Frequency:", pd.infer_freq(source.index))
    print("Duplicate timestamps:", source.index.duplicated().sum())
    print("Missing measurements:\n", source.isna().sum())
    print("Numeric, finite, physical-range, coordinate and wind-speed checks passed.")

    # 2. Shared labels verify all eight exact future three-hourly timestamps.
    # Future maximum includes +3h through +24h, with skipna=False.
    # Normal < 2.5; Elevated includes 2.5 and 3.0; High > 3.0 metres.
    print("Target hours / future observations:", HORIZON_HOURS, HORIZON_STEPS)
    labeled = build_labels(source)
    verification = pd.DataFrame(
        {
            "current_VHM0": source["VHM0"],
            "future_3h": source["VHM0"].shift(-1),
            "future_24h": source["VHM0"].shift(-HORIZON_STEPS),
            "future_max_wave_height": labeled["future_max_wave_height"],
            "target_class": labeled["target_class"],
        }
    )
    print("\nTarget verification:\n", verification.head())
    print("\nIncomplete final horizons:\n", verification.tail(HORIZON_STEPS))
    print("Complete future horizons:", len(labeled))
    print("Final incomplete horizons excluded:", len(source) - len(labeled))

    # 3. Fixed chronological 70/10/20 split, with eight label-boundary purges each.
    partitions = split_labels(labeled)
    rows = []
    for name, part in partitions.items():
        rows.append(
            {
                "partition": name,
                "eligible_label_rows": len(part),
                "boundary_purged": HORIZON_STEPS if name != "Testing" else 0,
                "history_excluded_later": 8 if name == "Training" else 0,
                "start_UTC": part.index[0],
                "end_UTC": part.index[-1],
            }
        )
    print(
        "\nPhase 2 partitions:\n", pd.DataFrame(rows).set_index("partition").to_string()
    )
    print(
        "Boundary exclusions:",
        len(labeled) - sum(len(part) for part in partitions.values()),
    )

    # Descriptive class counts reproduce Phase 2, not classifier test evaluation.
    for name, part in partitions.items():
        counts = part["target_class"].value_counts().reindex([0, 1, 2], fill_value=0)
        print(f"\n{name} label counts (diagnostic only):")
        print(pd.DataFrame({"count": counts, "percent": counts / len(part) * 100}))

    # 4. Future maximum is a classification-label diagnostic, never an input.
    print("\nFuture-maximum descriptive statistics:")
    print(
        labeled["future_max_wave_height"].describe(
            percentiles=[0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
        )
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(
        partitions["Training"]["future_max_wave_height"], bins=50, edgecolor="white"
    )
    ax.axvline(2.5, linestyle="--", color="orange", label="2.5 m")
    ax.axvline(3.0, linestyle="--", color="red", label="3.0 m")
    ax.set(
        title="Training future-maximum distribution",
        xlabel="Significant height (m)",
        ylabel="Count",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
