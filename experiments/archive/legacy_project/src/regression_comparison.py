"""Compare frozen regression horizons using saved predictions and metric tables."""

import json

import pandas as pd

from regression_pipeline import (
    METHODS,
    REGRESSION_DIR,
    persistence_improvements,
    regression_metrics,
)


def main():
    """Report both primary periods and common origins without choosing new models."""
    rows = []
    recorded = {}
    for horizon in [6, 24]:
        directory = REGRESSION_DIR / f"{horizon}_hour"
        metadata = json.loads((directory / "model_metadata.json").read_text())
        results = pd.read_csv(directory / "test_results.csv")
        predictions = pd.read_csv(
            directory / "test_predictions.csv",
            index_col="timestamp",
            float_precision="round_trip",
        )
        predictions.index = pd.DatetimeIndex(
            pd.to_datetime(predictions.index, utc=True)
        ).as_unit("ns")
        recorded[horizon] = predictions
        learned = results.loc[results.family.ne("Persistence")]
        best_ml = learned.loc[learned.rmse.idxmin(), "family"]
        best_method = results.loc[results.rmse.idxmin(), "family"]
        for result in results.to_dict("records"):
            rows.append(
                {
                    "horizon_hours": horizon,
                    "target": metadata["target"],
                    "training_rows": metadata["partitions"]["Training"]["rows"],
                    "validation_rows": metadata["partitions"]["Validation"]["rows"],
                    "test_rows": metadata["partitions"]["Testing"]["rows"],
                    "best_ml_by_test_rmse_reporting_only": best_ml,
                    "best_method_by_test_rmse_reporting_only": best_method,
                    **result,
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REGRESSION_DIR / "horizon_comparison.csv", index=False)
    print(
        "Primary test results (MAE/RMSE in metres):\n",
        comparison.pivot(
            index="family", columns="horizon_hours", values=["mae", "rmse", "r2"]
        )
        .reindex(METHODS)
        .to_string(),
    )

    common = recorded[6].index.intersection(recorded[24].index)
    pd.testing.assert_series_equal(
        recorded[6].loc[common, "current_VHM0"],
        recorded[24].loc[common, "current_VHM0"],
        check_exact=True,
    )
    common_rows = []
    for horizon in [6, 24]:
        predictions = recorded[horizon].loc[common]
        results = persistence_improvements(
            pd.DataFrame(
                [
                    {
                        "family": name,
                        **regression_metrics(
                            predictions.actual_VHM0, predictions[name]
                        ),
                    }
                    for name in METHODS
                ]
            )
        )
        for result in results.to_dict("records"):
            common_rows.append(
                {
                    "horizon_hours": horizon,
                    "common_origins": len(common),
                    "origin_start": str(common[0]),
                    "origin_end": str(common[-1]),
                    **result,
                }
            )
    common_results = pd.DataFrame(common_rows)
    common_results.to_csv(REGRESSION_DIR / "common_origin_comparison.csv", index=False)
    print(
        "\nSupplementary common-origin comparison:\n",
        common_results[
            [
                "horizon_hours",
                "family",
                "common_origins",
                "mae",
                "rmse",
                "r2",
                "high_rmse",
            ]
        ].to_string(index=False),
    )
    print(
        "Origins match, but future target timestamps and independently fitted models differ."
    )
    print(
        "Test rankings are descriptive only; no refitting or hyperparameter changes occur."
    )


if __name__ == "__main__":
    main()
