"""Consolidate controlled results and create descriptive comparisons, without fitting."""

import json

import matplotlib.pyplot as plt
import pandas as pd

from controlled_pipeline import OUTPUT_DIR
from regression_evaluation import save_figure
from regression_pipeline import METHODS, persistence_improvements, regression_metrics

HOURS = [6, 24, 48, 72]
FEATURE_SETS = ["full", "compact"]


def collect_results():
    """One master row per frozen ML configuration; preserve all scores and settings."""
    rows, rankings, partitions, predictions = [], [], [], {}
    for horizon in HOURS:
        for feature_set in FEATURE_SETS:
            directory = OUTPUT_DIR / f"{horizon}_hour" / feature_set
            metadata = json.loads((directory / "model_metadata.json").read_text())
            test = pd.read_csv(directory / "test_results.csv").set_index("family")
            validation = pd.read_csv(directory / "validation_results.csv").set_index(
                "family"
            )
            categories = pd.read_csv(directory / "category_errors.csv")
            predictions[horizon, feature_set] = pd.read_csv(
                directory / "test_predictions.csv",
                index_col="timestamp",
                float_precision="round_trip",
            )
            predictions[horizon, feature_set].index = pd.DatetimeIndex(
                pd.to_datetime(predictions[horizon, feature_set].index, utc=True)
            ).as_unit("ns")
            if feature_set == "full":
                for name, partition in metadata["partitions"].items():
                    partitions.append(
                        {
                            "horizon_hours": horizon,
                            "partition": name,
                            "cv_gap": horizon // 3,
                            **partition,
                        }
                    )
            learned = test.drop(index="Persistence").sort_values("rmse", kind="stable")
            for rank, name in enumerate(learned.index, start=1):
                rankings.append(
                    {
                        "horizon_hours": horizon,
                        "feature_set": feature_set,
                        "method": name,
                        "ml_rank": rank,
                        "rmse": test.loc[name, "rmse"],
                        "method_type": "learned",
                    }
                )
                high = categories.loc[
                    categories.family.eq(name) & categories.category.eq("High")
                ].iloc[0]
                info = metadata["models"][name]
                rows.append(
                    {
                        "horizon_hours": horizon,
                        "feature_set": feature_set,
                        "feature_count": metadata["feature_count"],
                        "model": name,
                        "train_rows": metadata["partitions"]["Training"]["rows"],
                        "validation_rows": metadata["partitions"]["Validation"]["rows"],
                        "test_rows": metadata["partitions"]["Testing"]["rows"],
                        "cv_rmse": info["mean_cv_rmse"],
                        "cv_rmse_std": info["std_cv_rmse"],
                        "selected_parameters": json.dumps(info["selected_parameters"]),
                        "validation_mae": validation.loc[name, "mae"],
                        "validation_rmse": validation.loc[name, "rmse"],
                        "validation_r2": validation.loc[name, "r2"],
                        "test_mae": test.loc[name, "mae"],
                        "test_rmse": test.loc[name, "rmse"],
                        "test_r2": test.loc[name, "r2"],
                        "test_bias": test.loc[name, "mean_signed_error"],
                        "persistence_rmse": test.loc["Persistence", "rmse"],
                        "rmse_improvement_pct": test.loc[
                            name, "rmse_improvement_percent"
                        ],
                        "mae_improvement_pct": test.loc[
                            name, "mae_improvement_percent"
                        ],
                        "high_wave_test_count": int(high.sample_count),
                        "high_wave_rmse": high.rmse,
                        "high_wave_mae": high.mae,
                        "high_wave_bias": high.mean_signed_error,
                        "high_wave_underprediction_frequency": high.underprediction_frequency,
                        "high_wave_mean_underprediction": high.mean_underprediction_magnitude,
                    }
                )
            rankings.append(
                {
                    "horizon_hours": horizon,
                    "feature_set": feature_set,
                    "method": "Persistence",
                    "ml_rank": None,
                    "rmse": test.loc["Persistence", "rmse"],
                    "method_type": "baseline",
                }
            )
    master = pd.DataFrame(rows).sort_values(["horizon_hours", "feature_set", "model"])
    master.to_csv(OUTPUT_DIR / "master_results.csv", index=False)
    pd.DataFrame(rankings).to_csv(OUTPUT_DIR / "algorithm_rankings.csv", index=False)
    pd.DataFrame(partitions).to_csv(OUTPUT_DIR / "partition_summary.csv", index=False)
    return master, predictions


def comparison_tables(master):
    """Define signs explicitly for feature reduction and XGBoost versus linear errors."""
    full = master.loc[master.feature_set.eq("full")]
    compact = master.loc[master.feature_set.eq("compact")]
    differences = full[["horizon_hours", "model", "test_rmse"]].merge(
        compact[["horizon_hours", "model", "test_rmse"]],
        on=["horizon_hours", "model"],
        suffixes=("_full", "_compact"),
    )
    differences["rmse_change_compact_minus_full"] = (
        differences.test_rmse_compact - differences.test_rmse_full
    )
    differences["rmse_change_percent_of_full"] = (
        100 * differences.rmse_change_compact_minus_full / differences.test_rmse_full
    )
    differences.to_csv(OUTPUT_DIR / "full_vs_compact.csv", index=False)
    xgb = master.loc[
        master.model.eq("XGBoost"), ["horizon_hours", "feature_set", "test_rmse"]
    ]
    linear = master.loc[
        master.model.eq("Linear Regression"),
        ["horizon_hours", "feature_set", "test_rmse"],
    ]
    paired = linear.merge(
        xgb, on=["horizon_hours", "feature_set"], suffixes=("_linear", "_xgb")
    )
    paired["rmse_reduction_linear_minus_xgb"] = (
        paired.test_rmse_linear - paired.test_rmse_xgb
    )
    paired["xgb_improvement_percent_over_linear"] = (
        100 * paired.rmse_reduction_linear_minus_xgb / paired.test_rmse_linear
    )
    paired.to_csv(OUTPUT_DIR / "xgboost_vs_linear.csv", index=False)
    outcomes = []
    for (horizon, features), group in master.groupby(["horizon_hours", "feature_set"]):
        scores = group.set_index("model").test_rmse
        outcomes.append(
            {
                "horizon_hours": horizon,
                "feature_set": features,
                "linear_beats_tree": scores["Linear Regression"]
                < scores["Decision Tree"],
                "forest_beats_tree": scores["Random Forest"] < scores["Decision Tree"],
                "xgb_beats_forest": scores["XGBoost"] < scores["Random Forest"],
                "xgb_beats_linear": scores["XGBoost"] < scores["Linear Regression"],
                "methods_beating_persistence": ", ".join(
                    group.loc[group.rmse_improvement_pct.gt(0), "model"]
                ),
            }
        )
    pd.DataFrame(outcomes).to_csv(OUTPUT_DIR / "algorithm_comparisons.csv", index=False)
    return differences, paired


def common_origin_results(predictions):
    """Reuse stored predictions at the test origins eligible for every horizon."""
    common = predictions[6, "full"].index
    for frame in predictions.values():
        common = common.intersection(frame.index)
    rows = []
    for (horizon, features), frame in predictions.items():
        selected = frame.loc[common]
        result = persistence_improvements(
            pd.DataFrame(
                [
                    {
                        "family": method,
                        **regression_metrics(selected.actual_VHM0, selected[method]),
                    }
                    for method in METHODS
                ]
            )
        )
        for row in result.to_dict("records"):
            rows.append(
                {
                    "horizon_hours": horizon,
                    "feature_set": features,
                    "common_test_origins": len(common),
                    "start": str(common[0]),
                    "end": str(common[-1]),
                    **row,
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "common_origin_results.csv", index=False)
    print("Supplementary common test origins:", len(common), common[0], common[-1])


def plot_comparisons(master, differences, paired):
    """Use fixed full/compact panels and metre axes beginning at zero."""
    figures = OUTPUT_DIR / "figures"
    figures.mkdir(exist_ok=True)
    rmse_upper = 1.08 * max(master.test_rmse.max(), master.persistence_rmse.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, feature_set in zip(axes, FEATURE_SETS):
        group = master.loc[master.feature_set.eq(feature_set)]
        for name in METHODS[1:]:
            curve = group.loc[group.model.eq(name)].sort_values("horizon_hours")
            ax.plot(curve.horizon_hours, curve.test_rmse, marker="o", label=name)
        baseline = group.groupby("horizon_hours").persistence_rmse.first()
        ax.plot(baseline.index, baseline, "k--", label="Persistence")
        ax.set(
            title=feature_set.title(),
            xlabel="Forecast horizon (h)",
            ylabel="Test RMSE (m)",
            xticks=HOURS,
            ylim=(0, rmse_upper),
        )
        ax.legend(fontsize=8)
    fig.suptitle("Native eligible test origins: controlled exploratory regression")
    save_figure(fig, figures, "rmse_vs_horizon.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, horizon in zip(axes.flat, HOURS):
        group = differences.loc[differences.horizon_hours.eq(horizon)].set_index(
            "model"
        )
        group[["test_rmse_full", "test_rmse_compact"]].rename(
            columns={"test_rmse_full": "Full (46)", "test_rmse_compact": "Compact (18)"}
        ).plot.bar(ax=ax, rot=15)
        ax.set(
            title=f"{horizon}h",
            ylabel="Test RMSE (m)",
            xlabel="Algorithm",
            ylim=(0, rmse_upper),
        )
    save_figure(fig, figures, "full_vs_compact_rmse.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for features, style in [("full", "-"), ("compact", "--")]:
        group = paired.loc[paired.feature_set.eq(features)].sort_values("horizon_hours")
        ax.plot(
            group.horizon_hours,
            group.test_rmse_linear,
            style,
            marker="o",
            color="steelblue",
            label=f"Linear, {features}",
        )
        ax.plot(
            group.horizon_hours,
            group.test_rmse_xgb,
            style,
            marker="s",
            color="darkorange",
            label=f"XGBoost, {features}",
        )
    ax.set(
        title="XGBoost versus ordinary Linear Regression",
        xlabel="Horizon (h)",
        ylabel="Test RMSE (m)",
        xticks=HOURS,
        ylim=(0, rmse_upper),
    )
    ax.legend()
    save_figure(fig, figures, "xgboost_vs_linear.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    high_upper = 1.08 * max(
        master.high_wave_rmse.max(),
        max(
            pd.read_csv(
                OUTPUT_DIR / f"{h}_hour/persistence_results.csv"
            ).high_rmse.iloc[0]
            for h in HOURS
        ),
    )
    for ax, features in zip(axes, FEATURE_SETS):
        for name in METHODS[1:]:
            group = master.loc[
                master.feature_set.eq(features) & master.model.eq(name)
            ].sort_values("horizon_hours")
            ax.plot(
                group.horizon_hours, group.rmse_improvement_pct, marker="o", label=name
            )
        ax.axhline(0, color="black", linestyle="--", label="Persistence reference")
        ax.set(
            title=features.title(),
            xlabel="Horizon (h)",
            ylabel="RMSE improvement over persistence (%)",
            xticks=HOURS,
        )
        ax.legend(fontsize=8)
    save_figure(fig, figures, "persistence_vs_ml.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, features in zip(axes, FEATURE_SETS):
        for name in METHODS[1:]:
            group = master.loc[
                master.feature_set.eq(features) & master.model.eq(name)
            ].sort_values("horizon_hours")
            ax.plot(group.horizon_hours, group.high_wave_rmse, marker="o", label=name)
        baseline = [
            pd.read_csv(
                OUTPUT_DIR / f"{h}_hour/persistence_results.csv"
            ).high_rmse.iloc[0]
            for h in HOURS
        ]
        ax.plot(HOURS, baseline, "k--", label="Persistence")
        ax.set(
            title=features.title(),
            xlabel="Horizon (h)",
            ylabel="Actual-High subset RMSE (m)",
            xticks=HOURS,
            ylim=(0, high_upper),
        )
        ax.legend(fontsize=8)
    save_figure(fig, figures, "high_wave_rmse_vs_horizon.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, horizon in zip(axes.flat, HOURS):
        for name in METHODS[1:]:
            group = master.loc[
                master.horizon_hours.eq(horizon) & master.model.eq(name)
            ].sort_values("feature_count")
            ax.plot(group.feature_count, group.test_rmse, marker="o", label=name)
        ax.set(
            title=f"{horizon}h",
            xlabel="Number of fixed input features",
            ylabel="Test RMSE (m)",
            xticks=[18, 46],
            ylim=(0, rmse_upper),
        )
        ax.legend(fontsize=8)
    save_figure(fig, figures, "feature_count_vs_rmse.png")


def main():
    master, predictions = collect_results()
    differences, paired = comparison_tables(master)
    common_origin_results(predictions)
    plot_comparisons(master, differences, paired)
    scores = master.pivot(
        index=["horizon_hours", "feature_set"], columns="model", values="test_rmse"
    )
    print("Consolidated controlled test RMSE:\n", scores.to_string())
    print(
        "Compact-minus-full differences; positive means compact is worse:\n",
        differences.to_string(index=False),
    )
    print(
        "XGBoost improvement over linear; positive means XGBoost is better:\n",
        paired.to_string(index=False),
    )


if __name__ == "__main__":
    main()
