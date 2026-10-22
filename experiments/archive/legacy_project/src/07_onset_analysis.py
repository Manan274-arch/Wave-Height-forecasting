"""Analyze frozen 24-hour predictions and compare exploratory forecast horizons."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from feature_pipeline import EXPERIMENT_DIR, classify_heights, prepare_features
from model_metrics import metrics

ROOT_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT_DIR / "experiments/6_hour/models"
ML_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


def onset_metrics(current_height, truth, prediction):
    """Measure observation-level onset detection only among currently non-High origins."""
    if not current_height.index.equals(truth.index) or not prediction.index.equals(
        truth.index
    ):
        raise ValueError("Onset-analysis timestamps must match exactly.")
    nonhigh = current_height.le(3.0)
    future_high = truth.eq(2)
    onset = nonhigh & future_high
    alerted = nonhigh & prediction.eq(2)
    detected = int((onset & prediction.eq(2)).sum())
    missed = int((onset & prediction.ne(2)).sum())
    false_alerts = int((alerted & ~future_high).sum())
    negative_origins = int((nonhigh & ~future_high).sum())
    if detected + missed != int(onset.sum()):
        raise ValueError("Incorrect onset denominator.")
    return {
        "new_high_origins": int(onset.sum()),
        "new_high_detected": detected,
        "new_high_missed": missed,
        "onset_recall": detected / int(onset.sum()) if onset.any() else np.nan,
        "currently_nonhigh_high_predictions": int(alerted.sum()),
        "currently_nonhigh_false_high": false_alerts,
        "currently_nonhigh_high_precision": detected / int(alerted.sum())
        if alerted.any()
        else np.nan,
        "currently_nonhigh_false_high_rate": false_alerts / negative_origins
        if negative_origins
        else np.nan,
        "current_high_future_high": int((~nonhigh & future_high).sum()),
        "current_nonhigh_future_high": int(onset.sum()),
        "current_nonhigh_future_nonhigh": negative_origins,
        "current_high_future_nonhigh": int((~nonhigh & ~future_high).sum()),
    }


def plot_onsets(analysis):
    """Compare onset recall and detected/missed counts for the five frozen methods."""
    scores = analysis.set_index("family")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scores["onset_recall"].plot.bar(
        ax=axes[0], rot=25, color=["darkorange"] + ["steelblue"] * 4
    )
    axes[0].set(title="New High onset recall", ylabel="Fraction detected", ylim=(0, 1))
    axes[0].bar_label(axes[0].containers[0], fmt="%.3f", padding=3)
    scores[["new_high_detected", "new_high_missed"]].rename(
        columns={"new_high_detected": "Detected", "new_high_missed": "Missed"}
    ).plot.bar(ax=axes[1], rot=25)
    axes[1].set(title="New High origins: observation counts", ylabel="Origins")
    fig.suptitle("24-hour exploratory onset detection; not independent storm events")
    fig.tight_layout()
    fig.savefig(EXPERIMENT_DIR / "figures/onset_detection.png", dpi=160)
    plt.show()
    plt.close(fig)


def compare_horizons(X_test, y_test, predictions):
    """Report primary horizons and supplementary common-origin results, without tuning."""
    _, _, _, _, X_six, y_six = prepare_features(horizon_steps=2)
    six_metadata = json.loads((ARCHIVE_DIR / "model_metadata.json").read_text())
    six_saved = pd.read_csv(ARCHIVE_DIR / "test_comparison.csv").set_index("family")
    six_predictions = {"Persistence": classify_heights(X_six["VHM0"])}
    # Archived six-hour models are loaded only for diagnostic onset/common-origin scores.
    # The original archived results are checked, never rewritten.
    with threadpool_limits(limits=4):
        for name, info in six_metadata["models"].items():
            path = ARCHIVE_DIR / info["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError("Archived model hash mismatch.")
            model = joblib.load(path)
            if list(model.feature_names_in_) != X_six.columns.tolist():
                raise ValueError("Archived model feature order mismatch.")
            prediction = pd.Series(model.predict(X_six), index=y_six.index)
            six_predictions[name] = prediction
            result = metrics(y_six, prediction)
            np.testing.assert_allclose(
                result["macro_f1"], six_saved.loc[name, "macro_f1"], rtol=0, atol=1e-12
            )
    test24 = pd.read_csv(EXPERIMENT_DIR / "test_comparison.csv").set_index("family")
    rows = []
    for hours, X, truth, method_predictions, saved in [
        (6, X_six, y_six, six_predictions, six_saved),
        (24, X_test, y_test, predictions, test24),
    ]:
        # Highest test score identifies a reporting row only; configurations stay frozen.
        best = saved["macro_f1"].idxmax()
        best_metrics = metrics(truth, method_predictions[best])
        persistence = metrics(truth, method_predictions["Persistence"])
        onset = onset_metrics(X["VHM0"], truth, method_predictions[best])
        high = X["VHM0"].gt(3.0)
        threshold_runs = int((high & ~high.shift(1, fill_value=False)).sum())
        rows.append(
            {
                "horizon_hours": hours,
                "future_observations": hours // 3,
                "test_rows": len(truth),
                "test_start": str(truth.index[0]),
                "test_end": str(truth.index[-1]),
                "class_0_count": int(truth.eq(0).sum()),
                "class_1_count": int(truth.eq(1).sum()),
                "class_2_count": int(truth.eq(2).sum()),
                "persistence_macro_f1": persistence["macro_f1"],
                "persistence_high_recall": persistence["high_recall"],
                "persistence_false_high": persistence["false_high_alerts"],
                "category_agreement": float(
                    method_predictions["Persistence"].eq(truth).mean()
                ),
                "best_ml_algorithm": best,
                "best_ml_macro_f1": best_metrics["macro_f1"],
                "best_ml_high_recall": best_metrics["high_recall"],
                "best_ml_false_high": best_metrics["false_high_alerts"],
                "best_ml_macro_f1_minus_persistence": best_metrics["macro_f1"]
                - persistence["macro_f1"],
                "new_high_origins": onset["new_high_origins"],
                "best_ml_onset_recall": onset["onset_recall"],
                "contiguous_current_high_runs": threshold_runs,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(EXPERIMENT_DIR / "horizon_comparison.csv", index=False)
    common = y_six.index.intersection(y_test.index)
    supplementary = []
    for hours, X, truth, method_predictions in [
        (6, X_six, y_six, six_predictions),
        (24, X_test, y_test, predictions),
    ]:
        for name in ["Persistence"] + ML_NAMES:
            prediction = method_predictions[name].loc[common]
            result = metrics(truth.loc[common], prediction)
            onset = onset_metrics(X.loc[common, "VHM0"], truth.loc[common], prediction)
            supplementary.append(
                {
                    "horizon_hours": hours,
                    "family": name,
                    "common_test_origins": len(common),
                    "start": str(common[0]),
                    "end": str(common[-1]),
                    **result,
                    **onset,
                }
            )
    pd.DataFrame(supplementary).to_csv(
        EXPERIMENT_DIR / "common_origin_comparison.csv", index=False
    )
    print("Primary horizon comparison:\n", comparison.to_string(index=False))
    print(
        "Supplementary common-origin comparison:",
        len(common),
        "identical origins; different targets.",
    )
    print(
        "Contiguous threshold-exceedance runs are not independently declustered storms."
    )


def main():
    """Reuse the exact saved Phase 5 predictions; do not refit or retune models."""
    _, _, _, _, X_test, y_test = prepare_features()
    recorded = pd.read_csv(
        EXPERIMENT_DIR / "test_predictions.csv",
        index_col="timestamp",
        float_precision="round_trip",
    )
    recorded.index = pd.DatetimeIndex(pd.to_datetime(recorded.index, utc=True)).as_unit(
        "ns"
    )
    pd.testing.assert_series_equal(recorded["target_class"], y_test, check_exact=True)
    pd.testing.assert_series_equal(
        recorded["current_VHM0"], X_test["VHM0"], check_names=False, check_exact=True
    )
    predictions = {"Persistence": classify_heights(X_test["VHM0"])}
    for name in ML_NAMES:
        prediction = recorded[name]
        if not prediction.isin([0, 1, 2]).all():
            raise ValueError("Invalid stored prediction classes.")
        predictions[name] = prediction
    rows = []
    for name, prediction in predictions.items():
        rows.append(
            {"family": name, **onset_metrics(X_test["VHM0"], y_test, prediction)}
        )
    analysis = pd.DataFrame(rows)
    analysis.to_csv(EXPERIMENT_DIR / "onset_analysis.csv", index=False)
    print("New High onset analysis:\n", analysis.to_string(index=False))
    print(
        "Undefined restricted precision is NaN when no non-High-origin High alerts exist."
    )
    plot_onsets(analysis)
    compare_horizons(X_test, y_test, predictions)
    print("Exploratory analysis complete; no thresholds, strategies or models changed.")


if __name__ == "__main__":
    main()
