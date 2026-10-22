"""Direct binary classifiers on the frozen compact regression origins.

Run with the existing virtual-environment Python and -B. All writes stay here.
"""

import argparse
import ast
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from controlled_pipeline import prepare_controlled  # noqa: E402
from feature_pipeline import DATA_PATH, load_source, make_features  # noqa: E402

HORIZONS = [6, 24, 48, 72]
PAIRS = {
    "Persistence": "Persistence",
    "Logistic Regression": "Linear Regression",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}
FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary(heights):
    """Normal=0 below 2.5 m; Elevated=1 at or above 2.5 m."""
    values = np.asarray(heights)
    if not np.isfinite(values).all():
        raise ValueError("Cannot label missing or nonfinite heights.")
    return (values >= 2.5).astype("int64")


def metrics(truth, predicted):
    """Fixed [Normal, Elevated] order; absent-class divisions return zero."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "sample_count": len(truth),
        "elevated_count": int(support[1]),
        "elevated_percent": 100 * float(support[1]) / len(truth),
        "accuracy": float((truth == predicted).mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "normal_precision": precision[0],
        "normal_recall": recall[0],
        "normal_f1": f1[0],
        "elevated_precision": precision[1],
        "elevated_recall": recall[1],
        "elevated_f1": f1[1],
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def estimators_and_grids():
    """Four prespecified candidates per family, unchanged across horizons."""
    linear = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=3000)),
        ]
    )
    tree_grid, forest_grid, xgb_grid = [], [], []
    for weight in [None, "balanced"]:
        for depth, leaf, split in [(4, 20, 20), (8, 10, 10)]:
            tree_grid.append(
                {
                    "max_depth": [depth],
                    "min_samples_leaf": [leaf],
                    "min_samples_split": [split],
                    "class_weight": [weight],
                }
            )
        for count, depth, leaf in [(120, 10, 5), (160, 16, 2)]:
            forest_grid.append(
                {
                    "n_estimators": [count],
                    "max_depth": [depth],
                    "min_samples_leaf": [leaf],
                    "max_features": [0.7],
                    "class_weight": [weight],
                }
            )
    for weight in [1, 3]:
        for count, depth, rate, penalty in [(160, 2, 0.07, 5), (220, 4, 0.05, 8)]:
            xgb_grid.append(
                {
                    "n_estimators": [count],
                    "max_depth": [depth],
                    "learning_rate": [rate],
                    "reg_lambda": [penalty],
                    "scale_pos_weight": [weight],
                }
            )
    return {
        "Logistic Regression": (
            linear,
            {
                "classifier__C": [0.1, 1.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), tree_grid),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=4),
            forest_grid,
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
            ),
            xgb_grid,
        ),
    }


def read_data(horizon):
    """Change only y: threshold the existing exact-future continuous targets."""
    data = prepare_controlled(horizon, "compact")
    X, y, Xv, yv, Xt, yt, summary = data
    source = load_source(DATA_PATH)
    for frame, target in [(X, y), (Xv, yv), (Xt, yt)]:
        np.testing.assert_array_equal(
            target, source.loc[frame.index + pd.Timedelta(hours=horizon), "VHM0"]
        )
    regression = pd.read_csv(
        ROOT
        / f"experiments/controlled_regression/{horizon}_hour/compact/test_predictions.csv",
        index_col="timestamp",
        float_precision="round_trip",
    )
    regression.index = pd.DatetimeIndex(
        pd.to_datetime(regression.index, utc=True)
    ).as_unit("ns")
    pd.testing.assert_index_equal(Xt.index, regression.index)
    np.testing.assert_array_equal(yt, regression.actual_VHM0)
    np.testing.assert_array_equal(Xt.VHM0, regression.Persistence)
    return (X, binary(y), Xv, binary(yv), Xt, binary(yt), summary), regression


def start_run():
    """Fingerprint all previous files and freeze the design before fitting."""
    manifest_path = OUTPUT / "original_files_sha256.json"
    if manifest_path.exists():
        return
    manifest = {}
    for base, directories, filenames in os.walk(ROOT):
        directories[:] = [
            d
            for d in directories
            if d
            not in {
                ".git",
                ".venv",
                "__pycache__",
                ".ruff_cache",
                ".pytest_cache",
                ".ipynb_checkpoints",
            }
            and Path(base, d) != OUTPUT
        ]
        for filename in filenames:
            path = Path(base, filename)
            manifest[str(path.relative_to(ROOT))] = sha(path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    definitions = {
        "horizons_hours": HORIZONS,
        "height_threshold_m": 2.5,
        "probability_threshold": 0.5,
        "selection": "Highest mean training-only CV Elevated F1; lower sample std then earlier candidate on ties.",
        "weighting": "balanced class weights calculated separately during each fit; XGBoost compares fixed scale_pos_weight=1 and 3. No full-training ratio leaks into folds.",
        "candidates": {
            name: list(ParameterGrid(grid))
            for name, (_, grid) in estimators_and_grids().items()
        },
        "feature_names": prepare_controlled(72, "compact")[0].columns.tolist(),
        "source_sha256": sha(DATA_PATH),
        "study_status": "Exploratory: earlier test results informed experimental questions; no independent confirmatory claim.",
        "versions": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "xgboost",
                "joblib",
                "matplotlib",
                "threadpoolctl",
            ]
        },
    }
    (OUTPUT / "experiment_definition.json").write_text(
        json.dumps(definitions, indent=2)
    )
    source = load_source(DATA_PATH)
    changed = source.copy()
    changed.iloc[15001:] *= 1.1
    pd.testing.assert_frame_equal(
        make_features(source).iloc[:15001], make_features(changed).iloc[:15001]
    )
    print(
        "Preservation manifest and design frozen. Future-perturbation causality check passed.",
        flush=True,
    )


def train(horizon):
    """Tune on training only, save validation predictions, then freeze final refits."""
    directory = OUTPUT / f"{horizon}_hour"
    if (directory / "model_metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite fitted classifiers: {directory}")
    directory.mkdir(exist_ok=True)
    (directory / "models").mkdir(exist_ok=True)
    (X, y, Xv, yv, _, _, summary), _ = read_data(horizon)
    folds = list(TimeSeriesSplit(n_splits=3, gap=horizon // 3).split(X))
    for training, validation in folds:
        assert (
            X.index[training[-1]] + pd.Timedelta(hours=horizon) < X.index[validation[0]]
        )
    tuning, selected, validation_rows = [], {}, []
    validation_predictions = pd.DataFrame(
        {"actual_label": yv, "Persistence_label": binary(Xv.VHM0)}, index=Xv.index
    )
    validation_rows.append({"method": "Persistence", **metrics(yv, binary(Xv.VHM0))})
    metadata = {
        **summary,
        "feature_names": X.columns.tolist(),
        "probability_threshold": 0.5,
        "source_sha256": sha(DATA_PATH),
        "cv_gap": horizon // 3,
        "refit_rows": len(X) + len(Xv),
        "models": {},
    }
    for name, (estimator, grid) in estimators_and_grids().items():
        candidates, candidate_rows = [], []
        for candidate, parameters in enumerate(ParameterGrid(grid)):
            rows = []
            for fold, (training, validation) in enumerate(folds, start=1):
                model = clone(estimator).set_params(**parameters)
                model.fit(X.iloc[training], y[training])
                probability = model.predict_proba(X.iloc[validation])[:, 1]
                rows.append(
                    {
                        "method": name,
                        "candidate": candidate,
                        "parameters": json.dumps(parameters),
                        "fold": fold,
                        "training_end": str(X.index[training[-1]]),
                        "validation_start": str(X.index[validation[0]]),
                        **metrics(y[validation], (probability >= 0.5).astype(int)),
                    }
                )
            scores = np.array([row["elevated_f1"] for row in rows])
            mean, std = float(scores.mean()), float(scores.std(ddof=1))
            candidates.append((mean, std, candidate, parameters))
            candidate_rows.extend(
                {**row, "mean_cv_elevated_f1": mean, "std_cv_elevated_f1": std}
                for row in rows
            )
        best = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
        tuning.extend(
            {**row, "selected": row["candidate"] == best[2]} for row in candidate_rows
        )
        selected[name] = clone(estimator).set_params(**best[3])
        selected[name].fit(X, y)
        probability = selected[name].predict_proba(Xv)[:, 1].astype("float64")
        labels = (probability >= 0.5).astype(int)
        validation_predictions[name + "_probability"] = probability
        validation_predictions[name + "_label"] = labels
        validation_rows.append({"method": name, **metrics(yv, labels)})
        metadata["models"][name] = {
            "selected_parameters": best[3],
            "mean_cv_elevated_f1": best[0],
            "std_cv_elevated_f1": best[1],
            "candidate_count": len(candidates),
        }
        print("SELECTED", horizon, name, metadata["models"][name], flush=True)
    pd.DataFrame(tuning).to_csv(directory / "cv_results.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(
        directory / "validation_metrics.csv", index=False
    )
    validation_predictions.to_csv(
        directory / "validation_predictions.csv", index_label="timestamp"
    )
    combined_X, combined_y = pd.concat([X, Xv]), np.concatenate([y, yv])
    assert combined_X.index[-1] + pd.Timedelta(hours=horizon) < pd.Timestamp(
        summary["nominal_test_start"]
    )
    for name, estimator in selected.items():
        final = clone(estimator)
        final.fit(combined_X, combined_y)
        path = directory / "models" / FILENAMES[name]
        joblib.dump(final, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(
            final.predict_proba(X.head(32)), loaded.predict_proba(X.head(32))
        )
        assert list(loaded.feature_names_in_) == X.columns.tolist()
        metadata["models"][name].update(
            {"file": "models/" + FILENAMES[name], "sha256": sha(path)}
        )
    (directory / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("TRAINING COMPLETE", horizon, "No direct test predictions made.", flush=True)


def evaluate(horizon):
    """Load frozen classifiers and pair results with existing regression predictions."""
    directory = OUTPUT / f"{horizon}_hour"
    (X, _, _, _, Xt, yt, _), regression = read_data(horizon)
    metadata = json.loads((directory / "model_metadata.json").read_text())
    assert metadata["source_sha256"] == sha(DATA_PATH)
    predictions = pd.DataFrame(
        {
            "actual_height": regression.actual_VHM0,
            "actual_label": yt,
            "target_timestamp": Xt.index + pd.Timedelta(hours=horizon),
            "Persistence_label": binary(Xt.VHM0),
        },
        index=Xt.index,
    )
    for name, info in metadata["models"].items():
        path = directory / info["file"]
        assert sha(path) == info["sha256"]
        model = joblib.load(path)
        assert (
            list(model.feature_names_in_)
            == X.columns.tolist()
            == metadata["feature_names"]
        )
        probability = model.predict_proba(Xt)[:, 1].astype("float64")
        predictions[name + "_probability"] = probability
        predictions[name + "_label"] = (probability >= 0.5).astype(int)
    direct, derived, comparisons, matrices = [], [], [], []
    for name, regressor in PAIRS.items():
        result = metrics(yt, predictions[name + "_label"])
        reference = metrics(yt, binary(regression[regressor]))
        direct.append({"horizon_hours": horizon, "method": name, **result})
        derived.append(
            {
                "horizon_hours": horizon,
                "method": name,
                "regressor": regressor,
                **reference,
            }
        )
        comparison = {"horizon_hours": horizon, "method": name, "regressor": regressor}
        for metric in [
            "elevated_precision",
            "elevated_recall",
            "elevated_f1",
            "macro_f1",
            "fp",
            "fn",
        ]:
            comparison.update(
                {
                    "direct_" + metric: result[metric],
                    "derived_" + metric: reference[metric],
                    "change_" + metric: result[metric] - reference[metric],
                }
            )
        comparisons.append(comparison)
        matrix = confusion_matrix(yt, predictions[name + "_label"], labels=[0, 1])
        matrices.append((name, matrix))
    predictions.to_csv(directory / "test_predictions.csv", index_label="timestamp")
    pd.DataFrame(direct).to_csv(directory / "test_metrics.csv", index=False)
    pd.DataFrame(derived).to_csv(
        directory / "regression_derived_metrics.csv", index=False
    )
    pd.DataFrame(comparisons).to_csv(directory / "comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": name,
                "actual": actual,
                "predicted": predicted,
                "count": int(matrix[actual, predicted]),
            }
            for name, matrix in matrices
            for actual in [0, 1]
            for predicted in [0, 1]
        ]
    ).to_csv(directory / "confusion_matrices.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    maximum = max(matrix.max() for _, matrix in matrices)
    for ax, (name, matrix) in zip(axes.flat, matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                color="white" if count > maximum / 2 else "black",
            )
        ax.set(
            title=name,
            xlabel="Predicted",
            ylabel="Actual",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Elevated"],
            yticklabels=["Normal", "Elevated"],
        )
    axes.flat[-1].axis("off")
    fig.suptitle(f"{horizon}h exact target: direct classifiers at probability >= 0.5")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    print(
        "TEST",
        horizon,
        pd.DataFrame(direct)[
            [
                "method",
                "elevated_precision",
                "elevated_recall",
                "elevated_f1",
                "macro_f1",
                "fp",
                "fn",
            ]
        ].to_string(index=False),
        flush=True,
    )


def consolidate():
    """Concatenate results and save rankings without selecting new configurations."""
    for name in ["test_metrics", "regression_derived_metrics", "comparison"]:
        combined = pd.concat(
            [pd.read_csv(OUTPUT / f"{h}_hour/{name}.csv") for h in HORIZONS],
            ignore_index=True,
        )
        combined.to_csv(OUTPUT / f"{name}.csv", index=False)
    direct = pd.read_csv(OUTPUT / "test_metrics.csv")
    ranks = []
    for horizon, group in direct.groupby("horizon_hours"):
        for metric in ["elevated_f1", "elevated_recall", "macro_f1"]:
            for row in group.assign(
                rank=group[metric].rank(method="min", ascending=False)
            ).itertuples():
                ranks.append(
                    {
                        "horizon_hours": horizon,
                        "metric": metric,
                        "method": row.method,
                        "rank": int(row.rank),
                        "score": getattr(row, metric),
                    }
                )
    pd.DataFrame(ranks).to_csv(OUTPUT / "rankings.csv", index=False)


def verify():
    """Audit preservation, reloads, labels, CV arithmetic and independent scores."""
    manifest = json.loads((OUTPUT / "original_files_sha256.json").read_text())
    for relative, expected in manifest.items():
        assert sha(ROOT / relative) == expected, relative
    definition = json.loads((OUTPUT / "experiment_definition.json").read_text())
    for package, expected in definition["versions"].items():
        assert version(package) == expected
    comparison = pd.read_csv(OUTPUT / "comparison.csv")
    model_count = 0
    for horizon in HORIZONS:
        directory = OUTPUT / f"{horizon}_hour"
        (X, _, Xv, yv, Xt, yt, _), regression = read_data(horizon)
        assert X.columns.tolist() == definition["feature_names"]
        metadata = json.loads((directory / "model_metadata.json").read_text())
        for filename, xx, truth in [
            ("test_predictions.csv", Xt, yt),
            ("validation_predictions.csv", Xv, yv),
        ]:
            frame = pd.read_csv(
                directory / filename,
                index_col="timestamp",
                float_precision="round_trip",
            )
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).as_unit("ns"),
                xx.index,
            )
            np.testing.assert_array_equal(frame.actual_label, truth)
            saved = pd.read_csv(
                directory
                / (
                    "test_metrics.csv"
                    if filename.startswith("test")
                    else "validation_metrics.csv"
                )
            ).set_index("method")
            for name in PAIRS:
                predicted = frame[name + "_label"].to_numpy()
                if name != "Persistence":
                    np.testing.assert_array_equal(
                        predicted,
                        (frame[name + "_probability"].to_numpy() >= 0.5).astype(int),
                    )
                tp = int(((truth == 1) & (predicted == 1)).sum())
                fp = int(((truth == 0) & (predicted == 1)).sum())
                tn = int(((truth == 0) & (predicted == 0)).sum())
                fn = int(((truth == 1) & (predicted == 0)).sum())

                def divide(a, b):
                    return a / b if b else 0.0

                p, r = divide(tp, tp + fp), divide(tp, tp + fn)
                np_, nr = divide(tn, tn + fn), divide(tn, tn + fp)
                f1, nf1 = divide(2 * p * r, p + r), divide(2 * np_ * nr, np_ + nr)
                calculated = {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "accuracy": (tp + tn) / len(truth),
                    "balanced_accuracy": np.mean(
                        [
                            value
                            for value, support in [(r, tp + fn), (nr, tn + fp)]
                            if support
                        ]
                    ),
                    "normal_precision": np_,
                    "normal_recall": nr,
                    "normal_f1": nf1,
                    "elevated_precision": p,
                    "elevated_recall": r,
                    "elevated_f1": f1,
                    "macro_f1": (f1 + nf1) / 2,
                    "weighted_f1": ((tp + fn) * f1 + (tn + fp) * nf1) / len(truth),
                }
                for metric, value in calculated.items():
                    np.testing.assert_allclose(
                        saved.loc[name, metric], value, rtol=1e-12, atol=1e-12
                    )
                if filename.startswith("test") and name != "Persistence":
                    info = metadata["models"][name]
                    path = directory / info["file"]
                    assert sha(path) == info["sha256"]
                    model = joblib.load(path)
                    assert list(model.feature_names_in_) == X.columns.tolist()
                    np.testing.assert_allclose(
                        model.predict_proba(Xt)[:, 1],
                        frame[name + "_probability"],
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    model_count += 1
        derived = pd.read_csv(directory / "regression_derived_metrics.csv").set_index(
            "method"
        )
        direct = pd.read_csv(directory / "test_metrics.csv").set_index("method")
        matrices = pd.read_csv(directory / "confusion_matrices.csv")
        for name, regressor in PAIRS.items():
            prediction = binary(regression[regressor])
            tp = int(((yt == 1) & (prediction == 1)).sum())
            fp = int(((yt == 0) & (prediction == 1)).sum())
            tn = int(((yt == 0) & (prediction == 0)).sum())
            fn = int(((yt == 1) & (prediction == 0)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            nf1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
            independent = {
                "elevated_precision": p,
                "elevated_recall": r,
                "elevated_f1": f1,
                "macro_f1": (f1 + nf1) / 2,
                "fp": fp,
                "fn": fn,
            }
            paired = comparison.loc[
                comparison.horizon_hours.eq(horizon) & comparison.method.eq(name)
            ].iloc[0]
            for metric, value in independent.items():
                np.testing.assert_allclose(
                    derived.loc[name, metric], value, atol=1e-12, rtol=1e-12
                )
                np.testing.assert_allclose(
                    paired["derived_" + metric], value, atol=1e-12, rtol=1e-12
                )
                np.testing.assert_allclose(
                    paired["direct_" + metric],
                    direct.loc[name, metric],
                    atol=1e-12,
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    paired["change_" + metric],
                    direct.loc[name, metric] - value,
                    atol=1e-12,
                    rtol=1e-12,
                )
            saved_matrix = matrices.loc[matrices.method.eq(name)].pivot(
                index="actual", columns="predicted", values="count"
            )
            np.testing.assert_array_equal(
                saved_matrix,
                [
                    [direct.loc[name, "tn"], direct.loc[name, "fp"]],
                    [direct.loc[name, "fn"], direct.loc[name, "tp"]],
                ],
            )
            if name == "Persistence":
                pd.testing.assert_series_equal(
                    direct.loc[name],
                    derived.loc[name].drop("regressor"),
                    check_dtype=False,
                )
        cv = pd.read_csv(directory / "cv_results.csv", float_precision="round_trip")
        for name, group in cv.groupby("method"):
            ranking = []
            for candidate, rows in group.groupby("candidate"):
                assert len(rows) == 3
                mean, std = rows.elevated_f1.mean(), rows.elevated_f1.std(ddof=1)
                np.testing.assert_allclose(rows.mean_cv_elevated_f1, mean, atol=1e-12)
                np.testing.assert_allclose(rows.std_cv_elevated_f1, std, atol=1e-12)
                assert (
                    pd.to_datetime(rows.training_end, utc=True)
                    + pd.Timedelta(hours=horizon)
                    < pd.to_datetime(rows.validation_start, utc=True)
                ).all()
                ranking.append((-mean, std, candidate))
            best = min(ranking)[2]
            assert group.loc[group.selected, "candidate"].eq(best).all()
            assert len(group.loc[group.selected]) == 3
    tree = ast.parse(Path(__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in [
            "evaluate",
            "verify",
            "consolidate",
            "read_data",
        ]:
            assert not any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in ["fit", "fit_transform", "fit_predict"]
                for call in ast.walk(node)
            )
    assert model_count == 16
    report = {
        "status": "passed",
        "horizons": HORIZONS,
        "models_reloaded": model_count,
        "original_files_unchanged": len(manifest),
        "dataset_sha256": sha(DATA_PATH),
        "independent_test_and_validation_scores": True,
        "independent_derived_comparisons_and_saved_confusion_counts": True,
        "evaluation_contains_no_fit_calls": True,
        "fixed_feature_count": 18,
        "exact_future_targets_and_regression_test_origins_identical": True,
        "threshold": 0.5,
        "test_based_tuning": False,
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["run", "train", "evaluate", "verify"], default="run"
    )
    stage = parser.parse_args().stage
    with threadpool_limits(limits=4):
        if stage in ["run", "train"]:
            start_run()
            for horizon in HORIZONS:
                train(horizon)
        if stage in ["run", "evaluate"]:
            for horizon in HORIZONS:
                assert (OUTPUT / f"{horizon}_hour/model_metadata.json").exists(), (
                    "Complete every training configuration before testing."
                )
            for horizon in HORIZONS:
                evaluate(horizon)
            consolidate()
        if stage in ["run", "evaluate", "verify"]:
            verify()


if __name__ == "__main__":
    main()
