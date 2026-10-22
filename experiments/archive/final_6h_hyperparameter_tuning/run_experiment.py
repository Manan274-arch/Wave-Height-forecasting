"""Isolated six-hour regression searches and nested chronological early stopping."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
OLD = ROOT / "experiments/controlled_regression/6_hour/compact"
sys.path.insert(0, str(ROOT / "src"))
from controlled_pipeline import prepare_controlled  # noqa: E402
from feature_pipeline import DATA_PATH, load_source  # noqa: E402
from regression_pipeline import regression_metrics  # noqa: E402

FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}
EARLY_ROUNDS = 40


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.item()), encoding="utf-8"
    )


def binary_metrics(truth, prediction):
    """Threshold unmodified heights at 2.5 m, with Elevated positive."""
    actual, predicted = (
        (np.asarray(truth) >= 2.5).astype(int),
        (np.asarray(prediction) >= 2.5).astype(int),
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float((actual == predicted).mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "elevated_precision": precision[1],
        "elevated_recall": recall[1],
        "elevated_f1": f1[1],
        "macro_f1": float(f1.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def data():
    """Reuse all original compact X/y values and chronological exclusions."""
    objects = prepare_controlled(6, "compact")
    source = load_source(DATA_PATH)
    for position in [0, 2, 4]:
        X, y = objects[position : position + 2]
        np.testing.assert_array_equal(
            y, source.loc[X.index + pd.Timedelta(hours=6), "VHM0"]
        )
    assert [len(objects[i]) for i in [0, 2, 4]] == [20446, 2920, 5844]
    assert objects[0].shape[1] == 18
    return objects


def estimators():
    return {
        "Linear Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("regressor", LinearRegression(fit_intercept=True)),
            ]
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=4),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="rmse",
        ),
    }


def initial_candidates():
    """Prespecified bounded spaces plus previous settings as incumbent anchors."""
    previous = json.loads((OLD / "model_metadata.json").read_text())["models"]
    tree = {
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 16, None],
        "min_samples_split": [2, 5, 10, 20, 40],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": [None, 0.7, 0.9],
        "ccp_alpha": [0, 0.00001, 0.0001, 0.001],
    }
    forest = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 8, 10, 12, 16, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 3, 5, 10],
        "max_features": [0.5, 0.7, 0.9, 1.0],
    }
    xgb = {
        "n_estimators": randint(100, 1001),
        "max_depth": randint(2, 8),
        "learning_rate": loguniform(0.01, 0.15),
        "subsample": uniform(0.65, 0.35),
        "colsample_bytree": uniform(0.65, 0.35),
        "min_child_weight": randint(1, 13),
        "reg_lambda": loguniform(0.1, 30),
        "reg_alpha": [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2],
        "gamma": [0, 0, 0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1],
    }
    candidates = {"Linear Regression": [{}]}
    for name, space, count in [
        ("Decision Tree", tree, 40),
        ("Random Forest", forest, 24),
        ("XGBoost", xgb, 59),
    ]:
        candidates[name] = [previous[name]["selected_parameters"]] + list(
            ParameterSampler(space, n_iter=count, random_state=42)
        )
    # Expand the XGBoost incumbent defaults so refinement has all nine search keys.
    candidates["XGBoost"][0] = {
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0,
        "gamma": 0,
        **candidates["XGBoost"][0],
    }
    return candidates


def focused_candidates(promising):
    """Twenty seeded neighborhoods around three strongest broad candidates."""
    rng = np.random.default_rng(43)
    result = []
    for i in range(20):
        base = promising[i % len(promising)]
        result.append(
            {
                "n_estimators": int(
                    np.clip(
                        round(base["n_estimators"] * rng.uniform(0.6, 1.5)), 100, 1000
                    )
                ),
                "max_depth": int(
                    np.clip(base["max_depth"] + rng.integers(-2, 3), 2, 7)
                ),
                "learning_rate": float(
                    np.clip(base["learning_rate"] * rng.uniform(0.5, 1.8), 0.01, 0.15)
                ),
                "subsample": float(
                    np.clip(base["subsample"] + rng.normal(0, 0.12), 0.65, 1)
                ),
                "colsample_bytree": float(
                    np.clip(base["colsample_bytree"] + rng.normal(0, 0.12), 0.65, 1)
                ),
                "min_child_weight": int(
                    np.clip(base["min_child_weight"] + rng.integers(-4, 5), 1, 12)
                ),
                "reg_lambda": float(
                    np.clip(base["reg_lambda"] * rng.uniform(0.3, 3), 0.1, 30)
                ),
                "reg_alpha": 0.0
                if rng.random() < 0.5
                else float(np.clip(base["reg_alpha"] + rng.normal(0, 0.35), 0, 2)),
                "gamma": 0.0
                if rng.random() < 0.5
                else float(np.clip(base["gamma"] + rng.normal(0, 0.12), 0, 1)),
            }
        )
    return result


def stopping_split(X):
    """Use the last 15% of an outer training partition for early stopping."""
    position = int(np.floor(0.85 * len(X)))
    stopping_start = X.index[position]
    core = np.flatnonzero(X.index + pd.Timedelta(hours=6) < stopping_start)
    stopping = np.arange(position, len(X))
    assert position - len(core) == 2
    assert X.index[core[-1]] + pd.Timedelta(hours=6) < X.index[stopping[0]]
    return core, stopping


def search(name, candidates, stage, X, y, folds, offset=0):
    """Score untouched chronological folds; XGBoost stops on nested training tails."""
    rows, summaries = [], []
    started = time.perf_counter()
    for index, parameters in enumerate(candidates, start=offset):
        fold_rows = []
        for fold, (training, scoring) in enumerate(folds, start=1):
            outer_X, outer_y = X.iloc[training], y.iloc[training]
            assert outer_X.index[-1] + pd.Timedelta(hours=6) < X.index[scoring[0]]
            model = clone(estimators()[name]).set_params(**parameters)
            detail = {}
            if name == "XGBoost":
                core, stopping = stopping_split(outer_X)
                model.set_params(early_stopping_rounds=EARLY_ROUNDS)
                model.fit(
                    outer_X.iloc[core],
                    outer_y.iloc[core],
                    eval_set=[(outer_X.iloc[stopping], outer_y.iloc[stopping])],
                    verbose=False,
                )
                detail = {
                    "core_rows": len(core),
                    "early_stopping_rows": len(stopping),
                    "core_end": str(outer_X.index[core[-1]]),
                    "early_stopping_start": str(outer_X.index[stopping[0]]),
                    "early_stopping_end": str(outer_X.index[stopping[-1]]),
                    "best_iteration": int(model.best_iteration),
                    "best_rounds": int(model.best_iteration) + 1,
                    "best_early_stopping_rmse": float(model.best_score),
                    "fitted_rounds": model.get_booster().num_boosted_rounds(),
                }
            else:
                model.fit(outer_X, outer_y)
            prediction = model.predict(X.iloc[scoring])
            score = regression_metrics(y.iloc[scoring], prediction)
            fold_rows.append(
                {
                    "model": name,
                    "stage": stage,
                    "candidate": index,
                    "parameters": json.dumps(parameters, default=lambda x: x.item()),
                    "fold": fold,
                    "training_end": str(outer_X.index[-1]),
                    "scoring_start": str(X.index[scoring[0]]),
                    "scoring_end": str(X.index[scoring[-1]]),
                    "fold_rmse": score["rmse"],
                    "fold_mae": score["mae"],
                    **detail,
                }
            )
        scores = np.array([row["fold_rmse"] for row in fold_rows])
        mean, std = float(scores.mean()), float(scores.std(ddof=1))
        rows.extend(
            {**row, "mean_cv_rmse": mean, "std_cv_rmse": std} for row in fold_rows
        )
        summaries.append(
            {
                "model": name,
                "stage": stage,
                "candidate": index,
                "parameters": parameters,
                "mean_cv_rmse": mean,
                "std_cv_rmse": std,
            }
        )
        if (index - offset + 1) % 5 == 0 or index == offset + len(candidates) - 1:
            print(
                "SEARCH",
                name,
                stage,
                index - offset + 1,
                "/",
                len(candidates),
                "best CV RMSE",
                min(s["mean_cv_rmse"] for s in summaries),
                "elapsed seconds",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
    return rows, summaries


def initialize():
    """Preserve checksums and verify prior metrics without altering old artifacts."""
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
    write_json(OUTPUT / "original_files_sha256.json", manifest)
    X, y, Xv, yv, Xt, yt, summary = data()
    previous = pd.read_csv(
        OLD / "test_predictions.csv",
        index_col="timestamp",
        float_precision="round_trip",
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(pd.to_datetime(previous.index, utc=True)).as_unit("ns"),
        Xt.index,
    )
    np.testing.assert_array_equal(previous.actual_VHM0, yt)
    saved = pd.read_csv(OLD / "test_results.csv").set_index("family")
    for name in FILES:
        for metric, value in regression_metrics(yt, previous[name]).items():
            np.testing.assert_allclose(
                saved.loc[name, metric], value, rtol=1e-12, atol=1e-12
            )
    candidates = initial_candidates()
    write_json(
        OUTPUT / "design.json",
        {
            "features": X.columns.tolist(),
            "partitions": summary,
            "target": "VHM0(t+6h), not a future maximum",
            "cv_gap": 2,
            "cv_splits": 3,
            "early_stopping_rounds": EARLY_ROUNDS,
            "inner_training_fraction": 0.85,
            "candidate_lists": candidates,
            "refinement_trials": 20,
            "selection": "Lowest training-only CV mean RMSE; lower sample std then earlier candidate on ties.",
            "source_sha256": sha(DATA_PATH),
            "versions": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "xgboost",
                    "joblib",
                    "threadpoolctl",
                ]
            },
        },
    )
    print(
        "Original metrics verified; design and preservation manifest frozen.",
        flush=True,
    )
    return candidates


def train():
    """Finish searches, validation and four refits before any new test prediction."""
    if (OUTPUT / "model_metadata.json").exists():
        raise FileExistsError("Refusing to overwrite completed tuned models.")
    candidates = initialize()
    X, y, Xv, yv, _, _, summary = data()
    folds = list(TimeSeriesSplit(n_splits=3, gap=2).split(X))
    all_rows, all_summaries = [], []
    selected = {}
    for name in FILES:
        rows, summaries = search(
            name,
            candidates[name],
            "A" if name == "XGBoost" else "randomized",
            X,
            y,
            folds,
        )
        if name == "XGBoost":
            top = sorted(
                summaries,
                key=lambda s: (s["mean_cv_rmse"], s["std_cv_rmse"], s["candidate"]),
            )[:3]
            focused = focused_candidates([s["parameters"] for s in top])
            write_json(
                OUTPUT / "xgboost_refinement_candidates.json",
                {"parent_candidates": top, "candidates": focused},
            )
            focused_rows, focused_summaries = search(
                name, focused, "B", X, y, folds, offset=len(candidates[name])
            )
            rows += focused_rows
            summaries += focused_summaries
        selected[name] = min(
            summaries,
            key=lambda s: (s["mean_cv_rmse"], s["std_cv_rmse"], s["candidate"]),
        )
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        pd.DataFrame(all_rows).to_csv(OUTPUT / "cv_results.csv", index=False)
        write_json(OUTPUT / "candidate_results.json", all_summaries)
        print("SELECTED", name, selected[name], flush=True)
    validation_predictions = pd.DataFrame({"actual_VHM0": yv}, index=Xv.index)
    validation_rows, final_settings, stopping_results = [], {}, []
    for name, info in selected.items():
        parameters = info["parameters"].copy()
        model = clone(estimators()[name]).set_params(**parameters)
        if name == "XGBoost":
            core, stopping = stopping_split(X)
            model.set_params(early_stopping_rounds=EARLY_ROUNDS)
            model.fit(
                X.iloc[core],
                y.iloc[core],
                eval_set=[(X.iloc[stopping], y.iloc[stopping])],
                verbose=False,
            )
            training_rounds = int(model.best_iteration) + 1
            stopping_results.append(
                {
                    "purpose": "training_tail_round_selection",
                    "best_iteration": int(model.best_iteration),
                    "best_rounds": training_rounds,
                    "fitted_rounds": model.get_booster().num_boosted_rounds(),
                    "fit_end": str(X.index[core[-1]]),
                    "stopping_start": str(X.index[stopping[0]]),
                    "stopping_end": str(X.index[stopping[-1]]),
                    "best_rmse": float(model.best_score),
                }
            )
            model = clone(estimators()[name]).set_params(
                **{**parameters, "n_estimators": training_rounds}
            )
            model.fit(X, y)
            no_stopping = clone(estimators()[name]).set_params(**parameters)
            no_stopping.fit(X, y)
            no_stopping_prediction = no_stopping.predict(Xv).astype("float64")
            validation_predictions["XGBoost_no_early_stopping"] = no_stopping_prediction
            calibration = clone(estimators()[name]).set_params(
                **parameters, early_stopping_rounds=EARLY_ROUNDS
            )
            calibration.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
            final_rounds = int(calibration.best_iteration) + 1
            validation_predictions["XGBoost_validation_calibration"] = (
                calibration.predict(Xv).astype("float64")
            )
            stopping_results.append(
                {
                    "purpose": "original_validation_final_round_selection",
                    "best_iteration": int(calibration.best_iteration),
                    "best_rounds": final_rounds,
                    "fitted_rounds": calibration.get_booster().num_boosted_rounds(),
                    "fit_end": str(X.index[-1]),
                    "stopping_start": str(Xv.index[0]),
                    "stopping_end": str(Xv.index[-1]),
                    "best_rmse": float(calibration.best_score),
                    "no_early_stopping_validation_rmse": regression_metrics(
                        yv, no_stopping_prediction
                    )["rmse"],
                }
            )
            parameters["n_estimators"] = final_rounds
        else:
            model.fit(X, y)
        prediction = model.predict(Xv).astype("float64")
        validation_predictions[name] = prediction
        validation_rows.append({"model": name, **regression_metrics(yv, prediction)})
        final_settings[name] = parameters
    pd.DataFrame(validation_rows).to_csv(OUTPUT / "validation_metrics.csv", index=False)
    validation_predictions.to_csv(
        OUTPUT / "validation_predictions.csv", index_label="timestamp"
    )
    pd.DataFrame(stopping_results).to_csv(
        OUTPUT / "early_stopping_summary.csv", index=False
    )
    pd.DataFrame([row for row in all_rows if row["model"] == "XGBoost"]).to_csv(
        OUTPUT / "xgboost_early_stopping_cv.csv", index=False
    )
    metadata = {
        "summary": summary,
        "feature_names": X.columns.tolist(),
        "refit_rows": len(X) + len(Xv),
        "source_sha256": sha(DATA_PATH),
        "test_used_for_selection": False,
        "models": {},
    }
    (OUTPUT / "models").mkdir(exist_ok=True)
    for name, parameters in final_settings.items():
        final = clone(estimators()[name]).set_params(**parameters)
        final.fit(pd.concat([X, Xv]), pd.concat([y, yv]))
        path = OUTPUT / "models" / FILES[name]
        joblib.dump(final, path)
        loaded = joblib.load(path)
        np.testing.assert_allclose(
            loaded.predict(X.head(32)), final.predict(X.head(32)), rtol=0, atol=1e-12
        )
        metadata["models"][name] = {
            **selected[name],
            "final_parameters": parameters,
            "file": "models/" + FILES[name],
            "sha256": sha(path),
        }
        print("FROZEN", name, parameters, flush=True)
    write_json(OUTPUT / "model_metadata.json", metadata)


def evaluate():
    """Only reload/predict; never fit or change choices using test errors."""
    _, _, _, _, X, y, _ = data()
    metadata = json.loads((OUTPUT / "model_metadata.json").read_text())
    assert sha(DATA_PATH) == metadata["source_sha256"]
    design = json.loads((OUTPUT / "design.json").read_text())
    for package, expected in design["versions"].items():
        assert version(package) == expected
    previous = pd.read_csv(
        OLD / "test_predictions.csv",
        index_col="timestamp",
        float_precision="round_trip",
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(pd.to_datetime(previous.index, utc=True)).as_unit("ns"),
        X.index,
    )
    np.testing.assert_array_equal(previous.actual_VHM0, y)
    predictions = pd.DataFrame(
        {"actual_VHM0": y, "target_timestamp": X.index + pd.Timedelta(hours=6)},
        index=X.index,
    )
    rows, classes, comparisons, class_comparisons = [], [], [], []
    for name, info in metadata["models"].items():
        path = OUTPUT / info["file"]
        assert sha(path) == info["sha256"]
        model = joblib.load(path)
        assert (
            list(model.feature_names_in_)
            == X.columns.tolist()
            == metadata["feature_names"]
        )
        prediction = model.predict(X).astype("float64")
        predictions[name] = prediction
        current = regression_metrics(y, prediction)
        old = regression_metrics(y, previous[name])
        rows.append({"model": name, **current})
        classifications = binary_metrics(y, prediction)
        original_classes = binary_metrics(y, previous[name])
        classes.append({"model": name, **classifications})
        comparisons.append(
            {
                "model": name,
                **{"old_" + k: old[k] for k in ["mae", "rmse", "r2"]},
                **{"tuned_" + k: current[k] for k in ["mae", "rmse", "r2"]},
                "rmse_improvement_percent": 100
                * (old["rmse"] - current["rmse"])
                / old["rmse"],
            }
        )
        class_comparisons.append(
            {
                "model": name,
                **{"old_" + k: v for k, v in original_classes.items()},
                **{"tuned_" + k: v for k, v in classifications.items()},
                **{
                    "change_" + k: classifications[k] - v
                    for k, v in original_classes.items()
                },
            }
        )
    predictions.to_csv(OUTPUT / "test_predictions.csv", index_label="timestamp")
    pd.DataFrame(rows).to_csv(OUTPUT / "test_regression_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(
        OUTPUT / "derived_classification_metrics.csv", index=False
    )
    pd.DataFrame(comparisons).to_csv(OUTPUT / "regression_comparison.csv", index=False)
    pd.DataFrame(class_comparisons).to_csv(
        OUTPUT / "classification_comparison.csv", index=False
    )
    paired = pd.DataFrame(comparisons).set_index("model")
    xgb_comparison = []
    for era in ["old", "tuned"]:
        for name in ["Linear Regression", "Random Forest"]:
            reference, xgb = (
                paired.loc[name, era + "_rmse"],
                paired.loc["XGBoost", era + "_rmse"],
            )
            xgb_comparison.append(
                {
                    "configuration": era,
                    "reference_model": name,
                    "reference_rmse": reference,
                    "xgboost_rmse": xgb,
                    "reference_minus_xgboost_rmse": reference - xgb,
                    "xgboost_advantage_percent": 100 * (reference - xgb) / reference,
                }
            )
    pd.DataFrame(xgb_comparison).to_csv(OUTPUT / "xgboost_comparison.csv", index=False)
    print(
        "TEST REGRESSION", pd.DataFrame(comparisons).to_string(index=False), flush=True
    )
    print(
        "TEST CLASSIFICATION", pd.DataFrame(classes).to_string(index=False), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run", "train", "evaluate"], default="run")
    stage = parser.parse_args().stage
    with threadpool_limits(limits=4):
        if stage in ["run", "train"]:
            train()
        if stage in ["run", "evaluate"]:
            evaluate()


if __name__ == "__main__":
    main()
