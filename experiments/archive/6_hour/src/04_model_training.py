"""Tune baselines on training CV, select strategies on validation, save four refits."""

import hashlib
import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from feature_pipeline import prepare_features
from model_metrics import metrics

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data/udupi_wave_wind_2015_2024.csv"
MODEL_DIR = ROOT_DIR / "models"


def smote_targets(labels):
    """Raise each eligible minority class to 10% of the fit-local majority."""
    counts = labels.value_counts().reindex([0, 1, 2], fill_value=0)
    if counts.min() < 4:
        raise ValueError("SMOTE with three neighbors requires four real examples.")
    goal = int(np.ceil(0.1 * counts.max()))
    return {label: goal for label in [1, 2] if counts[label] < goal}


def tune_baseline(name, model, grid, X_train, y_train, folds):
    """Choose by CV mean, then lower sample std, then earlier candidate order."""
    scorer = make_scorer(f1_score, labels=[0, 1, 2], average="macro", zero_division=0)
    search = GridSearchCV(
        model,
        grid,
        scoring=scorer,
        cv=folds,
        refit=False,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    results = search.cv_results_
    fold_scores = np.column_stack(
        [results[f"split{fold}_test_score"] for fold in range(3)]
    )
    means = fold_scores.mean(axis=1)
    standard_deviations = fold_scores.std(axis=1, ddof=1)
    best = max(
        range(len(means)),
        key=lambda index: (
            means[index],
            -standard_deviations[index],
            -index,
        ),
    )
    parameters = results["params"][best]
    tuned = clone(model).set_params(**parameters)
    rows = []
    for number, settings in enumerate(results["params"]):
        for fold, (train_positions, validation_positions) in enumerate(folds):
            rows.append(
                {
                    "family": name,
                    "configuration": number,
                    "parameters": json.dumps(settings),
                    "fold": fold + 1,
                    "macro_f1": fold_scores[number, fold],
                    "mean_macro_f1": means[number],
                    "std_macro_f1": standard_deviations[number],
                    "selected_baseline": number == best,
                    "training_rows": len(train_positions),
                    "validation_rows": len(validation_positions),
                }
            )
    summary = {
        "tuned_parameters": {
            key.removeprefix("classifier__"): value for key, value in parameters.items()
        },
        "cv_mean_macro_f1": float(means[best]),
        "cv_std_macro_f1": float(standard_deviations[best]),
    }
    print("TUNED BASELINE", name, summary, flush=True)
    return tuned, summary, rows


def compare_strategies(name, baseline, X_train, y_train, X_val, y_val):
    """Hold tuned hyperparameters fixed while changing imbalance handling."""
    baseline.fit(X_train, y_train)
    baseline_prediction = baseline.predict(X_val)

    weighted = clone(baseline)
    if isinstance(weighted, XGBClassifier):
        weighted.fit(
            X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train)
        )
    else:
        weight_parameter = "class_weight"
        if isinstance(weighted, Pipeline):
            weight_parameter = "classifier__class_weight"
        weighted.set_params(**{weight_parameter: "balanced"})
        weighted.fit(X_train, y_train)
    weighted_prediction = weighted.predict(X_val)

    classifier = baseline
    if isinstance(baseline, Pipeline):
        classifier = baseline.named_steps["classifier"]
    smote = ImbPipeline(
        [
            ("scale", StandardScaler()),
            (
                "smote",
                SMOTE(
                    sampling_strategy=smote_targets(y_train),
                    k_neighbors=3,
                    random_state=42,
                ),
            ),
            ("classifier", clone(classifier)),
        ]
    )
    smote.fit(X_train, y_train)
    smote_prediction = smote.predict(X_val)

    candidates = []
    # This order also defines exact validation Macro-F1 ties: baseline, weighting, SMOTE.
    for strategy, model, prediction in [
        ("baseline", baseline, baseline_prediction),
        ("class-weighted", weighted, weighted_prediction),
        ("SMOTE", smote, smote_prediction),
    ]:
        matrix = confusion_matrix(y_val, prediction, labels=[0, 1, 2])
        result = metrics(y_val, prediction)
        result["confusion_matrix"] = json.dumps(matrix.tolist())
        candidates.append(
            {"family": name, "strategy": strategy, "model": model, "metrics": result}
        )
        print(name, strategy, "validation confusion matrix:\n", matrix)
    return candidates


def refit_and_save(
    selected,
    summaries,
    X_train,
    y_train,
    X_val,
    y_val,
    test_start,
    validation_rows,
    tuning_rows,
):
    """Verify every refit in staging before replacing any valid model artifact."""
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    if len(X_combined) != 23365:
        raise ValueError("Expected 23,365 eligible refit observations.")
    if X_combined.index[-1] + pd.Timedelta(hours=6) >= test_start:
        raise ValueError("Refit targets overlap the test partition.")
    filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }
    metadata = {
        "methodology": "baseline-only CV tuning; fixed-parameter validation strategy selection",
        "selection": "CV: mean Macro-F1, lower sample std, earlier candidate. Validation: Macro-F1, then baseline/class-weighted/SMOTE order.",
        "feature_names": X_train.columns.tolist(),
        "classes": {
            "0": "Normal: below 2.5 m",
            "1": "Elevated: 2.5 through 3.0 m inclusive",
            "2": "High: above 3.0 m",
        },
        "target": "max(VHM0[t+3h], VHM0[t+6h]); provisional research thresholds",
        "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "cv": {
            "n_splits": 3,
            "gap": 2,
            "candidates_per_algorithm": 3,
            "baseline_only": True,
            "total_fold_fits": 36,
        },
        "refit_rows": len(X_combined),
        "refit_start": str(X_combined.index[0]),
        "refit_end": str(X_combined.index[-1]),
        "validation_start": str(X_val.index[0]),
        "validation_end": str(X_val.index[-1]),
        "test_used_for_selection": False,
        "models": {},
    }
    MODEL_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".phase4_", dir=MODEL_DIR) as temporary:
        stage = Path(temporary)
        for name, candidate in selected.items():
            model = clone(candidate["model"])
            strategy = candidate["strategy"]
            if strategy == "SMOTE":
                model.set_params(smote__sampling_strategy=smote_targets(y_combined))
            if isinstance(model, XGBClassifier) and strategy == "class-weighted":
                model.fit(
                    X_combined,
                    y_combined,
                    sample_weight=compute_sample_weight("balanced", y_combined),
                )
            else:
                model.fit(X_combined, y_combined)
            path = stage / filenames[name]
            joblib.dump(model, path)
            loaded = joblib.load(path)
            probe = X_train.head(32)
            np.testing.assert_array_equal(model.predict(probe), loaded.predict(probe))
            if list(loaded.feature_names_in_) != X_train.columns.tolist():
                raise ValueError("Saved feature names/order changed.")
            classifier = model
            if hasattr(model, "named_steps"):
                classifier = model.named_steps["classifier"]
            parameters = {
                key: value
                for key, value in classifier.get_params().items()
                if value is None
                or isinstance(value, (str, bool, int))
                or (isinstance(value, float) and np.isfinite(value))
            }
            metadata["models"][name] = {
                "file": filenames[name],
                "strategy": strategy,
                **summaries[name],
                "estimator_parameters": parameters,
                "validation_metrics_before_refit": candidate["metrics"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            print("Refit, saved and reloaded:", name, flush=True)
        (stage / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        pd.DataFrame(validation_rows).to_csv(
            stage / "validation_comparison.csv", index=False
        )
        pd.DataFrame(tuning_rows).to_csv(
            stage / "chronological_tuning.csv", index=False
        )
        # All four fits and reload checks succeeded before publication starts.
        for path in stage.iterdir():
            path.replace(MODEL_DIR / path.name)
    print("Published four verified refits. Phase 4 made no test predictions.")


def main():
    """Tune four baselines, compare twelve strategies, and save four selections."""
    warnings.simplefilter("error", ConvergenceWarning)
    with threadpool_limits(limits=4):
        X_train, y_train, X_val, y_val, X_test, _ = prepare_features(DATA_PATH)
        print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
        print("Training classes:\n", y_train.value_counts().sort_index())
        print("Validation classes:\n", y_val.value_counts().sort_index())
        folds = list(TimeSeriesSplit(n_splits=3, gap=2).split(X_train))
        for train_positions, validation_positions in folds:
            horizon = X_train.index[train_positions[-1]] + pd.Timedelta(hours=6)
            if horizon >= X_train.index[validation_positions[0]]:
                raise ValueError("CV labels cross the fold boundary.")
            if set(y_train.iloc[validation_positions]) != {0, 1, 2}:
                raise ValueError("CV validation fold lacks a class.")

        # Explicit baseline definitions preserve the established estimator settings.
        lr = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=5000,
                        tol=0.0001,
                        random_state=42,
                    ),
                ),
            ]
        )

        dt = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=12,
            min_samples_split=20,
            random_state=42,
        )

        rf = RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=4,
            random_state=42,
        )

        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=160,
            max_depth=3,
            learning_rate=0.07,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            n_jobs=4,
            random_state=42,
        )

        # Three complete candidate settings per algorithm; no weighted/SMOTE CV.
        lr_grid = [{}, {"classifier__C": [0.3]}, {"classifier__C": [3.0]}]
        dt_grid = [
            {},
            {"max_depth": [6], "min_samples_leaf": [20]},
            {"max_depth": [10], "min_samples_leaf": [8]},
        ]
        rf_grid = [
            {},
            {"max_depth": [10], "min_samples_leaf": [5]},
            {
                "n_estimators": [220],
                "max_depth": [16],
                "min_samples_leaf": [3],
                "max_features": [0.7],
            },
        ]
        xgb_grid = [
            {},
            {
                "n_estimators": [120],
                "max_depth": [2],
                "learning_rate": [0.1],
                "reg_lambda": [8],
            },
            {
                "n_estimators": [220],
                "max_depth": [4],
                "learning_rate": [0.05],
                "reg_lambda": [8],
            },
        ]
        lr, lr_summary, lr_cv = tune_baseline(
            "Logistic Regression", lr, lr_grid, X_train, y_train, folds
        )
        dt, dt_summary, dt_cv = tune_baseline(
            "Decision Tree", dt, dt_grid, X_train, y_train, folds
        )
        rf, rf_summary, rf_cv = tune_baseline(
            "Random Forest", rf, rf_grid, X_train, y_train, folds
        )
        xgb, xgb_summary, xgb_cv = tune_baseline(
            "XGBoost", xgb, xgb_grid, X_train, y_train, folds
        )
        summaries = {
            "Logistic Regression": lr_summary,
            "Decision Tree": dt_summary,
            "Random Forest": rf_summary,
            "XGBoost": xgb_summary,
        }
        candidates = []
        candidates.extend(
            compare_strategies(
                "Logistic Regression", lr, X_train, y_train, X_val, y_val
            )
        )
        candidates.extend(
            compare_strategies("Decision Tree", dt, X_train, y_train, X_val, y_val)
        )
        candidates.extend(
            compare_strategies("Random Forest", rf, X_train, y_train, X_val, y_val)
        )
        candidates.extend(
            compare_strategies("XGBoost", xgb, X_train, y_train, X_val, y_val)
        )
        rows = []
        selected = {}
        for candidate in candidates:
            name = candidate["family"]
            rows.append(
                {
                    "family": name,
                    "strategy": candidate["strategy"],
                    **candidate["metrics"],
                }
            )
            # Strict improvement preserves the declared order for exact ties.
            if (
                name not in selected
                or candidate["metrics"]["macro_f1"]
                > selected[name]["metrics"]["macro_f1"]
            ):
                selected[name] = candidate
        print(
            "\nAll twelve recalculated validation results:\n",
            pd.DataFrame(rows).to_string(index=False),
        )
        for name, candidate in selected.items():
            result = candidate["metrics"]
            print(
                "SELECTED",
                name,
                summaries[name]["tuned_parameters"],
                candidate["strategy"],
                "Macro-F1:",
                result["macro_f1"],
                "High precision/recall:",
                result["high_precision"],
                result["high_recall"],
            )
        refit_and_save(
            selected,
            summaries,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test.index[0],
            rows,
            lr_cv + dt_cv + rf_cv + xgb_cv,
        )


if __name__ == "__main__":
    main()
