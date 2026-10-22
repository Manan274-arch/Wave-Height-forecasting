"""Training-only RMSE tuning, validation and four frozen regression refits."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from feature_pipeline import DATA_PATH
from regression_pipeline import (
    REGRESSION_DIR,
    parse_horizon,
    persistence_improvements,
    prepare_regression,
    regression_metrics,
)


def tune_model(name, model, grid, X_train, y_train, folds):
    """Choose lowest mean CV RMSE, then lower sample std, then earlier setting."""
    search = GridSearchCV(
        model,
        grid,
        cv=folds,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        refit=False,
        error_score="raise",
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    results = search.cv_results_
    scores = -np.column_stack([results[f"split{k}_test_rmse"] for k in range(3)])
    means = scores.mean(axis=1)
    deviations = scores.std(axis=1, ddof=1)
    best = min(
        range(len(means)), key=lambda index: (means[index], deviations[index], index)
    )
    rows = []
    for candidate, parameters in enumerate(results["params"]):
        for fold, (train, validation) in enumerate(folds):
            rows.append(
                {
                    "family": name,
                    "candidate": candidate,
                    "parameters": json.dumps(parameters),
                    "fold": fold + 1,
                    "fold_rmse": scores[candidate, fold],
                    "fold_mae": -results[f"split{fold}_test_mae"][candidate],
                    "mean_cv_rmse": means[candidate],
                    "std_cv_rmse": deviations[candidate],
                    "selected": candidate == best,
                    "training_end": str(X_train.index[train[-1]]),
                    "validation_start": str(X_train.index[validation[0]]),
                }
            )
    information = {
        "selected_parameters": results["params"][best],
        "mean_cv_rmse": float(means[best]),
        "std_cv_rmse": float(deviations[best]),
        "candidate_count": len(means),
    }
    print("CV", name, information, flush=True)
    return clone(model).set_params(**results["params"][best]), information, rows


def save_refits(models, information, validation, tuning_rows, data, result_dir):
    """Publish artifacts only after all four refits and reload checks succeed."""
    X_train, y_train, X_val, y_val, X_test, _, summary = data
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    horizon = pd.Timedelta(hours=summary["horizon_hours"])
    if X_combined.index[-1] + horizon >= X_test.index[0]:
        raise ValueError("Refit targets overlap the test period.")
    filenames = {
        "Linear Regression": "linear_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }
    metadata = {
        **summary,
        "feature_names": X_train.columns.tolist(),
        "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "study_status": "Synopsis-prespecified regression methodology implemented after exploratory classification; existing test period informed prior design decisions.",
        "selection": "Training CV RMSE only; lower sample std then earlier candidate on ties. Validation and test never tune hyperparameters.",
        "cv": {
            "n_splits": 3,
            "gap": summary["forecast_steps"],
            "fold_fits": len(tuning_rows),
        },
        "refit_rows": len(X_combined),
        "refit_start": str(X_combined.index[0]),
        "refit_end": str(X_combined.index[-1]),
        "test_used_for_selection": False,
        "versions": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "xgboost",
                "joblib",
            ]
        },
        "models": {},
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training_", dir=result_dir) as temporary:
        stage = Path(temporary)
        (stage / "models").mkdir()
        for name, estimator in models.items():
            final_model = clone(estimator)
            final_model.fit(X_combined, y_combined)
            path = stage / "models" / filenames[name]
            joblib.dump(final_model, path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(
                loaded.predict(X_train.head(32)),
                final_model.predict(X_train.head(32)),
                rtol=0,
                atol=1e-12,
            )
            if list(loaded.feature_names_in_) != X_train.columns.tolist():
                raise ValueError("Saved feature order changed.")
            parameters = estimator.get_params(deep=False)
            if isinstance(estimator, Pipeline):
                parameters = estimator.named_steps["regressor"].get_params()
            parameters = {
                key: value
                for key, value in parameters.items()
                if value is None
                or isinstance(value, (str, bool, int))
                or (isinstance(value, float) and np.isfinite(value))
            }
            metadata["models"][name] = {
                "file": "models/" + filenames[name],
                **information[name],
                "estimator_parameters": parameters,
                "validation_metrics_before_refit": validation.loc[name].to_dict(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if name == "Linear Regression":
                metadata["models"][name]["linear_design_rank"] = int(
                    loaded.named_steps["regressor"].rank_
                )
            print("Refit, saved, reloaded:", name, len(X_combined), flush=True)
        (stage / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        validation.reset_index().to_csv(stage / "validation_results.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(stage / "tuning_results.csv", index=False)
        for path in stage.rglob("*"):
            if path.is_file():
                destination = result_dir / path.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)


def main():
    """Run one independent horizon; no classifier or resampling is involved."""
    horizon = parse_horizon()
    data = prepare_regression(horizon)
    X_train, y_train, X_val, y_val, _, _, summary = data
    print(json.dumps(summary, indent=2))
    folds = list(TimeSeriesSplit(n_splits=3, gap=horizon // 3).split(X_train))
    for train, validation in folds:
        if (
            X_train.index[train[-1]] + pd.Timedelta(hours=horizon)
            >= X_train.index[validation[0]]
        ):
            raise ValueError("CV target timestamps overlap validation origins.")
    linear = Pipeline([("scale", StandardScaler()), ("regressor", LinearRegression())])
    tree = DecisionTreeRegressor(random_state=42)
    forest = RandomForestRegressor(random_state=42, n_jobs=4)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
    )
    # Three coherent settings per tree family, including moderate and deeper trees.
    tree_grid = [
        {"max_depth": [4], "min_samples_split": [20], "min_samples_leaf": [20]},
        {"max_depth": [8], "min_samples_split": [10], "min_samples_leaf": [10]},
        {"max_depth": [12], "min_samples_split": [4], "min_samples_leaf": [4]},
    ]
    forest_grid = [
        {
            "n_estimators": [120],
            "max_depth": [10],
            "min_samples_leaf": [5],
            "max_features": [0.7],
        },
        {
            "n_estimators": [160],
            "max_depth": [16],
            "min_samples_leaf": [2],
            "max_features": [1.0],
        },
        {
            "n_estimators": [220],
            "max_depth": [None],
            "min_samples_leaf": [1],
            "max_features": [0.7],
        },
    ]
    xgb_grid = [
        {
            "n_estimators": [160],
            "max_depth": [2],
            "learning_rate": [0.07],
            "reg_lambda": [5],
        },
        {
            "n_estimators": [220],
            "max_depth": [4],
            "learning_rate": [0.05],
            "reg_lambda": [8],
        },
        {
            "n_estimators": [300],
            "max_depth": [6],
            "learning_rate": [0.05],
            "reg_lambda": [5],
        },
    ]
    with threadpool_limits(limits=4):
        # Linear Regression has no search: its single default setting is CV-scored.
        linear, linear_info, linear_cv = tune_model(
            "Linear Regression", linear, [{}], X_train, y_train, folds
        )
        tree, tree_info, tree_cv = tune_model(
            "Decision Tree", tree, tree_grid, X_train, y_train, folds
        )
        forest, forest_info, forest_cv = tune_model(
            "Random Forest", forest, forest_grid, X_train, y_train, folds
        )
        xgb, xgb_info, xgb_cv = tune_model(
            "XGBoost", xgb, xgb_grid, X_train, y_train, folds
        )
        linear.fit(X_train, y_train)
        tree.fit(X_train, y_train)
        forest.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        predictions = {
            "Persistence": X_val["VHM0"].to_numpy(),
            "Linear Regression": linear.predict(X_val),
            "Decision Tree": tree.predict(X_val),
            "Random Forest": forest.predict(X_val),
            "XGBoost": xgb.predict(X_val),
        }
        validation = persistence_improvements(
            pd.DataFrame(
                [
                    {"family": name, **regression_metrics(y_val, prediction)}
                    for name, prediction in predictions.items()
                ]
            )
        )
        print("Validation results:\n", validation.to_string(index=False), flush=True)
        models = {
            "Linear Regression": linear,
            "Decision Tree": tree,
            "Random Forest": forest,
            "XGBoost": xgb,
        }
        information = {
            "Linear Regression": linear_info,
            "Decision Tree": tree_info,
            "Random Forest": forest_info,
            "XGBoost": xgb_info,
        }
        save_refits(
            models,
            information,
            validation.set_index("family"),
            linear_cv + tree_cv + forest_cv + xgb_cv,
            data,
            REGRESSION_DIR / f"{horizon}_hour",
        )
    print("Training complete; test predictions have not been made.")


if __name__ == "__main__":
    main()
