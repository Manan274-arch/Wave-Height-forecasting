"""Tune and refit eight controlled configurations before any controlled testing."""

import argparse
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from controlled_pipeline import OUTPUT_DIR, prepare_controlled
from regression_pipeline import persistence_improvements, regression_metrics
from regression_training import save_refits, tune_model


def candidate_grids():
    """Keep the original compact search spaces identical for every configuration."""
    tree = [
        {"max_depth": [4], "min_samples_split": [20], "min_samples_leaf": [20]},
        {"max_depth": [8], "min_samples_split": [10], "min_samples_leaf": [10]},
        {"max_depth": [12], "min_samples_split": [4], "min_samples_leaf": [4]},
    ]
    forest = [
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
    xgb = [
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
    return tree, forest, xgb


def train_configuration(horizon, feature_set):
    """Run the same CV, validation and refit policy for one declared configuration."""
    directory = OUTPUT_DIR / f"{horizon}_hour" / feature_set
    if (directory / "model_metadata.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite completed controlled models: {directory}"
        )
    data = prepare_controlled(horizon, feature_set)
    X, y, X_val, y_val, _, _, summary = data
    print("CONFIGURATION", horizon, feature_set, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    folds = list(TimeSeriesSplit(n_splits=3, gap=horizon // 3).split(X))
    for train, validation in folds:
        if X.index[train[-1]] + pd.Timedelta(hours=horizon) >= X.index[validation[0]]:
            raise ValueError("CV targets overlap the next fold.")
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
    tree_grid, forest_grid, xgb_grid = candidate_grids()
    linear, linear_info, linear_cv = tune_model(
        "Linear Regression", linear, [{}], X, y, folds
    )
    tree, tree_info, tree_cv = tune_model("Decision Tree", tree, tree_grid, X, y, folds)
    forest, forest_info, forest_cv = tune_model(
        "Random Forest", forest, forest_grid, X, y, folds
    )
    xgb, xgb_info, xgb_cv = tune_model("XGBoost", xgb, xgb_grid, X, y, folds)
    linear.fit(X, y)
    tree.fit(X, y)
    forest.fit(X, y)
    xgb.fit(X, y)
    predictions = {
        "Persistence": X_val.VHM0.to_numpy(),
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
    print(
        "Validation:\n",
        validation[["family", "mae", "rmse", "r2"]].to_string(index=False),
        flush=True,
    )
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
        directory,
    )
    print(
        "COMPLETED",
        horizon,
        feature_set,
        "No controlled test predictions made.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--horizon", type=int, choices=[6, 24, 48, 72])
    parser.add_argument("--features", choices=["full", "compact"])
    options = parser.parse_args()
    if options.all:
        configurations = [(h, f) for h in [6, 24, 48, 72] for f in ["full", "compact"]]
    elif options.horizon and options.features:
        configurations = [(options.horizon, options.features)]
    else:
        parser.error("Use --all or provide --horizon and --features.")
    # Publish the fixed grids before test scoring, alongside the frozen subset.
    grids = candidate_grids()
    path = OUTPUT_DIR / "candidate_grids.json"
    expected_grids = {
        "Linear Regression": [{}],
        "Decision Tree": grids[0],
        "Random Forest": grids[1],
        "XGBoost": grids[2],
    }
    if path.exists() and json.loads(path.read_text()) != expected_grids:
        raise ValueError("Refusing to change the frozen candidate grids.")
    if not path.exists():
        path.write_text(json.dumps(expected_grids, indent=2))
    with threadpool_limits(limits=4):
        for horizon, feature_set in configurations:
            train_configuration(horizon, feature_set)


if __name__ == "__main__":
    main()
