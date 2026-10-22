"""Fit fixed original 12h regressors: training validation, then train+val refits."""

from importlib import import_module
from importlib.metadata import version
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

features = import_module("03_feature_engineering")
evaluation = import_module("05_model_evaluation")
ROOT = features.preprocessing.ROOT


def make_models():
    """No search, balancing, early stopping or test-based selection."""
    return {
        "Linear Regression": Pipeline(
            [("scale", StandardScaler()), ("regressor", LinearRegression())]
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=10, min_samples_split=10, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=42,
            n_jobs=4,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            reg_lambda=8,
            objective="reg:squarederror",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
        ),
    }


def main():
    X, y, Xv, yv, _, _, partitions = features.prepare_features()
    Xfinal, yfinal = pd.concat([X, Xv]), pd.concat([y, yv])
    assert len(Xfinal) == 23362
    predictions = pd.DataFrame({"actual_VHM0": yv}, index=Xv.index)
    metadata = {
        "horizon_hours": 12,
        "target": "VHM0(t+12h)",
        "binary_threshold_m": 2.5,
        "feature_names": X.columns.tolist(),
        "partitions": partitions,
        "refit_rows": len(Xfinal),
        "refit_origin_start": str(Xfinal.index[0]),
        "refit_origin_end": str(Xfinal.index[-1]),
        "refit_target_end": str(Xfinal.index[-1] + pd.Timedelta(hours=12)),
        "source_sha256": evaluation.sha(features.preprocessing.DATA_PATH),
        "versions": {name: version(name) for name in evaluation.PACKAGES},
        "procedure": "Fixed configurations; validate training-only fit; fresh clone refitted on eligible train+validation. No tuning or test selection.",
        "models": {},
    }
    (ROOT / "models").mkdir(exist_ok=True)
    rows = []
    with threadpool_limits(limits=4):
        for name, prototype in make_models().items():
            validation_model = clone(prototype).fit(X, y)
            predicted = np.asarray(validation_model.predict(Xv), dtype=np.float64)
            predictions[name] = predicted
            rows.append({"model": name, **evaluation.regression_metrics(yv, predicted)})
            model = clone(prototype).fit(Xfinal, yfinal)
            path = ROOT / "models" / evaluation.MODEL_FILES[name]
            joblib.dump(model, path)
            metadata["models"][name] = {
                "file": str(path.relative_to(ROOT)),
                "sha256": evaluation.sha(path),
                "parameters": evaluation.parameter_record(model),
            }
            print(f"Saved {name}: {len(Xfinal):,} refit rows")
    (ROOT / "models/model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for directory in ["metrics", "predictions"]:
        (ROOT / "results" / directory).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        ROOT / "results/metrics/validation_regression_metrics.csv", index=False
    )
    predictions.to_csv(
        ROOT / "results/predictions/validation_predictions.csv", index_label="timestamp"
    )


if __name__ == "__main__":
    main()
