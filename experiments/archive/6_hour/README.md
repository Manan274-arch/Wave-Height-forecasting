# Machine Learning-Based Forecasting of High-Wave Events Along the Udupi Coast

An undergraduate comparative study of Logistic Regression, Decision Tree, Random Forest and XGBoost. The objective is to classify the maximum significant wave-height category at the next two three-hourly observations near offshore Udupi, supporting research into high-wave conditions.

**Status:** Phases 1-5 are implemented and executed. Hyperparameters were tuned on chronological training folds, strategies selected on validation data, and four frozen refits evaluated on the independent 2023-2024 test partition. Test scores did not influence configuration selection.

## Dataset

`data/udupi_wave_wind_2015_2024.csv` contains **29,224 rows and 20 columns**, sampled every three hours from **2015-01-01 00:00 through 2024-12-31 21:00 UTC**. The fixed offshore wave cell is near **13.4 N, 74.6 E**. It has no missing observations or duplicate timestamps.

| Source | Copernicus Marine dataset ID | Version used |
|---|---|---|
| Wave model/reanalysis | `cmems_mod_glo_wav_my_0.2deg_PT3H-i` | 202411 |
| Hourly gridded wind | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | 202211 |

The CSV includes total significant height (`VHM0`), primary/secondary swell and wind-wave heights (`VHM0_SW1`, `VHM0_SW2`, `VHM0_WW`), their from-directions (`VMDR*`), spectral wave periods (`VTM*`, `VTPK`), signed eastward/northward stress-equivalent 10 m wind components, calculated wind speed, UTC timestamps and coordinates. Heights are in metres, periods in seconds, directions in degrees clockwise from north, and winds in m/s.

`prepare_data.py` retains the original acquisition workflow: select the offshore wave cell, interpolate wind components spatially, calculate speed after interpolation, align exact UTC times and validate the CSV. The ML workflow uses the existing CSV without downloads or changes. Historical raw files are outside this project at `../Wave Forecasting Raw Archive 2026-10-21/raw/`.

## Project Structure and Phases

```text
Wave Forecasting/
|-- src/
|   |-- 01_eda.py
|   |-- 02_preprocessing.py
|   |-- 03_feature_engineering.py
|   |-- 04_model_training.py
|   |-- 05_model_evaluation.py
|   |-- feature_pipeline.py
|   `-- model_metrics.py
|-- data/udupi_wave_wind_2015_2024.csv
|-- models/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- random_forest.joblib
|   |-- xgboost.joblib
|   |-- model_metadata.json
|   |-- chronological_tuning.csv
|   |-- validation_comparison.csv
|   |-- test_comparison.csv
|   |-- validation_test_comparison.csv
|   `-- figures/                      # Four evaluation PNGs
|-- prepare_data.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

The local `.venv/` is preserved. Phases 1-3 inspect the observations, construct labels/purged splits, and explain/check the exact 46 causal features. Phase 4 tunes, compares strategies and saves four refits. Phase 5 loads those artifacts, checks compatibility and reports final test results without fitting. `feature_pipeline.py` supplies deterministic data preparation; `model_metrics.py` gives both stages the same metric definitions. There are no notebook dependencies or intermediate feature datasets.

## Target and Chronology

```python
future_max_wave_height = max(VHM0[t+3h], VHM0[t+6h])
```

| Class | Research category | Future maximum |
|---|---|---|
| 0 | Normal | Below 2.5 m |
| 1 | Elevated / Watch | 2.5 through 3.0 m inclusive |
| 2 | High / Alert | Above 3.0 m |

These are **provisional research thresholds, not official marine warnings**. Both future timestamps must exist; `skipna=False` excludes the final two incomplete horizons. The target is the maximum of two sampled significant heights, not the largest individual wave or continuous six-hour maximum.

Nominal chronological 70%/10%/20% splitting purges two rows at each training/validation boundary so target horizons end strictly before the next partition. Eight initial training rows lack feature history.

| Partition | Final shape | UTC range |
|---|---|---|
| Training | (20445, 46) | 2015-01-02 00:00 to 2021-12-31 12:00 |
| Validation | (2920, 46) | 2021-12-31 21:00 to 2022-12-31 18:00 |
| Testing | (5845, 46) | 2023-01-01 03:00 to 2024-12-31 15:00 |

Class counts (Normal / Elevated / High) are 19,023 / 872 / 550 for training, 2,674 / 153 / 93 for validation, and 5,448 / 275 / 122 for testing. The validation partition is predominantly 2022, with its first observation on 2021-12-31.

The original 46 features and ordering remain unchanged: 13 current wave/wind values, 8 circular direction encodings, 9 historical lags, 5 changes, 9 trailing statistics and 2 cyclical month encodings. Rolling windows include current measurements and require complete history; standard deviation uses `ddof=1`. A 24h window contains eight observations from t-21h through t. Future heights, labels and constant coordinates are excluded from predictors.

## Finalized Training Methodology

1. Tune each **baseline only**, using the existing three candidate settings and `TimeSeriesSplit(n_splits=3, gap=2)`. GridSearchCV fits Logistic Regression scaling inside training folds. This produces **36 fold fits**, nine per algorithm.
2. Select baseline parameters by highest mean CV Macro-F1; exact ties use lower sample standard deviation, then earlier candidate order.
3. Hold those estimator parameters fixed and fit baseline, balanced class-weighted and conservative SMOTE versions on training data. Multiclass XGBoost uses training-label-derived sample weights.
4. Compare all 12 models on the same fixed validation partition. Select the highest Macro-F1 strategy per algorithm; exact ties prefer baseline, then class-weighted, then SMOTE. No additional hyperparameter search uses validation data.
5. Refit only those four selections on **23,365 eligible training-plus-validation rows**. Recalculate weights/SMOTE targets and fit scaling inside each final pipeline. All four models are staged and reloaded successfully before replacing existing artifacts.
6. Freeze the models and evaluate the same held-out test observations. Test sorting is reporting only and does not change configurations.

SMOTE uses `StandardScaler -> SMOTE -> classifier`, three neighbors and seed 42. Minority classes below 10% of the fit-local majority count are increased to that target. Weighting and SMOTE are separate strategies; neither resamples validation/test observations. No learned scaling uses validation/test data during strategy selection. During final refitting, validation becomes part of eligible training data.

Both stages report accuracy, balanced accuracy, Macro-F1, weighted-F1, class-wise precision/recall/F1, High precision/recall/F1, missed High observations, false High alerts and confusion matrices in Normal/Elevated/High order. Macro-F1 is primary because the data are strongly imbalanced.

## Baseline Tuning and Selected Strategies

| Algorithm | Selected baseline settings | CV Macro-F1 | CV sample std | Selected strategy |
|---|---|---|---|---|
| Logistic Regression | C=0.3 | 0.878833 | 0.048880 | baseline |
| Decision Tree | max_depth=6, min_samples_leaf=20, min_samples_split=20 | 0.899767 | 0.028061 | baseline |
| Random Forest | n_estimators=160, max_depth=12, min_samples_leaf=3, max_features=sqrt | 0.900684 | 0.043398 | SMOTE |
| XGBoost | n_estimators=120, max_depth=2, learning_rate=0.1, reg_lambda=8 | 0.897458 | 0.039305 | baseline |

Other fixed settings remain explicit in the training script and saved estimator metadata. The Random Forest CV winner retained its original baseline settings.

## Recalculated Validation Results

These scores were generated by the finalized workflow, replacing the previous strategy-specific tuning experiment.

| Algorithm | Strategy | Accuracy | Macro-F1 | High precision | High recall |
|---|---|---|---|---|---|
| Logistic Regression | baseline | 0.983562 | 0.886556 | 0.810526 | 0.827957 |
| Logistic Regression | class-weighted | 0.970890 | 0.854405 | 0.802083 | 0.827957 |
| Logistic Regression | SMOTE | 0.979452 | 0.867505 | 0.764706 | 0.838710 |
| Decision Tree | baseline | 0.982534 | 0.878074 | 0.845238 | 0.763441 |
| Decision Tree | class-weighted | 0.968151 | 0.844169 | 0.797872 | 0.806452 |
| Decision Tree | SMOTE | 0.980822 | 0.874117 | 0.820225 | 0.784946 |
| Random Forest | baseline | 0.983904 | 0.887759 | 0.867470 | 0.774194 |
| Random Forest | class-weighted | 0.975342 | 0.870734 | 0.860465 | 0.795699 |
| Random Forest | SMOTE | 0.984247 | 0.895192 | 0.846154 | 0.827957 |
| XGBoost | baseline | 0.984932 | 0.901404 | 0.858696 | 0.849462 |
| XGBoost | class-weighted | 0.971918 | 0.871632 | 0.851064 | 0.860215 |
| XGBoost | SMOTE | 0.983562 | 0.898354 | 0.821782 | 0.892473 |

## Final Held-Out Test Results

Scores describe the four frozen refits evaluated on 5,845 observations.

| Algorithm | Strategy | Accuracy | Balanced accuracy | Macro-F1 | High precision | High recall | High F1 | Missed High | False High |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | baseline | 0.989564 | 0.919285 | 0.933008 | 0.911290 | 0.926230 | 0.918699 | 9 | 11 |
| Logistic Regression | baseline | 0.989564 | 0.920067 | 0.932009 | 0.910569 | 0.918033 | 0.914286 | 10 | 11 |
| Decision Tree | baseline | 0.988879 | 0.916000 | 0.930490 | 0.925000 | 0.909836 | 0.917355 | 11 | 9 |
| Random Forest | SMOTE | 0.988537 | 0.920069 | 0.922500 | 0.869231 | 0.926230 | 0.896825 | 9 | 17 |

Validation versus test (changes are test minus validation):

| Algorithm | Validation Macro-F1 | Test Macro-F1 | Change | Validation High recall | Test High recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.886556 | 0.932009 | +0.045453 | 0.827957 | 0.918033 |
| Decision Tree | 0.878074 | 0.930490 | +0.052416 | 0.763441 | 0.909836 |
| Random Forest | 0.895192 | 0.922500 | +0.027308 | 0.827957 | 0.926230 |
| XGBoost | 0.901404 | 0.933008 | +0.031604 | 0.849462 | 0.926230 |

XGBoost and Random Forest each detect 113 of 122 High observations, with 11 versus 17 false High alerts. Decision Tree has nine false High alerts but misses 11 High observations. Logistic Regression detects 112 High observations with 11 false High alerts. These counts are observations, not independent storm events. All algorithms scored higher on this test period than on validation; differing periods, class support and final refitting can contribute. The close ranking does not establish statistical significance or justify further tuning on these test results.

## Execution and Artifacts

Use the existing Python 3.11 Windows environment. On a fresh installation, create `.venv` and install `requirements.txt`. Run from the project root:

```powershell
.venv\Scripts\python.exe src/01_eda.py
.venv\Scripts\python.exe src/02_preprocessing.py
.venv\Scripts\python.exe src/03_feature_engineering.py
.venv\Scripts\python.exe src/04_model_training.py
.venv\Scripts\python.exe src/05_model_evaluation.py
```

The first three phases work independently. Close Matplotlib windows to continue. Phase 4 replaces the four joblib files, metadata and validation/CV tables only after successful refit verification. Phase 5 verifies source/model hashes, dependency versions, feature names/order and chronology, then writes the two test comparison tables and four PNGs under `models/figures/`; it also displays the plots. It never refits models or changes their metadata. Models should remain frozen for final reporting.

```python
import joblib
import pandas as pd

df = pd.read_csv("data/udupi_wave_wind_2015_2024.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
model = joblib.load("models/logistic_regression.joblib")
# Predict using the 46 engineered columns in model_metadata.json order.
```

Metadata records the source hash, ordered predictors, thresholds, versions, tuned parameter overrides, complete estimator settings, selected strategies, pre-refit validation metrics and refit dates/count. Saved pipelines apply fitted scaling during prediction; SMOTE is inactive. Only load trusted joblib files with compatible packages.

## Scientific Limitations

The reference represents offshore modeled waves and gridded wind, not direct beach conditions, breaking waves or run-up. Research categories are not official warnings. Synthetic SMOTE combinations can violate physical predictor relationships. The search is modest; only 122 test observations are High, and adjacent observations/overlapping six-hour horizons are correlated. Observation-level metrics do not establish independent event detection or operational capability. No confidence intervals, independent-event analysis or real-time deployment validation are implemented. Test results are a final comparison, not another tuning opportunity.
