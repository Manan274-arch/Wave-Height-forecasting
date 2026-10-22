# Machine Learning-Based Forecasting of High-Wave Events Along the Udupi Coast

An undergraduate comparison of persistence, Linear Regression, Decision Tree Regressor, Random Forest Regressor and XGBoost Regressor for significant wave-height forecasting near offshore Udupi.

**Primary methodology:** continuous significant wave height exactly **six hours ahead**. **Secondary experiment:** exactly **24 hours ahead**. Both experiments have been executed, including training-only tuning, validation, frozen-model tests, persistence, conditional High-wave errors and figures.

This regression methodology comes from the original project synopsis, but implementation followed exploratory six-hour and 24-hour maximum-category classification studies. The existing test period therefore is not entirely independent of earlier project-design decisions. Classification code and artifacts remain unchanged and accessible below.

## Dataset and Features

`data/udupi_wave_wind_2015_2024.csv` is unchanged: **29,224 rows and 20 columns**, every three hours from **2015-01-01 00:00 to 2024-12-31 21:00 UTC**, near **13.4 N, 74.6 E**. There are no missing observations or duplicate timestamps.

| Source | Copernicus Marine dataset ID | Version used |
|---|---|---|
| Offshore modeled/reanalysis waves | `cmems_mod_glo_wav_my_0.2deg_PT3H-i` | 202411 |
| Hourly gridded wind | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | 202211 |

Original columns describe total significant height (`VHM0`), swell/wind-wave heights (`VHM0_SW1`, `VHM0_SW2`, `VHM0_WW`), from-directions (`VMDR*`), spectral periods (`VTM*`, `VTPK`), signed stress-equivalent 10 m wind components, speed, coordinates and UTC timestamps. Units are metres, degrees clockwise from north, seconds and m/s as applicable.

`prepare_data.py` remains unchanged. It selects the offshore wave cell, spatially interpolates wind components, calculates speed after interpolation and aligns exact timestamps. The ML workflows use the existing CSV without downloads. Historical raw files remain outside the project at `../Wave Forecasting Raw Archive 2026-10-21/raw/`.

All algorithms and both regression horizons retain the exact **46 causal predictors**: 13 current measurements, 8 circular encodings, 9 historical lags, 5 changes, 9 trailing statistics and 2 cyclical month encodings. Rolling windows include current measurements and require complete past context; sample standard deviation uses `ddof=1`. Future measurements/targets never enter X. Current VHM0 remains an important legitimate predictor. Classification source validation and feature construction are reused without modification.

## Point Targets and Common Split Boundaries

```python
six_hour_target = source["VHM0"].shift(-2)
twenty_four_hour_target = source["VHM0"].shift(-8)
```

These are heights **exactly at** t+6h and t+24h, not maxima over future windows, current heights or class labels. Shifted timestamps are explicitly checked against the requested future timestamp. Invalid alignments are rejected; trailing incomplete targets are excluded.

Both regression experiments share nominal 70%/10%/20% boundaries calculated on all raw timestamps **before horizon exclusions**. Nominal validation starts 2022-01-01 00:00 UTC and test starts 2023-01-01 06:00 UTC. This differs deliberately from the earlier classification convention of splitting complete labels; it keeps regression boundaries identical across horizons while preserving chronology. No classification split or result was changed.

| Horizon | Partition | Rows | UTC start | UTC end | Purged | History excluded | Incomplete targets |
|---|---|---|---|---|---|---|---|
| 6 | Training | 20446 | 2015-01-02 00:00:00+00:00 | 2021-12-31 15:00:00+00:00 | 2 | 8 | 0 |
| 6 | Validation | 2920 | 2022-01-01 00:00:00+00:00 | 2022-12-31 21:00:00+00:00 | 2 | 0 | 0 |
| 6 | Testing | 5844 | 2023-01-01 06:00:00+00:00 | 2024-12-31 15:00:00+00:00 | 0 | 0 | 2 |
| 24 | Training | 20440 | 2015-01-02 00:00:00+00:00 | 2021-12-30 21:00:00+00:00 | 8 | 8 | 0 |
| 24 | Validation | 2914 | 2022-01-01 00:00:00+00:00 | 2022-12-31 03:00:00+00:00 | 8 | 0 | 0 |
| 24 | Testing | 5838 | 2023-01-01 06:00:00+00:00 | 2024-12-30 21:00:00+00:00 | 0 | 0 | 8 |

Every matrix has 46 features. Target-alignment rejection counts are zero. Eight initial training origins lack full historical features. Boundary purges remove two origins per boundary at six hours and eight at 24 hours. Target endpoints remain strictly before the next nominal partition. Final refits use **23,366 rows at six hours** and **23,354 at 24 hours**, retaining the purge gaps.

## Training and Evaluation Methodology

- Ordinary Linear Regression uses `StandardScaler -> LinearRegression`, with defaults and no arbitrary hyperparameter search. Its single setting is scored on the same three folds for context. All 46 inputs are retained; the final linear design has rank 39 because some engineered predictors are deterministic linear combinations. Standard least-squares handles this; coefficients should not be interpreted as independent causal effects.
- Decision Tree explores three coherent shallow/moderate/deeper settings in depth, split and leaf size. Random Forest explores three settings in tree count, depth, leaf size and feature fraction, including an unrestricted-depth candidate. XGBoost explores three settings in tree count, depth, learning rate and L2 regularization with `reg:squarederror`. Complete candidate settings are visible in `regression_training.py` and `tuning_results.csv`.
- Training-only `TimeSeriesSplit(n_splits=3, gap=2)` is used for six hours and `gap=8` for 24 hours. Timestamp separation is asserted. Each horizon has **30 fold fits**: three for the untuned linear baseline and nine for each tree family. GridSearchCV fits scaling inside folds.
- Select the lowest mean CV RMSE; ties prefer lower sample standard deviation, then earlier candidate order. Validation never selects additional configurations. No SMOTE, class weighting, class loss or classification score is used in regression training.
- Fit the four chosen configurations on training data, record validation metrics, then independently refit all four on eligible training+validation. Scaling is refitted inside the linear pipeline on this combined training set. All four staged models must reload successfully before publication.
- Test evaluation loads frozen joblib models without fitting. Persistence predicts current VHM0 on exactly the same origins. No predictions are clipped, class-thresholded for training or manually adjusted. All predictions in these runs were nonnegative.

Primary metrics are MAE (m), RMSE (m) and R?. Tables also contain signed error (prediction minus actual), median/maximum absolute error, and actual-High subset counts/MAE/RMSE. Positive improvement means lower error than persistence:

```text
absolute RMSE improvement = persistence_RMSE - model_RMSE
percentage improvement = 100 * absolute improvement / persistence_RMSE
```

MAE uses the same convention. Percentage improvement is undefined if the baseline error is zero. These models were not tuned using High-category subsets or derived classification scores.

## Training-Only CV Results

| Horizon | Algorithm | Selected parameters | CV RMSE (m) | Sample std (m) |
|---|---|---|---|---|
| 6 | Linear Regression | Untuned defaults | 0.084408 | 0.007855 |
| 6 | Decision Tree | {"max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 10} | 0.098287 | 0.002266 |
| 6 | Random Forest | {"max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | 0.086898 | 0.006085 |
| 6 | XGBoost | {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 220, "reg_lambda": 8} | 0.085607 | 0.009286 |
| 24 | Linear Regression | Untuned defaults | 0.193453 | 0.020018 |
| 24 | Decision Tree | {"max_depth": 4, "min_samples_leaf": 20, "min_samples_split": 20} | 0.209965 | 0.017564 |
| 24 | Random Forest | {"max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | 0.197484 | 0.014697 |
| 24 | XGBoost | {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 220, "reg_lambda": 8} | 0.189339 | 0.014357 |

Other fixed estimator settings and versions are recorded in metadata.

## Validation Results

| Horizon | Method | MAE (m) | RMSE (m) | R? |
|---|---|---|---|---|
| 6 | Persistence | 0.048339 | 0.073950 | 0.990154 |
| 6 | Linear Regression | 0.047577 | 0.071279 | 0.990853 |
| 6 | Decision Tree | 0.052013 | 0.080696 | 0.988276 |
| 6 | Random Forest | 0.046809 | 0.070978 | 0.990930 |
| 6 | XGBoost | 0.045004 | 0.068117 | 0.991646 |
| 24 | Persistence | 0.107684 | 0.158255 | 0.954925 |
| 24 | Linear Regression | 0.108314 | 0.154023 | 0.957303 |
| 24 | Decision Tree | 0.117376 | 0.167309 | 0.949619 |
| 24 | Random Forest | 0.105886 | 0.155312 | 0.956585 |
| 24 | XGBoost | 0.103711 | 0.150871 | 0.959033 |

## Frozen Test Results

| Method | 6h MAE (m) | 6h RMSE (m) | 6h R? | 24h MAE (m) | 24h RMSE (m) | 24h R? |
|---|---|---|---|---|---|---|
| Persistence | 0.048299 | 0.070679 | 0.990053 | 0.109274 | 0.164192 | 0.946369 |
| Linear Regression | 0.044513 | 0.065403 | 0.991483 | 0.105479 | 0.153510 | 0.953120 |
| Decision Tree | 0.049663 | 0.074729 | 0.988880 | 0.114885 | 0.167565 | 0.944143 |
| Random Forest | 0.044158 | 0.065713 | 0.991401 | 0.105438 | 0.156393 | 0.951343 |
| XGBoost | 0.042027 | 0.063851 | 0.991882 | 0.100743 | 0.148001 | 0.956425 |

Error improvement over persistence:

| Horizon | Method | MAE reduction (m) | MAE improvement | RMSE reduction (m) | RMSE improvement |
|---|---|---|---|---|---|
| 6 | XGBoost | +0.006272 | +12.985% | +0.006828 | +9.660% |
| 6 | Linear Regression | +0.003786 | +7.838% | +0.005276 | +7.465% |
| 6 | Random Forest | +0.004141 | +8.574% | +0.004965 | +7.025% |
| 6 | Decision Tree | -0.001363 | -2.823% | -0.004050 | -5.730% |
| 24 | XGBoost | +0.008531 | +7.807% | +0.016191 | +9.861% |
| 24 | Linear Regression | +0.003795 | +3.473% | +0.010681 | +6.505% |
| 24 | Random Forest | +0.003836 | +3.510% | +0.007799 | +4.750% |
| 24 | Decision Tree | -0.005611 | -5.135% | -0.003373 | -2.054% |

XGBoost has the lowest overall test RMSE at both horizons. Linear Regression outperforms Random Forest by overall RMSE at both horizons, although Random Forest has marginally lower MAE. Both ensembles improve on the single Decision Tree. Linear Regression, Random Forest and XGBoost improve on persistence overall; Decision Tree does not. Differences between the leading learned models are small in absolute metres and have not been tested for statistical significance.

Errors are higher at 24 hours for all five methods in these data. `common_origin_comparison.csv` repeats the comparison on **5,838 identical test origins** and retains this pattern. Future target timestamps and fitted models still differ; equal origins do not make this a perfectly controlled comparison of difficulty.

## Actual High-Wave Behaviour

Evaluation subsets use actual future height: Normal <2.5 m, Elevated 2.5-3.0 m inclusive, High >3.0 m. These are research-defined offshore categories, **not official warning thresholds**, and never define regression training labels. `category_errors.csv` records each category's support, MAE, RMSE, signed error, underprediction frequency and average underprediction magnitude among strict underpredictions.

Both test sets contain **116 actual High targets**. Validation contains 83 High targets at each horizon.

| Horizon | Method | High MAE (m) | High RMSE (m) | High bias (m) | Underprediction frequency | Mean magnitude when underpredicting (m) |
|---|---|---|---|---|---|---|
| 6 | Persistence | 0.137931 | 0.180493 | -0.023793 | 49.14% | 0.164561 |
| 6 | Linear Regression | 0.125534 | 0.170302 | -0.032100 | 52.59% | 0.149882 |
| 6 | Decision Tree | 0.157860 | 0.213324 | -0.035443 | 55.17% | 0.175181 |
| 6 | Random Forest | 0.132380 | 0.180011 | -0.043687 | 56.90% | 0.154726 |
| 6 | XGBoost | 0.133059 | 0.187407 | -0.011907 | 50.86% | 0.142508 |
| 24 | Persistence | 0.340172 | 0.420324 | -0.138448 | 60.34% | 0.396571 |
| 24 | Linear Regression | 0.323475 | 0.400630 | -0.204701 | 72.41% | 0.364693 |
| 24 | Decision Tree | 0.364520 | 0.439751 | -0.258888 | 75.00% | 0.415605 |
| 24 | Random Forest | 0.305252 | 0.383705 | -0.200326 | 73.28% | 0.344982 |
| 24 | XGBoost | 0.295589 | 0.371637 | -0.189407 | 68.97% | 0.351622 |

All methods have negative mean signed error on actual High targets. Linear Regression has the lowest High-subset RMSE at six hours. XGBoost has the lowest High-subset RMSE at 24 hours, but underpredicts 80/116 High targets (68.97%) with mean High bias -0.1894 m. Strong aggregate R? does not remove the need to study extreme-wave errors. Optional derived classification analysis is not implemented; regression remains the primary evaluation.

## Structure and Reproduction

```text
src/
|-- 01_eda.py ... 07_onset_analysis.py    # Existing classification workflow, unchanged
|-- feature_pipeline.py                 # Existing source validation and causal features
|-- model_metrics.py                    # Existing classification metrics
|-- regression_pipeline.py              # Exact targets, common splits, regression metrics
|-- regression_training.py              # CV, validation, four refits per horizon
|-- regression_evaluation.py            # Frozen inference, diagnostics, nine figures
`-- regression_comparison.py            # Primary and common-origin horizon comparisons

experiments/
|-- 6_hour/                            # Original classification archive, unchanged
|-- 24_hour/                           # Maximum-category classification, unchanged
`-- regression/
    |-- 6_hour/                        # Primary exact-6h regression
    |   |-- models/                    # Four joblib models
    |   |-- figures/                   # Nine PNGs
    |   |-- model_metadata.json
    |   |-- tuning_results.csv
    |   |-- validation_results.csv
    |   |-- test_results.csv
    |   |-- persistence_comparison.csv
    |   |-- validation_test_comparison.csv
    |   |-- category_errors.csv
    |   |-- feature_importance.csv
    |   `-- test_predictions.csv
    |-- 24_hour/                       # Same artifacts for secondary exact-24h regression
    |-- horizon_comparison.csv
    `-- common_origin_comparison.csv
```

The raw CSV, `prepare_data.py`, requirements and existing classification source/artifacts are unchanged. No raw data copy is made. All regression output paths are separate from classification paths. Metadata records targets, timestamps/counts/exclusions, source hash, exact feature order, CV settings, tuned parameters, validation metrics, refit sizes, versions and model hashes.

Use the existing Python 3.11 Windows `.venv`. From the project root:

```powershell
.venv\Scripts\python.exe src/regression_pipeline.py --horizon 6
.venv\Scripts\python.exe src/regression_training.py --horizon 6
.venv\Scripts\python.exe src/regression_evaluation.py --horizon 6
.venv\Scripts\python.exe src/regression_pipeline.py --horizon 24
.venv\Scripts\python.exe src/regression_training.py --horizon 24
.venv\Scripts\python.exe src/regression_evaluation.py --horizon 24
.venv\Scripts\python.exe src/regression_comparison.py
```

Plots display normally; close each window to continue. For headless verification set `MPLBACKEND=Agg`. Each horizon produces time series, scatterplots, residual distributions, MAE/RMSE comparisons, validation/test RMSE, persistence-relative RMSE improvement, High-wave errors and tree feature importances. Time-series panels show the full test period plus **July 1-14 of the first test year**, selected by a fixed calendar rule irrespective of model errors. Feature importance describes the fitted model (impurity reduction for tree/forest, gain for XGBoost), not causal effects; correlated predictors can share importance.

Models can be loaded with joblib from `experiments/regression/6_hour/models/` or `24_hour/models/`. Supply the exact 46 columns from the corresponding metadata. Only load trusted artifacts with compatible dependencies. Final test sorting is reporting only; do not tune configurations after inspecting test scores.

## Preserved Exploratory Classification Studies

The prior classification targets predict the maximum category over all future observations in the next six or 24 hours. They are different questions from exact-future regression, so their scores are not directly comparable to regression errors.

| Classification horizon | Persistence Macro-F1 | Best ML Macro-F1 | Algorithm | Test origins |
|---|---|---|---|---|
| 6 hours | 0.936989 | 0.933008 | XGBoost | 5845 |
| 24 hours | 0.856664 | 0.852398 | XGBoost | 5844 |

Six-hour models/results, original source snapshots and documentation remain in [experiments/6_hour](experiments/6_hour/). Twenty-four-hour classification results, predictions, onset analyses and figures remain in [experiments/24_hour](experiments/24_hour/). Their entry points (`04_model_training.py` through `07_onset_analysis.py`) remain usable and retain their original output paths. The legacy root `models/` copy remains untouched.

## Scientific Limits

The measurements represent offshore modeled/reanalysis waves and gridded wind, not beach run-up or direct instrument observations at the reference cell. Short-horizon sea states are strongly persistent, and the seasonal range contributes to high aggregate R?. Only 116 test targets are High at either horizon; neighbouring samples and forecasting errors are autocorrelated, and they are not independent storm events. No uncertainty intervals, independent-event analysis, real-time availability study or operational deployment validation are implemented.

The searches are compact and their outcomes depend on the declared candidates; no universal algorithm ranking is established. The existing test period influenced prior exploratory design decisions, even though regression was prespecified in the synopsis. All actual regression tuning stayed within training. These results require further event-level validation before claims about extreme-wave forecasting or operational marine warnings.
