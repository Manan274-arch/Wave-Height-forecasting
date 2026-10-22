# Machine Learning-Based Forecasting of High-Wave Events Along the Udupi Coast

## Project overview

This undergraduate machine learning project predicts offshore significant wave height near the Udupi coast, Karnataka, exactly **12 hours ahead**. It compares four regression models using **18 causal features** derived from historical wave and wind measurements.

Continuous wave-height predictions are also converted into Normal/Elevated categories to study high-wave detection. Such forecasting can support research into marine conditions and coastal hazards; this project is **not a validated operational coastal warning system**.

## Dataset

The historical dataset is stored at `data/udupi_wave_wind_2015_2024.csv`.

| Property | Value |
|---|---|
| Coverage | 2015-01-01 00:00 UTC to 2024-12-31 21:00 UTC |
| Sampling interval | Three hours |
| Size | 29,224 observations and 20 columns |
| Reference location | Offshore Udupi, approximately 13.4 degrees N, 74.6 degrees E |
| Wave source | Copernicus Marine wave reanalysis |
| Wind source | Copernicus Marine hourly wind data |

Original source dataset identifiers:

- Waves: `cmems_mod_glo_wav_my_0.2deg_PT3H-i`, version `202411`.
- Wind: `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`, version `202211`.

Historical data preparation selected the offshore wave grid point, interpolated wind components onto its coordinates, aligned wind observations with the three-hourly wave timestamps and combined the measurements into the final CSV. Wind speed was calculated from the interpolated components. The active ML workflow reads this existing CSV and does not download data or modify it.

### Main variables

| Variables | Meaning | Unit |
|---|---|---|
| `VHM0` | Total significant wave height | m |
| `VHM0_SW1`, `VHM0_SW2` | Primary and secondary swell significant heights | m |
| `VHM0_WW` | Wind-wave significant height | m |
| `VMDR`, `VMDR_SW1`, `VMDR_SW2`, `VMDR_WW` | Total and partition mean wave from-directions | degrees |
| `VTM01_SW1`, `VTM01_SW2`, `VTM01_WW` | Partition mean periods from spectral moments (0, 1) | s |
| `VTM02` | Total mean period from spectral moments (0, 2) | s |
| `VTM10` | Total mean period from spectral moments (-1, 0) | s |
| `VTPK` | Peak wave period | s |
| `eastward_wind`, `northward_wind` | Interpolated stress-equivalent 10 m wind components | m/s |
| `wind_speed` | Magnitude of the interpolated wind components | m/s |

The remaining columns identify the UTC timestamp, latitude and longitude. Coordinates are metadata and are not model inputs.

## Forecast objective and binary labels

For a forecast issued at time `t`, the continuous target is:

```python
future_VHM0 = VHM0(t + 12 hours)
```

After verifying the complete three-hourly timestamp grid, this is equivalent to `VHM0.shift(-4)`. The implementation constructs the target through timestamp lookup and verifies agreement with positional shifting. The target is the height at one future time, rather than the maximum over a future window.

Both actual and predicted future heights use the same fixed binary threshold:

- **Normal:** height < 2.5 m.
- **Elevated:** height >= 2.5 m.

Elevated is the positive class. Labels are derived directly from unrounded, unclipped regression predictions; no separate classifiers are trained. The 2.5 m boundary is a chosen research threshold, not an official coastal warning threshold.

## Repository structure

```text
Wave Forecasting/
|-- data/udupi_wave_wind_2015_2024.csv
|-- src/
|   |-- 01_eda.py
|   |-- 02_preprocessing.py
|   |-- 03_feature_engineering.py
|   |-- 04_model_training.py
|   `-- 05_model_evaluation.py
|-- models/
|   |-- linear_regression.joblib
|   |-- decision_tree.joblib
|   |-- random_forest.joblib
|   |-- xgboost.joblib
|   `-- model_metadata.json
|-- results/
|   |-- metrics/
|   |-- figures/eda/
|   `-- predictions/
|-- experiments/archive/
|-- README.md
|-- requirements.txt
`-- .gitignore
```

The five scripts form the implemented workflow. Archived experiments are preserved for provenance and are not imported by active scripts. Model metadata records feature order, constructor settings, training dates and counts, package versions, and model/data checksums.

## Five-phase workflow

### 1. Exploratory data analysis

`src/01_eda.py` inspects dataset dimensions, columns, data types, missing values, timestamp coverage and observation intervals. It reports descriptive wave-height statistics, monthly/yearly patterns and Elevated-wave frequency, and saves distribution, seasonal and wind-wave relationship figures.

Full-history EDA is descriptive. It does not constitute independent model validation.

### 2. Data preprocessing

`src/02_preprocessing.py` parses timestamps as UTC, sorts observations chronologically and validates the original schema and complete three-hourly grid. It checks for duplicate timestamps, missing/nonfinite measurements, invalid heights or directions, unexpected coordinates and inconsistent wind magnitude.

No imputation is performed: unexpected missing data raise an error. The original CSV remains unchanged. The script also defines the raw chronological split boundaries before feature warm-up and target-overlap exclusions.

### 3. Feature engineering

`src/03_feature_engineering.py` constructs these 18 features in the fixed order used by every model:

```python
COMPACT_FEATURES = [
    "VHM0", "VHM0_SW1", "VHM0_WW", "VTM02", "VTPK",
    "eastward_wind", "northward_wind", "wind_speed",
    "VMDR_sin", "VMDR_cos",
    "VHM0_lag_6h", "VHM0_lag_24h", "VHM0_change_3h",
    "VHM0_roll_mean_12h", "VHM0_roll_std_12h",
    "wind_speed_roll_mean_12h", "month_sin", "month_cos",
]
```

| Feature group | Definition |
|---|---|
| Current measurements | Five wave variables and three wind variables available at origin `t` |
| Direction encoding | Sine and cosine of `VMDR`, converted from degrees to radians |
| Height lags | Heights two and eight observations before `t`, corresponding to 6 h and 24 h |
| Height change | Current height minus the preceding three-hour observation |
| Rolling summaries | Four observations ending at `t`; wave-height mean/std and wind-speed mean |
| Month encoding | Sine and cosine of `2*pi*month/12` |

Rolling windows include the current observation and the preceding three observations, with `min_periods=4`. Rolling standard deviation uses `ddof=1`. CSV wind speed is retained exactly after checking it against `hypot(eastward_wind, northward_wind)`.

All predictors use information available at the forecast origin. The first eight origins are excluded for insufficient lag history; the final four lack future targets. A causality check perturbs future measurements and confirms that earlier feature values remain unchanged.

### 4. Model training

`src/04_model_training.py` uses the original fixed controlled configurations:

| Model | Main configuration |
|---|---|
| Linear Regression | StandardScaler followed by ordinary LinearRegression with an intercept |
| Decision Tree | Maximum depth 8; minimum leaf size 10; minimum split size 10 |
| Random Forest | 120 trees; maximum depth 10; minimum leaf size 5; feature fraction 0.7 |
| XGBoost | 220 trees; maximum depth 4; learning rate 0.05; L2 regularization 8 |

XGBoost also uses row and column subsampling of 0.8, a squared-error objective and histogram trees. Tree models use random seed 42; Random Forest and XGBoost use four workers. Complete settings are recorded in `models/model_metadata.json`.

Each model is first fitted on training data to produce validation predictions. A fresh clone is then refitted on eligible training plus validation observations and saved. Linear Regression scaling is fitted within its pipeline using only the corresponding fitting partition. The canonical workflow does not perform hyperparameter searches, early stopping, balancing or test-based tuning.

### 5. Model evaluation

`src/05_model_evaluation.py` reloads the saved models, checks their provenance and evaluates test predictions without fitting.

Regression metrics include MAE, RMSE, R-squared, mean bias and median absolute error. Binary evaluation includes accuracy, balanced accuracy, precision/recall/F1 for both classes, macro-F1, weighted-F1 and confusion counts. Class order is fixed as `[Normal, Elevated]`, with safe zero-division handling.

## Chronological partitions and leakage precautions

The original floor-based 70/10/20 split boundaries are retained. Four origins are purged at each training/validation boundary so retained targets occur strictly before the next partition's first forecast origin. Data are never shuffled.

| Partition | Eligible observations | First forecast origin (UTC) | Last forecast origin (UTC) |
|---|---:|---|---|
| Training | 20,444 | 2015-01-02 00:00 | 2021-12-31 09:00 |
| Validation | 2,918 | 2022-01-01 00:00 | 2022-12-31 15:00 |
| Test | 5,842 | 2023-01-01 06:00 | 2024-12-31 09:00 |

Final models are refitted on **23,362** training-plus-validation observations. Test targets extend from 2023-01-01 18:00 UTC to 2024-12-31 21:00 UTC. Past measurements across partition boundaries may provide causal feature history, but no test observation is used to fit a model or scaler.

## Final 12-hour results

The following values are calculated on the 5,842 test observations. Lower MAE/RMSE and higher R-squared/F1 indicate better measured performance.

| Model | MAE (m) | RMSE (m) | R-squared | Elevated F1 |
|---|---:|---:|---:|---:|
| Linear Regression | 0.071560 | 0.103027 | 0.978870 | 0.879240 |
| Decision Tree | 0.075992 | 0.112207 | 0.974937 | 0.859175 |
| Random Forest | 0.069742 | 0.102122 | 0.979240 | 0.884298 |
| XGBoost | 0.068209 | 0.099548 | 0.980273 | 0.885559 |

### Regression-derived Elevated detection

The test set contains **380 Elevated observations (6.5046%)** and **5,462 Normal observations**. Accuracy should therefore be considered alongside class-specific recall and F1.

| Model | Accuracy | Balanced accuracy | Elevated precision | Elevated recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.984765 | 0.923295 | 0.907563 | 0.852632 | 0.935555 |
| Decision Tree | 0.983054 | 0.895446 | 0.934985 | 0.794737 | 0.925080 |
| Random Forest | 0.985621 | 0.920080 | 0.927746 | 0.844737 | 0.938316 |
| XGBoost | 0.985621 | 0.924977 | 0.918079 | 0.855263 | 0.938944 |

XGBoost has the lowest measured regression RMSE. Its Elevated F1 advantage over Random Forest is small; these observed differences alone do not establish statistical significance. Full numerical metrics and confusion counts are retained under `results/metrics/`.

## Installation and execution

The recorded environment uses **Python 3.11.9**. From the repository root in PowerShell, create an environment only if `.venv` does not already exist:

```powershell
py -3.11 -m venv .venv
```

Install the pinned dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the five implemented phases in order:

```powershell
.\.venv\Scripts\python.exe -B src/01_eda.py
.\.venv\Scripts\python.exe -B src/02_preprocessing.py
.\.venv\Scripts\python.exe -B src/03_feature_engineering.py
.\.venv\Scripts\python.exe -B src/04_model_training.py
.\.venv\Scripts\python.exe -B src/05_model_evaluation.py
```

Training refreshes the four active models and validation outputs. Evaluation refreshes test metrics, predictions and figures. To evaluate the existing saved models without retraining, run only `05_model_evaluation.py`. Script paths are resolved relative to their file locations; no additional datasets or intermediate feature CSVs are required.

### Load the dataset with Pandas

From the repository root:

```python
import pandas as pd

df = pd.read_csv("data/udupi_wave_wind_2015_2024.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
print(df.shape)
print(df.head())
```

## Output files and reproducibility

| Location | Contents |
|---|---|
| `models/` | Four final regression models and `model_metadata.json` |
| `results/metrics/regression_metrics.csv` | Final test regression metrics |
| `results/metrics/classification_metrics.csv` | Both classes' binary metrics and confusion counts |
| `results/metrics/confusion_matrices.csv` | TN, FP, FN and TP for each model |
| `results/metrics/validation_regression_metrics.csv` | Training-only fits evaluated on validation data |
| `results/predictions/validation_predictions.csv` | Continuous validation predictions |
| `results/predictions/test_predictions.csv` | Actual/predicted heights, origin/target timestamps and binary labels |
| `results/figures/eda/` | Wave-height distribution, seasonal patterns and wind-wave relationships |
| `results/figures/` | Model comparisons, confusion matrices, actual/predicted heights and residuals |
| `results/metrics/verification.json` | Existing consolidation and reproducibility audit record |

Consolidation verified exact agreement of feature/target partitions with the earlier 12-hour reference, matching estimator settings, independent metric calculations and predictions within an absolute tolerance of `1e-12 m`. The original CSV checksum and archived historical contents were preserved.

Historical temporary-file paths in `verification.json` identify the earlier audit's inventory and relocation records. They are provenance references, not dependencies required by the canonical scripts. The archive remains separate from the active workflow.

## Methodological limitations

- The dataset represents offshore model/reanalysis conditions at one reference cell, not direct beach measurements, nearshore transformation or wave run-up.
- Origin-time input availability and operational data delays have not been established. Historical skill does not automatically imply operational forecasting capability.
- The 12-hour horizon was selected after exploratory comparisons, and the test period has been inspected during development. Reported metrics are **exploratory, not independent prospective validation**.
- Class imbalance, serial correlation, seasonal variation and rare extreme conditions limit conclusions drawn from aggregate scores.
- The fixed 2.5 m threshold is a research choice and must not be presented as an official marine warning criterion.
