# Machine Learning-Based Forecasting of High-Wave Events Along the Udupi Coast

An undergraduate machine learning project for **direct multiclass classification of the highest experimental wave-warning category reached over the next six hours** near offshore Udupi, Karnataka, India. This is a classification study, not a regression study.

**Current status:** historical CSV preparation, Phase 1 EDA and Phase 2 preprocessing are complete. The project has finalized the 2.5 m / 3.0 m experimental classification boundaries. `notebooks/03_feature_engineering.ipynb` contains deterministic causal feature engineering and X/y preparation code, supplied unexecuted for manual review. Notebooks 04 and 05 remain Markdown placeholders. No classifier has been trained or evaluated, and no operational warning capability has been validated.

## Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Project Structure](#project-structure)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Target Definition](#target-definition)
- [How to Run](#how-to-run)
- [Evaluation Methodology](#evaluation-methodology)
- [Limitations](#limitations)

## Project Overview

High-wave conditions can affect fishing, navigation, coastal activities, and marine safety along the Udupi coast. Forecasting changes in wave height can support planning and provide a basis for studying potentially hazardous offshore conditions.

The research objective is to classify the highest experimental warning category reached at the next two three-hourly timestamps, using information available at prediction time. A future maximum significant height is calculated only as an intermediate for assigning candidate categories; it is not a regression training target. Significant wave height is a sea-state measure; this target does not describe the largest individual wave or beach run-up. The project is a research study, not an operational marine warning service.

## Dataset

The final dataset is stored at:

```text
data/udupi_wave_wind_2015_2024.csv
```

| Property | Value |
|---|---|
| Historical coverage | 2015-01-01 00:00 to 2024-12-31 21:00 UTC |
| Sampling interval | 3 hours |
| Rows | 29,224, including leap years |
| Columns | 20, including timestamp and coordinates |
| Reference location | Offshore near 13.4° N, 74.6° E |
| Exact wave-cell coordinates | 13.399999618530273° N, 74.5999984741211° E |
| Validation status | Complete expected timestamps; no duplicates or missing values |

### Copernicus Marine sources

| Source | Dataset ID | Product | Version used | Source resolution |
|---|---|---|---|---|
| Waves | `cmems_mod_glo_wav_my_0.2deg_PT3H-i` | `GLOBAL_MULTIYEAR_WAV_001_032` | `202411` | Approximately 0.2°, three-hourly |
| Wind | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | `WIND_GLO_PHY_L4_MY_012_006` | `202211` | 0.125°, hourly |

Wave variables describe offshore model conditions. Wind variables are gridded stress-equivalent 10 m components derived from scatterometer observations and atmospheric reanalysis. Stress-equivalent winds express surface stress using a reference air-density convention. Final wind values are interpolated estimates at the wave cell, not direct measurements taken there.

### Variables

| Columns | Meaning | Units |
|---|---|---|
| `timestamp` | UTC observation timestamp | UTC datetime |
| `latitude`, `longitude` | Shared offshore reference coordinates | Degrees |
| `VHM0` | Total significant wave height | m |
| `VHM0_SW1`, `VHM0_SW2` | Primary and secondary swell significant heights | m |
| `VHM0_WW` | Wind-wave significant height | m |
| `VMDR`, `VMDR_SW1`, `VMDR_SW2`, `VMDR_WW` | Total, primary swell, secondary swell, and wind-wave from-directions | Degrees clockwise from north |
| `VTM01_SW1`, `VTM01_SW2`, `VTM01_WW` | Mean periods of the swell and wind-wave partitions | s |
| `VTM02` | Total mean period from spectral moments (0, 2) | s |
| `VTM10` | Total period from spectral moments (-1, 0) | s |
| `VTPK` | Peak wave period | s |
| `eastward_wind`, `northward_wind` | Interpolated stress-equivalent wind components | m/s |
| `wind_speed` | Magnitude calculated from interpolated components | m/s |

The CSV contains original wave variables and aligned wind estimates. Engineered predictors, future wave-height values, and classification labels have **not** been added.

## Data Preparation

The self-contained `prepare_data.py` script implements the following workflow:

1. Inspect the specified Copernicus datasets and download required variables in annual NetCDF chunks when network acquisition is explicitly enabled.
2. Select the nearest valid offshore wave cell around the requested reference location and retain its exact coordinates across years.
3. Retrieve surrounding wind cells and bilinearly interpolate the eastward and northward components independently onto that exact wave coordinate.
4. Calculate wind speed **after** component interpolation: `wind_speed = sqrt(eastward_wind**2 + northward_wind**2)`.
5. Match hourly wind estimates to exact three-hourly wave timestamps in UTC. No nearest-time or future-wind substitution is used.
6. Combine the original wave variables, interpolated components, speed, and reference coordinates into one CSV.
7. Validate expected coverage, duplicates, missing measurements, coordinates, interpolation bounds, and wind-speed arithmetic. Reload the written CSV and check numerical equivalence before publishing it.

The verified surrounding wind coordinates are latitudes 13.3125°/13.4375° N and longitudes 74.5625°/74.6875° E. The script rejects extrapolation and invalid required neighbors; it does not invent or fill measurements.

Raw NetCDF files were moved out of the active project after verification. The maintainer's local archive is the sibling folder `../Wave Forecasting Raw Archive 2026-10-21/raw/`; it is not part of this repository. Its 22 NetCDF files and two legacy metadata files were verified against a size/checksum inventory. The existing final CSV can be used directly without restoring this archive.

## Project Structure

```text
Wave Forecasting/
|-- .venv/                         # Local environment; excluded from version control
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- prepare_data.py                # Implemented CSV preparation
|-- notebooks/
|   |-- 01_eda.ipynb               # Completed EDA; preserved unchanged
|   |-- 02_preprocessing.ipynb     # Completed validation, target investigation and chronological split
|   |-- 03_feature_engineering.ipynb # Causal features and X/y objects; unexecuted
|   |-- 04_model_training.ipynb    # Markdown placeholder
|   `-- 05_model_evaluation.ipynb  # Markdown placeholder
`-- data/
    `-- udupi_wave_wind_2015_2024.csv
```

The notebook workflow replaces the previous docstring-only `src/` modules and `main.py`. `prepare_data.py` remains the independent data-acquisition script. There are no separate configuration or reporting frameworks. `requirements.txt` retains all data-preparation dependencies and adds `ipykernel` for VS Code notebook execution and Matplotlib for inline plots. Model-training dependencies will be added when training is implemented.

## Machine Learning Workflow

| Phase | Notebook | Status |
|---|---|---|
| 1. Dataset inspection and EDA | `01_eda.ipynb` | Completed; original cells and outputs preserved |
| 2. Preprocessing and target investigation | `02_preprocessing.ipynb` | Completed; original cells and outputs preserved |
| 3. Causal feature engineering | `03_feature_engineering.ipynb` | Implemented, awaiting manual execution/review |
| 4. Classifier training | `04_model_training.ipynb` | Markdown placeholder; not implemented |
| 5. Classifier evaluation | `05_model_evaluation.ipynb` | Markdown placeholder; not implemented |

Phase 2 validated all 29,224 source observations, found no missing measurements/duplicate timestamps/irregular intervals, and retained 29,222 complete future horizons. Four label-boundary exclusions left 20,453 training, 2,920 validation and 5,845 test observations. Its historical threshold investigation is preserved unchanged; the subsequent project decision finalizes the original 2.5 m / 3.0 m pair.

### Phase 3 feature methodology

Phase 3 independently reads the original CSV, reconstructs exact future labels, and verifies the Phase 2 counts and UTC dates before building inputs. It contains one straightforward deterministic `make_features` function with explicit feature specifications and a complete in-notebook feature dictionary.

| Input group | Definition | Count |
|---|---|---:|
| Current measurements | Four heights, six periods, eastward/northward wind and speed | 13 |
| Wave directions | Sine/cosine of VMDR, VMDR_SW1, VMDR_SW2 and VMDR_WW, replacing raw degrees | 8 |
| Historical lags | VHM0 at 3/6/12/24h; speed at 3/6/12h; primary swell height at 6h; VTM02 at 3h | 9 |
| Changes | Current minus earlier VHM0 at 3/6/12h and speed at 3/6h | 5 |
| Trailing statistics | VHM0 mean at 6/12/24h, max at 12/24h, sample std at 12h; speed mean at 6/12h, max at 12h | 9 |
| Seasonality | Sine/cosine of UTC month; no raw month or year | 2 |
| **Total** | Explicitly selected numerical inputs | **46** |

The current numeric measurements are `VHM0`, `VHM0_SW1`, `VHM0_SW2`, `VHM0_WW`, `VTM01_SW1`, `VTM01_SW2`, `VTM01_WW`, `VTM02`, `VTM10`, `VTPK`, `eastward_wind`, `northward_wind` and `wind_speed`. Constant latitude/longitude are metadata, not predictors. There is no separate wind-direction angle column; signed components retain wind orientation. Speed is dependent on U/V but provides magnitude explicitly, a useful representation for tree models. This redundancy is documented, not removed through learned feature selection.

All lag offsets are positive. Changes are differences in source units, not percentage changes or rates. A trailing 24-hour rolling window contains **8 observations from t-21h through t**, including current; 6h/12h windows contain 2/4 observations. Windows require complete history, use `center=False`, and standard deviation uses `ddof=1`. Wind histories use the available three-hourly CSV, not the archived hourly wind series.

Features are built across the full chronological source before partition selection, preserving valid history across years and split starts. No missing history is filled. The longest lag requires eight earlier rows, so the expected first complete timestamp is **2015-01-02 00:00 UTC**. Only eight initial training rows should be excluded; validation/test retain all Phase 2 rows and their exact boundaries.

Expected model-ready shapes, checked by the notebook when manually executed:

| Partition | Phase 2 rows | History exclusions | X shape | y shape |
|---|---:|---:|---|---|
| Training | 20,453 | 8 | `(20445, 46)` | `(20445,)` |
| Validation | 2,920 | 0 | `(2920, 46)` | `(2920,)` |
| Testing | 5,845 | 0 | `(5845, 46)` | `(5845,)` |

These are deterministic expected dimensions, not claimed execution results for the new notebook. All six objects remain in memory. Feature previews, class distributions, a training-only correlation heatmap and a fixed training-window wave/trend plot appear inline when run. Correlation is descriptive, not proof of predictive importance; no test statistics select features.

### Leakage prevention

- Future heights/timestamps, `future_max_wave_height` and `target_class` never enter X; negative shifts are isolated to label construction.
- Positive feature shifts and trailing complete-history rolling windows use only information available at or before prediction time.
- The notebook reproduces Phase 2 boundaries and label-horizon purging, then applies history exclusions without recalculating split cutoffs.
- Validation/test may use genuinely available earlier source measurements, including rows excluded as supervised boundary samples. Earlier-partition labels cannot use observations from the next partition.
- An in-notebook future-perturbation check requires all earlier inputs to remain identical after later measurements are changed in a temporary copy.
- No scaling, normalization, imputation, SMOTE, oversampling, class weighting, learned feature selection or model fitting occurs. Future learned transformations must be fitted on training only.

Planned classifiers remain **Logistic Regression, Decision Tree, Random Forest and XGBoost**. These will be implemented only in Phase 4 after the feature outputs are reviewed.

## Target Definition

At timestamp t, the label-construction diagnostic is:

```text
future_max_wave_height = max(VHM0[t+3h], VHM0[t+6h])
```

Both future timestamps must match the exact offsets and contain valid heights. The maximum uses `skipna=False`; final incomplete horizons remain unclassified and are excluded. This is the maximum of **two sampled future significant wave heights**, not the maximum individual wave during continuous observation. The project trains classifiers, not a regression model on this diagnostic.

### Finalized experimental offshore categories

| Class | Category | Future maximum significant wave height |
|---|---|---|
| 0 | Normal | Below 2.5 m |
| 1 | Elevated / Watch | 2.5-3.0 m inclusive |
| 2 | High / Alert | Above 3.0 m |

The thresholds are **finalized for this project**. Exactly 2.5 m and exactly 3.0 m map to Class 1. They define experimental offshore research categories and are **not official operational marine warnings**. Phase 3 does not change or investigate alternative thresholds. Future diagnostics and integer classification labels remain separate from all input predictors.

## How to Run

### Environment setup

Use Python 3.11. From the project root, activate the existing local environment and install the currently declared dependencies if needed:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On a fresh installation without `.venv/`, create it first:

```powershell
py -3.11 -m venv .venv
```

The current requirements support CSV preparation, a Python notebook kernel, and EDA plots. Model-training libraries are not yet declared because training is not implemented. No packages were installed as part of the notebook restructuring; run the installation command yourself before selecting the kernel if needed.

### Load the existing CSV

Data preparation does not need to be rerun before analysis:

```python
import pandas as pd

df = pd.read_csv(
    'data/udupi_wave_wind_2015_2024.csv',
    parse_dates=['timestamp'],
)
df = df.set_index('timestamp')
print(df.shape)  # (29224, 19): timestamp is now the index
```

### Optional data regeneration

To regenerate offline, first restore the archived `raw/` folder to `data/raw/`. Then run from the project root:

```powershell
.\.venv\Scripts\python.exe -B prepare_data.py --full-approved
```

For an explicitly approved acquisition when raw files are unavailable, authenticate interactively if needed and enable downloads:

```powershell
.\.venv\Scripts\copernicusmarine.exe login
.\.venv\Scripts\python.exe -B prepare_data.py --full-approved --download
```

Copernicus handles authentication; do not put credentials in source files. Network acquisition is disabled without `--download`. Running the script without flags selects the January 2022 sample and requires suitable cached raw files; those caches are currently archived.

### Execute notebooks in VS Code

1. Enable the **Python** and **Jupyter** extensions in VS Code.
2. Open the project folder and then `notebooks/01_eda.ipynb`.
3. Use **Select Kernel** to choose the project `.venv` Python 3.11 environment. If it is unavailable, install the declared requirements in that environment first.
4. Run cells manually from top to bottom with the cell run button or **Shift+Enter**. DataFrames, numerical tables and figures appear below their cells; no output files are exported.
5. With Phases 1 and 2 complete, open `notebooks/03_feature_engineering.ipynb` and run it manually from top to bottom. It loads the source independently and checks the saved Phase 2 split. Review feature descriptions, leakage checks, shapes and training-only plots before proceeding. Notebooks 04 and 05 remain Markdown placeholders.

The notebooks support a kernel working directory of either `notebooks/` or the project root. Phase 3 has been authored without automatic execution: all saved code cells have empty outputs and null execution counts. It produces only in-memory objects and inline displays, with no new datasets, model files or reports.

Notebooks do not automatically share variables. Phase 4 must reproduce the deterministic source loading, finalized targets, fixed splits, feature specifications and integrity checks before training. The handoff mechanism will be agreed during Phase 4; no pickle/joblib/CSV intermediate or automatic cross-notebook execution has been added.

## Evaluation Methodology

Phase 2 implements a **strict chronological 70/10/20 split** of complete-horizon observations. It uses `floor(0.70*N)` initial training rows, `floor(0.10*N)` subsequent validation rows, and assigns the remainder to testing. This is a proportion-based split, not a calendar-year split; there is no shuffling or random train/test split. The completed Phase 2 outputs establish the exact boundaries; Phase 3 reconstructs and verifies them before feature exclusions.

The Phase 2 retained prediction-time ranges are training **2015-01-01 00:00 to 2021-12-31 12:00 UTC**, validation **2021-12-31 21:00 to 2022-12-31 18:00 UTC**, and testing **2023-01-01 03:00 to 2024-12-31 15:00 UTC**. Phase 3 keeps the same partition cutoffs; the first usable training timestamp advances by 24 hours solely for historical context.

To prevent cross-partition label leakage, training rows whose `t+6h` reaches the nominal validation start and validation rows whose horizon reaches the nominal test start are excluded. Boundaries are not recalculated after this purge. The notebook reports excluded rows, usable counts, date ranges, percentages of labeled observations and percentages of retained observations. Small departures from 70/10/20 after purging are expected.

Future historical inputs must remain causal: earlier available observations can provide history, but no future values may enter a predictor. No learned transformation is fitted in Phases 2 or 3. The requested test class counts expose holdout prevalence for diagnostic review only; test counts and test-period statistics must not drive feature selection or tuning. The project thresholds are now fixed independently of Phase 3 feature inspection.

Planned classifier evaluation will report confusion matrices, precision, recall, F1-score and support for each class, plus macro F1. All four classifiers will use consistent partitions and the finalized 2.5 m / 3.0 m classification definition. The validation period may support model selection; the final test period must not drive tuning.

No models, classification predictions, confusion matrices, measured performance scores or operational capabilities have been produced. Neither Phase 2 nor Phase 3 implements regression training or regression evaluation metrics.

## Limitations

- The reference represents offshore model/reanalysis and gridded wind conditions, not direct beach measurements. Local bathymetry, breaking waves, run-up, tides, and beach-specific hazards are not resolved by this point dataset.
- Interpolated stress-equivalent wind is an estimate at the wave coordinate; it is not a colocated instrument measurement or an ordinary unadjusted wind observation.
- Forecasting performance has not been established. Model choice, feature usefulness, class balance, and generalization remain to be investigated.
- The research thresholds are not official coastal or marine warning criteria.
- Historical retrospective evaluation does not automatically establish operational forecasting capability. Source latency, availability at prediction time, reanalysis revisions, changing conditions, and prospective validation must be considered separately.
- Phases 1 and 2 are complete. Phase 3 feature code awaits manual execution/review; training and evaluation remain unimplemented. Thresholds are finalized for this experimental classification task, while model results and operational performance remain unestablished.
