# Machine Learning-Based Forecasting of High-Wave Events Along the Udupi Coast

An undergraduate machine learning project for **direct multiclass classification of the highest experimental wave-warning category reached over the next six hours** near offshore Udupi, Karnataka, India. This is a classification study, not a regression study.

**Current status:** historical CSV preparation and Phase 1 EDA are complete. `notebooks/02_preprocessing.ipynb` now contains validation, intermediate future-maximum construction, candidate-category analysis and chronological 70/10/20 partitioning code. Phase 2 has not been executed automatically; its results and warning thresholds remain to be reviewed. Notebooks 03, 04 and 05 contain Markdown placeholders only. No models have been trained or operational warning capability validated.

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
|   |-- 02_preprocessing.ipynb     # Phase 2 validation, target analysis and splits; unexecuted
|   |-- 03_feature_engineering.ipynb # Markdown placeholder
|   |-- 04_model_training.ipynb    # Markdown placeholder
|   `-- 05_model_evaluation.ipynb  # Markdown placeholder
`-- data/
    `-- udupi_wave_wind_2015_2024.csv
```

The notebook workflow replaces the previous docstring-only `src/` modules and `main.py`. `prepare_data.py` remains the independent data-acquisition script. There are no separate configuration or reporting frameworks. `requirements.txt` retains all data-preparation dependencies and adds `ipykernel` for VS Code notebook execution and Matplotlib for inline plots. Model-training dependencies will be added when training is implemented.

## Machine Learning Workflow

| Phase | Notebook | Status |
|---|---|---|
| 1. Dataset inspection and EDA | `01_eda.ipynb` | Completed; existing cells and outputs preserved |
| 2. Validation and warning-target investigation | `02_preprocessing.ipynb` | Implemented code, awaiting manual execution and review |
| 3. Historical feature engineering | `03_feature_engineering.ipynb` | Markdown placeholder; not implemented |
| 4. Classifier training | `04_model_training.ipynb` | Markdown placeholder; not implemented |
| 5. Classifier evaluation | `05_model_evaluation.ipynb` | Markdown placeholder; not implemented |

Phase 2 reads the CSV independently, parses UTC timestamps, sorts in memory, and checks schema, missingness, duplicates, complete three-hourly coverage, numeric/finite/nonnegative heights and physical consistency. Unexpected issues stop processing rather than being silently repaired. The original CSV remains unchanged.

It then constructs a future-height diagnostic, excludes incomplete final horizons, creates chronological partitions and purges target windows crossing their boundaries. The notebook compares the original threshold pair and two fixed sensitivity pairs, without finalizing any of them. Candidate labels exist only in memory for analysis; no final label column or new CSV is produced.

Planned feature engineering will use only current/past information: wave/wind histories, trailing statistics, changes, circular directions and seasonal features. The planned classifiers are **Logistic Regression, Decision Tree, Random Forest and XGBoost**. Training-only fitting of any learned transforms, feature selection or imbalance handling remains future work. No scaling, imputation, SMOTE, feature selection, model training or tuning is performed in Phase 2.

## Target Definition

The intermediate diagnostic at prediction time `t` is:

```text
future_max_wave_height = max(VHM0[t+3h], VHM0[t+6h])
```

Both future observations must exist at exactly the requested offsets and contain valid heights. The notebook verifies timestamp continuity and explicit +3h/+6h matches before using shifts, and requires both values with `skipna=False`. The final two source observations cannot form complete labels and are excluded from the in-memory labeled dataset, while the original data are preserved.

This is the maximum of **two future three-hourly significant wave-height observations**, not the maximum individual wave during continuous observation of a six-hour period. It is an intermediate for direct multiclass classification, **not a regression target for training**. Future measurements may define supervised outcomes, but every future-height/time diagnostic and candidate label must be excluded from model inputs.

### Original provisional thresholds: comparison only

| Class | Experimental category | Future maximum significant height |
|---|---|---|
| 0 | Normal | Below 2.5 m |
| 1 | Elevated | 2.5-3.0 m inclusive |
| 2 | High | Above 3.0 m |

**Thresholds remain under investigation. These are not finalized criteria or official marine safety warnings.** Phase 2 also compares fixed pairs `(2.25, 2.75)` and `(2.0, 2.5)` metres, using the same lower-inclusive/upper-inclusive Elevated convention. These are sensitivity checks, not quantile-balanced targets or scientifically validated recommendations.

Training-only comparisons show counts, percentages, minority-class size and monthly/yearly Elevated/High distributions. The notebook also shows requested full-period descriptive statistics and partition class counts, including test prevalence, strictly as diagnostics. **Do not use test statistics or full-period quantiles to choose thresholds, features or models.** Final decisions must use a physical/research rationale and training data, consulting validation only if necessary. More statistically convenient class balance does not validate a coastal-warning level.

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
5. With Phase 1 complete, open `notebooks/02_preprocessing.ipynb` and run its cells manually from top to bottom. It independently loads the CSV, so no shared kernel state from EDA is required. Review its validation summary, future-height verification, threshold comparisons and purged split before progressing to `03_feature_engineering.ipynb`, `04_model_training.ipynb` and `05_model_evaluation.ipynb`. Those later notebooks remain Markdown-only placeholders.

The implemented notebooks use relative paths supporting a kernel started from either `notebooks/` or the project root. Phase 2 has been authored without automatic execution: its saved code cells have no execution counts or outputs. Running it displays DataFrames, numerical summaries, a histogram figure and a generated Markdown findings summary inline. It does not save datasets, reports or models, download observations, or modify the source CSV. It is a research review step, not an operational warning pipeline.

## Evaluation Methodology

Phase 2 implements a **strict chronological 70/10/20 split** of complete-horizon observations. It uses `floor(0.70*N)` initial training rows, `floor(0.10*N)` subsequent validation rows, and assigns the remainder to testing. This is a proportion-based split, not a calendar-year split; there is no shuffling or random train/test split. Exact row counts and UTC boundaries are displayed when the notebook is run.

To prevent cross-partition label leakage, training rows whose `t+6h` reaches the nominal validation start and validation rows whose horizon reaches the nominal test start are excluded. Boundaries are not recalculated after this purge. The notebook reports excluded rows, usable counts, date ranges, percentages of labeled observations and percentages of retained observations. Small departures from 70/10/20 after purging are expected.

Future historical inputs must remain causal: earlier available observations can provide history, but no future values may enter a predictor. No learned transformation is fitted in Phase 2. The requested test class counts expose holdout prevalence for diagnostic review only; thresholds and tuning must not be selected using those counts or test-period descriptive statistics.

Planned classifier evaluation will report confusion matrices, precision, recall, F1-score and support for each class, plus macro F1. All four classifiers will use consistent partitions and the eventual agreed classification definition. The validation period may support model selection; the final test period must not drive tuning.

No models, classification predictions, confusion matrices, measured performance scores or operational capabilities have been produced. Phase 2 generates no regression training target or regression evaluation metrics.

## Limitations

- The reference represents offshore model/reanalysis and gridded wind conditions, not direct beach measurements. Local bathymetry, breaking waves, run-up, tides, and beach-specific hazards are not resolved by this point dataset.
- Interpolated stress-equivalent wind is an estimate at the wave coordinate; it is not a colocated instrument measurement or an ordinary unadjusted wind observation.
- Forecasting performance has not been established. Model choice, feature usefulness, class balance, and generalization remain to be investigated.
- The research thresholds are not official coastal or marine warning criteria.
- Historical retrospective evaluation does not automatically establish operational forecasting capability. Source latency, availability at prediction time, reanalysis revisions, changing conditions, and prospective validation must be considered separately.
- The repository provides validated historical data, completed EDA and an unexecuted Phase 2 review notebook. Final thresholds, feature methodology and all model results remain undecided. The README will be updated with reviewed findings as implementation progresses.
