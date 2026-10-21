# Machine Learning-Based Forecasting of High-Wave Events Along the Udupi Coast

An undergraduate machine learning project investigating whether historical offshore wave and wind conditions can predict the **maximum significant wave-height category over the next six hours** near Udupi, Karnataka, India.

**Current status:** historical data preparation is complete. The final CSV is available and validated. `notebooks/01_eda.ipynb` contains initial inspection and plotting code, intentionally not executed yet. The other four notebooks contain descriptive Markdown only. Preprocessing, feature engineering, training and evaluation remain planned; no models have been trained and no forecasting performance has been established.

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

The research objective is to classify the highest significant wave height occurring at the next two three-hourly timestamps, using information available at prediction time. Significant wave height is a sea-state measure; this target does not describe the largest individual wave or beach run-up. The project is a research study, not an operational marine warning service.

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
|   |-- 01_eda.ipynb               # Initial EDA code; unexecuted
|   |-- 02_preprocessing.ipynb     # Markdown placeholder
|   |-- 03_feature_engineering.ipynb # Markdown placeholder
|   |-- 04_model_training.ipynb    # Markdown placeholder
|   `-- 05_model_evaluation.ipynb  # Markdown placeholder
`-- data/
    `-- udupi_wave_wind_2015_2024.csv
```

The notebook workflow replaces the previous docstring-only `src/` modules and `main.py`. `prepare_data.py` remains the independent data-acquisition script. There are no separate configuration or reporting frameworks. `requirements.txt` retains all data-preparation dependencies and adds `ipykernel` for VS Code notebook execution and Matplotlib for inline plots. Model-training dependencies will be added when training is implemented.

## Machine Learning Workflow

Initial EDA code is available in the first notebook but has not been executed. All subsequent stages are **planned**, not implemented. The intended order is:

1. **Exploratory data analysis - `01_eda.ipynb`:** run the initial read-only inspection cells to display shape, column types, missing values, UTC coverage, three-hourly continuity, numerical statistics, current-height range counts and monthly/yearly summaries. Three inline plots show the height histogram, monthly distributions and monthly exceedance percentages. These are current-height ranges, not future-target classes. Later EDA can build on the initial inspection after discussion.
2. **Data preprocessing - `02_preprocessing.ipynb`:** agree on data-quality handling and chronological separation before implementing transformations. No preprocessing is currently performed.
3. **Feature engineering:** construct causal wave/wind histories, trailing statistics, changes, circular direction encodings, and seasonal features. Use only measurements available at or before prediction time.
4. **Six-hour target construction:** obtain both exact future wave heights and assign the provisional category defined below.
5. **Chronological splitting:** allocate earlier observations to training, subsequent observations to validation, and the latest observations to testing. Exact dates will be agreed before implementation.
6. **Model training:** compare Logistic Regression, Decision Tree, Random Forest, and XGBoost under a consistent validation protocol. Fit any learned preprocessing only on training data.
7. **Model evaluation:** assess overall and class-wise performance on later, unseen observations using the methodology below.

No random temporal mixing, full-dataset scaling, or inclusion of future-derived columns in predictors is intended. These safeguards will need explicit implementation and verification in future tasks.

## Target Definition

For a prediction issued at timestamp `t`, the planned continuous diagnostic is:

```text
future_max_height = max(VHM0[t+3h], VHM0[t+6h])
```

Both exact future timestamps must exist and contain valid heights. The current height at `t` is not included in this future maximum. Missing future values must not be fabricated or interpolated to create labels.

| Class | Category | Future maximum significant wave height |
|---|---|---|
| 0 | Normal | Below 2.5 m |
| 1 | Elevated | 2.5–3.0 m inclusive |
| 2 | High-wave | Above 3.0 m |

These are **provisional research thresholds, not official marine warning thresholds**. They are defined for this study and have not yet been implemented as labels. Future-height diagnostics and labels must remain separate from predictor columns.

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
5. Review the EDA results before agreeing on preprocessing. Continue in filename order: `01_eda.ipynb` -> `02_preprocessing.ipynb` -> `03_feature_engineering.ipynb` -> `04_model_training.ipynb` -> `05_model_evaluation.ipynb`, implementing each later stage only after its methodology is discussed.

The EDA notebook uses relative paths supporting a kernel started from either `notebooks/` or the project root. It reads the existing CSV and does not download, fill, export, construct future targets or train models. Its stored code cells have no outputs or execution counts. The other four notebooks contain only descriptive Markdown headings and purpose notes, so there is no runnable training/evaluation workflow or command yet. Notebooks are independent documents; a future mechanism for passing data between ML stages has not been implemented.

## Evaluation Methodology

The planned evaluation will preserve time order rather than randomly shuffle observations. Training data will precede validation data, and a final chronological test period will remain unseen during model and hyperparameter selection.

Split boundaries must account for the six-hour target horizon: training labels must not reach into a later validation/test period. Historical predictors may use earlier observations that would have been available at prediction time. Learned scaling, imputation, or feature selection will be fitted on training data only.

Planned outputs include:

- **Confusion matrices** to inspect errors between the three categories.
- **Precision and recall per class**, with attention to missed high-wave events and false alarms.
- **Class-wise F1-scores and support**, alongside macro F1 to avoid hiding weak minority-class performance behind majority-class results.
- **Comparable validation and test results** for all four planned models using the same splits and target definition.

No split dates, fitted models, confusion matrices, accuracy scores, or other measured performance results exist yet. Decisions about imbalance handling and tuning will be made using training/validation data, without using the final test period for selection.

## Limitations

- The reference represents offshore model/reanalysis and gridded wind conditions, not direct beach measurements. Local bathymetry, breaking waves, run-up, tides, and beach-specific hazards are not resolved by this point dataset.
- Interpolated stress-equivalent wind is an estimate at the wave coordinate; it is not a colocated instrument measurement or an ordinary unadjusted wind observation.
- Forecasting performance has not been established. Model choice, feature usefulness, class balance, and generalization remain to be investigated.
- The research thresholds are not official coastal or marine warning criteria.
- Historical retrospective evaluation does not automatically establish operational forecasting capability. Source latency, availability at prediction time, reanalysis revisions, changing conditions, and prospective validation must be considered separately.
- This repository currently provides validated historical data, unexecuted initial EDA code and a notebook scaffold for the remaining ML stages. The README will be updated with actual methods, commands, and results as each ML stage is implemented.
