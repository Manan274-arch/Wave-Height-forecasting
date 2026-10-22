# Direct binary classification versus regression-derived categories

Controlled exploratory experiment for the Udupi Wave Height Forecasting project. Executed locally on 22 October 2026 with the existing virtual environment. Four horizons, 16 saved direct classifiers and four deterministic persistence evaluations completed. Previous source files, models, results, documentation and the original CSV were preserved; repository cleanup was not performed.

## Target, features and chronology

Normal=0 when VHM0 < 2.5 m; Elevated=1 when VHM0 >= 2.5 m. Targets are labels of the exact future measurements at t+6/24/48/72 hours, respectively. They are not labels of a future maximum and do not use the old three-class target.

`run_experiment.py` imports `src/controlled_pipeline.py::prepare_controlled` without modifying it. The only adaptation is converting its existing continuous y into binary y. Feature values, order, origins, continuous-target alignment, raw split boundaries, initial history exclusions and horizon-specific purging are unchanged. Regression-derived predictions are checked for exact matching origins and actual future heights before paired evaluation.

All models use the same existing 18 compact features:

```text
VHM0, VHM0_SW1, VHM0_WW, VTM02, VTPK,
eastward_wind, northward_wind, wind_speed, VMDR_sin, VMDR_cos,
VHM0_lag_6h, VHM0_lag_24h, VHM0_change_3h,
VHM0_roll_mean_12h, VHM0_roll_std_12h, wind_speed_roll_mean_12h,
month_sin, month_cos
```

Raw boundaries remain positions 20,456 and 23,378 (validation 2022-01-01 00:00 UTC; testing 2023-01-01 06:00 UTC). Targets must remain strictly before the next partition. Rolling windows are trailing, include t and retain sample std ddof=1; no future wave/wind observation enters X. The existing full-feature builder is reused solely to obtain the established compact subset; feature selection was not repeated. A future-perturbation check passed.

| horizon_hours | partition | rows | start | end | purged | history_excluded | incomplete_target_excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Training | 20446 | 2015-01-02 00:00:00+00:00 | 2021-12-31 15:00:00+00:00 | 2 | 8 | 0 |
| 6 | Validation | 2920 | 2022-01-01 00:00:00+00:00 | 2022-12-31 21:00:00+00:00 | 2 | 0 | 0 |
| 6 | Testing | 5844 | 2023-01-01 06:00:00+00:00 | 2024-12-31 15:00:00+00:00 | 0 | 0 | 2 |
| 24 | Training | 20440 | 2015-01-02 00:00:00+00:00 | 2021-12-30 21:00:00+00:00 | 8 | 8 | 0 |
| 24 | Validation | 2914 | 2022-01-01 00:00:00+00:00 | 2022-12-31 03:00:00+00:00 | 8 | 0 | 0 |
| 24 | Testing | 5838 | 2023-01-01 06:00:00+00:00 | 2024-12-30 21:00:00+00:00 | 0 | 0 | 8 |
| 48 | Training | 20432 | 2015-01-02 00:00:00+00:00 | 2021-12-29 21:00:00+00:00 | 16 | 8 | 0 |
| 48 | Validation | 2906 | 2022-01-01 00:00:00+00:00 | 2022-12-30 03:00:00+00:00 | 16 | 0 | 0 |
| 48 | Testing | 5830 | 2023-01-01 06:00:00+00:00 | 2024-12-29 21:00:00+00:00 | 0 | 0 | 16 |
| 72 | Training | 20424 | 2015-01-02 00:00:00+00:00 | 2021-12-28 21:00:00+00:00 | 24 | 8 | 0 |
| 72 | Validation | 2898 | 2022-01-01 00:00:00+00:00 | 2022-12-29 03:00:00+00:00 | 24 | 0 | 0 |
| 72 | Testing | 5822 | 2023-01-01 06:00:00+00:00 | 2024-12-28 21:00:00+00:00 | 0 | 0 | 24 |

Refitting uses exactly the eligible original training+validation origins: 23,366 / 23,354 / 23,338 / 23,322 at 6 / 24 / 48 / 72 hours. The test sets have 380 actual Elevated observations each (6.502%, 6.509%, 6.518%, 6.527% respectively). Native eligible final-origin counts differ slightly across horizons; comparisons between approaches within a horizon always use identical timestamps and targets.

## Prespecified models and tuning

Logistic Regression has StandardScaler inside its pipeline; each fold fits its own scaler. Models use random_state=42. Forest/XGBoost use four workers; candidate/fold searches are sequential.

Four candidates per family, identical across horizons:

- Logistic Regression: C in {0.1, 1.0} crossed with class_weight in {None, balanced}; max_iter=3000, otherwise ordinary defaults.
- Decision Tree: (depth=4, leaf=20, split=20) or (depth=8, leaf=10, split=10), each unweighted or balanced.
- Random Forest: (trees=120, depth=10, leaf=5) or (trees=160, depth=16, leaf=2), each unweighted or balanced; max_features=0.7.
- XGBoost: (trees=160, depth=2, rate=0.07, lambda=5) or (trees=220, depth=4, rate=0.05, lambda=8), each with scale_pos_weight=1 or 3. Objective binary:logistic, histogram trees, subsample=0.8, colsample_bytree=0.8.

Balanced weights are calculated by each estimator using only its current fitting partition. XGBoost compares a fixed unweighted candidate with moderate fixed positive weighting of 3; no full-training class ratio is calculated and passed into earlier folds. Weighting is selected by CV, never assumed beneficial. No SMOTE or new dependencies are used.

Use three chronological folds with gap=horizon_hours/3 (2, 8, 16, 24), explicitly checking that each last training target precedes its fold-validation origin. Select highest mean CV Elevated F1, then lower sample standard deviation (ddof=1), then earlier candidate on exact ties. Every fold also records precision, recall, both class F1 scores and confusion counts. The searches total 192 CV fits, followed by 16 training-only validation fits and 16 final refits.

Every prediction is classified using probability >= 0.5. There is no threshold optimization or tuned-threshold result. This rule is used consistently in CV, validation and test evaluation, including exact probability ties. Probabilities from weighted models are not asserted to be calibrated event probabilities. Persistence directly thresholds current VHM0; it is not fitted and no artificial probability is supplied.

Validation predictions and probabilities are retained before final refitting. All 16 final models were frozen before exposing any new direct test results. Test inference reloads models, checks hashes and feature order, and never fits. No test scores were used to alter grids, features, weights or thresholds.

Selected parameters and pre-refit evidence:

| horizon_hours | method | selected_parameters | cv_mean_elevated_f1 | cv_sample_std | cv_mean_precision | cv_mean_recall | validation_elevated_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Logistic Regression | {"classifier__C": 1.0, "classifier__class_weight": null} | 0.889763 | 0.052803 | 0.920267 | 0.861987 | 0.953586 |
| 6 | Decision Tree | {"class_weight": null, "max_depth": 4, "min_samples_leaf": 20, "min_samples_split": 20} | 0.910728 | 0.045264 | 0.924971 | 0.897033 | 0.957806 |
| 6 | Random Forest | {"class_weight": null, "max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | 0.902048 | 0.058974 | 0.934673 | 0.872006 | 0.957806 |
| 6 | XGBoost | {"learning_rate": 0.07, "max_depth": 2, "n_estimators": 160, "reg_lambda": 5, "scale_pos_weight": 1} | 0.911155 | 0.048816 | 0.932802 | 0.890675 | 0.953782 |
| 24 | Logistic Regression | {"classifier__C": 0.1, "classifier__class_weight": null} | 0.742418 | 0.067029 | 0.806177 | 0.688504 | 0.876890 |
| 24 | Decision Tree | {"class_weight": null, "max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 10} | 0.709396 | 0.084487 | 0.750095 | 0.673138 | 0.848614 |
| 24 | Random Forest | {"class_weight": "balanced", "max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | 0.737209 | 0.080401 | 0.657685 | 0.843443 | 0.872180 |
| 24 | XGBoost | {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 220, "reg_lambda": 8, "scale_pos_weight": 3} | 0.753372 | 0.075409 | 0.755805 | 0.751795 | 0.879208 |
| 48 | Logistic Regression | {"classifier__C": 0.1, "classifier__class_weight": null} | 0.576440 | 0.040228 | 0.713525 | 0.487213 | 0.792541 |
| 48 | Decision Tree | {"class_weight": null, "max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 10} | 0.587105 | 0.064520 | 0.596297 | 0.588825 | 0.683603 |
| 48 | Random Forest | {"class_weight": null, "max_depth": 16, "max_features": 0.7, "min_samples_leaf": 2, "n_estimators": 160} | 0.574618 | 0.080424 | 0.661193 | 0.509881 | 0.752381 |
| 48 | XGBoost | {"learning_rate": 0.07, "max_depth": 2, "n_estimators": 160, "reg_lambda": 5, "scale_pos_weight": 3} | 0.593030 | 0.050754 | 0.588234 | 0.609476 | 0.838966 |
| 72 | Logistic Regression | {"classifier__C": 0.1, "classifier__class_weight": "balanced"} | 0.485168 | 0.069354 | 0.346375 | 0.837817 | 0.641711 |
| 72 | Decision Tree | {"class_weight": "balanced", "max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 10} | 0.464515 | 0.015488 | 0.386275 | 0.607673 | 0.580460 |
| 72 | Random Forest | {"class_weight": "balanced", "max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | 0.506665 | 0.029838 | 0.430167 | 0.639358 | 0.759227 |
| 72 | XGBoost | {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 220, "reg_lambda": 8, "scale_pos_weight": 3} | 0.516634 | 0.046903 | 0.521291 | 0.530202 | 0.777778 |

## A. Direct test classification

All metrics use a 0-1 scale. Elevated is positive, class order is [Normal, Elevated], and zero_division=0. Balanced accuracy averages recalls of classes present in the actual sample; fixed-class macro-F1 includes both classes. All actual partitions here contain both classes.

| horizon_hours | method | accuracy | balanced_accuracy | elevated_precision | elevated_recall | elevated_f1 | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Persistence | 0.990418 | 0.960596 | 0.926316 | 0.926316 | 0.926316 | 0.960596 |
| 6 | Logistic Regression | 0.991786 | 0.952758 | 0.963687 | 0.907895 | 0.934959 | 0.965288 |
| 6 | Decision Tree | 0.991444 | 0.953799 | 0.955801 | 0.910526 | 0.932615 | 0.964023 |
| 6 | Random Forest | 0.990760 | 0.947312 | 0.957865 | 0.897368 | 0.926630 | 0.960850 |
| 6 | XGBoost | 0.991786 | 0.953982 | 0.961111 | 0.910526 | 0.935135 | 0.965375 |
| 24 | Persistence | 0.977732 | 0.908519 | 0.828947 | 0.828947 | 0.828947 | 0.908519 |
| 24 | Logistic Regression | 0.977903 | 0.871885 | 0.893417 | 0.750000 | 0.815451 | 0.901849 |
| 24 | Decision Tree | 0.972765 | 0.859344 | 0.831832 | 0.728947 | 0.776999 | 0.881248 |
| 24 | Random Forest | 0.975334 | 0.939065 | 0.764574 | 0.897368 | 0.825666 | 0.906197 |
| 24 | XGBoost | 0.976704 | 0.923884 | 0.796117 | 0.863158 | 0.828283 | 0.907894 |
| 48 | Persistence | 0.965009 | 0.856432 | 0.731579 | 0.731579 | 0.731579 | 0.856432 |
| 48 | Logistic Regression | 0.968096 | 0.804225 | 0.854015 | 0.615789 | 0.715596 | 0.849348 |
| 48 | Decision Tree | 0.952830 | 0.756890 | 0.675585 | 0.531579 | 0.594993 | 0.784975 |
| 48 | Random Forest | 0.966038 | 0.790884 | 0.842105 | 0.589474 | 0.693498 | 0.837761 |
| 48 | XGBoost | 0.965695 | 0.877607 | 0.719512 | 0.776316 | 0.746835 | 0.864218 |
| 72 | Persistence | 0.951220 | 0.800111 | 0.626316 | 0.626316 | 0.626316 | 0.800111 |
| 72 | Logistic Regression | 0.897801 | 0.937989 | 0.388370 | 0.984211 | 0.556962 | 0.749600 |
| 72 | Decision Tree | 0.919443 | 0.889594 | 0.439783 | 0.855263 | 0.580876 | 0.768158 |
| 72 | Random Forest | 0.939368 | 0.890461 | 0.522241 | 0.834211 | 0.642351 | 0.804613 |
| 72 | XGBoost | 0.953624 | 0.845458 | 0.625571 | 0.721053 | 0.669927 | 0.822493 |

`test_metrics.csv` also contains Normal precision/recall/F1, weighted-F1, Elevated counts/percentages and TP/FP/TN/FN for all 20 combinations.

## B. Regression-derived -> direct comparison

Logistic Regression is paired with regression-derived Linear Regression; other model families pair by name. Entries below show derived -> direct. The persistence row is identical under both approaches. `comparison.csv` also records explicit direct-minus-derived deltas; positive precision/recall/F1 deltas are increases, while negative FP/FN deltas mean fewer errors.

| horizon_hours | method | Elevated precision | Elevated recall | Elevated F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| 6 | Persistence | 0.926316 -> 0.926316 | 0.926316 -> 0.926316 | 0.926316 -> 0.926316 | 0.960596 -> 0.960596 |
| 6 | Logistic Regression | 0.943089 -> 0.963687 | 0.915789 -> 0.907895 | 0.929239 -> 0.934959 | 0.962197 -> 0.965288 |
| 6 | Decision Tree | 0.962857 -> 0.955801 | 0.886842 -> 0.910526 | 0.923288 -> 0.932615 | 0.959089 -> 0.964023 |
| 6 | Random Forest | 0.960114 -> 0.957865 | 0.886842 -> 0.897368 | 0.922025 -> 0.926630 | 0.958411 -> 0.960850 |
| 6 | XGBoost | 0.953297 -> 0.961111 | 0.913158 -> 0.910526 | 0.932796 -> 0.935135 | 0.964113 -> 0.965375 |
| 24 | Persistence | 0.828947 -> 0.828947 | 0.828947 -> 0.828947 | 0.828947 -> 0.828947 | 0.908519 -> 0.908519 |
| 24 | Logistic Regression | 0.870482 -> 0.893417 | 0.760526 -> 0.750000 | 0.811798 -> 0.815451 | 0.899788 -> 0.901849 |
| 24 | Decision Tree | 0.762443 -> 0.831832 | 0.886842 -> 0.728947 | 0.819951 -> 0.776999 | 0.903158 -> 0.881248 |
| 24 | Random Forest | 0.871951 -> 0.764574 | 0.752632 -> 0.897368 | 0.807910 -> 0.825666 | 0.897755 -> 0.906197 |
| 24 | XGBoost | 0.895522 -> 0.796117 | 0.789474 -> 0.863158 | 0.839161 -> 0.828283 | 0.914335 -> 0.907894 |
| 48 | Persistence | 0.731579 -> 0.731579 | 0.731579 -> 0.731579 | 0.731579 -> 0.731579 | 0.856432 -> 0.856432 |
| 48 | Logistic Regression | 0.815436 -> 0.854015 | 0.639474 -> 0.615789 | 0.716814 -> 0.715596 | 0.849666 -> 0.849348 |
| 48 | Decision Tree | 0.856631 -> 0.675585 | 0.628947 -> 0.531579 | 0.725341 -> 0.594993 | 0.854444 -> 0.784975 |
| 48 | Random Forest | 0.848375 -> 0.842105 | 0.618421 -> 0.589474 | 0.715373 -> 0.693498 | 0.849189 -> 0.837761 |
| 48 | XGBoost | 0.833333 -> 0.719512 | 0.657895 -> 0.776316 | 0.735294 -> 0.746835 | 0.859450 -> 0.864218 |
| 72 | Persistence | 0.626316 -> 0.626316 | 0.626316 -> 0.626316 | 0.626316 -> 0.626316 | 0.800111 -> 0.800111 |
| 72 | Logistic Regression | 0.775934 -> 0.388370 | 0.492105 -> 0.984211 | 0.602254 -> 0.556962 | 0.789923 -> 0.749600 |
| 72 | Decision Tree | 0.794393 -> 0.439783 | 0.447368 -> 0.855263 | 0.572391 -> 0.580876 | 0.774702 -> 0.768158 |
| 72 | Random Forest | 0.751004 -> 0.522241 | 0.492105 -> 0.834211 | 0.594595 -> 0.642351 | 0.785722 -> 0.804613 |
| 72 | XGBoost | 0.771218 -> 0.625571 | 0.550000 -> 0.721053 | 0.642089 -> 0.669927 | 0.810447 -> 0.822493 |

## C. Missed observations and false positives

| horizon_hours | method | derived_fn | direct_fn | change_fn | derived_fp | direct_fp | change_fp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Persistence | 28 | 28 | 0 | 28 | 28 | 0 |
| 6 | Logistic Regression | 32 | 35 | 3 | 21 | 13 | -8 |
| 6 | Decision Tree | 43 | 34 | -9 | 13 | 16 | 3 |
| 6 | Random Forest | 43 | 39 | -4 | 14 | 15 | 1 |
| 6 | XGBoost | 33 | 34 | 1 | 17 | 14 | -3 |
| 24 | Persistence | 65 | 65 | 0 | 65 | 65 | 0 |
| 24 | Logistic Regression | 91 | 95 | 4 | 43 | 34 | -9 |
| 24 | Decision Tree | 43 | 103 | 60 | 105 | 56 | -49 |
| 24 | Random Forest | 94 | 39 | -55 | 42 | 105 | 63 |
| 24 | XGBoost | 80 | 52 | -28 | 35 | 84 | 49 |
| 48 | Persistence | 102 | 102 | 0 | 102 | 102 | 0 |
| 48 | Logistic Regression | 137 | 146 | 9 | 55 | 40 | -15 |
| 48 | Decision Tree | 141 | 178 | 37 | 40 | 97 | 57 |
| 48 | Random Forest | 145 | 156 | 11 | 42 | 42 | 0 |
| 48 | XGBoost | 130 | 85 | -45 | 50 | 115 | 65 |
| 72 | Persistence | 142 | 142 | 0 | 142 | 142 | 0 |
| 72 | Logistic Regression | 193 | 6 | -187 | 54 | 589 | 535 |
| 72 | Decision Tree | 210 | 55 | -155 | 44 | 414 | 370 |
| 72 | Random Forest | 193 | 63 | -130 | 62 | 290 | 228 |
| 72 | XGBoost | 171 | 106 | -65 | 62 | 164 | 102 |

## D. Direct model hierarchy

Higher scores rank first; equals signs represent actual ties, not rounded near-ties.

| horizon_hours | elevated_f1 | elevated_recall | macro_f1 |
| --- | --- | --- | --- |
| 6 | XGBoost > Logistic Regression > Decision Tree > Random Forest > Persistence | Persistence > Decision Tree = XGBoost > Logistic Regression > Random Forest | XGBoost > Logistic Regression > Decision Tree > Random Forest > Persistence |
| 24 | Persistence > XGBoost > Random Forest > Logistic Regression > Decision Tree | Random Forest > XGBoost > Persistence > Logistic Regression > Decision Tree | Persistence > XGBoost > Random Forest > Logistic Regression > Decision Tree |
| 48 | XGBoost > Persistence > Logistic Regression > Random Forest > Decision Tree | XGBoost > Persistence > Logistic Regression > Random Forest > Decision Tree | XGBoost > Persistence > Logistic Regression > Random Forest > Decision Tree |
| 72 | XGBoost > Random Forest > Persistence > Decision Tree > Logistic Regression | Logistic Regression > Decision Tree > Random Forest > XGBoost > Persistence | XGBoost > Random Forest > Persistence > Decision Tree > Logistic Regression |

At 6h, XGBoost and Logistic Regression are effectively tied: Elevated F1 differs by only 0.000176 (0.0176 percentage points). Persistence leads recall. At 24h persistence has the highest F1 and macro-F1; XGBoost trails persistence by only 0.000665 F1, and Random Forest has the highest recall. At 48h XGBoost leads all three rankings. At 72h XGBoost leads F1/macro-F1, but Logistic Regression leads recall by producing many more false positives.

## Interpretation

1. Direct classification is not uniformly better than thresholding regressors. Elevated F1 improves in 10 of 16 matched trained-model comparisons, with declines in the other six. At 6h gains are small for every family. At 48h only XGBoost improves Elevated F1. Classification improvements do not imply any improvement in regression RMSE: these estimators do not predict continuous heights.
2. Six hours gives the strongest overall Elevated F1 and macro-F1, for both approaches. Longer lead time makes detection harder. Highest recall alone is achieved by 72h balanced Logistic Regression (98.42%), but with 589 false positives and precision only 38.84%; it is not the strongest balanced detector.
3. Twenty-four-hour direct classification is not substantially better overall. Direct XGBoost recall rises from 78.95% to 86.32%, but precision falls from 89.55% to 79.61%, F1 drops from 0.839161 to 0.828283, and false positives rise from 35 to 84. Direct Random Forest improves recall to 89.74%, but incurs 105 false positives; its F1=0.825666 remains below persistence=0.828947 and regression-derived XGBoost=0.839161.
4. These results alone do not justify changing the project from continuous wave-height forecasting to classification as its primary objective. Direct classification may be appropriate if a binary decision and its miss/false-positive costs are specified in advance. No such operational cost function has been established here, and the 6h XGBoost F1 gain over its regression-derived counterpart is only 0.002339 (0.234 percentage points).
5. For the existing forecasting objective, retain 6h regression with derived binary evaluation as a clear undergraduate project, with persistence and Linear Regression as serious references. The direct classifiers provide a useful exploratory comparison. If the stated objective is changed explicitly to binary classification, 6h direct classification is a defensible alternative, with Logistic Regression and XGBoost essentially tied rather than a compelling complexity hierarchy. A 24h minimum lead-time requirement would be an application constraint, not a conclusion that direct classification is superior.

At 48/72h direct XGBoost trades precision for recall. At 72h F1 increases from 0.642089 to 0.669927 and misses fall from 171 to 106, but false positives rise from 62 to 164. Whether this trade-off is useful requires predefined application costs. Persistence remains important across horizons.

The dataset is one offshore historical series, with roughly 6.5% Elevated test samples. Accuracy is dominated by Normal observations; use class-specific errors as well. Samples are temporally dependent, no event-block confidence intervals were calculated, and no statistical significance is claimed. These fixed thresholds are research definitions, not official warnings. The comparison is exploratory because prior test results have already been inspected and informed the research questions. Choosing an approach/horizon here is not independent test-set validation and does not establish operational coastal-warning capability.

## Files and execution

```text
direct_binary_classification/
|-- run_experiment.py
|-- README.md
|-- experiment_definition.json       # fixed design, features, grids, versions
|-- original_files_sha256.json       # 269 protected pre-existing files
|-- test_metrics.csv                 # 20 direct method/horizon rows
|-- regression_derived_metrics.csv   # 20 paired baseline rows
|-- comparison.csv                  # paired metrics and explicit deltas
|-- rankings.csv
|-- verification.json
|-- {6,24,48,72}_hour/
    |-- models/                     # four classifiers per horizon
    |-- model_metadata.json
    |-- cv_results.csv              # 48 candidate/fold rows
    |-- validation_metrics.csv
    |-- validation_predictions.csv
    |-- test_metrics.csv
    |-- test_predictions.csv        # labels, 4 positive-class probabilities
    |-- regression_derived_metrics.csv
    |-- comparison.csv
    |-- confusion_matrices.csv      # actual rows, predicted columns
    |-- confusion_matrices.png
```

Executed from the repository root using existing Python 3.11.9:

```powershell
.\.venv\Scripts\python.exe -B -u experiments/direct_binary_classification/run_experiment.py
```

The training stage refuses to overwrite completed classifier models. To recalculate outputs using only frozen models, or rerun the independent audit without training:

```powershell
.\.venv\Scripts\python.exe -B -u experiments/direct_binary_classification/run_experiment.py --stage evaluate
.\.venv\Scripts\python.exe -B -u experiments/direct_binary_classification/run_experiment.py --stage verify
```

The code also supports --stage train for the initial training stage. Evaluation requires all four training horizons to be complete. Package versions are recorded in experiment_definition.json; the existing environment was used without installation. Do not rerun training on these completed outputs or change settings based on test performance.

Verification passed: all 16 model reloads and probabilities, identical compact feature names/order, exact future labels, paired test origins/targets, strict chronological gaps, independently recalculated direct test and validation metrics, derived comparison deltas and confusion counts. All 269 pre-existing files remain byte-identical. Dataset SHA-256: `421199b25b1b7c475d8f09b65317f37b0aa0bf19a3385da0696e510d8a575ed8`. No original source/results/models were overwritten, no repository cleanup occurred and no Git commands were executed.
