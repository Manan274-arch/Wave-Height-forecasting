# Controlled 12-hour wave-height regression

Executed locally on 22 October 2026. This experiment changes the forecast horizon from 6 to 12 hours while preserving the four original controlled six-hour compact-feature estimator configurations. Four 12h models, native/common-origin comparisons, derived binary metrics, paired moving-block uncertainty intervals and seven figures were produced.

No existing workflow or model was replaced, no hyperparameter search was performed, and no files outside this experiment were changed. Expanded six-hour tuning settings were not used.

## Data, features and exact target

The unchanged original CSV contains 29,224 observations and 20 columns from 2015-01-01 00:00 to 2024-12-31 21:00 UTC. `feature_pipeline.load_source` verifies the complete three-hour grid, schema, finite values, coordinates and physical checks. There are no timestamp gaps or duplicates. The original loader deliberately rejects irregular replacement data because the existing lag/rolling definitions require a uniform grid.

The experiment-local `prepare_12h()` reuses the unchanged loader and feature builder, selecting the same 18 columns in the original order. It reproduces the raw-boundary splitting/purging logic locally because the existing shared horizon guard does not accept 12h; that shared code was left untouched. Targets are looked up at exact origin+12h timestamps and independently checked against VHM0.shift(-4) on the verified regular grid. No future maximum or future-derived predictor is used.

```text
VHM0, VHM0_SW1, VHM0_WW, VTM02, VTPK,
eastward_wind, northward_wind, wind_speed, VMDR_sin, VMDR_cos,
VHM0_lag_6h, VHM0_lag_24h, VHM0_change_3h,
VHM0_roll_mean_12h, VHM0_roll_std_12h, wind_speed_roll_mean_12h,
month_sin, month_cos
```

Every feature matrix exactly matches existing compact features at shared origins. Rolling statistics retain causal trailing windows and ddof=1; seasonal values use origin UTC. Future-perturbation checks passed.

Raw boundaries remain positions 20,456 and 23,378, calculated before exclusions. Remove eight initial origins for feature history, four training/validation boundary origins each, and four final unavailable test targets. Require origin+12h strictly before the next partition start. Historical observations before a partition begins remain available to its predictors.

Inclusive prediction-origin dates, in UTC:

| partition | nominal_rows | rows | start | end | history_excluded | purged | unavailable_targets |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training | 20456 | 20444 | 2015-01-02 00:00:00+00:00 | 2021-12-31 09:00:00+00:00 | 8 | 4 | 0 |
| Validation | 2922 | 2918 | 2022-01-01 00:00:00+00:00 | 2022-12-31 15:00:00+00:00 | 0 | 4 | 0 |
| Testing | 5846 | 5842 | 2023-01-01 06:00:00+00:00 | 2024-12-31 09:00:00+00:00 | 0 | 0 | 4 |

Training targets end 2021-12-31 21:00 UTC, before validation starts 2022-01-01 00:00. Validation targets end 2023-01-01 03:00, before testing starts 2023-01-01 06:00. Test targets run 2023-01-01 18:00 through 2024-12-31 21:00 UTC.

## Fixed model configurations and fitting

Clone each saved estimator from `experiments/controlled_regression/6_hour/compact/models/`. This preserves all original hyperparameters, including defaults and the Linear Regression pipeline, while discarding learned six-hour state. Parameter equality, feature order and original model hashes were checked.

| model | original_selected_parameters | original_model_sha256 |
| --- | --- | --- |
| Linear Regression | {} | 70329ee93b2b237505ddbe463c76ea64125b37a2270f3992340f971632f261e9 |
| Decision Tree | {"max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 10} | fb546a97aa592a3f66c45b0b7b34fd74d1d9dd060f2fc53b0887c6fcf6edbab1 |
| Random Forest | {"max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | 20a21770584cf8a6f1325760e5e4ae86dbc445b4897609855566b43f1f6bcc72 |
| XGBoost | {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 220, "reg_lambda": 8} | 31aab3eced688deddc8b453d70c3596567da2ee2bf2f637ebbe6814dfc4f2556 |

Linear Regression retains StandardScaler and an intercept. Decision Tree uses depth=8, leaf=10, split=10; Forest uses 120 trees, depth=10, leaf=5 and max_features=0.7; XGBoost uses 220 trees, depth=4, learning_rate=0.05, lambda=8, subsample=0.8, colsample_bytree=0.8 and the original other defaults. Tree ensembles retain random_state=42 and four workers. XGBoost retains squared-error regression and histogram trees; no new early stopping is introduced.

Fit each configuration on the 20,444 eligible training observations and record chronological validation metrics/predictions. Validation is diagnostic and does not select settings or model families. Then refit fresh clones on the same 23,362 eligible training+validation origins, retaining the boundary gap. The scaler is fitted on training for validation, and training+validation for final test inference. No test data enter fitting. All four models are saved before test evaluation, then reloaded for prediction. No new CV/tuning is needed for this fixed-configuration comparison.

## A. Native 12h regression results

Errors are in metres; signed bias is prediction minus actual.

| model | mae | rmse | r2 | mean_signed_error | median_absolute_error |
| --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.071560 | 0.103027 | 0.978870 | 0.003134 | 0.050789 |
| Decision Tree | 0.075992 | 0.112207 | 0.974937 | -0.000693 | 0.052650 |
| Random Forest | 0.069742 | 0.102122 | 0.979240 | 0.001692 | 0.048750 |
| XGBoost | 0.068209 | 0.099548 | 0.980273 | 0.002985 | 0.048569 |

## B. Native 12h derived binary classification

Normal <2.5 m, Elevated >=2.5 m. Apply the same fixed threshold to unrounded continuous predictions and actual future heights. Elevated is positive; class order [Normal, Elevated]; zero_division=0. No separate classifiers or threshold selection were used.

There are 380 actual Elevated observations (6.504622%) and 5,462 Normal observations in the 5,842-row native 12h test set.

| horizon_hours | model | sample_count | normal_count | elevated_count | elevated_percent | accuracy | balanced_accuracy | normal_precision | normal_recall | normal_f1 | elevated_precision | elevated_recall | elevated_f1 | macro_f1 | weighted_f1 | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | Linear Regression | 5842 | 5462 | 380 | 6.504622 | 0.984765 | 0.923295 | 0.989790 | 0.993958 | 0.991870 | 0.907563 | 0.852632 | 0.879240 | 0.935555 | 0.984544 | 324 | 33 | 5429 | 56 |
| 12 | Decision Tree | 5842 | 5462 | 380 | 6.504622 | 0.983054 | 0.895446 | 0.985867 | 0.996155 | 0.990984 | 0.934985 | 0.794737 | 0.859175 | 0.925080 | 0.982411 | 302 | 21 | 5441 | 78 |
| 12 | Random Forest | 5842 | 5462 | 380 | 6.504622 | 0.985621 | 0.920080 | 0.989265 | 0.995423 | 0.992334 | 0.927746 | 0.844737 | 0.884298 | 0.938316 | 0.985307 | 321 | 25 | 5437 | 59 |
| 12 | XGBoost | 5842 | 5462 | 380 | 6.504622 | 0.985621 | 0.924977 | 0.989978 | 0.994691 | 0.992329 | 0.918079 | 0.855263 | 0.885559 | 0.938944 | 0.985384 | 325 | 29 | 5433 | 55 |

Confusion matrices are [[TN, FP], [FN, TP]], with actual rows and predicted columns. Every matrix totals 5,842. Counts and the combined four-model plot are saved.

## C. Native 6h versus 12h regression

The original six-hour predictions and metrics were independently verified. These primary results use 5,844 eligible origins at 6h and 5,842 at 12h. Differences below are 12h minus 6h; larger MAE/RMSE and smaller R-squared indicate reduced accuracy.

| model | six_hour_mae | twelve_hour_mae | change_12h_minus_6h_mae | six_hour_rmse | twelve_hour_rmse | change_12h_minus_6h_rmse | six_hour_r2 | twelve_hour_r2 | change_12h_minus_6h_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.045017 | 0.071560 | 0.026543 | 0.065885 | 0.103027 | 0.037141 | 0.991356 | 0.978870 | -0.012486 |
| Decision Tree | 0.048910 | 0.075992 | 0.027081 | 0.072639 | 0.112207 | 0.039569 | 0.989494 | 0.974937 | -0.014557 |
| Random Forest | 0.044447 | 0.069742 | 0.025294 | 0.065874 | 0.102122 | 0.036248 | 0.991359 | 0.979240 | -0.012120 |
| XGBoost | 0.043060 | 0.068209 | 0.025149 | 0.064838 | 0.099548 | 0.034710 | 0.991629 | 0.980273 | -0.011356 |

## D. Native 6h versus 12h classification

| model | six_hour_elevated_precision | twelve_hour_elevated_precision | change_12h_minus_6h_elevated_precision | six_hour_elevated_recall | twelve_hour_elevated_recall | change_12h_minus_6h_elevated_recall | six_hour_elevated_f1 | twelve_hour_elevated_f1 | change_12h_minus_6h_elevated_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.943089 | 0.907563 | -0.035526 | 0.915789 | 0.852632 | -0.063158 | 0.929239 | 0.879240 | -0.049999 |
| Decision Tree | 0.962857 | 0.934985 | -0.027873 | 0.886842 | 0.794737 | -0.092105 | 0.923288 | 0.859175 | -0.064113 |
| Random Forest | 0.960114 | 0.927746 | -0.032368 | 0.886842 | 0.844737 | -0.042105 | 0.922025 | 0.884298 | -0.037727 |
| XGBoost | 0.953297 | 0.918079 | -0.035218 | 0.913158 | 0.855263 | -0.057895 | 0.932796 | 0.885559 | -0.047237 |

At both horizons the actual Elevated count is 380, but corresponding forecast origins/targets differ. XGBoost misses rise from 33 at 6h to 55 at 12h; false positives rise from 17 to 29. Its Elevated recall decreases 5.789474 percentage points and F1 decreases 4.723712 percentage points. Forest's F1 decrease is smaller (3.772710 percentage points), but it does not become the highest-F1 model at 12h.

## E. Model separation

Positive reference-minus-XGBoost RMSE gaps favor XGBoost; percentage advantages use each reference model's RMSE as denominator.

| horizon_hours | best_rmse | best_mae | best_elevated_f1 | best_elevated_precision | best_elevated_recall | best_worst_rmse_gap | Linear Regression_minus_xgboost_rmse | xgboost_advantage_percent_over_Linear Regression | Random Forest_minus_xgboost_rmse | xgboost_advantage_percent_over_Random Forest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | XGBoost | XGBoost | XGBoost | Decision Tree | Linear Regression | 0.007801 | 0.001047 | 1.589698 | 0.001036 | 1.572186 |
| 12 | XGBoost | XGBoost | XGBoost | Decision Tree | XGBoost | 0.012660 | 0.003479 | 3.376991 | 0.002574 | 2.520492 |

At both horizons, RMSE (ascending), MAE (ascending), and R-squared (descending) all rank: XGBoost, Random Forest, Linear Regression, Decision Tree. The 6h Forest/Linear RMSE difference is only about 0.000012 m; the 12h difference is about 0.000905 m.

Not every metric favors XGBoost. Decision Tree has the highest Elevated precision at both horizons and the fewest 12h false positives (21). Linear Regression has the highest 6h Elevated recall (0.915789). Forest ties XGBoost for highest 12h accuracy (0.985621), and has slightly higher Normal F1, 0.992334 versus 0.992329; that Normal-F1 difference is negligible. Forest also has higher 12h Elevated precision than XGBoost, although Decision Tree leads both. XGBoost's 12h Elevated F1 advantage over Forest is only 0.001261, so classification separation is much smaller than an algorithm-ranking table alone might suggest.

## Common-origin comparison

The common eligible forecast origins number 5,842, from 2023-01-01 06:00 to 2024-12-31 09:00 UTC. Reuse already-fitted model predictions; there is no retraining. Each horizon retains its own future target, so this is a common-origin comparison, not an identical-target task.

| horizon_hours | model | sample_count | mae | rmse | r2 | elevated_precision | elevated_recall | elevated_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Linear Regression | 5842 | 0.045023 | 0.065894 | 0.991356 | 0.943089 | 0.915789 | 0.929239 |
| 6 | Decision Tree | 5842 | 0.048924 | 0.072651 | 0.989492 | 0.962857 | 0.886842 | 0.923288 |
| 6 | Random Forest | 5842 | 0.044457 | 0.065884 | 0.991358 | 0.960114 | 0.886842 | 0.922025 |
| 6 | XGBoost | 5842 | 0.043067 | 0.064847 | 0.991628 | 0.953297 | 0.913158 | 0.932796 |
| 12 | Linear Regression | 5842 | 0.071560 | 0.103027 | 0.978870 | 0.907563 | 0.852632 | 0.879240 |
| 12 | Decision Tree | 5842 | 0.075992 | 0.112207 | 0.974937 | 0.934985 | 0.794737 | 0.859175 |
| 12 | Random Forest | 5842 | 0.069742 | 0.102122 | 0.979240 | 0.927746 | 0.844737 | 0.884298 |
| 12 | XGBoost | 5842 | 0.068209 | 0.099548 | 0.980273 | 0.918079 | 0.855263 | 0.885559 |

Removing the two extra 6h origins has negligible influence on regression differences and leaves its Elevated precision/recall/F1 unchanged.

## Paired moving-block uncertainty

Use ordinary noncircular moving-block bootstrap over common origins, with 2,000 replicates for each prespecified block length: 56 observations (7 days) as primary and 112 observations (14 days) as sensitivity. RNG seed=42. Draw uniformly selected contiguous blocks with replacement, concatenate and truncate to the original sample length. The exact same resampled origin blocks are used for both models in each contrast and both horizons. Frozen predictions are resampled; models are not refitted. Calculate percentile 95% intervals for XGBoost RMSE minus reference RMSE; negative favors XGBoost.

| horizon_hours | reference_model | common_origins | block_observations | block_days | replicates | xgboost_minus_reference_rmse | percentile_95_lower | percentile_95_upper | interval_includes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Linear Regression | 5842 | 56 | 7.000000 | 2000 | -0.001047 | -0.002719 | 0.001014 | True |
| 6 | Random Forest | 5842 | 56 | 7.000000 | 2000 | -0.001037 | -0.002169 | 0.000216 | True |
| 12 | Linear Regression | 5842 | 56 | 7.000000 | 2000 | -0.003479 | -0.005718 | -0.000910 | False |
| 12 | Random Forest | 5842 | 56 | 7.000000 | 2000 | -0.002574 | -0.004266 | -0.000771 | False |
| 6 | Linear Regression | 5842 | 112 | 14.000000 | 2000 | -0.001047 | -0.002715 | 0.001065 | True |
| 6 | Random Forest | 5842 | 112 | 14.000000 | 2000 | -0.001037 | -0.002213 | 0.000290 | True |
| 12 | Linear Regression | 5842 | 112 | 14.000000 | 2000 | -0.003479 | -0.005508 | -0.001177 | False |
| 12 | Random Forest | 5842 | 112 | 14.000000 | 2000 | -0.002574 | -0.004146 | -0.000860 | False |

The 6h intervals include zero, so this analysis does not establish a clear RMSE separation from Linear Regression or Forest there. The 12h intervals lie below zero for both contrasts under both chosen block lengths, supporting the direction of the observed 12h differences conditional on this bootstrap procedure. The absolute gaps remain only about 3.48 mm versus Linear Regression and 2.57 mm versus Forest. This does not establish a practically important or universally superior model.

Blocks account for local serial dependence, but one coastal series, seasonal nonstationarity, limited extreme episodes and block-length assumptions remain limitations. Earlier test inspection and multiple exploratory comparisons prevent treating these intervals as independent confirmatory validation. No IID confidence intervals, universal significance claims or operational warning claims are made.

## Recommendation

**B: propose 12-hour forecasting as the mini-project's research focus, while retaining six hours as a benchmark. No existing workflow has been switched or replaced.**

This is an explicit research-design judgment that values six additional hours of forecast lead time. XGBoost's 12h RMSE is about 0.100 m and Elevated recall remains 85.53%, with precision 91.81% and F1 0.885559. Doubling lead time costs 0.034710 m RMSE, 0.025149 m MAE and 22 additional missed Elevated observations compared with its original 6h model. Those costs must be shown rather than hidden behind a larger algorithm gap.

Twelve hours provides an intermediate lead-time/performance trade-off with the same simple fixed methodology. Its RMSE separation is more consistently directional under the paired block analysis, but that is supporting evidence rather than the reason by itself to change horizon. The classification gap between XGBoost and Forest remains marginal, and Decision Tree/Linear Regression retain specific strengths. Presenting these competing strengths makes a more defensible comparison than claiming XGBoost wins every metric.

Six hours remains preferable if minimizing absolute error and missed observations is the overriding objective. No application-specific cost function or minimum warning lead time has been established, so neither horizon is universally superior. A future operational choice needs such requirements and genuinely new evaluation data. Any proposed switch remains exploratory because the same historical test period has already informed project decisions. The 2.5 m category is a research threshold, not an official warning level.

## Figures and reproducibility

Seven 220-dpi PNGs were generated and visually checked. Model colors are consistent; paired bars use horizon hatching. Regression and classification comparison axes start at zero, and R-squared/F1 axes preserve the full score scale. The time-series interval is the prespecified July 1-14 of the first test year, identical for all four models; its x-axis is forecast-origin UTC and actual height is the 12h future target.

```text
figures/
|-- rmse_6h_vs_12h.png
|-- mae_6h_vs_12h.png
|-- r2_6h_vs_12h.png
|-- elevated_f1_6h_vs_12h.png
|-- precision_recall_12h.png
|-- confusion_matrices_12h.png
|-- actual_predicted_12h.png
```

Other outputs: run_experiment.py, verify_results.py, four joblib models, design/model metadata, validation metrics/predictions, test predictions, regression/classification metrics, native comparisons, differences, rankings, model separation, common-origin comparison, bootstrap intervals, original-file checksum manifest and verification.json. No duplicated raw/intermediate dataset is generated.

Using the existing Python 3.11.9 environment, executed from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -u experiments/controlled_12h_regression/run_experiment.py
.\.venv\Scripts\python.exe -B -u experiments/controlled_12h_regression/verify_results.py
```

Training refuses to overwrite completed models. Frozen evaluation without fitting is available through:

```powershell
.\.venv\Scripts\python.exe -B -u experiments/controlled_12h_regression/run_experiment.py --stage evaluate
```

Package versions are recorded in design.json; no packages were installed. Verification independently recalculated native/common regression and binary metrics, validation metrics, confusion counts, horizon differences and bootstrap intervals; checked all four reloads, complete original estimator parameters (including NaN-valued XGBoost missing defaults), feature order and temporal guards; and confirmed all 358 pre-existing files byte-identical. Original dataset SHA-256 remains `421199b25b1b7c475d8f09b65317f37b0aa0bf19a3385da0696e510d8a575ed8`. No Git commands or repository cleanup were performed.
