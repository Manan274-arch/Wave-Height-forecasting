# Final six-hour regression hyperparameter optimization

Exploratory Udupi Wave Height Forecasting experiment executed locally on 22 October 2026. Four frozen regressors were saved and evaluated. No existing source, model, result or dataset was overwritten, and no cleanup or Git operations were performed.

**Main finding:** expanded tuning did not improve XGBoost test RMSE. It increased from 0.0648380684 m to 0.0649661807 m (0.197588% worse), while MAE improved slightly. Decision Tree RMSE improved 1.260877%; Random Forest retained its incumbent settings; ordinary Linear Regression remained unchanged. XGBoost still leads the four-model RMSE table, but its advantage became smaller. No settings were changed after observing these results.

## Frozen data and target

Reuse `src/controlled_pipeline.py::prepare_controlled(6, "compact")`, including the existing feature engineering and raw-timeline split logic. The target is exactly VHM0(t+6 hours), implemented by shift(-2) after verifying three-hour timestamp alignment. It is not a future maximum.

All models use the same ordered 18 features:

```text
VHM0, VHM0_SW1, VHM0_WW, VTM02, VTPK,
eastward_wind, northward_wind, wind_speed, VMDR_sin, VMDR_cos,
VHM0_lag_6h, VHM0_lag_24h, VHM0_change_3h,
VHM0_roll_mean_12h, VHM0_roll_std_12h, wind_speed_roll_mean_12h,
month_sin, month_cos
```

Raw boundaries are positions 20,456 and 23,378, before exclusions. Eight initial historical origins and two origins at each train/validation boundary are excluded; the last two test targets are unavailable. Rolling windows retain their original causal definitions, including ddof=1 and UTC month encoding. All timestamps below are inclusive UTC prediction origins.

| Partition | Rows | Start | End |
| --- | --- | --- | --- |
| Training | 20,446 | 2015-01-02 00:00 | 2021-12-31 15:00 |
| Validation | 2,920 | 2022-01-01 00:00 | 2022-12-31 21:00 |
| Test | 5,844 | 2023-01-01 06:00 | 2024-12-31 15:00 |

Final refits use exactly 23,366 eligible training+validation origins, retaining purge exclusions. Test timestamps and actual targets exactly match the previous controlled compact six-hour experiment. No CSV contents or feature definitions were changed.

## Prespecified searches

All selection uses training-only TimeSeriesSplit(n_splits=3, gap=2), checking target endpoints against the next scoring origin. Minimize mean fold RMSE; ties prefer lower sample standard deviation (ddof=1), then earlier candidate index. Scaling is fitted inside the Linear Regression pipeline on each fitting fold. Seeds are 42; refinement uses an independent fixed seed 43. Forest and XGBoost use four workers; candidate searches are sequential.

- Linear Regression: one ordinary default setting, fit_intercept=True, with StandardScaler. No regularized substitute.
- Decision Tree: 40 ParameterSampler configurations plus the previous incumbent, 41 candidates/123 fold fits. Search depth {3,4,5,6,8,10,12,16,None}, split {2,5,10,20,40}, leaf {1,2,5,10,20}, max_features {None,0.7,0.9}, ccp_alpha {0,0.00001,0.0001,0.001}.
- Random Forest: 24 randomized configurations plus the previous incumbent, 25 candidates/75 fold fits. Search trees {100,200,300,500}, depth {5,8,10,12,16,None}, split {2,5,10,20}, leaf {1,2,3,5,10}, max_features {0.5,0.7,0.9,1.0}. The additional incumbent has 120 trees, preserving the prior valid configuration rather than excluding it because it is outside the new tree-count grid.
- XGBoost Stage A: 60 configurations (the incumbent plus 59 seeded randomized settings), 180 fold fits. Uniform integer trees 100-1000, depth 2-7, child weight 1-12; log-uniform learning rate 0.01-0.15 and lambda 0.1-30; uniform row/column fractions 0.65-1.0. Alpha samples {0,0.01,0.05,0.1,0.25,0.5,1,2}; gamma samples {0,0.001,0.01,0.05,0.1,0.25,0.5,1}, giving zero extra sampling mass. These bounded choices include both unregularized and regularized settings.
- XGBoost Stage B: 20 seeded configurations around the three lowest-CV-RMSE Stage A candidates, 60 fold fits. Vary depth by up to two, tree caps by factors 0.6-1.5, learning rate by 0.5-1.8, row/column fractions with std 0.12, child weight by up to four, lambda by 0.3-3, and alpha/gamma with zero-inclusive perturbations, bounded to the original allowed ranges. Candidate lists are frozen before validation/test use.

Total: 147 candidate configurations and 441 chronological scoring fits. These are bounded searches, not exhaustive optimization guarantees. XGBoost uses objective=reg:squarederror, tree_method=hist and RMSE evaluation. There is no class weighting, feature selection, artificial noise or prediction clipping.

## Nested early stopping and final rounds

For every XGBoost CV candidate/fold:

1. The outer training prefix is divided at floor(0.85 * training_rows).
2. The last 15% becomes a chronologically later early-stopping segment. Remove two origins from the preceding fitting core so its last six-hour target is strictly before the stopping segment.
3. Fit only the core, monitoring only the stopping segment, with early_stopping_rounds=40.
4. Predict the untouched outer CV scoring fold using the best-iteration prefix. The CV scoring fold is never supplied to fit/eval_set. No outer-fold refit is performed; XGBoost therefore fits on a smaller core than the non-early-stopped algorithms during CV, with the remainder reserved for stopping. All final models use identical eligible refit origins.

For the selected hyperparameters, select a training-only round count using the same 85/15 split of the entire training partition. Refit on all training data using that fixed count and evaluate the original validation partition. This gives the XGBoost diagnostic validation RMSE in Table B, without using validation to choose its hyperparameters or training-tail rounds.

Next, fit the selected hyperparameters on all training observations and use the original validation partition solely as an early-stopping calibration segment to select the final fixed number of boosting rounds. This calibration error is a selection diagnostic, not independent validation evidence. Finally refit on eligible training+validation using that fixed count, with no early stopping/eval_set and no test information. All four final models are frozen before new test prediction.

Selected XGBoost settings: depth=3, learning_rate=0.047512748101637955, subsample=0.8409658762605772, colsample_bytree=0.9535196267442514, min_child_weight=3, reg_lambda=1.365048163666801, reg_alpha=0, gamma=0.001. Selected candidate tree cap=479; final refit count=369.

The winning Stage A candidate has mean CV RMSE 0.0861690273 m; Stage B's best is 0.0864006505 m, so refinement did not improve the winner. Selected candidate outer-fold results:

| fold | fold_rmse | best_iteration | best_rounds | fitted_rounds | core_end | early_stopping_start | early_stopping_end | scoring_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.090194 | 367.000000 | 368.000000 | 408.000000 | 2016-06-27 15:00:00+00:00 | 2016-06-28 00:00:00+00:00 | 2016-10-01 18:00:00+00:00 | 2016-10-02 03:00:00+00:00 |
| 2 | 0.075444 | 352.000000 | 353.000000 | 393.000000 | 2017-12-22 15:00:00+00:00 | 2017-12-23 00:00:00+00:00 | 2018-07-02 15:00:00+00:00 | 2018-07-03 00:00:00+00:00 |
| 3 | 0.092870 | 413.000000 | 414.000000 | 454.000000 | 2019-06-18 18:00:00+00:00 | 2019-06-19 03:00:00+00:00 | 2020-04-01 12:00:00+00:00 | 2020-04-01 21:00:00+00:00 |

Final round-selection records:

| purpose | best_iteration | best_rounds | fitted_rounds | fit_end | stopping_start | stopping_end | best_rmse | no_early_stopping_validation_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| training_tail_round_selection | 448 | 449 | 479 | 2020-12-13 00:00:00+00:00 | 2020-12-13 09:00:00+00:00 | 2021-12-31 15:00:00+00:00 | 0.090125 | nan |
| original_validation_final_round_selection | 368 | 369 | 409 | 2021-12-31 15:00:00+00:00 | 2022-01-01 00:00:00+00:00 | 2022-12-31 21:00:00+00:00 | 0.068776 | 0.068509 |

Here best_iteration is zero-based, so best_rounds=best_iteration+1. The training-tail fit selected 449 rounds (best_iteration=448); validation calibration selected 369 (best_iteration=368) and stopped after fitting 409 rounds. The final model contains exactly 369 rounds and no early-stopping configuration.

**Early stopping did not demonstrate an accuracy benefit in this run.** The selected 479-round model trained without early stopping has validation RMSE 0.0685093655 m, compared with 0.0686325990 m using the training-tail-selected 449 rounds and 0.0687759490 m for validation-calibrated 369 rounds. Patience 40 terminated before later improvements achievable by continuing to the cap. We did not change patience or round-selection rules after inspecting the test results.

Early stopping did reduce CV computation: 175/240 XGBoost fits ended before their caps, fitting 68,802 rounds rather than the total caps of 144,555 (about 52.4% fewer rounds). This is a computational benefit, not evidence of better generalization. Final test differences cannot be attributed to early stopping alone because other hyperparameters also changed.

## A. Regression comparison

Positive RMSE improvement means lower tuned error. Tiny Random Forest floating-point changes represent unchanged performance.

| model | old_rmse | tuned_rmse | old_mae | tuned_mae | old_r2 | tuned_r2 | rmse_improvement_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.065885 | 0.065885 | 0.045017 | 0.045017 | 0.991356 | 0.991356 | 0.000000 |
| Decision Tree | 0.072639 | 0.071723 | 0.048910 | 0.049227 | 0.989494 | 0.989757 | 1.260877 |
| Random Forest | 0.065874 | 0.065874 | 0.044447 | 0.044447 | 0.991359 | 0.991359 | -0.000000 |
| XGBoost | 0.064838 | 0.064966 | 0.043060 | 0.042964 | 0.991629 | 0.991596 | -0.197588 |

## B. Selected hyperparameters, CV and validation

Parameters omitted from these dictionaries retain the declared estimator defaults. Linear Regression remains scaled ordinary least squares with an intercept. Exact values are retained in model_metadata.json.

| Model | Original searched parameters | Selected final parameters | Mean CV RMSE | CV sample std | Validation RMSE |
| --- | --- | --- | --- | --- | --- |
| Linear Regression | {} | {} | 0.084713 | 0.007855 | 0.071125 |
| Decision Tree | {"max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 10} | {"min_samples_split": 10, "min_samples_leaf": 20, "max_features": null, "max_depth": 16, "ccp_alpha": 1e-05} | 0.094009 | 0.006578 | 0.077188 |
| Random Forest | {"max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | {"max_depth": 10, "max_features": 0.7, "min_samples_leaf": 5, "n_estimators": 120} | 0.086936 | 0.006204 | 0.070801 |
| XGBoost | {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 220, "reg_lambda": 8} | {"colsample_bytree": 0.9535196267442514, "gamma": 0.001, "learning_rate": 0.047512748101637955, "max_depth": 3, "min_child_weight": 3, "n_estimators": 369, "reg_alpha": 0, "reg_lambda": 1.365048163666801, "subsample": 0.8409658762605772} | 0.086169 | 0.009384 | 0.068633 |

The XGBoost CV row refers to the selected 479-round-cap nested-early-stopping candidate; the final parameter dictionary uses 369 rounds. Its table validation metric is the training-tail-selected 449-round diagnostic. The separate final round-calibration result is reported above. Historical XGBoost CV used a different fitting protocol without a nested stopping tail, so old and new CV scores are not an isolated comparison of early stopping.

## C. Derived binary classification

Normal <2.5 m; Elevated >=2.5 m, using full-precision unmodified predictions. Elevated is positive, class order [Normal, Elevated], zero_division=0. No classifier or threshold optimization is involved. There are 380 actual Elevated observations among 5,844 test origins.

| model | accuracy | balanced_accuracy | elevated_precision | elevated_recall | elevated_f1 | macro_f1 | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.990931 | 0.955973 | 0.943089 | 0.915789 | 0.929239 | 0.962197 | 348 | 21 | 5443 | 32 |
| Decision Tree | 0.990589 | 0.942323 | 0.965616 | 0.886842 | 0.924554 | 0.959768 | 337 | 12 | 5452 | 43 |
| Random Forest | 0.990246 | 0.942140 | 0.960114 | 0.886842 | 0.922025 | 0.958411 | 337 | 14 | 5450 | 43 |
| XGBoost | 0.991615 | 0.956339 | 0.953425 | 0.915789 | 0.934228 | 0.964875 | 348 | 17 | 5447 | 32 |

Comparison with previous classification:

| model | old_elevated_f1 | tuned_elevated_f1 | change_elevated_f1 | old_fp | tuned_fp | old_fn | tuned_fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 0.929239 | 0.929239 | 0.000000 | 21 | 21 | 32 | 32 |
| Decision Tree | 0.923288 | 0.924554 | 0.001267 | 13 | 12 | 43 | 43 |
| Random Forest | 0.922025 | 0.922025 | 0.000000 | 14 | 14 | 43 | 43 |
| XGBoost | 0.932796 | 0.934228 | 0.001432 | 17 | 17 | 33 | 32 |

## D. XGBoost versus Linear Regression and Random Forest

Positive absolute/percentage differences favor XGBoost; percentages use each reference model's RMSE as denominator.

| configuration | reference_model | reference_rmse | xgboost_rmse | reference_minus_xgboost_rmse | xgboost_advantage_percent |
| --- | --- | --- | --- | --- | --- |
| old | Linear Regression | 0.065885 | 0.064838 | 0.001047 | 1.589698 |
| old | Random Forest | 0.065874 | 0.064838 | 0.001036 | 1.572186 |
| tuned | Linear Regression | 0.065885 | 0.064966 | 0.000919 | 1.395251 |
| tuned | Random Forest | 0.065874 | 0.064966 | 0.000908 | 1.377704 |

XGBoost's absolute lead over Linear Regression shrank from 0.001047380 m to 0.000919268 m, a decrease of 0.000128112 m. Its percentage lead shrank from 1.589698% to 1.395251%. Its lead over Random Forest shrank from 1.572186% to 1.377704%. It is not a clearer winner after this optimization.

## Conclusions and limitations

- XGBoost test RMSE worsened 0.197588%, while its MAE improved about 0.000096 m. More search effort did not guarantee a better held-out score.
- Early stopping shortened many CV fits, but the matched validation diagnostics did not show an error benefit. Do not claim it caused a test improvement.
- Decision Tree RMSE improved 1.260877%, but MAE worsened from 0.048910 to 0.049227 m. No randomized Forest candidate beat the incumbent on CV; its final settings and errors are unchanged.
- XGBoost remains the lowest-RMSE model among the four, with only a 1.40% advantage over Linear Regression and 1.38% over Random Forest. These are small descriptive differences, not statistical significance claims.
- Derived Elevated F1 increases slightly for Decision Tree (0.923288 -> 0.924554) and XGBoost (0.932796 -> 0.934228). XGBoost misses one fewer Elevated observation, with the same 17 false positives; its classification gain does not imply improved regression RMSE. Linear Regression/Forest classification is unchanged.
- For a prespecified CV-selected final workflow, retain the four configurations selected by this procedure, including the tuned Tree and unchanged linear/Forest settings, and transparently report the XGBoost negative test finding. The old XGBoost remains the lower observed test-RMSE benchmark and is preserved. Do not switch hyperparameters back and forth on this already-examined test set or present such a switch as independent validation. Any subsequent choice between the two XGBoost configurations needs training/validation evidence or genuinely new unseen evaluation data.

The experiment is exploratory because historical test results already informed research decisions. No uncertainty/significance analysis or operational coastal-warning validation was performed. One offshore historical series and dependent three-hourly observations limit generalization; binary thresholds are research definitions rather than official warnings. A new untouched evaluation period is required to assess future methodological changes independently.

## Outputs and commands

All new files are exclusively inside this directory:

```text
final_6h_hyperparameter_tuning/
|-- run_experiment.py
|-- verify_results.py
|-- README.md
|-- design.json
|-- original_files_sha256.json
|-- cv_results.csv
|-- candidate_results.json
|-- xgboost_refinement_candidates.json
|-- xgboost_early_stopping_cv.csv
|-- early_stopping_summary.csv
|-- validation_predictions.csv
|-- validation_metrics.csv
|-- models/                         # exactly four final regressors
|-- model_metadata.json
|-- test_predictions.csv
|-- test_regression_metrics.csv
|-- derived_classification_metrics.csv
|-- regression_comparison.csv
|-- classification_comparison.csv
|-- xgboost_comparison.csv
|-- verification.json
```

Using the existing Python 3.11.9 environment, executed from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -u experiments/final_6h_hyperparameter_tuning/run_experiment.py
.\.venv\Scripts\python.exe -B -u experiments/final_6h_hyperparameter_tuning/verify_results.py
```

The training entry point refuses to overwrite completed models. To evaluate the four saved models without fitting:

```powershell
.\.venv\Scripts\python.exe -B -u experiments/final_6h_hyperparameter_tuning/run_experiment.py --stage evaluate
```

Search-only execution is available through --stage train on a fresh experiment. Package versions and candidate lists are saved in design.json. No dependencies were installed or changed. No persistence model is included in the final four-model tables.

Independent verification passed: four model reloads/predictions, original ordered 18 inputs, exact six-hour targets, original test origins, strict partition/CV/nested-stopping separation, 369 final fixed XGBoost rounds, validation/test regression metrics, binary metrics and comparisons. All 334 pre-existing files are byte-identical. Original CSV SHA-256: `421199b25b1b7c475d8f09b65317f37b0aa0bf19a3385da0696e510d8a575ed8`. No original artifacts were overwritten; no Git commands were executed.
