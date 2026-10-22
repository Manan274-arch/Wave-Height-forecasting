# Archived six-hour experiment

Target: max(VHM0[t+3h], VHM0[t+6h]); classes <2.5 m, 2.5-3.0 m inclusive, >3.0 m.
Chronological splits, two-origin purges and TimeSeriesSplit(3, gap=2).
46 causal predictors; train/validation/test sizes 20445/2920/5845.
Final refits use 23365 eligible training+validation origins.
Test Macro-F1: persistence 0.936989; XGBoost 0.933008; Logistic Regression
0.932009; Decision Tree 0.930490; Random Forest 0.922500.
Category agreement 98.9735%; 110 future High origins already High; 12 new High origins.

Artifacts, figures, metadata and source snapshots are preserved byte-for-byte.
The source snapshot records provenance; do not run its scripts directly against
archive-relative data paths. The unchanged source CSV remains in project data/.
Original root models/ also remains unchanged as a legacy copy; active scripts
now write only experiments/24_hour/. Do not alter this archive.
