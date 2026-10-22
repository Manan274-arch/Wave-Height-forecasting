# Historical experiments and superseded project files

This archive was created only after the canonical five-script 12-hour workflow reproduced the verified experiment. Original file contents are preserved byte-for-byte. Nothing here is imported by active source code.

| Directory | Historical role |
|---|---|
| `6_hour/` | Earlier six-hour classification workflow and outputs |
| `24_hour/` | Earlier 24-hour classification workflow and outputs |
| `regression/` | Early six-hour and 24-hour wave-height regression experiments |
| `controlled_regression/` | Controlled forecast-horizon and feature-set comparisons, including 6, 24, 48 and 72 hours |
| `direct_binary_classification/` | Controlled direct classifiers; not part of the finalized regression workflow |
| `final_6h_hyperparameter_tuning/` | Expanded six-hour regression hyperparameter searches; settings are not used by the final project |
| `controlled_12h_regression/` | Original independently verified 12-hour reference experiment, retained unchanged after successful canonical comparison |
| `legacy_project/` | Original root documentation, dependency list, ignore file, acquisition script, superseded source modules and legacy root model artifacts |

Historical scripts retain their original contents, including original relative paths. Relocation may invalidate their entry-point paths and imports; they are provenance records and are not advertised as directly runnable from this archive. Reconstruct their original directory layout in a separate workspace if reproducing those workflows. The historical dependency list is `legacy_project/requirements.txt`; final root requirements intentionally cover only the active five-phase workflow. Raw NetCDF downloads remain outside this consolidation scope.

The original CSV remains in the active `data/` directory without changes. The pre-cleanup inventory, complete relocation map and independent verification script are held in the external temporary preservation directory named in `results/metrics/verification.json`. Every inventoried original file is checked at either its unchanged active path or its mapped archive path. No Git commands were executed.
