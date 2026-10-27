# CoastCast — Vercel deployment

A React dashboard and a read-only Python inference function. No Docker, database, API keys, or training jobs are required.

## Run locally

From the repository root, in separate terminals:

```powershell
.venv\Scripts\python.exe web\api\index.py
```

```powershell
cd web
npm.cmd install
npm.cmd run dev
```

Open the Vite URL printed in the terminal. Vite proxies `/api` to port 8000. Use the project's existing Python environment, or install `web/requirements.txt` in a separate environment. Python 3.11 or newer is required by the pinned scientific packages.

## Deploy on Vercel

1. Push this repository, including `web/artifacts`, to your Git provider.
2. Import it in Vercel and set **Root Directory** to `web`.
3. Select the **Vite** framework preset. Build command: `npm run build`; output: `dist`.
4. Deploy. `vercel.json` routes `/api/*` to the Python function. The dashboard and inference share one domain.
5. Check `/api/health`, then run an Elevated event replay. Confirm all four predictions appear and reveal the outcome.

The CPU-only XGBoost distribution avoids GPU dependencies in the Linux deployment. Scientific-library versions match model metadata. The deployed Linux bundle size and wheel availability must be verified by Vercel's build; local Windows package sizes are not a guarantee. If Vercel reports a size limit, use its supported Large Functions option where available. Do not silently change scikit-learn versions to bypass installation failures: retrain or re-export with a documented compatible environment instead.

## Verify

From repository root:

```powershell
.venv\Scripts\python.exe web\scripts\verify_inference.py
npm.cmd run build --prefix web
```

Verification compares the API with the original feature pipeline and saved predictions for all models at six origins, including first/last test dates and the scenario presets. It also checks invalid origins. The API loads saved joblib models; it does not return cached predictions as its forecast.

## Demo sequence

1. Begin with the Elevated event preset: describe the 48-hour history and forecast origin.
2. Run the forecast, compare the four models, then reveal the actual +12 h measurement.
3. Show a calm period and the largest XGBoost error. These are descriptive test examples, not independent validation or unseen trials.
4. Open Model Comparison: explain RMSE, event recall, class imbalance, and confusion counts.
5. Open Data Explorer: discuss seasonality and the separate chart axes.
6. Open Methodology: explain causal features, purged chronological partitions, and refitting.

All times are UTC. The API only accepts existing test origins. It computes features from history through the origin; future measurements are returned separately for reveal. The default forecast is XGBoost; selecting another model updates the displayed model from the same four-model run. Changing the origin requires Run forecast. CSV exports include actual height only after reveal.

## Artifacts

`web/artifacts` is a deployable snapshot of the project's saved models, metadata, raw observations, metrics and test predictions. If canonical artifacts change, copy the new files into this directory and rerun verification. No model retraining happens in this app. Joblib files must come only from this trusted project.

This is a research demonstration for one offshore Udupi location. The 2.5 m threshold is a chosen research threshold. The app has no live data feed and is not an operational warning system.

Prediction time selection: the date picker and example cards now select the target time. The API subtracts exactly 12 hours to find the model input origin. Recorded outcomes, exported target times and predictions all refer to the selected time. Restart the local Python API after this update.
