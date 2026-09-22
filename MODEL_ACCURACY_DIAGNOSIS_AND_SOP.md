# Bitcoin Price Prediction — Model Accuracy Diagnosis, Fixes, and Run SOP

This document is a static-code review of the price-prediction pipeline in this repository. It:

1. Maps the **five** distinct "models" that exist in the codebase and which one actually runs in production.
2. Diagnoses the **model accuracy problem** with concrete root causes and `file:line` evidence.
3. Lists **possible fixes**, prioritized, as recommendations (no code is changed by this document).
4. Provides a **standard operating procedure (SOP)** for running the model and validating its accuracy.

> Scope note: this is a written report only. Every fix below is a recommendation with severity/effort; it has not been applied.

---

## 1. The pipeline landscape (read this first)

The word "model" is overloaded in this repo. There are **five** independent entry points, and they are **not** consistent with each other:

| # | Entry point | What it does | Used by |
|---|---|---|---|
| 1 | `python/run_prediction.py` | Fetches ~60 one-minute candles, builds features, fits **one** `RandomForestClassifier`, returns `predicted_direction` + probabilities | **The Node server (live path)** via `services/predictionService.js` → `PythonShell.run('run_prediction.py')` |
| 2 | `priceprediction.py` | Daily candles + blockchain/mempool data, 7-model ensemble, walk-forward CV, next-day ±1% prediction | `demo.py`, `test_runner.py` (standalone only) |
| 3 | `enhanced_prediction.py` | 1-minute data, 6-model ensemble + rule-based LONG/SHORT signals + a game-theory voting engine | Standalone; `run_prediction.py` imports `create_enhanced_features` from it |
| 4 | `enhanced_forecasting.py` | 6/12/24-hour price-level forecast at 15-min steps via `MultiOutputRegressor` | Imported by `priceprediction.py` and `continuous_training.py` |
| 5 | `continuous_training.py` | Threaded "continuous" retraining loop | `priceprediction.py` demo; intended to be the retraining engine |

**The single most important fact:** the model that the web app actually shows to users is **entry point #1** (`run_prediction.py`), *not* the elaborate ensembles in `priceprediction.py` / `enhanced_prediction.py`. Almost all of the sophistication (ensembles, walk-forward CV, calibration, feature selection, baseline reporting) lives in the standalone scripts that the server **never calls**.

---

## 2. Diagnosis — root causes of the accuracy problem

### D1. The live model is a single shallow tree ensemble trained on ~59 rows, with zero validation

`run_prediction.py` fetches a `window_size=60` slice of one-minute candles (`fetch_recent_market_data`, `run_prediction.py:39-78`) and then:

```python
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
y_train = enhanced_df['target'][:-1]
X_train = enhanced_df[feature_cols][:-1]
model.fit(X_train, y_train)          # ~59 training rows
probs = model.predict_proba(X)[0]
```

(`run_prediction.py:140-153`)

- ~59 rows against several dozen engineered features is severely under-determined; the model overfits noise and its predictions are unstable minute-to-minute.
- There is **no train/validation split, no cross-validation, and no baseline comparison** in the live path. The `confidence` field is just the raw `predict_proba` max — it is not an accuracy estimate.
- Every scheduled run (every 60 s via `predictionService.js:151-166`) fits a **fresh** model from scratch. There is no persistence and no incremental update, so run-to-run variance is high.

**Consequence:** the number shown to the user is essentially an unvalidated, uncalibrated guess.

### D2. Class-probability mapping bug (wrong probabilities when a class is absent)

`run_prediction.py:158-166` maps probabilities **positionally**:

```python
if len(probs) == 2:              # Binary classification
    decrease_prob = probs[0]
    increase_prob = probs[1]
    no_change_prob = 0
else:                            # Multi-class classification
    decrease_prob = probs[0]
    no_change_prob = probs[1]
    increase_prob = probs[2]
```

This is correct **only if** the training window contains all three classes `{-1, 0, 1}` and `model.classes_ == [-1, 0, 1]`. With ~59 samples it is common for a class to be absent; sklearn then returns a **different** `classes_` ordering and a `predict_proba` matrix with fewer columns, so the code silently labels, e.g., `P(no change)` as `decrease_prob`. The displayed Increase/Decrease/No-change probabilities can therefore be flat-out wrong without any error being raised.

**Fix:** map probabilities through `model.classes_` (see §3, F3), never positionally.

### D3. Stale-prediction bug — the model predicts on the *second-to-last* row

`create_enhanced_features` ends with `return df.dropna()` (`enhanced_prediction.py:389`). The current (last) row has `target = NaN` because `future_price = price.shift(-1)` has no next bar, so `dropna()` removes it. Then `run_prediction.py:138` predicts on `iloc[-1:]` of the **already-dropped** frame:

```python
X = enhanced_df[feature_cols].iloc[-1:]   # second-to-last original row
```

Meanwhile the JSON response reports the *current* FIX price (`run_prediction.py:170`). The features used for inference are **one bar stale** relative to the reported price.

**Fix:** keep the current row for inference and drop NaN only from the training rows (see §3, F3).

### D4. The classification label is noise (0.1% on a 1-minute bar)

The live path calls `create_enhanced_features(df, pct_threshold=0.001)` (`run_prediction.py:133`). In `enhanced_prediction.py:376-383` the target is:

```python
df.loc[valid_future & (df['next_return'] >= pct_threshold), 'target'] = 1    # ≥ +0.1%
df.loc[valid_future & (df['next_return'] <= -pct_threshold), 'target'] = -1  # ≤ -0.1%
```

±0.1% over a **one-minute** interval sits inside the bid-ask spread and tick noise band. The label is close to a coin flip, so no classifier can learn real signal from it. (This contrasts with `priceprediction.py:304-311`, which uses ±1.0% on a *daily* horizon — a very different problem.)

**Consequence:** even a "correct" implementation would report accuracy near the majority-class baseline.

### D5. Prediction-horizon mismatch across the codebase

There is no single agreed horizon or threshold:

| File | Horizon | Threshold |
|---|---|---|
| `run_prediction.py:179` | `"1 minute"` | ±0.1% |
| `priceprediction.py:304-311` | next **day** | ±1.0% |
| `enhanced_prediction.py:392-459` | grid-selected (1–60 min) | grid-selected (0.1–0.3%) |
| `enhanced_forecasting.py` | 6 / 12 / 24 hours | n/a (regression) |

The server, the standalone scripts, and the README all describe different things as "the prediction." Any accuracy number is therefore not comparable across entry points.

### D6. Reported forecast accuracy is fabricated from synthetic/interpolated data

- `priceprediction.py:create_15min_data` (lines 655-712) builds "15-minute" data by **interpolating daily closes and adding random noise**. `priceprediction.py:1145` then feeds this to `run_enhanced_forecasting`, so the 12-hour forecast is trained and evaluated on invented data.
- `priceprediction.py:fetch_blockchain_data` falls back to **random** blockchain columns on any error (`priceprediction.py:462-506`), so the "blockchain features" are often pure noise.
- `run_prediction.py:generate_synthetic_data` is a random walk used whenever the live fetch fails in demo mode.

Any accuracy printed on these paths measures the model's ability to reproduce its own injected randomness, not real predictive skill.

### D7. The user-facing "Prediction Accuracy" is faked

`client/src/pages/PredictionAnalysis.js:114-128` computes the "actual" direction with **`Math.random()`**:

```js
const actualDirection = Math.random() > 0.7 ? prediction.predicted_direction :
                        (Math.random() > 0.5 ? 1 : -1);
```

And `services/predictionService.js:getHistoricalPredictions` (lines 174-228) returns purely synthetic/random history in demo mode and is unimplemented (`501`) in live mode.

**Consequence:** the "Prediction Accuracy" card in the UI is a random number and has no relationship to real model performance. This is arguably the single most misleading thing in the app.

### D8. The continuous-training system is broken (calls a non-existent method)

`continuous_training.py:266` and `:324` call:

```python
forecaster.generate_12_hour_forecast(test_data)   # no such method
```

`EnhancedBitcoinForecaster` only defines `generate_forecast(...)` (`enhanced_forecasting.py:252`). Therefore `_check_model_agreement()` and `get_latest_prediction()` **always raise `AttributeError`** (caught and logged as warnings), and the "continuous training" engine never actually produces an ensemble prediction. Its "new data" is also a synthetic random walk (`_generate_new_data_point`, `continuous_training.py:110-133`).

### D9. Look-ahead leakage in the forecasting feature pipeline

`enhanced_forecasting.py` builds features on the **full** series before splitting:

- `preprocess_data` (`:100-115`) applies `.resample().interpolate(method='time').ffill().bfill()` across the whole series.
- `create_prediction_features` (`:194-198`) applies `.ffill().bfill().fillna(0.0)` across the whole series.
- `prepare_multistep_data` (`:200-215`) computes all features before `dropna`/train-test split.

`bfill()` (backward fill) uses **future** values to fill past NaNs, and interpolation across the full range can also look ahead. This inflates apparent accuracy on the `train_test_split(shuffle=False)` holdout.

### D10. No prediction-vs-actual ledger

There is no code path that persists a prediction (`timestamp`, `predicted_direction`, `price`, `confidence`, `horizon`) and later scores it against the realized close. Without this, **true out-of-sample accuracy is unmeasurable**, which is why every "accuracy" surface in the app is synthetic.

### D11. Uncalibrated confidence + arbitrary gates

- `CalibratedClassifierCV` and `print_confidence_report` exist in `priceprediction.py` / `enhanced_prediction.py`, but **not** in the live `run_prediction.py`.
- The 0.7 / 0.6 confidence gates (`priceprediction.py:1119-1124`, `enhanced_prediction.py:1480`) are hardcoded and never validated against realized outcomes.

### D12. Mixed naive/aware datetimes

`run_prediction.py:44,69` uses `datetime.utcnow()` and `datetime.utcfromtimestamp` (naive UTC), while `enhanced_prediction.py` uses timezone-aware UTC (`datetime.now(timezone.utc)`, `pd.to_datetime(..., utc=True)`). Mixing naive and aware timestamps is a recurring source of off-by-one bar / sorting bugs.

---

## 3. Possible fixes (prioritized — recommendations only)

Severity: **P0** = correctness/misleading-information, do first. **P1** = material accuracy. **P2** = robustness/quality.

### F1 (P0) — Stop faking accuracy; add a real prediction ledger
- **Problem:** D7, D10.
- **Change:** remove the `Math.random()` "actual direction" logic from `client/src/pages/PredictionAnalysis.js`; replace it with data from a persisted ledger.
- **Ledger design:** every `run_prediction.py` result (and the standalone runs) should write `{timestamp, price, predicted_direction, confidence, horizon, threshold}` to a store (Mongo is already in `docker-compose.yml`). After the horizon elapses, fetch the realized close and compute direction hit-rate, by-class precision/recall, and a confidence-calibration table. Expose real numbers via `/api/predictions/history`.

### F2 (P0) — Fix the broken continuous-training call
- **Problem:** D8.
- **Change:** `continuous_training.py:266,324` → `forecaster.generate_forecast(data, horizon_hours=12)`; also replace the synthetic `_generate_new_data_point` with a real market-data fetch (or drop the "continuous" claim until one exists).

### F3 (P0) — Harden the live inference path (`run_prediction.py`)
- **Problem:** D1, D2, D3, D4.
- **Changes:**
  1. Fetch a realistic history — at least 1–7 days of 1-minute candles — instead of `window_size=60`.
  2. Map probabilities through `model.classes_`:
     ```python
     class_to_prob = dict(zip(model.classes_, probs))
     decrease_prob = class_to_prob.get(-1, 0.0)
     no_change_prob = class_to_prob.get(0, 0.0)
     increase_prob = class_to_prob.get(1, 0.0)
     ```
  3. Keep the current row for inference: build features, then split `X_train = features[:-1]`, `X_latest = features[-1:]` rather than relying on `df.dropna()`.
  4. Reconsider the label: either raise `pct_threshold` to a value above the tick-noise band (and/or predict a longer horizon), or move to a regression target (next-bar return) — see F4.

### F4 (P1) — Choose one horizon/threshold and validate it properly
- **Problem:** D4, D5.
- **Change:** define a single product spec ("predict direction over the next N minutes for a move ≥ X%"). Reuse the walk-forward machinery already present (`make_time_series_split`, `walk_forward_scores`, `print_baseline_report`) and **report CV accuracy alongside the majority-class baseline**. Only ship the model if CV edge over baseline is meaningful and stable (low CV spread).

### F5 (P1) — Remove look-ahead leakage in `enhanced_forecasting.py`
- **Problem:** D9.
- **Change:** compute features only on the training portion (or use strictly expanding/`shift`-based rolling features). Remove full-series `.bfill()` from `preprocess_data` and `create_prediction_features`; use forward-fill only, and never fill across the train/test boundary.

### F6 (P1) — Use real 15-minute candles for the forecaster
- **Problem:** D6.
- **Change:** feed `enhanced_forecasting` real Coinbase `granularity=900` candles (the fetch helper already exists at `enhanced_forecasting.py:411-461`) instead of `create_15min_data`'s interpolation. Treat `create_15min_data` as demo-only, or delete it.

### F7 (P2) — Persist models and retrain on schedule / drift, not from scratch each call
- **Problem:** D1 (retrain-every-call).
- **Change:** cache the fitted model (e.g., `joblib`) keyed by a data-version/schema hash; retrain on new data or detected drift, reusing the existing `ContinuousTrainingSystem` scaffolding (once F2 is applied).

### F8 (P2) — Calibrate probabilities and gate on calibrated confidence
- **Problem:** D11.
- **Change:** wrap the live classifier in `CalibratedClassifierCV` (pattern already in `fit_calibrated_classifier`), and set decision gates using the calibration buckets printed by `print_confidence_report` rather than hardcoded 0.6/0.7.

### F9 (P2) — Metric logging + a real dashboard
- **Problem:** D10, D7.
- **Change:** log walk-forward CV, baseline, class balance, and (once the ledger exists) realized accuracy to `logs/`; surface those in the UI in place of synthetic history.

---

## 4. Standard Operating Procedure — running the model

### 4.1 Prerequisites / environment

**Python (the model runtime):**

```bash
# from the repo root
pip install -r requirements.txt
```

- CPU-only is sufficient; `requirements.txt` pins numpy/pandas/scikit-learn/scipy/requests.
- Optional GPU acceleration (RAPIDS cuML) is Linux/WSL-only:
  ```bash
  python setup_cuml_cuda.py            # auto-detect, install, verify
  python setup_cuml_cuda.py --verify-only
  ```
  On a Windows host, `setup_cuml_cuda.py` prints WSL2 + NVIDIA-driver steps and falls back to CPU sklearn (`enhanced_prediction.py` auto-detects and prints `Backend: CPU fallback active`).

**Node (the API/web layer):**

```bash
npm install
```

**Infrastructure (optional for live/demo):**

```bash
docker compose up -d     # mongo, mongo-express, redis, FIX simulator
```

### 4.2 Configuration

- `NODE_ENV` and `DEMO_MODE` control simulated data (`utils/runtimeMode.js:10-13`):
  - `DEMO_MODE=true` (or unset outside production) → synthetic fallbacks allowed.
  - `NODE_ENV=production` with `DEMO_MODE` unset → **fail-closed** (no synthetic data).
- Copy `.env.development` values as needed (port, Mongo/Redis URIs, Coinbase keys, FIX config).
- Confirm the Coinbase public candle endpoint is reachable from the run host:
  `https://api.exchange.coinbase.com/products/BTC-USD/candles`.

### 4.3 Choosing an entry point

| Goal | Command | Notes |
|---|---|---|
| Standalone batch demo (full pipeline incl. blockchain + forecasting) | `python demo.py` | Prints ensemble CV, baseline, forecast, and a 3-min continuous-training demo |
| Standalone enhanced 1-min model | `python enhanced_prediction.py` | 6-model ensemble + signals + voting engine; needs 24k 1-min points |
| Single live inference (what the server calls) | `python python/run_prediction.py --price 67000 --volume 5 --allow-synthetic` | Returns one JSON prediction |
| 12-hour forecaster alone | `python enhanced_forecasting.py` | Fetches 15-min candles, prints forecast table + stats |
| API + web app | `npm start` | Server on `PORT`; predictions scheduled every 60 s |

### 4.4 Pre-run checks

1. **Dependencies:** `python -c "import sklearn, pandas, numpy; print('ok')"`.
2. **Network:** confirm a live candle fetch succeeds (no `No candles returned from Coinbase API` error).
3. **Class balance:** the target should contain ≥2 classes; a single-class window means probabilities are degenerate (see D2).
4. **Mode awareness:** in live mode, confirm FIX market data is real and not flagged `simulated`; `predictionService.js` rejects simulated payloads outside demo mode (`predictionService.js:58-60`).

### 4.5 Running & monitoring

1. Start order: infrastructure → server (`npm start`) → client (`npm run client`), or the standalone Python entry point directly.
2. Watch the logs:
   - `logs/general.log` — prediction runs, backend selection, CV/baseline output.
   - `logs/error.log` — failures (`ImportError`, `AttributeError`, API errors).
3. In the standalone `enhanced_prediction.py` output, look for:
   - `Backend: CUDA/cuML active` or `CPU fallback active`.
   - `Mean CV Accuracy` vs the printed `Baseline majority-class accuracy`.
   - The confidence-calibration buckets (`Confidence Calibration Check`).

### 4.6 Validation / acceptance (how to judge accuracy)

1. Record **walk-forward CV accuracy** and the **majority-class baseline**. A model is only useful if CV accuracy exceeds baseline by a stable margin (small CV spread, few skipped folds).
2. Check the **confidence calibration** table: in the `≥70%` bucket, realized accuracy should be near/above the claimed confidence. If high-confidence predictions are no more accurate than low-confidence ones, the confidence is not meaningful.
3. With the prediction ledger in place (F1): score realized direction hit-rate by class and horizon; reject the model if directional hit-rate is not significantly above chance for the chosen horizon.

### 4.7 Troubleshooting

| Symptom | Likely cause | See |
|---|---|---|
| `AttributeError: 'EnhancedBitcoinForecaster' object has no attribute 'generate_12_hour_forecast'` | Broken method call in continuous training | D8 / F2 |
| `ImportError` in prediction JSON (`errorType: ImportError`) | Missing Python deps | `requirements.txt` |
| `No candles returned from Coinbase API` | Network/endpoint issue; demo mode falls back to synthetic | `run_prediction.py:61-66` |
| Increase/Decrease/No-change probabilities look wrong | Positional `probs` mapping when a class is absent | D2 / F3 |
| Prediction never reflects the latest price | `dropna()` drops the current row; prediction is one bar stale | D3 / F3 |
| "Prediction Accuracy" in UI changes randomly each refresh | Accuracy is computed with `Math.random()` | D7 / F1 |

### 4.8 Shutdown

1. Stop the web/server process (`Ctrl+C` on `npm start`).
2. Stop infrastructure if no longer needed: `docker compose down`.
3. Archive logs for post-run analysis: `logs/`.

---

## 5. Summary of highest-leverage actions

1. **F1 + D7/D10** — stop showing a random number as "accuracy"; build the prediction-vs-actual ledger. (This is a trust/UI issue before it is an ML issue.)
2. **F3** — fix the live `run_prediction.py` (enough history, `model.classes_` mapping, predict on the true last row).
3. **F4** — pick one horizon/threshold and gate on walk-forward CV vs baseline.
4. **F2** — repair `continuous_training.py` so its retraining/ensemble actually runs.
5. **F5/F6** — remove look-ahead leakage and stop training the forecaster on interpolated synthetic data.

Until items 1–4 are done, treat all printed accuracy/confidence figures in this system as **unvalidated**, and treat the web UI's "Prediction Accuracy" card as **not real**.
