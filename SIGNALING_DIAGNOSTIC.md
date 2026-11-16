# Signaling & Direction Accuracy - Diagnostic Report

## Issues Found ✅

### 1. **Cached Predictions Using Old 15-Minute Config**
- Latest prediction still shows "15 minutes" (generated before config change)
- Server cached this prediction in memory
- Configuration is updated but cached prediction hasn't expired

### 2. **Direction Accuracy IS Working** ✅
```
horizon_minutes: 15
total predictions: 7
scored: 3 (completed)
pending: 4 (waiting for windows to close)
direction_accuracy: 66.67% (3 out of 3 correct)
```

✅ System correctly calculated direction for 3 completed 15-minute windows
⚠️ But we're now using 60-minute predictions, so old accuracy is outdated

### 3. **Why Predictions Show As "Pending"**
- After prediction is made, system waits for the target window to close
- Then fetches actual OHLCV candles from Coinbase
- Only then can it score: actual price, direction correctness, error
- **60-minute predictions will take 60 minutes to score** (vs 15 minutes before)

**Timeline Example:**
```
13:00 UTC - Prediction generated for 13:00-14:00 window
          (predicted: up, confidence: 0.55)
14:00 UTC - Window closes, actual candles fetched from Coinbase
14:00+ UTC - Scored: actual direction confirmed or rejected
          (actual: up ✓ prediction correct)
```

### 4. **Signaling Configuration**
The system is configured with these signal types:
- `classifier` - Main prediction model direction
- `heldout_classifier` - Validation subset  
- `forecast_ensemble` - Regression forecasts
- `long` - Buy signals

Default predictions use `classifier` signal - this is working.

## Solution: Run Fresh Predictions with New Config

### Step 1: Force New Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predictions/run
```

This will:
- Load fresh config (60-minute horizon)
- Generate new 1-hour predictions
- Store in ledger for scoring in ~60 minutes

### Step 2: Monitor Accuracy Development
```bash
curl http://localhost:5000/api/v1/predictions/accuracy
```

Watch for new entries with `horizonMinutes: 60` to appear

### Step 3: Wait for Windows to Close
- First 60-minute prediction will be scorable at 60 minutes after generation
- Each hour, new prediction generated
- Accuracy builds over time as windows complete

## Current Status

```
✅ Direction accuracy calculation: WORKING (66.67% on 15m predictions)
✅ Signaling system: WORKING (classifier signal generating predictions)
✅ Scoring logic: WORKING (3 completed, 4 pending)
⚠️  Cached prediction: OLD (still showing 15m, needs refresh)
🔄 New 60m predictions: NEED TO GENERATE
```

## What Happens Next

### Immediate (After New Prediction Run)
1. `horizon_minutes: 60` predictions recorded
2. Status: `pending` (waiting 60 minutes)
3. Direction accuracy: N/A (waiting for window close)

### After 60 Minutes  
1. Target window closes
2. Actual candles fetched from Coinbase API
3. Direction scored: correct/incorrect calculated
4. Direction accuracy updates with real data

### Building Accuracy (Ongoing)
- Every hour: new prediction made
- Every hour: old prediction becomes scorable
- Direction accuracy smooths out over 10-20 predictions

## Data Retention

The system keeps all predictions in SQLite database at:
```
data/predictions.sqlite3
```

Tables:
- `runs`: prediction generation metadata
- `predictions`: individual predictions with scoring results

All direction correctness stored permanently for analysis.

## Summary

| Issue | Status | Solution |
|-------|--------|----------|
| Config updated to 60m | ✅ Done | Already applied |
| Old cached prediction | ⚠️ Stale | Run new prediction |
| Direction accuracy working | ✅ Yes | Already calculating |
| Scoring pending | ✅ Expected | Will complete in 60m |
| Signaling configured | ✅ Yes | Using `classifier` signal |

## Next Actions

1. **Run fresh prediction:**
   ```bash
   curl -X POST http://localhost:5000/api/v1/predictions/run
   ```

2. **Check latest prediction:**
   ```bash
   curl http://localhost:5000/api/v1/predictions/latest
   ```
   Should show `"timeframe": "1 hour"` and `"horizon_minutes": 60`

3. **Monitor accuracy in 60 minutes:**
   ```bash
   curl http://localhost:5000/api/v1/predictions/accuracy
   ```
   Should show new entry with `horizonMinutes: 60`

4. **Check full prediction history:**
   ```bash
   curl "http://localhost:5000/api/v1/predictions/history"
   ```

All systems are functional. Just need to generate fresh predictions with new config!
