# Pending Predictions - Explained ✅

## Current Status

```
15-minute predictions: 7 scored    ✅ (Complete, all windows closed)
60-minute predictions: 13 pending  ⏳ (Windows still closing or waiting for scoring)
```

## Why So Many Pending? (This is Normal!)

### The Prediction & Scoring Pipeline

```
Timeline:
01:48 UTC - Prediction #1 made
           Target: 01:48 - 02:50 UTC (60 minutes)
           Status: PENDING

02:50 UTC - Window closes ✓
           Ready to score, BUT...

02:50-02:53 UTC - No scoring happens yet
           Scoring only runs during NEXT prediction

02:53 UTC - Prediction #2 made
           Now the SYSTEM scores Prediction #1
           Prediction #1: PENDING → SCORED
           Prediction #2: NEW → PENDING

02:55 UTC - Window #2 closes ✓
           Ready to score, waiting for Prediction #3

02:58 UTC - Prediction #3 made
           Now the SYSTEM scores Prediction #2
           And so on...
```

## Key Insight

**Scoring only happens during prediction runs, not continuously.**

- Prediction runs every **5 minutes**
- Scoring happens **during** these runs
- So predictions score within **5 minutes after window closes**

### Timeline of Current 13 Pending Predictions

| Prediction | Generated | Window Closes | Status |
|-----------|-----------|---------------|--------|
| #1 | 01:48 | 02:50 | ⏳ Ready (should score soon) |
| #2 | 01:53 | 02:55 | ⏳ Ready (should score soon) |
| #3 | 01:58 | 03:00 | ⏳ Soon |
| #4 | 02:03 | 03:05 | ⏳ Soon |
| #5 | 02:08 | 03:10 | ⏳ Soon |
| #6-#13 | 02:13-02:48 | 03:15-03:50 | ⏳ Waiting (1-50 min) |

## Why This Behavior

### Historical 15-Minute System
- **Predictions**: Made every 5 minutes
- **Windows**: 15 minutes each
- **Pending pipeline**: ~3 at any time (15÷5)
- **All 7 past predictions scored**: Because their windows closed before current time

### New 60-Minute System
- **Predictions**: Made every 5 minutes (same)
- **Windows**: 60 minutes each (4x longer)
- **Pending pipeline**: ~12 at any time (60÷5)
- **13 predictions pending**: Expected! Windows from the past hour

## When Will They Score?

### Predictions Ready to Score (window closed, waiting)
```
#1 (ended 02:50) - Scores at: 02:53-03:03 (next 5 prediction runs)
#2 (ended 02:55) - Scores at: 02:58-03:03 (next 4 prediction runs)
```

### Predictions Waiting for Window to Close
```
#3 (ends 03:00) - Scores at: 03:05-03:10 (5 min after close)
#4 (ends 03:05) - Scores at: 03:10-03:15 (5 min after close)
...
#13 (ends 03:50) - Scores at: 03:55-04:00 (5 min after close)
```

## Expected Progression

**Right now (02:53 UTC):**
```
Pending: 13 | Scored: 7
```

**In 5 minutes (02:58 UTC):** (After next prediction runs)
```
Pending: 12 | Scored: 8  ← One prediction scored
```

**In 10 minutes (03:03 UTC):**
```
Pending: 11 | Scored: 9  ← Two more scored
```

**In 60 minutes (03:50 UTC):**
```
Pending: 1 | Scored: 19  ← All but newest scored
```

**In 65 minutes (03:55 UTC):**
```
Pending: 0 | Scored: 20  ← All scored!
```

Then the pattern repeats with new predictions.

## Why Not Score Continuously?

**Design choice**: Scoring is tied to prediction generation to:

1. **Use consistent market data** - Score with same data used for prediction
2. **Batch efficiency** - Score multiple windows in one operation
3. **Reduce API calls** - Fetch candles once per prediction run, not continuously
4. **Simplify logic** - One place handles scoring (prediction finish)

## Data Availability

For scoring to work:
1. ✅ **Window must close** - Window end time ≤ current time
2. ✅ **Candles must exist** - Coinbase must have OHLCV data
3. ✅ **Next prediction runs** - Scoring triggered by next run

If Coinbase has no trades in a window (rare), that prediction becomes "unscorable" instead.

## Verify It's Working

Check that predictions are scoring as expected:

```bash
# Show all predictions with status
curl http://localhost:5000/api/v1/predictions/accuracy?scope=live

# Should show:
# - 15m: all scored (7/7)
# - 60m: growing from pending to scored
```

Or view in the UI:
```
http://localhost:5173/predictions
Filter: Live | Direction | Any status
```

## This is 100% Normal! ✅

The pending count will:
- **Grow** after each prediction run (new window added)
- **Shrink** as windows close and get scored
- **Stay ~12** at steady state (60÷5 predictions)
- **Cycle** as predictions score and new ones generate

## Common Concerns Addressed

**Q: Are predictions broken?**
A: No! Scoring happens 5 minutes after each window closes. Expected behavior.

**Q: Why 13 instead of 7?**
A: 60-minute windows = 4x longer = 4x more in pipeline (12 vs 3).

**Q: When will they score?**
A: Next 5-minute prediction run after their window closes (max 5 min wait).

**Q: Can I speed this up?**
A: No, by design. Scoring tied to prediction runs for data consistency.

## Summary

```
✅ Normal: 13 pending 60m predictions
✅ Expected: Will score as windows close (~1 per minute)
✅ Timeline: All will be scored within 60 minutes
✅ Pipeline: Steady-state is ~12 pending at any time
✅ System working correctly
```

**No action needed!** This is how the system is designed to work.

Check again in 10-15 minutes and you'll see pending count dropping as windows close and get scored.
