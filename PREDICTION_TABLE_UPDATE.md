# Prediction History Table - New Columns Added

## Changes Made ✅

### File Modified
- `client/src/pages/Predictions.jsx`
  - Added 3 new columns to the prediction history table
  - Added icon imports for visual indicators

### New Columns Added

| Column | Shows | When Available |
|--------|-------|-----------------|
| **Actual** | Actual price at target window close (from Coinbase) | When status = "scored" |
| **% Diff** | Absolute percentage difference between predicted and actual price | When status = "scored" |
| **Signal Accurate** | Visual indicator (✓ Yes / ✗ No) showing if direction prediction was correct | When status = "scored" + direction prediction exists |

## Visual Indicators

### Signal Accurate Column
- **✓ Yes** (green) - Direction prediction was correct
- **✗ No** (red) - Direction prediction was incorrect  
- **–** (dash) - Prediction not yet scored or no direction was predicted

### Data Flow

```
1. Prediction Made
   ├─ Status: Pending
   └─ New columns: show "–" (not yet available)

2. Target Window Closes
   ├─ Status: Scored (or Unscorable)
   ├─ Actual: populated with live Coinbase price
   ├─ % Diff: calculated as |predicted - actual| / actual * 100
   └─ Signal Accurate: determined by comparing predicted_direction vs actual_direction

3. Display Updated
   └─ All three new columns now show data
```

## Example Data Display

### Live Prediction (Pending)
| Prediction | Target window | Status | Actual | % Diff | Signal Accurate |
|-----------|----------------|--------|--------|--------|-----------------|
| Up (55% confidence) | 14:00 - 15:00 | Pending | – | – | – |

### Scored Prediction (Completed)
| Prediction | Target window | Status | Actual | % Diff | Signal Accurate |
|-----------|----------------|--------|--------|--------|-----------------|
| Up (55% confidence) | 13:00 - 14:00 | Scored | $74,950.25 | +0.06% | ✓ Yes |

## Technical Details

### Actual Column
```javascript
{row.status === 'scored' && row.actual_price != null 
  ? formatUsd(row.actual_price) 
  : '–'}
```
- Shows formatted USD price from `actual_price` field in ledger
- Only displays when prediction is scored and price data available
- Data comes from Coinbase API after target window closes

### % Diff Column
```javascript
{row.status === 'scored' && row.abs_pct_error != null 
  ? formatPct(row.abs_pct_error, 2) 
  : '–'}
```
- Shows absolute percentage error (abs_pct_error field)
- Calculated by: `|predicted_price - actual_price| / actual_price * 100`
- Formatted to 2 decimal places
- Only displays when status is scored and error calculated

### Signal Accurate Column
```javascript
{row.status === 'scored' && row.direction_correct != null 
  ? (row.direction_correct === 1 ? '✓ Yes' : '✗ No') 
  : '–'}
```
- Shows green checkmark (✓ Yes) when `direction_correct === 1`
- Shows red X (✗ No) when `direction_correct === 0`
- Uses Material-UI icons: `CheckIcon` and `CloseIcon`
- Color coded for quick visual scanning
- Only displays for scored predictions with direction signals

## How It Works

### Scoring Timeline (1-Hour Predictions)
```
13:00 UTC - Model generates prediction for 13:00-14:00 window
           Direction: UP, Price: $74,950
           Status: Pending
           Columns show: "–" "–" "–"

14:00 UTC - Target window closes
           System fetches actual OHLCV from Coinbase
           Actual close price: $74,906
           
14:00+ UTC - Prediction scored
           Actual: $74,906 (from Coinbase API)
           % Diff: 0.06% (error calculation)
           Signal: ✓ Yes (prediction was correct)
           Columns now show: "$74,906" "+0.06%" "✓ Yes"
```

### Data Dependencies
- `actual_price` - Fetched from Coinbase after window closes
- `abs_pct_error` - Calculated by PredictionHistory.reconcile()
- `direction_correct` - Determined by comparing predicted vs actual direction
- All populated when `status` changes from `pending` to `scored`

## Benefits

1. **Quick Assessment** - See at a glance if predictions were correct
2. **Error Tracking** - Monitor percentage errors over time
3. **Performance Analysis** - Identify which signals need calibration
4. **Live Validation** - Verify live predictions against Coinbase data
5. **Historical Accuracy** - Compare 1-hour predictions with actual moves

## Filters Still Work

The prediction history table retains all filtering:
- **Scope**: Live vs Backtest predictions
- **Kind**: Direction, Price, High/Low, Rule signal, Trade
- **Status**: Pending, Scored, Unscorable
- **Pagination**: Navigate through large prediction sets

## Example Scenarios

### Scenario 1: Recent Direction Predictions
```
Filter: Live | Direction | All Status
Result: Shows all live direction predictions

For pending predictions:
  Actual: – | % Diff: – | Signal Accurate: –

For scored predictions:
  Actual: $74,906 | % Diff: +0.05% | Signal Accurate: ✓ Yes
  Actual: $74,920 | % Diff: -0.08% | Signal Accurate: ✗ No
```

### Scenario 2: Backtest Historical Validation
```
Filter: Backtest | Direction | Scored
Result: Shows completed backtest predictions with outcomes

Shows historical performance:
  Actual: $74,850 | % Diff: 0.03% | Signal Accurate: ✓ Yes
  Actual: $74,875 | % Diff: 0.12% | Signal Accurate: ✓ Yes
  Actual: $74,920 | % Diff: 0.08% | Signal Accurate: ✗ No

Can analyze historical accuracy trends
```

## UI/UX Features

1. **Smart Display** - Columns only show data when appropriate (status = scored)
2. **Color Coding** - Green for correct, red for incorrect signals
3. **Icons** - Material-UI icons for quick visual scanning
4. **Alignment** - Actual and % Diff right-aligned for numeric readability
5. **Responsive** - Table remains horizontally scrollable on small screens

## Browser View

The new columns will appear in this order:
```
Made | Kind | Signal | Horizon | Reference | Prediction | Target Window | Status | 
ACTUAL | % DIFF | SIGNAL ACCURATE | Outcome
```

The table is horizontally scrollable on smaller screens to accommodate all columns.

## Testing

To see the new columns working:

1. **Fresh predictions** (pending):
   ```bash
   curl -X POST http://localhost:5000/api/v1/predictions/run
   ```
   New columns will show "–" (not yet scorable)

2. **Wait 60 minutes** for window to close

3. **Check results**:
   ```
   Open http://localhost:5173/predictions
   Filter: Live | Direction | Scored
   ```
   New columns will now show actual prices and accuracy

4. **Historical data** (if available):
   ```
   Filter: Backtest
   ```
   Shows historical accuracy on past predictions

## Summary

✅ Three new data columns added to prediction history
✅ Displays actual price, percentage error, and signal accuracy
✅ Only shows data when predictions are scored and ready
✅ Visual indicators (checkmarks) for quick assessment
✅ Color coded for performance at a glance
✅ Maintains all existing filters and pagination
✅ Ready to use immediately

Rebuild complete and deployed to `client/dist/`!
