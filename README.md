# Enhanced Bitcoin Price Prediction System v3

A comprehensive Bitcoin price prediction system featuring 24-hour, 12-hour, and 6-hour forecasting, market sentiment analysis, multi-model training, and continuous data updates.

## 🚀 Features

### 🔮 Multi-Horizon Price Forecasting
- **15-minute intervals**: High-resolution predictions with 15-minute granularity
- **24-hour forecast**: 96 predictions covering the next 24 hours
- **12-hour forecast**: 48 predictions covering the next 12 hours  
- **6-hour forecast**: 24 predictions covering the next 6 hours
- **Multi-step prediction**: Uses recursive forecasting with ensemble models
- **Confidence intervals**: Provides prediction uncertainty estimates
- **Detailed output**: Shows exact times, predicted prices, target prices, and sell prices

### 📈 Market Sentiment Analysis
- **6-hour window analysis**: Determines market direction using 1% variance threshold
- **Sentiment classification**: BULLISH, BEARISH, or NEUTRAL market conditions
- **Trend strength**: Quantifies the strength of price movements
- **Volatility metrics**: Comprehensive market volatility analysis

### 📊 Statistical Analysis
- **High/Low identification**: Clearly marks predicted high and low prices with timestamps
- **Central tendency**: Calculates median and average predicted prices
- **Delta analysis**: Shows absolute and percentage changes from current price
- **Price range**: Comprehensive range analysis for all forecast horizons
- **Target/Sell prices**: Displays 1% profit target and 1% stop loss prices

### 🤖 Multi-Model Training System
- **5 ensemble models**: RandomForest (x2), GradientBoosting, ExtraTrees, Pipeline
- **Minimum 3 models**: Ensures data integrity with redundant models
- **Automatic retraining**: Models retrain when new data arrives
- **Performance monitoring**: Tracks model agreement and performance

### ⏰ Continuous Data Updates
- **5-minute intervals**: Automatic data updates every 5 minutes
- **Real-time processing**: Immediate model retraining on new data
- **Data integrity checks**: Validates data quality on each update
- **Rolling window**: Maintains optimal data size for performance

### 📋 Enhanced Output Format
- **Professional formatting**: Clear sections with emojis and tables
- **Real-time logging**: Comprehensive system status logging
- **Detailed tables**: 15-minute prediction tables with predicted, target, and sell prices
- **System metrics**: Model performance and health monitoring
- **Multiple horizons**: Shows 24-hour, 12-hour, and 6-hour forecasts

## 🏃 Quick Start

### Runtime Mode (Demo vs Live)
This project now uses an explicit runtime switch for simulated data paths:

- `DEMO_MODE=true`: Allows simulated/synthetic data fallbacks for demos.
- `DEMO_MODE=false`: Disables simulated/synthetic API fallbacks.
- Production default is fail-closed: if `NODE_ENV=production` and `DEMO_MODE` is not set, simulation is disabled.

### Run the Enhanced System
```bash
python3 demo.py
```

### Run Individual Components
```bash
# Basic prediction system
python3 test_runner.py

# Enhanced forecasting only
python3 -c "from enhanced_forecasting import *; print('Enhanced system ready')"

# Continuous training demo
python3 -c "from continuous_training import *; print('Continuous training ready')"
```

## 📁 File Structure

```
├── priceprediction.py           # Main prediction system
├── enhanced_forecasting.py      # 12-hour forecasting module
├── continuous_training.py       # Continuous training system
├── demo.py                      # Demo script
├── test_runner.py              # Test runner
└── README.md                   # This file
```

## 🔧 System Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn
- SciPy, Matplotlib
- Python-dateutil

Install dependencies:
```bash
pip install numpy pandas scikit-learn scipy matplotlib python-dateutil
```

## 📊 Sample Output

```
🔮 ENHANCED BITCOIN PRICE FORECAST - 24H, 12H, 6H PREDICTIONS
================================================================================

📊 CURRENT STATUS:
   Current Price: $38,911.89
   Target Price:  $39,300.81 (+1.0%)
   Sell Price:    $38,522.77 (-1.0%)
   Timestamp: 2025-09-10 18:46:52

📈 MARKET SENTIMENT (6-hour analysis):
   🐻 Market Direction: BEARISH
   💹 Price Change: -1.44% (6-hour)
   📊 Variance: 1.68%
   🎯 Trend Strength: 0.86

🔮 24-HOUR FORECAST SUMMARY:
   📈 Predicted HIGH: $39,127.16 at 04:30 (270 min)
   📉 Predicted LOW:  $38,909.88 at 02:00 (120 min)
   📊 Average Price:  $39,033.33
   📊 Median Price:   $39,064.23
   📏 Price Range:    $217.28 (0.56%)

📈 24-HOUR DELTA ANALYSIS (vs Current Price):
   📈 HIGH Delta:     $+215.27 (+0.55%)
   📉 LOW Delta:      $-2.01 (-0.01%)
   📊 AVERAGE Delta:  $+121.44 (+0.31%)
   📊 MEDIAN Delta:   $+152.35 (+0.39%)

⏰ DETAILED 15-MINUTE PREDICTIONS (24-HOUR):
--------------------------------------------------------------------------------
Time     Predicted Price  Target Price     Sell Price       Change     Change%
--------------------------------------------------------------------------------
00:15    $38,989.97       $39,300.81       $38,522.77       $+78.08    +0.20%
00:30    $38,956.82       $39,300.81       $38,522.77       $+44.93    +0.12%
00:45    $39,050.83       $39,300.81       $38,522.77       $+138.94   +0.36%  
...
```

## 🔄 Continuous Training System

The system includes a continuous training component that:

- Updates data every 5 minutes
- Retrains models automatically
- Maintains minimum 3 models for redundancy
- Performs data integrity checks
- Monitors model agreement
- Provides real-time system status

### Running Continuous Training
```python
from continuous_training import ContinuousTrainingSystem

# Initialize system
system = ContinuousTrainingSystem(update_interval=300, min_models=3)

# Start continuous training
system.start_continuous_system(initial_data)

# Get latest predictions
prediction = system.get_latest_prediction()

# Stop system
system.stop_continuous_system()
```

## 🎯 Key Innovations

1. **Multi-horizon forecasting**: Predicts 24, 12, and 6 hours ahead with high accuracy
2. **High-resolution intervals**: 15-minute intervals for precise trading decisions
3. **Ensemble approach**: Combines multiple models for robust predictions
4. **Target/Sell prices**: Shows 1% profit target and stop loss levels
5. **Real-time adaptation**: Continuously learns from new data
6. **Comprehensive analysis**: Provides statistical, sentiment, and technical analysis
7. **Production ready**: Includes monitoring, logging, and error handling

## 📈 Performance Features

- **Fast training**: Models train in 2-3 seconds
- **Efficient prediction**: Generates 96 predictions in under 1 second
- **Memory efficient**: Maintains rolling data window
- **Scalable**: Can handle continuous operation
- **Robust**: Handles missing data and network failures

## 🛠️ Customization

### Adjust prediction horizon
```python
# Change forecast horizons (supports 6, 12, or 24 hours)
forecast_24h = forecaster.generate_forecast(data, horizon_hours=24)
forecast_12h = forecaster.generate_forecast(data, horizon_hours=12)
forecast_6h = forecaster.generate_forecast(data, horizon_hours=6)
```

### Modify update interval
```python
# Update every 2 minutes instead of 5
system = ContinuousTrainingSystem(update_interval=120)
```

### Change sentiment analysis window
```python
# Use 4-hour window instead of 6-hour
sentiment = forecaster.analyze_market_sentiment(data, window_hours=4)
```

## Class imbalance experiment

Fresh classifier training defaults to `imbalanced-learn` random oversampling.
Install the updated requirements in the model's Python environment. The sampler
duplicates minority-class observations only inside each training fold; validation,
held-out data, and calibration class frequencies remain unchanged. Resampled models
have constructor class weights disabled and receive no balanced sample weights.
This applies to scaler evaluation, the direction ensemble, and confidence-booster
classifiers. The continuous price regressors keep their existing training method.

WSL/bash commands, after syncing the updated code:

```bash
python -m pip install -r requirements.txt
CLASS_BALANCING=random-over FAST_PATH_MODEL_STATE=0 python enhanced_prediction.py

# Compare against the previous weighting method on a fresh run
CLASS_BALANCING=sample-weight FAST_PATH_MODEL_STATE=0 python enhanced_prediction.py
```

The same options apply to `priceprediction.py`. Compare held-out balanced accuracy,
macro F1, and per-class precision/recall, then evaluate forward results in prediction
history. Oversampling is an experiment; an accuracy improvement is not assumed.
Each saved classifier records its balancing method. Fast-path runs retain the
72-hour cache policy and report the cached model's method; changing
`CLASS_BALANCING` does not silently retrain an otherwise valid cache. A fresh run
replaces the existing classifier cache, so comparisons do not maintain two caches.

## Prediction history and accuracy

Both `enhanced_prediction.py` and `priceprediction.py` automatically save their
outputs to `data/predictions.sqlite3`. Override the location with
`PREDICTION_DB_PATH`. The database is independent of the 72-hour model caches;
keep it when deleting or refreshing fitted models.

Each run first scores pending predictions against completed Coinbase five-minute
candles, fetching missing historical ranges as needed. A price forecast labeled
18:00 is scored against the close of the **18:00–18:15** window. Missing candles
remain pending; actual prices are never interpolated. Classifications keep their
native outcome horizon, while high/low forecasts use the next 15-minute window.
Scoring happens when you run a prediction again; no background scheduler is added.

The ledger retains overlapping forecasts from separate runs and the displayed,
baseline-adjusted ensemble prices. It records model identity, training cutoff,
cache use, confidence, and live/historical status. Historical test predictions
and outcomes already known when generated are excluded from live summaries.
The displayed historical trade outcome is an observed price, not a price forecast.
Older model caches may have an unknown training cutoff until refreshed.

From the repository root, using the same Python environment as the models:

```bash
# Live accuracy summary, grouped by output type and horizon
python -m services.prediction_history

# Export predictions alongside actuals, errors, and pending/scored status
python -m services.prediction_history --csv predictions.csv

# Six-hour price forecasts from runs in October (exclusive end date, UTC)
python -m services.prediction_history --kind price --horizon 360 \
  --start 2026-10-01 --end 2026-11-01 --csv six-hour.csv

# Include historical evaluations, or select one model version/database
python -m services.prediction_history --history all
python -m services.prediction_history --model-version VERSION --db /path/to/history.sqlite3
```

Percentage error is `100 * (predicted - actual) / actual`; success means absolute
error at most **1%**. Reports include MAE, RMSE, MAPE, direction accuracy and sample
counts, with high/low endpoint errors and range containment where applicable.
Trade reports show underlying market returns, not simulated trading profits.
CSV timestamps are UTC; model terminal timestamps remain America/New_York.
Date filters select run generation times. CSV exports include pending records.

Git and `scripts/sync-to-wsl.ps1` exclude SQLite databases and sidecar files so a
code sync cannot replace WSL history. Windows and WSL use separate local ledgers
by default. For custom filenames without a database extension, set
`PREDICTION_DB_PATH` in the sync shell too and add a Git ignore rule, or keep the
database outside the repository. Back up the runtime database separately.

Tests (no network or full model training):

```bash
python -m unittest test_prediction_history -v
python -m unittest test_history_integration test_forecasting_backends -v
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📞 Support

For questions or issues, please check the documentation or open an issue on the repository.
