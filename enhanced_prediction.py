# Optional heavy dependencies are imported lazily to allow module import without them.

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Optional, TypedDict
import importlib

StandardScaler = None
Pipeline = None
RandomForestClassifier = None
GradientBoostingClassifier = None
VotingClassifier = None
LogisticRegression = None
SVC = None
train_test_split = None
accuracy_score = None
classification_report = None
cross_val_score = None


def _ensure_sklearn():
    """Lazily import scikit-learn components when needed."""
    global StandardScaler, Pipeline
    global RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    global LogisticRegression, SVC
    global train_test_split, accuracy_score, classification_report, cross_val_score

    if RandomForestClassifier is not None:
        return

    try:
        from sklearn.ensemble import RandomForestClassifier as _RF, GradientBoostingClassifier as _GB, VotingClassifier as _VC
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.svm import SVC as _SVC
        from sklearn.model_selection import train_test_split as _tts, cross_val_score as _cvs
        from sklearn.metrics import accuracy_score as _acc, classification_report as _cls
        from sklearn.preprocessing import StandardScaler as _Scaler
        from sklearn.pipeline import Pipeline as _Pipeline
    except ImportError as exc:  # pragma: no cover - explicit guidance is more helpful than stack trace
        raise ImportError(
            "scikit-learn is required for the enhanced prediction script. "
            "Install it via 'pip install scikit-learn'."
        ) from exc

    StandardScaler = _Scaler
    Pipeline = _Pipeline
    RandomForestClassifier = _RF
    GradientBoostingClassifier = _GB
    VotingClassifier = _VC
    LogisticRegression = _LR
    SVC = _SVC
    train_test_split = _tts
    accuracy_score = _acc
    classification_report = _cls
    cross_val_score = _cvs

# Lazy loader for scipy.stats to avoid hard import at module import time (and avoid stub errors)
_stats = None
def _get_stats():
    """Lazily import scipy.stats and cache it. Returns None if unavailable."""
    global _stats
    if _stats is None:
        try:
            _stats = importlib.import_module("scipy.stats")
        except Exception:
            _stats = None
    return _stats

# matplotlib is not required for core functionality; avoid importing to reduce dependency surface
try:
    import requests
except Exception:
    requests = None
from datetime import datetime, timedelta
import time
from collections import deque


def _future_window_extrema(prices: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute future max/min prices and their indices within a fixed lookahead window."""
    n = len(prices)
    max_vals = np.full(n, np.nan)
    max_idx = np.full(n, -1, dtype=int)
    min_vals = np.full(n, np.nan)
    min_idx = np.full(n, -1, dtype=int)

    if window <= 0:
        return max_vals, max_idx, min_vals, min_idx

    max_deque = deque()
    min_deque = deque()

    for i in range(n - 1, -1, -1):
        while max_deque and (max_deque[0][1] - i) > window:
            max_deque.popleft()
        while min_deque and (min_deque[0][1] - i) > window:
            min_deque.popleft()

        if max_deque:
            max_vals[i] = max_deque[0][0]
            max_idx[i] = max_deque[0][1]
        if min_deque:
            min_vals[i] = min_deque[0][0]
            min_idx[i] = min_deque[0][1]

        while max_deque and max_deque[-1][0] <= prices[i]:
            max_deque.pop()
        max_deque.append((prices[i], i))

        while min_deque and min_deque[-1][0] >= prices[i]:
            min_deque.pop()
        min_deque.append((prices[i], i))

    return max_vals, max_idx, min_vals, min_idx


# TypedDict describing the trade plan returned by _find_optimal_trade
class TradePlan(TypedDict):
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    return_: float

def _find_optimal_trade(prices: np.ndarray, dates: np.ndarray, start_idx: int,
                        horizon: int, trade_type: str) -> Optional[TradePlan]:
    """Find the best trade (long/short) within the horizon using future data."""
    start_idx = int(start_idx)
    end_idx = min(len(prices), start_idx + horizon + 1)
    if start_idx >= len(prices) - 1 or end_idx - (start_idx + 1) <= 0:
        return None

    window_prices = prices[start_idx + 1:end_idx]
    window_dates = dates[start_idx + 1:end_idx]
    if window_prices.size == 0:
        return None

    best = None
    if trade_type == 'long':
        min_price = window_prices[0]
        min_idx_rel = 0
        best_return = -np.inf
        for idx, price in enumerate(window_prices):
            if price < min_price:
                min_price = price
                min_idx_rel = idx
            if min_price <= 0:
                continue
            ret = (price - min_price) / min_price
            if ret > best_return:
                best_return = ret
                best = (min_idx_rel, idx, ret)
    else:  # short
        max_price = window_prices[0]
        max_idx_rel = 0
        best_return = -np.inf
        for idx, price in enumerate(window_prices):
            if price > max_price:
                max_price = price
                max_idx_rel = idx
            if max_price <= 0:
                continue
            ret = (max_price - price) / max_price
            if ret > best_return:
                best_return = ret
                best = (max_idx_rel, idx, ret)

    if best is None or best[2] <= 0:
        return None

    entry_rel, exit_rel, ret = best
    entry_idx = start_idx + 1 + entry_rel
    exit_idx = start_idx + 1 + exit_rel
    return {
        'entry_time': pd.to_datetime(dates[entry_idx]),
        'entry_price': float(prices[entry_idx]),
        'exit_time': pd.to_datetime(dates[exit_idx]),
        'exit_price': float(prices[exit_idx]),
        'return_': float(ret)
    }

def create_enhanced_features(df, pct_threshold=0.01, interval_minutes=1, lookahead_minutes=24 * 60):
    """Create comprehensive technical indicators and 24h lookahead targets.

    This function will use scipy.stats.linregress for slope calculation when available.
    If scipy is not installed, it will fall back to numpy.polyfit to compute slopes.
    """
    if pd is None or np is None:
        raise ImportError("create_enhanced_features requires numpy and pandas to be installed.")

    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Basic time features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Number of 1-min intervals in a day
    intervals_in_day = 24 * 60
    
    # Price lags and returns (scaled for 1-min intervals)
    for lag in [1, 5, 15, 30, 60, intervals_in_day]:  # 1m, 5m, 15m, 30m, 1h, 1d
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
        df[f'return_{lag}'] = df['price'].pct_change(lag)
    
    # Moving averages and ratios (scaled for 1-min intervals)
    for window in [30, 60, 120, 360, intervals_in_day * 7]: # 30m, 1h, 2h, 6h, 1w
        df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['price'].ewm(span=window).mean()
        df[f'price_sma_ratio_{window}'] = df['price'] / df[f'sma_{window}']
        df[f'volatility_{window}'] = df['price'].rolling(window=window).std()
    
    # MACD indicators (using standard short-term periods, sensitive for 1-min data)
    ema_12 = df['price'].ewm(span=12).mean()
    ema_26 = df['price'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI (Relative Strength Index) (using standard periods)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = calculate_rsi(df['price'], 14)
    df['rsi_28'] = calculate_rsi(df['price'], 28)
    
    # Bollinger Bands (using standard period)
    bb_period = 20
    bb_std = 2
    bb_ma = df['price'].rolling(window=bb_period).mean()
    bb_std_dev = df['price'].rolling(window=bb_period).std()
    df['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
    df['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
    df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Price momentum and slopes (scaled for 1-min)
    for period in [30, 60, 120]: # 30m, 1h, 2h
        df[f'momentum_{period}'] = (df['price'] / df['price'].shift(period) - 1) * 100
        
        # Linear regression slope
        def calc_slope(window, period=period):
            if len(window) < period:
                return np.nan
            x = np.arange(len(window))
            stats = _get_stats()
            if stats is not None:
                # Use scipy's linregress when available
                try:
                    slope, _, _, _, _ = stats.linregress(x, window)
                    return slope
                except Exception:
                    pass
            # Fallback to numpy.polyfit if scipy is not available or fails
            try:
                slope = np.polyfit(x, window, 1)[0]
                return float(slope)
            except Exception:
                return np.nan
        
        df[f'lr_slope_{period}'] = df['price'].rolling(window=period).apply(
            lambda x: calc_slope(x, period), raw=True
        )
    
    # Volume proxies (using price volatility as proxy)
    df['volume_proxy'] = df['price'].rolling(window=20).std()
    df['price_volume_trend'] = df['price'] * df['volume_proxy']
    
    # Support/Resistance levels (scaled for 1-min)
    for window in [60, 120, 240]: # 1h, 2h, 4h
        df[f'resistance_{window}'] = df['price'].rolling(window=window).max()
        df[f'support_{window}'] = df['price'].rolling(window=window).min()
        df[f'price_position_{window}'] = (df['price'] - df[f'support_{window}']) / (df[f'resistance_{window}'] - df[f'support_{window}'])
    
    # Trend indicators (scaled for 1-min)
    df['price_trend_30'] = np.where(df['price'] > df['sma_30'], 1, 0)
    df['price_trend_120'] = np.where(df['price'] > df['sma_120'], 1, 0)
    df['trend_strength'] = df['price_trend_30'] + df['price_trend_120']
    
    # 24-hour lookahead statistics
    lookahead_steps = max(1, int(lookahead_minutes / max(interval_minutes, 1)))
    prices = df['price'].to_numpy()
    dates = df['date'].to_numpy(dtype='datetime64[ns]')

    future_max, future_max_idx, future_min, future_min_idx = _future_window_extrema(prices, lookahead_steps)

    safe_prices = np.where(prices == 0, np.nan, prices)
    future_max_return = (future_max - prices) / safe_prices
    future_min_return = (future_min - prices) / safe_prices

    nat_value = np.datetime64('NaT')
    future_max_time = np.full(len(df), nat_value, dtype='datetime64[ns]')
    future_min_time = np.full(len(df), nat_value, dtype='datetime64[ns]')

    valid_max = future_max_idx >= 0
    valid_min = future_min_idx >= 0

    if valid_max.any():
        future_max_time[valid_max] = dates[future_max_idx[valid_max]]
    if valid_min.any():
        future_min_time[valid_min] = dates[future_min_idx[valid_min]]

    df['future_max_price_24h'] = future_max
    df['future_min_price_24h'] = future_min
    df['future_max_return_24h'] = future_max_return
    df['future_min_return_24h'] = future_min_return
    df['future_max_time_24h'] = future_max_time
    df['future_min_time_24h'] = future_min_time

    # Classification target using best directional move within horizon
    df['target'] = 0
    long_mask = future_max_return >= pct_threshold
    short_mask = future_min_return <= -pct_threshold

    df.loc[long_mask & (~short_mask | (future_max_return >= np.abs(future_min_return))), 'target'] = 1
    df.loc[short_mask & (~long_mask | (np.abs(future_min_return) > future_max_return)), 'target'] = -1

    return df.dropna().reset_index(drop=True)

def fetch_bitcoin_data(num_points=36000, interval_minutes=1):
    """Fetch a specific number of Bitcoin price data points from Coinbase API.
    Falls back to simulated data on any error or if 'requests' is unavailable.
    Requires pandas and numpy when returning data.
    """
    if pd is None or np is None:
        raise ImportError("fetch_bitcoin_data requires numpy and pandas to be installed.")

    print(f"Fetching last {num_points} {interval_minutes}-minute data points...")
    all_data = []
    points_per_request = 300  # Coinbase API limit per request
    num_requests = (num_points + points_per_request - 1) // points_per_request
    granularity = interval_minutes * 60

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    end_time = datetime.utcnow()

    try:
        if requests is None:
            raise RuntimeError("'requests' library not available")
        for i in range(num_requests):
            start_time = end_time - timedelta(minutes=points_per_request * interval_minutes)
            
            params = {
                "granularity": granularity,
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
            
            print(f"Request {i+1}/{num_requests}: Fetching data from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print("   No more data available from API.")
                break
            
            all_data.extend(data)
            
            oldest_timestamp = data[-1][0]
            end_time = datetime.fromtimestamp(oldest_timestamp)

            time.sleep(0.5)

        if not all_data:
            raise ValueError("No data received from Coinbase API")

        df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
        df = df.sort_values("time", ascending=False).drop_duplicates(subset=['time']).sort_values("time")
        
        df["date"] = pd.to_datetime(df["time"], unit="s")
        df["price"] = pd.to_numeric(df["close"], errors='coerce')
        
        print(f"Successfully fetched {len(df)} unique {interval_minutes}-minute data points.")
        return df[["date", "price"]].dropna()
        
    except Exception as e:
        print(f"API Error: {e}. Using simulated data...")
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=num_points), periods=num_points, freq=f'{interval_minutes}T')
        prices = np.cumsum(np.random.randn(num_points) * 2) + 60000
        return pd.DataFrame({'date': dates, 'price': prices})

def main():
    """Enhanced Bitcoin prediction using scikit-learn ensemble on 1-minute data."""
    print("🚀 Enhanced Bitcoin 1-Minute Interval Prediction with Scikit-Learn")
    print("=" * 70)
    
    # Get data and create features
    interval_minutes = 1
    lookahead_minutes = 24 * 60
    lookahead_steps = lookahead_minutes // interval_minutes

    df = fetch_bitcoin_data(num_points=36000, interval_minutes=interval_minutes)
    print(f"📊 Fetched {len(df)} data points of Bitcoin data at 1-minute intervals")
    
    df = create_enhanced_features(
        df,
        pct_threshold=0.01,
        interval_minutes=interval_minutes,
        lookahead_minutes=lookahead_minutes
    )
    
    # Select features (exclude non-predictive columns)
    leakage_cols = {
        'date',
        'price',
        'target',
        'future_price',
        'next_return',
        'future_max_price_24h',
        'future_min_price_24h',
        'future_max_return_24h',
        'future_min_return_24h',
        'future_max_time_24h',
        'future_min_time_24h'
    }
    feature_cols = [col for col in df.columns if col not in leakage_cols]
    X = df[feature_cols].apply(lambda col: pd.to_numeric(col, errors='coerce')).astype(np.float64)
    y = df['target']
    
    print(f"⚙️  Created {len(feature_cols)} technical features")
    print(f"🎯 Using {len(feature_cols)} features for prediction")
    
    # Class distribution
    class_dist = y.value_counts().sort_index()
    print(f"\n📈 Class Distribution:")
    print(f"   Decrease ≥1.0% (24h): {class_dist.get(-1, 0)} ({class_dist.get(-1, 0)/len(y)*100:.1f}%)")
    print(f"   No Trade ≥1.0%: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
    print(f"   Increase ≥1.0% (24h): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
    
    # Time series split
    _ensure_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create ensemble of classifiers
    print(f"\n🤖 Training Scikit-Learn Ensemble...")
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
    )
    
    # Logistic Regression with scaling
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    
    # SVM with scaling
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
    ])
    
    # Ensemble voting classifier
    ensemble = VotingClassifier([
        ('rf', rf),
        ('gb', gb), 
        ('lr', lr_pipe),
        ('svm', svm_pipe)
    ], voting='soft')
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Predictions
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    confidence = np.max(y_proba, axis=1)
    
    # Performance evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=["Decrease ≥1.0%", "No Trade", "Increase ≥1.0%"],
                              labels=[-1, 0, 1], zero_division=0))
    
    # Feature importance from Random Forest
    rf.fit(X_train, y_train)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔥 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Predictions with timestamps
    test_df = df.iloc[-len(y_test):].copy()
    test_df['predicted'] = y_pred
    test_df['confidence'] = confidence
    
    print(f"\n🎯 Bitcoin 24h ≥1.0% Move Predictions:")
    print("=" * 80)
    
    # High confidence predictions
    high_conf_predictions = test_df[
        (test_df['predicted'] != 0) & (test_df['confidence'] >= 0.7)
    ]

    if len(high_conf_predictions) > 0:
        print(f"🔥 High Confidence Predictions (≥70%):")
        prices_array = df['price'].to_numpy()
        dates_array = df['date'].to_numpy(dtype='datetime64[ns]')
        for _, row in high_conf_predictions.iterrows():
            direction = "📈 INCREASE" if row['predicted'] == 1 else "📉 DECREASE"
            timestamp = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            trade_type = 'long' if row['predicted'] == 1 else 'short'
            trade_plan = _find_optimal_trade(
                prices_array,
                dates_array,
                row.name,
                lookahead_steps,
                trade_type
            )
            print(f"   Price ${row['price']:.2f} | {direction} by ≥1.0% | Confidence: {row['confidence']:.1%} | {timestamp}")
            if trade_plan:
                entry_ts = trade_plan['entry_time'].strftime('%Y-%m-%d %H:%M')
                exit_ts = trade_plan['exit_time'].strftime('%Y-%m-%d %H:%M')
                trade_label = 'Long' if trade_type == 'long' else 'Short'
                print(
                    f"      {trade_label} Entry: {entry_ts} @ ${trade_plan['entry_price']:.2f} → Exit: {exit_ts} @ ${trade_plan['exit_price']:.2f}"
                    f" | Return: {trade_plan['return_']:.2%}"
                )
            else:
                print("      No actionable trade plan found within 24h horizon")
    else:
        print("⚠️  No high-confidence significant moves predicted")

    # All predictions
    print(f"\n📋 All Test Predictions:")
    for _, row in test_df.iterrows():
        direction_map = {1: "Increase", -1: "Decrease", 0: "No Change"}
        direction = direction_map[row['predicted']]
        timestamp = row['date'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"Price ${row['price']:.2f} | Confidence: {row['confidence']:.3f} | {direction} by ≥1.0% (24h) {timestamp}")
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n🔄 Cross-Validation:")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    print(f"\n✅ Analysis Complete!")
    print(f"   📊 Dataset: {len(X_train)} train + {len(X_test)} test samples")
    print(f"   🤖 Ensemble: 4 algorithms (RF, GB, LR, SVM)")
    print(f"   📈 Features: {len(feature_cols)} technical indicators")

if __name__ == "__main__":
    main()
