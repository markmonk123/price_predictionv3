# Optional heavy dependencies are imported lazily to allow module import without them.
try:
    import numpy as np
except Exception:
    np = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception:
    RandomForestClassifier = GradientBoostingClassifier = VotingClassifier = None
    train_test_split = cross_val_score = None
    classification_report = confusion_matrix = accuracy_score = None
    LogisticRegression = SVC = None
    StandardScaler = None
    Pipeline = None
try:
    from scipy import stats
except Exception:
    stats = None
# matplotlib is not required for core functionality; avoid importing to reduce dependency surface
try:
    import requests
except Exception:
    requests = None
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import time

DISPLAY_TZ = ZoneInfo("America/New_York")


def format_display_time(timestamp):
    """Convert timestamps to America/New_York for user-facing output."""
    ts = pd.to_datetime(timestamp, utc=True, errors='coerce')
    if pd.isna(ts):
        return "Invalid timestamp"
    return ts.tz_convert(DISPLAY_TZ).strftime('%Y-%m-%d %I:%M:%S %p %Z')


def display_forecast_windows(price_df, horizon_hours=12):
    """Display 15-minute forecast windows when no trade signal is available."""
    try:
        from enhanced_forecasting import EnhancedBitcoinForecaster
    except Exception as e:
        print(f"\n⚠️ Forecast windows unavailable: {e}")
        return False

    try:
        forecast_input = price_df[['date', 'price']].copy()
        current_price = pd.to_numeric(forecast_input['price'], errors='coerce').dropna().iloc[-1]

        print("\n🔮 No high-confidence trade signal. Generating forecast windows instead...")
        forecaster = EnhancedBitcoinForecaster()

        if not forecaster.train_models(forecast_input, horizon_hours=horizon_hours):
            print("⚠️ Forecast window generation failed during model training.")
            return False

        forecast_df = forecaster.generate_forecast(forecast_input, horizon_hours=horizon_hours)
        if forecast_df is None or forecast_df.empty:
            print("⚠️ Forecast window generation failed during inference.")
            return False

        forecast_stats = forecaster.calculate_forecast_statistics(
            forecast_df, current_price=current_price, horizon_hours=horizon_hours
        )

        print(f"\n📅 Forecast Windows ({horizon_hours}h, 15-minute intervals):")
        for _, row in forecast_df.iterrows():
            forecast_delta_pct = ((row['predicted_price'] - current_price) / current_price) * 100
            timestamp = format_display_time(row['timestamp'])
            print(
                f"   +{int(row['interval_minutes']):>3}m | {timestamp} | "
                f"${row['predicted_price']:.2f} | Δ {forecast_delta_pct:+.2f}% | "
                f"σ {row['prediction_std']:.2f}"
            )

        print(f"\n📊 Forecast Summary:")
        print(f"   Current Price: ${forecast_stats['current_price']:.2f}")
        print(f"   Forecast Mean: ${forecast_stats['forecast_mean']:.2f}")
        print(f"   Forecast Range: ${forecast_stats['forecast_min']:.2f} to ${forecast_stats['forecast_max']:.2f}")
        print(f"   Forecast Volatility: {forecast_stats['forecast_volatility']:.4f}")
        print(f"   Forecast Choppiness: {forecast_stats['forecast_choppiness']:.2f}")
        return True

    except Exception as e:
        print(f"\n⚠️ Failed to generate forecast windows: {e}")
        return False

def create_enhanced_features(df, pct_threshold=0.002):
    """Create comprehensive technical indicators for Bitcoin 1-minute interval prediction.
    Requires numpy, pandas, and scipy.stats. Raises ImportError if unavailable.
    """
    if pd is None or np is None or stats is None:
        raise ImportError("create_enhanced_features requires numpy, pandas, and scipy to be installed.")

    df = df.copy()
    
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
        df[f'price_sma_ratio_{window}'] = df['price'] / (df[f'sma_{window}'] + 1e-8)
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
        rs = gain / (loss + 1e-8)
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
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_ma + 1e-8)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['price'] - df['bb_lower']) / (bb_range + 1e-8)
    # BB %B: where price sits within the band expressed as a percentage (0% = lower, 100% = upper)
    df['bb_pct'] = df['bb_position'] * 100

    # Short-term volatility (10-period rolling std) used for next high/low window sizing
    df['sma_volatility_10'] = df['price'].rolling(window=10).std()

    # Price momentum and slopes (scaled for 1-min)
    for period in [30, 60, 120]: # 30m, 1h, 2h
        df[f'momentum_{period}'] = (df['price'] / (df['price'].shift(period) + 1e-8) - 1) * 100
        
        # Linear regression slope
        def calc_slope(window, period=period):
            if len(window) < period:
                return np.nan
            x = np.arange(len(window))
            slope, _, _, _, _ = stats.linregress(x, window)
            return slope
        
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
        price_range = df[f'resistance_{window}'] - df[f'support_{window}']
        df[f'price_position_{window}'] = (df['price'] - df[f'support_{window}']) / (price_range + 1e-8)
    
    # Trend indicators (scaled for 1-min)
    df['price_trend_30'] = np.where(df['price'] > df['sma_30'], 1, 0)
    df['price_trend_120'] = np.where(df['price'] > df['sma_120'], 1, 0)
    df['trend_strength'] = df['price_trend_30'] + df['price_trend_120']

    # DPO (Detrended Price Oscillator): price minus shifted SMA removes long-term trend
    dpo_period = 20
    dpo_shift = dpo_period // 2 + 1
    df['dpo'] = df['price'] - df['price'].rolling(window=dpo_period).mean().shift(dpo_shift)

    # Directional Movement Index: ADX, DI+, DI-
    adx_period = 14
    if 'high' in df.columns and 'low' in df.columns:
        high_s = df['high']
        low_s = df['low']
    else:
        # Approximate high/low from close when not available
        high_s = df['price'].rolling(window=2).max()
        low_s = df['price'].rolling(window=2).min()
    tr1 = high_s - low_s
    tr2 = (high_s - df['price'].shift(1)).abs()
    tr3 = (low_s - df['price'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    dm_plus_raw = high_s.diff()
    dm_minus_raw = low_s.shift(1) - low_s
    dm_plus = dm_plus_raw.where((dm_plus_raw > dm_minus_raw) & (dm_plus_raw > 0), 0.0)
    dm_minus = dm_minus_raw.where((dm_minus_raw > dm_plus_raw) & (dm_minus_raw > 0), 0.0)
    atr_adx = tr.ewm(alpha=1 / adx_period, adjust=False).mean()
    df['di_plus'] = 100 * dm_plus.ewm(alpha=1 / adx_period, adjust=False).mean() / (atr_adx + 1e-8)
    df['di_minus'] = 100 * dm_minus.ewm(alpha=1 / adx_period, adjust=False).mean() / (atr_adx + 1e-8)
    dx = 100 * (df['di_plus'] - df['di_minus']).abs() / (df['di_plus'] + df['di_minus'] + 1e-8)
    df['adx'] = dx.ewm(alpha=1 / adx_period, adjust=False).mean()

    # Classification target
    df['future_price'] = df['price'].shift(-1)
    df['next_return'] = (df['future_price'] - df['price']) / (df['price'] + 1e-8)
    df['target'] = 0
    df.loc[df['next_return'] >= pct_threshold, 'target'] = 1
    df.loc[df['next_return'] <= -pct_threshold, 'target'] = -1
    
    # Replace inf and nan values
    df = df.replace([np.inf, -np.inf], np.nan)
    df['next_return'] = (df['future_price'] - df['price']) / df['price']
    df['target'] = 0
    df.loc[df['next_return'] >= pct_threshold, 'target'] = 1
    df.loc[df['next_return'] <= -pct_threshold, 'target'] = -1
    
    return df.dropna()

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
    end_time = datetime.now(timezone.utc)

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
            
            print(
                f"Request {i+1}/{num_requests}: Fetching data from "
                f"{start_time.strftime('%Y-%m-%d %H:%M UTC')} to {end_time.strftime('%Y-%m-%d %H:%M UTC')}"
            )
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print("   No more data available from API.")
                break
            
            all_data.extend(data)
            
            oldest_timestamp = data[-1][0]
            end_time = datetime.fromtimestamp(oldest_timestamp, tz=timezone.utc)

            time.sleep(0.5)

        if not all_data:
            raise ValueError("No data received from Coinbase API")

        df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
        df = df.sort_values("time", ascending=False).drop_duplicates(subset=['time']).sort_values("time")
        
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["price"] = pd.to_numeric(df["close"], errors='coerce')
        df["high"] = pd.to_numeric(df["high"], errors='coerce')
        df["low"] = pd.to_numeric(df["low"], errors='coerce')
        
        print(f"Successfully fetched {len(df)} unique {interval_minutes}-minute data points.")
        return df[["date", "price", "high", "low"]].dropna()
        
    except Exception as e:
        print(f"API Error: {e}. Using simulated data...")
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(minutes=num_points),
            periods=num_points,
            freq=f'{interval_minutes}min',
            tz='UTC'
        )
        prices = np.cumsum(np.random.randn(num_points) * 2) + 60000
        noise = np.abs(np.random.randn(num_points) * 15)
        return pd.DataFrame({'date': dates, 'price': prices, 'high': prices + noise, 'low': prices - noise})

def calculate_next_hl_window(row):
    """Use Bollinger Band %B and sma_volatility_10 to project the next high/low price window.

    Logic:
    - The short-term volatility (sma_volatility_10) defines the expected price move magnitude.
    - The projected next_high is price + sma_volatility_10, capped at the upper Bollinger Band.
    - The projected next_low  is price - sma_volatility_10, floored at the lower Bollinger Band.
    - BB %B context: when price is near the upper band (bb_pct > 80) the upside is limited;
      when near the lower band (bb_pct < 20) the downside is limited.
    - window_pct_diff is the full span of the projected window as % of current price.

    Returns a dict with next_high, next_low, window_pct_diff, and bb_pct.
    """
    price = row['price']
    vol = row['sma_volatility_10']
    bb_upper = row['bb_upper']
    bb_lower = row['bb_lower']
    bb_pct = row['bb_pct']

    raw_next_high = price + vol
    raw_next_low = price - vol

    # Respect band boundaries so the window never exceeds what the bands imply
    next_high = min(raw_next_high, bb_upper)
    next_low = max(raw_next_low, bb_lower)

    window_pct_diff = (next_high - next_low) / price * 100
    return {
        'next_high': next_high,
        'next_low': next_low,
        'window_pct_diff': window_pct_diff,
        'bb_pct': bb_pct,
    }


def identify_long_signal(df, adx_threshold=25, lr_period=30):
    """Identify Long Signals using rule-based conditions (all must be true simultaneously):
    - DPO > 0: Price above the detrended baseline
    - LR Slope > 0: Linear Regression slope positive (upward price momentum)
    - DI+ > DI-: Positive directional movement dominates
    - ADX >= adx_threshold: Trend strength confirmed (>= 25)
    - MACD Histogram > 0: MACD line above signal line (bullish momentum layer)
    Returns a DataFrame of rows satisfying all five conditions.
    """
    lr_col = f'lr_slope_{lr_period}'
    required = ['dpo', lr_col, 'di_plus', 'di_minus', 'adx', 'macd_histogram']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Long Signal identification: {missing}")
    long_mask = (
        (df['dpo'] > 0) &
        (df[lr_col] > 0) &
        (df['di_plus'] > df['di_minus']) &
        (df['adx'] >= adx_threshold) &
        (df['macd_histogram'] > 0)
    )
    result = df[long_mask].copy()
    result['signal'] = 'LONG'
    return result


def identify_short_signal(df, adx_threshold=25):
    """Identify Short Signals using rule-based conditions (all must be true simultaneously):
    - DI- > DI+: Negative directional movement dominates
    - DPO < 0: Price below the detrended baseline
    - ADX >= adx_threshold: Trend strength confirmed (>= 25)
    - MACD Histogram < 0: MACD line below signal line (bearish momentum layer)
    Returns a DataFrame of rows satisfying all four conditions.
    """
    required = ['dpo', 'di_plus', 'di_minus', 'adx', 'macd_histogram']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Short Signal identification: {missing}")
    short_mask = (
        (df['di_minus'] > df['di_plus']) &
        (df['dpo'] < 0) &
        (df['adx'] >= adx_threshold) &
        (df['macd_histogram'] < 0)
    )
    result = df[short_mask].copy()
    result['signal'] = 'SHORT'
    return result


def identify_short_signal_lrs(df, adx_threshold=25, lr_period=30):
    """Identify Short Signals via LR Slope divergence — fired when DPO and LRS do not align.
    Conditions (all must be true simultaneously):
    - DI- > DI+: Negative directional movement dominates
    - LR Slope < 0: Downward price momentum
    - ADX >= adx_threshold: Trend strength confirmed (>= 25)
    - MACD Histogram < 0: MACD confirms bearish momentum, compensating for absent DPO
    DPO is intentionally excluded: MACD fills its role when LRS and DPO are out of sync.
    Returns a DataFrame of rows satisfying all four conditions.
    """
    lr_col = f'lr_slope_{lr_period}'
    required = ['di_plus', 'di_minus', lr_col, 'adx', 'macd_histogram']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Short (LRS) Signal identification: {missing}")
    short_lrs_mask = (
        (df['di_minus'] > df['di_plus']) &
        (df[lr_col] < 0) &
        (df['adx'] >= adx_threshold) &
        (df['macd_histogram'] < 0)
    )
    result = df[short_lrs_mask].copy()
    result['signal'] = 'SHORT_LRS'
    return result


def identify_long_signal_dpo(df, adx_threshold=25):
    """Inverse of identify_short_signal — Long via DPO alignment (LRS not required).
    Fires when LRS and DPO are out of sync on the long side.
    Conditions (all must be true simultaneously):
    - DI+ > DI-: Positive directional movement dominates
    - DPO > 0: Price above the detrended baseline
    - ADX >= adx_threshold: Trend strength confirmed (>= 25)
    - MACD Histogram > 0: MACD confirms bullish momentum, compensating for absent LRS
    LRS is intentionally excluded: MACD fills its role when DPO and LRS are out of sync.
    Returns a DataFrame of rows satisfying all four conditions.
    """
    required = ['dpo', 'di_plus', 'di_minus', 'adx', 'macd_histogram']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Long (DPO) Signal identification: {missing}")
    long_dpo_mask = (
        (df['di_plus'] > df['di_minus']) &
        (df['dpo'] > 0) &
        (df['adx'] >= adx_threshold) &
        (df['macd_histogram'] > 0)
    )
    result = df[long_dpo_mask].copy()
    result['signal'] = 'LONG_DPO'
    return result


def identify_long_signal_lrs(df, adx_threshold=25, lr_period=30):
    """Inverse of identify_short_signal_lrs — Long via LR Slope alignment (DPO not required).
    Fires when DPO and LRS are out of sync on the long side.
    Conditions (all must be true simultaneously):
    - DI+ > DI-: Positive directional movement dominates
    - LR Slope > 0: Upward price momentum
    - ADX >= adx_threshold: Trend strength confirmed (>= 25)
    - MACD Histogram > 0: MACD confirms bullish momentum, compensating for absent DPO
    DPO is intentionally excluded: MACD fills its role when LRS and DPO are out of sync.
    Returns a DataFrame of rows satisfying all four conditions.
    """
    lr_col = f'lr_slope_{lr_period}'
    required = ['di_plus', 'di_minus', lr_col, 'adx', 'macd_histogram']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Long (LRS) Signal identification: {missing}")
    long_lrs_mask = (
        (df['di_plus'] > df['di_minus']) &
        (df[lr_col] > 0) &
        (df['adx'] >= adx_threshold) &
        (df['macd_histogram'] > 0)
    )
    result = df[long_lrs_mask].copy()
    result['signal'] = 'LONG_LRS'
    return result


def main():
    """Enhanced Bitcoin prediction using scikit-learn ensemble on 1-minute data."""
    print("🚀 Enhanced Bitcoin 1-Minute Interval Prediction with Scikit-Learn")
    print("=" * 70)
    
    # Get data and create features
    raw_df = fetch_bitcoin_data(num_points=36000, interval_minutes=1)
    print(f"📊 Fetched {len(raw_df)} data points of Bitcoin data at 1-minute intervals")
    
    df = create_enhanced_features(raw_df, pct_threshold=0.002) # Adjusted threshold for smaller timeframe
    print(f"⚙️  Created {len(df.columns)-3} technical features")  # -3 for date, price, target

    # --- Rule-Based LONG Signal: DPO > 0 AND LR Slope > 0 AND DI+ > DI- AND ADX >= 25 AND MACD Hist > 0 ---
    print("\n📡 Rule-Based LONG Signal (DPO > 0 AND LR Slope > 0 AND DI+ > DI- AND ADX ≥ 25 AND MACD Hist > 0):")    
    try:
        long_signals_rule = identify_long_signal(df, adx_threshold=25, lr_period=30)
        if not long_signals_rule.empty:
            latest = long_signals_rule.iloc[-1]
            hl_window = calculate_next_hl_window(latest)
            print(f"   ✅ LONG Signal Detected!")
            print(f"   Entry Price  : ${latest['price']:.2f}")
            print(f"   Timestamp    : {format_display_time(latest['date'])}")
            print(f"   DPO          : {latest['dpo']:.4f}  (> 0 ✓)")
            print(f"   LR Slope(30) : {latest['lr_slope_30']:.6f}  (> 0 ✓)")
            print(f"   DI+          : {latest['di_plus']:.2f}  > DI- {latest['di_minus']:.2f}  (Strong ✓)")
            print(f"   ADX          : {latest['adx']:.2f}  (≥ 25 ✓)")
            print(f"   MACD Hist    : {latest['macd_histogram']:.4f}  (> 0 ✓)  MACD {latest['macd']:.4f} / Sig {latest['macd_signal']:.4f}")
            print(f"   ── Next High/Low Window (BB + Volatility) ──────────────")
            print(f"   BB %B        : {hl_window['bb_pct']:.1f}%  (position within bands)")
            print(f"   BB Upper     : ${latest['bb_upper']:.2f}")
            print(f"   BB Lower     : ${latest['bb_lower']:.2f}")
            print(f"   Vol(10)      : ±${latest['sma_volatility_10']:.2f}")
            print(f"   Next High    : ${hl_window['next_high']:.2f}")
            print(f"   Next Low     : ${hl_window['next_low']:.2f}")
            print(f"   Window Δ     : {hl_window['window_pct_diff']:.3f}%")
            print(f"   Total Long Signals In Period: {len(long_signals_rule)}")
        else:
            print("   ⚠  No Long Signal: DPO > 0, LR Slope > 0, DI+ dominant, ADX ≥ 25, and MACD Hist > 0 not simultaneously met.")
    except ValueError as e:
        print(f"   ⚠  Signal identification error: {e}")

    # --- Rule-Based Short Signal: DI- > DI+, DPO < 0, ADX >= 25, MACD Hist < 0 ---
    print("\n📡 Rule-Based SHORT Signal (DI- > DI+ AND DPO < 0 AND ADX ≥ 25 AND MACD Hist < 0):")
    try:
        short_signals_rule = identify_short_signal(df, adx_threshold=25)
        if not short_signals_rule.empty:
            latest_s = short_signals_rule.iloc[-1]
            hl_window_s = calculate_next_hl_window(latest_s)
            print(f"   🔴 SHORT Signal Detected!")
            print(f"   Entry Price  : ${latest_s['price']:.2f}")
            print(f"   Timestamp    : {format_display_time(latest_s['date'])}")
            print(f"   DPO          : {latest_s['dpo']:.4f}  (< 0 ✓)")
            print(f"   DI-          : {latest_s['di_minus']:.2f}  > DI+ {latest_s['di_plus']:.2f}  (Dominant ✓)")
            print(f"   ADX          : {latest_s['adx']:.2f}  (≥ 25 ✓)")
            print(f"   MACD Hist    : {latest_s['macd_histogram']:.4f}  (< 0 ✓)  MACD {latest_s['macd']:.4f} / Sig {latest_s['macd_signal']:.4f}")
            print(f"   ── Next High/Low Window (BB + Volatility) ──────────────")
            print(f"   BB %B        : {hl_window_s['bb_pct']:.1f}%  (position within bands)")
            print(f"   BB Upper     : ${latest_s['bb_upper']:.2f}")
            print(f"   BB Lower     : ${latest_s['bb_lower']:.2f}")
            print(f"   Vol(10)      : ±${latest_s['sma_volatility_10']:.2f}")
            print(f"   Next High    : ${hl_window_s['next_high']:.2f}")
            print(f"   Next Low     : ${hl_window_s['next_low']:.2f}")
            print(f"   Window Δ     : {hl_window_s['window_pct_diff']:.3f}%")
            print(f"   Total Short Signals In Period: {len(short_signals_rule)}")
        else:
            print("   ⚠  No Short Signal: DI- dominant, DPO < 0, ADX ≥ 25, and MACD Hist < 0 not simultaneously met.")
    except ValueError as e:
        print(f"   ⚠  Signal identification error: {e}")

    # --- Short Signal (LRS Variant): DI- > DI+, LR Slope < 0, ADX >= 25, MACD Hist < 0 ---
    print("\n📡 SHORT Signal — LRS Divergence (DI- > DI+ AND LR Slope < 0 AND ADX ≥ 25 AND MACD Hist < 0):")
    try:
        short_lrs_rule = identify_short_signal_lrs(df, adx_threshold=25, lr_period=30)
        if not short_lrs_rule.empty:
            latest_sl = short_lrs_rule.iloc[-1]
            hl_window_sl = calculate_next_hl_window(latest_sl)
            print(f"   🔴 SHORT (LRS) Signal Detected!")
            print(f"   Entry Price  : ${latest_sl['price']:.2f}")
            print(f"   Timestamp    : {format_display_time(latest_sl['date'])}")
            print(f"   LR Slope(30) : {latest_sl['lr_slope_30']:.6f}  (< 0 ✓)")
            print(f"   DI-          : {latest_sl['di_minus']:.2f}  > DI+ {latest_sl['di_plus']:.2f}  (Dominant ✓)")
            print(f"   ADX          : {latest_sl['adx']:.2f}  (≥ 25 ✓)")
            print(f"   MACD Hist    : {latest_sl['macd_histogram']:.4f}  (< 0 ✓)  MACD {latest_sl['macd']:.4f} / Sig {latest_sl['macd_signal']:.4f}")
            print(f"   DPO          : {latest_sl['dpo']:.4f}  (not required — MACD fills momentum gap)")
            print(f"   ── Next High/Low Window (BB + Volatility) ──────────────")
            print(f"   BB %B        : {hl_window_sl['bb_pct']:.1f}%  (position within bands)")
            print(f"   BB Upper     : ${latest_sl['bb_upper']:.2f}")
            print(f"   BB Lower     : ${latest_sl['bb_lower']:.2f}")
            print(f"   Vol(10)      : ±${latest_sl['sma_volatility_10']:.2f}")
            print(f"   Next High    : ${hl_window_sl['next_high']:.2f}")
            print(f"   Next Low     : ${hl_window_sl['next_low']:.2f}")
            print(f"   Window Δ     : {hl_window_sl['window_pct_diff']:.3f}%")
            print(f"   Total Short (LRS) Signals In Period: {len(short_lrs_rule)}")
        else:
            print("   ⚠  No Short (LRS) Signal: DI- dominant, LR Slope < 0, ADX ≥ 25, and MACD Hist < 0 not simultaneously met.")
    except ValueError as e:
        print(f"   ⚠  Signal identification error: {e}")

    # --- Long Signal (DPO Variant): DI+ > DI-, DPO > 0, ADX >= 25, MACD Hist > 0 [inverse of SHORT primary] ---
    print("\n📡 LONG Signal — DPO Only (DI+ > DI- AND DPO > 0 AND ADX ≥ 25 AND MACD Hist > 0):")
    try:
        long_dpo_rule = identify_long_signal_dpo(df, adx_threshold=25)
        if not long_dpo_rule.empty:
            latest_ld = long_dpo_rule.iloc[-1]
            hl_window_ld = calculate_next_hl_window(latest_ld)
            print(f"   ✅ LONG (DPO) Signal Detected!")
            print(f"   Entry Price  : ${latest_ld['price']:.2f}")
            print(f"   Timestamp    : {format_display_time(latest_ld['date'])}")
            print(f"   DPO          : {latest_ld['dpo']:.4f}  (> 0 ✓)")
            print(f"   DI+          : {latest_ld['di_plus']:.2f}  > DI- {latest_ld['di_minus']:.2f}  (Dominant ✓)")
            print(f"   ADX          : {latest_ld['adx']:.2f}  (≥ 25 ✓)")
            print(f"   MACD Hist    : {latest_ld['macd_histogram']:.4f}  (> 0 ✓)  MACD {latest_ld['macd']:.4f} / Sig {latest_ld['macd_signal']:.4f}")
            print(f"   LR Slope(30) : {latest_ld['lr_slope_30']:.6f}  (not required — MACD fills momentum gap)")
            print(f"   ── Next High/Low Window (BB + Volatility) ──────────────")
            print(f"   BB %B        : {hl_window_ld['bb_pct']:.1f}%  (position within bands)")
            print(f"   BB Upper     : ${latest_ld['bb_upper']:.2f}")
            print(f"   BB Lower     : ${latest_ld['bb_lower']:.2f}")
            print(f"   Vol(10)      : ±${latest_ld['sma_volatility_10']:.2f}")
            print(f"   Next High    : ${hl_window_ld['next_high']:.2f}")
            print(f"   Next Low     : ${hl_window_ld['next_low']:.2f}")
            print(f"   Window Δ     : {hl_window_ld['window_pct_diff']:.3f}%")
            print(f"   Total Long (DPO) Signals In Period: {len(long_dpo_rule)}")
        else:
            print("   ⚠  No Long (DPO) Signal: DI+ dominant, DPO > 0, ADX ≥ 25, and MACD Hist > 0 not simultaneously met.")
    except ValueError as e:
        print(f"   ⚠  Signal identification error: {e}")

    # --- Long Signal (LRS Variant): DI+ > DI-, LR Slope > 0, ADX >= 25, MACD Hist > 0  [inverse of SHORT LRS] ---
    print("\n📡 LONG Signal — LRS Divergence (DI+ > DI- AND LR Slope > 0 AND ADX ≥ 25 AND MACD Hist > 0):")
    try:
        long_lrs_rule = identify_long_signal_lrs(df, adx_threshold=25, lr_period=30)
        if not long_lrs_rule.empty:
            latest_ll = long_lrs_rule.iloc[-1]
            hl_window_ll = calculate_next_hl_window(latest_ll)
            print(f"   ✅ LONG (LRS) Signal Detected!")
            print(f"   Entry Price  : ${latest_ll['price']:.2f}")
            print(f"   Timestamp    : {format_display_time(latest_ll['date'])}")
            print(f"   LR Slope(30) : {latest_ll['lr_slope_30']:.6f}  (> 0 ✓)")
            print(f"   DI+          : {latest_ll['di_plus']:.2f}  > DI- {latest_ll['di_minus']:.2f}  (Dominant ✓)")
            print(f"   ADX          : {latest_ll['adx']:.2f}  (≥ 25 ✓)")
            print(f"   MACD Hist    : {latest_ll['macd_histogram']:.4f}  (> 0 ✓)  MACD {latest_ll['macd']:.4f} / Sig {latest_ll['macd_signal']:.4f}")
            print(f"   DPO          : {latest_ll['dpo']:.4f}  (not required — MACD fills momentum gap)")
            print(f"   ── Next High/Low Window (BB + Volatility) ──────────────")
            print(f"   BB %B        : {hl_window_ll['bb_pct']:.1f}%  (position within bands)")
            print(f"   BB Upper     : ${latest_ll['bb_upper']:.2f}")
            print(f"   BB Lower     : ${latest_ll['bb_lower']:.2f}")
            print(f"   Vol(10)      : ±${latest_ll['sma_volatility_10']:.2f}")
            print(f"   Next High    : ${hl_window_ll['next_high']:.2f}")
            print(f"   Next Low     : ${hl_window_ll['next_low']:.2f}")
            print(f"   Window Δ     : {hl_window_ll['window_pct_diff']:.3f}%")
            print(f"   Total Long (LRS) Signals In Period: {len(long_lrs_rule)}")
        else:
            print("   ⚠  No Long (LRS) Signal: DI+ dominant, LR Slope > 0, ADX ≥ 25, and MACD Hist > 0 not simultaneously met.")
    except ValueError as e:
        print(f"   ⚠  Signal identification error: {e}")

    # Select features (exclude non-predictive columns)
    feature_cols = [col for col in df.columns if col not in ['date', 'price', 'high', 'low', 'future_price', 'next_return', 'target', 'signal', 'bb_pct']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"🎯 Using {len(feature_cols)} features for prediction")
    
    # Class distribution
    class_dist = y.value_counts().sort_index()
    print(f"\n📈 Class Distribution:")
    print(f"   Decrease ≥0.2%: {class_dist.get(-1, 0)} ({class_dist.get(-1, 0)/len(y)*100:.1f}%)")
    print(f"   No Change: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
    print(f"   Increase ≥0.2%: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
    
    # Time series split
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
                              target_names=["Decrease ≥0.2%", "No Change", "Increase ≥0.2%"],
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
    test_df['future_price'] = test_df['future_price']  # Already present

    # Find high-confidence long and short signals
    long_signals = test_df[(test_df['predicted'] == 1) & (test_df['confidence'] >= 0.7)]
    short_signals = test_df[(test_df['predicted'] == -1) & (test_df['confidence'] >= 0.7)]

    # Get the best long and short signal by confidence
    best_long = long_signals.sort_values('confidence', ascending=False).head(1)
    best_short = short_signals.sort_values('confidence', ascending=False).head(1)

    # Decide which to take: long or short
    if not best_long.empty and not best_short.empty:
        # Compare confidence, then expected return
        if best_long['confidence'].values[0] >= best_short['confidence'].values[0]:
            best_trade = best_long
            direction = "LONG (Buy)"
        else:
            best_trade = best_short
            direction = "SHORT (Sell)"
    elif not best_long.empty:
        best_trade = best_long
        direction = "LONG (Buy)"
    elif not best_short.empty:
        best_trade = best_short
        direction = "SHORT (Sell)"
    if best_long.empty and best_short.empty:
        print("⚠️ No high-confidence trade signals found.")
        display_forecast_windows(raw_df, horizon_hours=12)
    else:
        # Output target price and direction only when a trade signal exists.
        row = best_trade.iloc[0]
        print(f"\n🚩 Target Trade Signal:")
        print(f"   Direction: {direction}")
        print(f"   Entry Price: ${row['price']:.2f}")
        print(f"   Target Price (next interval): ${row['future_price']:.2f}")
        print(f"   Confidence: {row['confidence']:.1%}")
        print(f"   Timestamp: {format_display_time(row['date'])}")
        print(f"   Expected Return: {(row['future_price'] - row['price']) / row['price'] * 100:.2f}%")

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
