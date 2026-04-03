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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
    from sklearn.pipeline import Pipeline
except Exception:
    RandomForestClassifier = GradientBoostingClassifier = VotingClassifier = None
    train_test_split = cross_val_score = None
    classification_report = confusion_matrix = accuracy_score = None
    LogisticRegression = SVC = None
    StandardScaler = MinMaxScaler = RobustScaler = PowerTransformer = QuantileTransformer = None
    Pipeline = None

# ---------------------------------------------------------------------------
# CUDA / cuML acceleration — auto-detected on WSL with an NVIDIA GPU.
# cuML provides GPU-accelerated drop-in replacements for sklearn estimators.
# Falls back silently to sklearn CPU paths if CUDA / cuML is unavailable.
# WSL requirement: NVIDIA driver ≥ 525 + CUDA toolkit visible inside WSL.
# ---------------------------------------------------------------------------
CUDA_AVAILABLE = False
_gpu_count = 0
_cuml_RF = None
_cuml_LR = None
_cuml_StandardScaler = None
_cuml_MinMaxScaler = None
_cuml_RobustScaler = None
try:
    import cuml  # noqa: F401 — RAPIDS cuML
    from cuml.ensemble import RandomForestClassifier as _cuRF
    from cuml.linear_model import LogisticRegression as _cuLR
    import cuml.preprocessing as _cuml_prep
    import cupy as cp  # CUDA array library bundled with RAPIDS

    _gpu_count = cp.cuda.runtime.getDeviceCount()
    if _gpu_count > 0:
        CUDA_AVAILABLE = True
        # Return numpy arrays from all cuML ops so sklearn utilities remain compatible
        cuml.set_global_output_type('numpy')
        _cuml_RF = _cuRF
        _cuml_LR = _cuLR
        _cuml_StandardScaler = _cuml_prep.StandardScaler
        _cuml_MinMaxScaler = _cuml_prep.MinMaxScaler
        _cuml_RobustScaler = _cuml_prep.RobustScaler
except Exception:
    pass  # No CUDA / cuML available — silently continue with CPU sklearn

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
    Fails hard on any error or if 'requests' is unavailable.
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
        raise RuntimeError(f"Coinbase API fetch failed: {e}") from e

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


class SignalVotingEngine:
    """
    Multi-framework voting engine that aggregates three independent economic and
    game-theory lenses to produce a conviction-weighted final trade direction.

    Frameworks
    ----------
    Adam Smith (1776) — The Wealth of Nations
        The invisible hand drives prices toward their natural (equilibrium) level
        through self-interested supply and demand. When market price deviates above
        natural price demand is dominant (LONG); below, supply dominates (SHORT).
        Modelled via price-vs-SMA, BB position, and RSI heat.

    Bruce Bueno de Mesquita — Expected Utility (EU) Game Theory
        Stakeholders (bulls, bears) each have Capability (di_plus / di_minus),
        Salience (ADX, how much the trend matters), and a Position (reinforced by
        DPO and MACD histogram sign). EU_bull vs EU_bear determines the net
        expected utility: positive → LONG, negative → SHORT.

    John Nash (Nobel 1994) — Nash Equilibrium & Dominant Strategy
        A dominant strategy is one that is optimal regardless of what the opposing
        players do. When a clear majority of independent momentum indicators
        (DPO, LR Slope, MACD histogram, BB position) agree on a direction, a
        dominant strategy exists. When indicators are split, the market is in a
        mixed-strategy Nash equilibrium — no trade (abstain).
    """

    ADX_MIN = 25  # Minimum ADX for BdM to vote
    DOMINANT_THRESHOLD = 0.75  # Fraction of indicators that must agree for Nash dominant strategy
    VOTE_DEADBAND = 0.12  # Score must exceed this magnitude for a directional final vote

    # --- Framework 1: Adam Smith ---
    def _adam_smith_vote(self, row):
        """Invisible hand: market price vs natural price via supply/demand balance."""
        score = 0.0
        price = row.get('price', 1.0)

        # Supply/demand: price vs short and medium-term natural price (SMA)
        for sma_col, weight in [('sma_30', 0.20), ('sma_120', 0.20)]:
            sma_val = row.get(sma_col, None)
            if sma_val is not None and sma_val > 0:
                score += weight if price > sma_val else -weight

        # Bollinger Band position: 0=lower band, 1=upper band; 0.5=natural price
        bb_pos = row.get('bb_position', 0.5)
        score += (bb_pos - 0.5) * 0.80  # +0.4 max long, -0.4 max short

        # RSI: overbought/oversold as self-correcting market signal
        rsi = row.get('rsi_14', 50.0)
        if 30 <= rsi <= 70:
            score += 0.10 if rsi > 50 else -0.10  # Trend continuation in healthy range
        elif rsi > 70:
            score -= 0.25  # Overbought — Smith: market will self-correct downward
        elif rsi < 30:
            score += 0.25  # Oversold — Smith: market will self-correct upward

        direction = 1 if score > self.VOTE_DEADBAND else (-1 if score < -self.VOTE_DEADBAND else 0)
        conviction = min(abs(score), 1.0)
        return direction, conviction, f"score={score:+.3f}  price/SMA={'above' if price > row.get('sma_30', price) else 'below'}  BB%={bb_pos*100:.1f}  RSI={rsi:.1f}"

    # --- Framework 2: Bruce Bueno de Mesquita Expected Utility ---
    def _bdm_vote(self, row):
        """EU model: capability × salience × position for bulls vs bears."""
        adx = row.get('adx', 0.0)
        di_plus = row.get('di_plus', 0.0)
        di_minus = row.get('di_minus', 0.0)
        dpo = row.get('dpo', 0.0)
        macd_hist = row.get('macd_histogram', 0.0)

        if adx < self.ADX_MIN:
            return 0, 0.0, f"ADX={adx:.1f} below threshold — abstain (trend lacks conviction)"

        # Salience: how much does the trend strength matter right now?
        salience = min(adx / 100.0, 1.0)

        # Capability-position product for each stakeholder group
        # Partial credit (0.4) when a confirming indicator is absent to model uncertainty
        bull_eu = (
            (di_plus / 100.0)
            * salience
            * (1.0 if dpo > 0 else 0.4)
            * (1.0 if macd_hist > 0 else 0.4)
        )
        bear_eu = (
            (di_minus / 100.0)
            * salience
            * (1.0 if dpo < 0 else 0.4)
            * (1.0 if macd_hist < 0 else 0.4)
        )

        net_eu = bull_eu - bear_eu
        total_eu = bull_eu + bear_eu + 1e-8
        conviction = min(abs(net_eu) / total_eu, 1.0)

        direction = 1 if net_eu > 0 else (-1 if net_eu < 0 else 0)
        return direction, conviction, (
            f"EU_bull={bull_eu:.4f}  EU_bear={bear_eu:.4f}  net={net_eu:+.4f}  "
            f"ADX={adx:.1f}  DI+={di_plus:.1f}  DI-={di_minus:.1f}"
        )

    # --- Framework 3: John Nash Dominant Strategy ---
    def _nash_vote(self, row, lr_period=30):
        """Dominant strategy theory: unanimous indicator agreement → exploitable signal.
        Near Nash equilibrium (all indicators near zero) → mixed strategy → abstain.
        """
        price = row.get('price', 1.0) or 1.0
        lr_col = f'lr_slope_{lr_period}'

        # Collect independent directional votes from momentum/position indicators
        indicator_votes = {}
        for col, label in [
            ('dpo', 'DPO'),
            (lr_col, 'LRS'),
            ('macd_histogram', 'MACD'),
        ]:
            val = row.get(col, None)
            if val is not None:
                indicator_votes[label] = 1 if val > 0 else -1

        bb_pos = row.get('bb_position', None)
        if bb_pos is not None:
            indicator_votes['BB'] = 1 if bb_pos > 0.5 else -1

        if not indicator_votes:
            return 0, 0.0, 'Nash: no indicators available'

        votes = list(indicator_votes.values())
        total = len(votes)
        bull_count = sum(1 for v in votes if v == 1)
        bear_count = total - bull_count
        agreement_ratio = max(bull_count, bear_count) / total

        # Check for near-equilibrium (Nash NE): small DPO and MACD relative to price
        dpo_pct = abs(row.get('dpo', 1.0)) / price
        macd_pct = abs(row.get('macd_histogram', 1.0)) / price
        near_eq = dpo_pct < 0.001 and macd_pct < 0.0001

        if near_eq or agreement_ratio < self.DOMINANT_THRESHOLD:
            summary = '  '.join(f"{k}={'↑' if v==1 else '↓'}" for k, v in indicator_votes.items())
            reason = ('near equilibrium' if near_eq else f'mixed strategy (agree={agreement_ratio:.0%})')
            return 0, 0.0, f'Nash: {reason}  [{summary}]'

        direction = 1 if bull_count > bear_count else -1
        conviction = agreement_ratio
        summary = '  '.join(f"{k}={'↑' if v==1 else '↓'}" for k, v in indicator_votes.items())
        return direction, conviction, f'Nash: dominant strategy (agree={agreement_ratio:.0%})  [{summary}]'

    # --- Aggregation ---
    def evaluate(self, row, lr_period=30):
        """Aggregate all three framework votes into a final conviction-weighted signal.

        Returns a dict with keys:
            direction       int   +1 LONG / -1 SHORT / 0 NEUTRAL
            recommendation  str   'LONG' / 'SHORT' / 'NEUTRAL'
            weighted_score  float conviction-weighted net score
            conviction      float 0.0 – 1.0
            frameworks      list  [(name, direction, conviction, reason), ...]
        """
        smith_dir, smith_conv, smith_reason = self._adam_smith_vote(row)
        bdm_dir,   bdm_conv,   bdm_reason   = self._bdm_vote(row)
        nash_dir,  nash_conv,  nash_reason  = self._nash_vote(row, lr_period)

        frameworks = [
            ('Adam Smith',          smith_dir, smith_conv, smith_reason),
            ('BdM Exp. Utility',    bdm_dir,   bdm_conv,   bdm_reason),
            ('Nash Equilibrium',    nash_dir,  nash_conv,  nash_reason),
        ]

        # Conviction-weighted vote aggregation
        weighted_sum = sum(d * c for _, d, c, _ in frameworks if d != 0)
        total_weight = sum(c for _, d, c, _ in frameworks if d != 0)

        if total_weight < 1e-8:
            final_dir, final_conv, raw_score = 0, 0.0, 0.0
        else:
            raw_score = weighted_sum / total_weight
            final_conv = min(abs(raw_score), 1.0)
            final_dir = 1 if raw_score > self.VOTE_DEADBAND else (-1 if raw_score < -self.VOTE_DEADBAND else 0)

        return {
            'direction':      final_dir,
            'recommendation': 'LONG' if final_dir == 1 else ('SHORT' if final_dir == -1 else 'NEUTRAL'),
            'weighted_score': raw_score,
            'conviction':     final_conv,
            'frameworks':     frameworks,
        }

    def evaluate_latest(self, df, lr_period=30):
        """Evaluate the most recent row of the feature DataFrame."""
        return self.evaluate(df.iloc[-1], lr_period=lr_period)


# ---------------------------------------------------------------------------
# GPU-aware factory helpers
# Each helper transparently selects the cuML GPU implementation when CUDA is
# available, stripping parameters that cuML does not support, then falls back
# to the standard sklearn implementation on CPU.
# ---------------------------------------------------------------------------

def _gpu_rf(**kwargs):
    """Return a RandomForestClassifier, preferring cuML when CUDA is available.

    cuML RF does not accept `n_jobs` (GPU parallelism is implicit).
    All other sklearn-compatible kwargs are forwarded as-is.
    """
    if CUDA_AVAILABLE and _cuml_RF is not None:
        gpu_kw = {k: v for k, v in kwargs.items() if k != 'n_jobs'}
        return _cuml_RF(**gpu_kw)
    return RandomForestClassifier(**kwargs)


def _gpu_lr(**kwargs):
    """Return a LogisticRegression, preferring cuML when CUDA is available.

    cuML LR does not accept `class_weight` or `random_state`.
    All other sklearn-compatible kwargs (e.g. max_iter, C) are forwarded.
    """
    if CUDA_AVAILABLE and _cuml_LR is not None:
        gpu_kw = {k: v for k, v in kwargs.items() if k not in ('class_weight', 'random_state')}
        return _cuml_LR(**gpu_kw)
    return LogisticRegression(**kwargs)


def _gpu_scalers():
    """Build the five-scaler dict, substituting cuML GPU scalers where supported.

    PowerTransformer and QuantileTransformer have no cuML equivalent and always
    run on CPU via sklearn regardless of CUDA availability.
    """
    if CUDA_AVAILABLE and _cuml_StandardScaler is not None:
        return {
            'StandardScaler':     _cuml_StandardScaler(),
            'MinMaxScaler':       _cuml_MinMaxScaler(),
            'RobustScaler':       _cuml_RobustScaler(),
            'PowerTransformer':   PowerTransformer(method='yeo-johnson'),   # CPU only
            'QuantileTransformer': QuantileTransformer(output_distribution='normal'),  # CPU only
        }
    return {
        'StandardScaler':     StandardScaler(),
        'MinMaxScaler':       MinMaxScaler(),
        'RobustScaler':       RobustScaler(),
        'PowerTransformer':   PowerTransformer(method='yeo-johnson'),
        'QuantileTransformer': QuantileTransformer(output_distribution='normal'),
    }


def _cpu_scaler_by_name(name):
    """Return the sklearn CPU scaler instance for a given scaler label."""
    if name == 'StandardScaler':
        return StandardScaler()
    if name == 'MinMaxScaler':
        return MinMaxScaler()
    if name == 'RobustScaler':
        return RobustScaler()
    if name == 'PowerTransformer':
        return PowerTransformer(method='yeo-johnson')
    if name == 'QuantileTransformer':
        return QuantileTransformer(output_distribution='normal')
    raise ValueError(f"Unknown scaler name: {name}")


class ScalerEvaluator:
    """Evaluates 5 different scalers and ranks them by cross-validation performance."""
    
    @staticmethod
    def evaluate(X_train, y_train, cv=3):
        """Rank scalers by average cross-validation accuracy.
        
        Returns:
            list: [(scaler_name, avg_cv_score, scaler_object), ...] sorted descending by score
        """
        from sklearn.model_selection import cross_val_score
        
        results = []
        # _gpu_scalers() returns cuML implementations when CUDA is available
        scalers = _gpu_scalers()
        for name, scaler in scalers.items():
            try:
                X_scaled = scaler.fit_transform(X_train)
                rf = _gpu_rf(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                scores = cross_val_score(rf, X_scaled, y_train, cv=cv, scoring='accuracy')
                avg_score = scores.mean()
                results.append((name, avg_score, scaler))
            except Exception as e:
                print(f"    ⚠  {name} GPU-path eval failed: {e}. Retrying on CPU scaler.")
                try:
                    cpu_scaler = _cpu_scaler_by_name(name)
                    X_scaled_cpu = cpu_scaler.fit_transform(X_train)
                    rf_cpu = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                    scores_cpu = cross_val_score(rf_cpu, X_scaled_cpu, y_train, cv=cv, scoring='accuracy')
                    avg_score_cpu = scores_cpu.mean()
                    results.append((name, avg_score_cpu, cpu_scaler))
                except Exception as e_cpu:
                    print(f"    ⚠  {name} CPU fallback eval failed: {e_cpu}")
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class EnsembleConfidenceBooster:
    """Multi-scaler ensemble to verify and boost voting engine confidence.
    
    Uses the top 3 scalers, each paired with RF, GB, LR for a 3×3 parallel ensemble.
    Each model votes on direction; majority vote + average confidence boost the signal.
    """
    
    def __init__(self, scalers_list, X_train, y_train, random_state=42):
        """Initialize with top 3 (scaler, X_scaled) pairs and train models.
        
        Args:
            scalers_list: list of (scaler_name, avg_cv_score, scaler_object) tuples (top 3)
            X_train: training features
            y_train: training labels
        """
        self.models = {}
        
        for scaler_name, _, scaler in scalers_list[:3]:
            try:
                X_scaled = scaler.fit_transform(X_train)
            except Exception as e:
                print(f"    ⚠  {scaler_name} scaler failed in booster: {e}. Falling back to CPU scaler.")
                scaler = _cpu_scaler_by_name(scaler_name)
                X_scaled = scaler.fit_transform(X_train)
            self.models[scaler_name] = {}
            
            # Random Forest
            # _gpu_rf strips n_jobs (implicit on GPU); all other kwargs forwarded
            try:
                rf = _gpu_rf(
                    n_estimators=100, max_depth=12, min_samples_split=5,
                    class_weight='balanced', random_state=random_state, n_jobs=-1
                )
            except Exception:
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=12, min_samples_split=5,
                    class_weight='balanced', random_state=random_state, n_jobs=-1
                )
            rf.fit(X_scaled, y_train)
            self.models[scaler_name]['rf'] = (rf, scaler)
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=80, learning_rate=0.1, max_depth=6,
                random_state=random_state
            )
            gb.fit(X_scaled, y_train)
            self.models[scaler_name]['gb'] = (gb, scaler)
            
            # Logistic Regression — _gpu_lr strips class_weight/random_state for cuML
            try:
                lr = _gpu_lr(
                    class_weight='balanced', random_state=random_state, max_iter=1000
                )
            except Exception:
                lr = LogisticRegression(
                    class_weight='balanced', random_state=random_state, max_iter=1000
                )
            lr.fit(X_scaled, y_train)
            self.models[scaler_name]['lr'] = (lr, scaler)
    
    def predict_with_boost(self, X_test_latest):
        """Predict on single row, returning (direction, boosted_confidence).
        
        Args:
            X_test_latest: single row feature vector (1D array or Series)
        
        Returns:
            (direction, boosted_confidence, breakdown_str)
        """
        X_row = X_test_latest.values.reshape(1, -1) if hasattr(X_test_latest, 'values') else X_test_latest.reshape(1, -1)
        
        votes = []
        confidences = []
        breakdown = []
        
        for scaler_name in sorted(self.models.keys()):
            for model_type in ['rf', 'gb', 'lr']:
                model, scaler = self.models[scaler_name][model_type]
                X_scaled = scaler.transform(X_row)
                pred = model.predict(X_scaled)[0]
                votes.append(pred)
                
                # Get prediction probability/confidence
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    conf = np.max(proba)
                else:
                    conf = 0.5  # LR always has proba, but fallback just in case
                confidences.append(conf)
                breakdown.append(f"{scaler_name[:8]}_{model_type.upper()}={pred:+d}@{conf:.2f}")
        
        # Voting: majority direction
        vote_sum = sum(votes)
        final_direction = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else 0)
        
        # Confidence boost: average confidences, weighted by vote agreement
        avg_conf = np.mean(confidences)
        vote_agreement = abs(vote_sum) / len(votes)  # How unanimous?
        boosted_conf = avg_conf * vote_agreement
        
        breakdown_str = ' | '.join(breakdown)
        return final_direction, boosted_conf, breakdown_str


def fuse_voter_and_ensemble(vote, boost_dir, boost_conf):
    """Fuse rule/game-theory voter output with ensemble verification.

    Decision policy:
    - Strong agreement with meaningful confidence -> ACTION.
    - Disagreement -> HOLD to avoid conflicting regime assumptions.
    - Low confidence from either side -> HOLD.
    """
    voter_dir = vote.get('direction', 0)
    voter_conf = vote.get('conviction', 0.0)

    min_voter_conf = 0.35
    min_ens_conf = 0.25

    if voter_dir == 0 or boost_dir == 0:
        return {
            'action': 'HOLD',
            'reason': 'One side is neutral',
            'score': 0.0,
            'direction': 0,
        }

    if voter_dir != boost_dir:
        return {
            'action': 'HOLD',
            'reason': 'Voter and ensemble disagree',
            'score': -abs(voter_conf - boost_conf),
            'direction': 0,
        }

    if voter_conf < min_voter_conf or boost_conf < min_ens_conf:
        return {
            'action': 'HOLD',
            'reason': 'Agreement exists but confidence is too low',
            'score': (voter_conf + boost_conf) / 2,
            'direction': 0,
        }

    fused_conf = (0.6 * voter_conf) + (0.4 * boost_conf)
    action = 'BUY/LONG' if voter_dir == 1 else 'SELL/SHORT'
    return {
        'action': action,
        'reason': 'Voter-first signal verified by ensemble',
        'score': fused_conf,
        'direction': voter_dir,
    }


def main():
    """Enhanced Bitcoin prediction using scikit-learn ensemble on 1-minute data."""
    print("🚀 Enhanced Bitcoin 1-Minute Interval Prediction with Scikit-Learn")
    print("=" * 70)
    if CUDA_AVAILABLE:
        print(f"⚡ Backend: CUDA/cuML active ({_gpu_count} GPU(s) detected)")
    else:
        print("🧰 Backend: CPU fallback active (CUDA/cuML unavailable)")
    
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

    # --- Economic & Game Theory Voting Engine ---
    print("\n🏙  Economic & Game Theory Signal Voting Engine")
    print("   Adam Smith (Supply/Demand)  ·  BdM Expected Utility  ·  Nash Equilibrium")
    print("   " + "─" * 62)
    try:
        engine = SignalVotingEngine()
        vote = engine.evaluate_latest(df, lr_period=30)
        for name, direction, conviction, reason in vote['frameworks']:
            if direction == 1:
                arrow, colour = "↑ LONG ", "✅"
            elif direction == -1:
                arrow, colour = "↓ SHORT", "🔴"
            else:
                arrow, colour = "○ NEUT ", "⚪"
            print(f"   {colour} {name:<22} {arrow}  conviction={conviction:.2f}")
            print(f"       └ {reason}")
        print("   " + "─" * 62)
        rec   = vote['recommendation']
        score = vote['weighted_score']
        conv  = vote['conviction']
        rec_icon = "✅" if rec == 'LONG' else ("🔴" if rec == 'SHORT' else "⚪")
        print(f"   {rec_icon} FINAL VERDICT  : {rec}")
        print(f"      Weighted Score : {score:+.4f}")
        print(f"      Conviction     : {conv:.2%}")
    except Exception as e:
        print(f"   ⚠  Voting engine error: {e}")

    # Select features (exclude non-predictive columns)
    feature_cols = [col for col in df.columns if col not in ['date', 'price', 'high', 'low', 'future_price', 'next_return', 'target', 'signal', 'bb_pct', 'macd', 'macd_signal']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"\n🎯 Using {len(feature_cols)} features for prediction")
    
    # Class distribution
    class_dist = y.value_counts().sort_index()
    print(f"\n📈 Class Distribution:")
    print(f"   Decrease ≥0.2%: {class_dist.get(-1, 0)} ({class_dist.get(-1, 0)/len(y)*100:.1f}%)")
    print(f"   No Change: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
    print(f"   Increase ≥0.2%: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
    
    # Time series split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # --- Scaler Evaluation & Multi-Scaler Ensemble Confidence Booster ---
    print("\n🔍 Evaluating 5 scalers for optimal feature preprocessing...")
    scaler_results = ScalerEvaluator.evaluate(X_train, y_train, cv=3)
    print(f"   Top 3 Scalers (by CV accuracy):")
    for i, (name, score, _) in enumerate(scaler_results[:3], 1):
        print(f"   {i}. {name:<25} CV Accuracy: {score:.4f}")
    
    print(f"\n🤝 Multi-Scaler Ensemble Confidence Booster (3 scalers × 3 models = 9 voters)")
    booster = EnsembleConfidenceBooster(scaler_results, X_train, y_train)
    
    # Get latest test row for ensemble prediction
    latest_test_row = X_test.iloc[-1] if len(X_test) > 0 else X_test.iloc[-1]
    boost_dir, boost_conf, boost_breakdown = booster.predict_with_boost(latest_test_row)
    
    print(f"   Ensemble Verdict: {'+1 LONG' if boost_dir==1 else ('-1 SHORT' if boost_dir==-1 else '0 NEUTRAL')}")
    print(f"   Boosted Confidence: {boost_conf:.2%}")
    print(f"   Breakdown: {boost_breakdown}")
    
    # Synergy check: does ensemble agree with voting engine?
    voting_rec = vote['recommendation']
    ensemble_rec = 'LONG' if boost_dir == 1 else ('SHORT' if boost_dir == -1 else 'NEUTRAL')
    synergy = "✅ STRONG" if voting_rec == ensemble_rec else "⚠  DIVERGENT"
    print(f"   Voting Engine vs Ensemble: {synergy}  (VE={voting_rec}, Ens={ensemble_rec})")
    
    # Final decision: voter-first, ensemble-verified
    fused = fuse_voter_and_ensemble(vote, boost_dir, boost_conf)
    print(f"   Final Action: {fused['action']}")
    print(f"   Fusion Score: {fused['score']:.2%}")
    print(f"   Decision Rationale: {fused['reason']}")
    print()
    
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
