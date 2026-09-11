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
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
    from sklearn.pipeline import Pipeline
except Exception:
    RandomForestClassifier = GradientBoostingClassifier = VotingClassifier = None
    train_test_split = cross_val_score = None
    classification_report = confusion_matrix = accuracy_score = None
    LogisticRegression = SGDClassifier = SVC = None
    StandardScaler = None
    Pipeline = None
try:
    from scipy import stats
except Exception:
    stats = None
try:
    # imbalanced-learn for resampling inside CV pipelines
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    RandomOverSampler = None
    ImbPipeline = None
# matplotlib is not required for core functionality; avoid importing to reduce dependency surface
try:
    import requests
except Exception:
    requests = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import torch
except Exception:
    torch = None
try:
    from numba import cuda as numba_cuda
except Exception:
    numba_cuda = None
from datetime import datetime, timedelta, timezone
import os
import time
import logging


def configure_logging():
    level = os.getenv('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("enhanced_prediction")


logger = configure_logging()


def detect_cuda_support():
    """Return True if CUDA appears available via torch, numba, or cupy."""
    if os.getenv('CUDA_DISABLED', '').lower() in ['1', 'true', 'yes']:
        return False

    if torch is not None:
        try:
            if torch.cuda.is_available():
                return True
        except Exception:
            pass

    if numba_cuda is not None:
        try:
            if numba_cuda.is_available():
                return True
        except Exception:
            pass

    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return True
    except Exception:
        pass

    return False


def resolve_product_id():
    """Return the Coinbase product identifier to query."""
    explicit = os.getenv('MARKET_SYMBOL') or os.getenv('PRODUCT_ID')
    if explicit:
        return explicit.strip().upper()

    base = (os.getenv('BASE_SYMBOL') or 'BTC').strip().upper()
    quote = (os.getenv('DEFAULT_QUOTE') or 'USD').strip().upper()
    return f"{base}-{quote}" if quote else base

def create_enhanced_features(df, pct_threshold=0.002, future_steps=1):
    """Create comprehensive technical indicators for minute-level price prediction.
    Requires numpy, pandas, and scipy.stats. Raises ImportError if unavailable.
    """
    if pd is None or np is None or stats is None:
        raise ImportError("create_enhanced_features requires numpy, pandas, and scipy to be installed.")

    logger.info("Starting feature engineering rows=%s pct_threshold=%s", len(df), pct_threshold)
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
    # Bollinger Band Percent (0..1 within bands)
    df['bbp'] = (df['price'] - df['bb_lower']) / (bb_range + 1e-8)
    # Maintain original normalized position name for backward-compat
    df['bb_position'] = df['bbp']
    # Heuristic flags near bands
    df['bbp_low'] = (df['bbp'] <= 0.2).astype(int)
    df['bbp_high'] = (df['bbp'] >= 0.8).astype(int)
    
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
    
    # Classification target: allow predicting N steps ahead (future_steps intervals)
    df['future_price'] = df['price'].shift(-future_steps)
    df['next_return'] = (df['future_price'] - df['price']) / (df['price'] + 1e-8)
    df['target'] = 0
    df.loc[df['next_return'] >= pct_threshold, 'target'] = 1
    df.loc[df['next_return'] <= -pct_threshold, 'target'] = -1
    
    # Replace inf and nan values
    df = df.replace([np.inf, -np.inf], np.nan)
    # Recompute next_return using the same future_steps and reassess target
    df['next_return'] = (df['future_price'] - df['price']) / df['price']
    df['target'] = 0
    df.loc[df['next_return'] >= pct_threshold, 'target'] = 1
    df.loc[df['next_return'] <= -pct_threshold, 'target'] = -1
    
    engineered = df.dropna()
    logger.info("Feature engineering complete rows=%s cols=%s", len(engineered), len(engineered.columns))
    return engineered

def fetch_product_data_rest(product_id, num_points=36000, interval_minutes=1):
    """Fetch Coinbase REST OHLC data for the given product_id.
    Falls back to simulated data on any error or if 'requests' is unavailable.
    Requires pandas and numpy when returning data.
    """
    if pd is None or np is None:
        raise ImportError("fetch_product_data_rest requires numpy and pandas to be installed.")

    logger.info("REST fetch start product=%s points=%s interval=%sm", product_id, num_points, interval_minutes)
    all_data = []
    points_per_request = 150  # smaller pages reduce boundary overlap
    num_requests = (num_points + points_per_request - 1) // points_per_request
    max_pages_env = os.getenv('REST_MAX_PAGES')
    max_pages = int(max_pages_env) if max_pages_env else None
    if max_pages is not None and num_requests > max_pages:
        logger.info("REST fetch capped to %s pages via REST_MAX_PAGES", max_pages)
        num_requests = max_pages
    granularity = interval_minutes * 60

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    now = datetime.utcnow().replace(tzinfo=timezone.utc, microsecond=0)

    def align_dt(dt: datetime) -> datetime:
        epoch = int(dt.timestamp())
        floored = epoch - (epoch % granularity)
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    end_time = align_dt(now - timedelta(seconds=granularity))

    try:
        if requests is None:
            raise RuntimeError("'requests' library not available")
        prev_oldest_ts = None
        for i in range(num_requests):
            stride = timedelta(seconds=points_per_request * granularity)
            start_time = align_dt(end_time - stride)

            if start_time >= end_time:
                break

            params = {
                "granularity": granularity,
                "start": start_time.isoformat().replace('+00:00','Z'),
                "end": end_time.isoformat().replace('+00:00','Z')
            }

            logger.info("REST page %s/%s window=%s->%s stride=%s", i+1, num_requests, start_time, end_time, points_per_request)

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning("REST page %s empty, advancing window back further", i+1)
                end_time = start_time - timedelta(seconds=granularity)
                continue

            all_data.extend(data)

            oldest_timestamp = data[-1][0]
            if prev_oldest_ts is not None and oldest_timestamp >= prev_oldest_ts:
                logger.warning("REST pagination safety hit at %s; stopping fetch", datetime.fromtimestamp(oldest_timestamp, tz=timezone.utc))
                break
            prev_oldest_ts = oldest_timestamp

            end_time = start_time - timedelta(seconds=granularity)

            time.sleep(0.3)

        if not all_data:
            raise ValueError("No data received from Coinbase API")

        df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
        df = df.sort_values("time", ascending=False).drop_duplicates(subset=['time']).sort_values("time")
        
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["price"] = pd.to_numeric(df["close"], errors='coerce')
        
        logger.info("REST fetch completed rows=%s max_date=%s product=%s", len(df), df['date'].max(), product_id)
        print(f"data_source=coinbase_rest product={product_id}")
        return df[["date", "price"]].dropna()
        
    except Exception as e:
        allow_sim = os.getenv('ALLOW_SIMULATION', 'true').lower() not in ['0', 'false', 'no']
        if not allow_sim:
            raise
        logger.exception("REST fetch failed, switching to simulated data for %s", product_id)
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=num_points), periods=num_points, freq=f'{interval_minutes}T')
        prices = np.cumsum(np.random.randn(num_points) * 2) + 60000
        print(f"data_source=simulated product={product_id}")
        return pd.DataFrame({'date': dates, 'price': prices})

def fetch_product_data_cdp(product_id, num_points=36000, interval_minutes=1):
    """Fetch historical Coinbase Advanced Trade data for product_id via coinbase-advanced-py REST client.
    Requires COINBASE_API_KEY/COINBASE_API_SECRET in the environment.
    """
    if pd is None:
        raise RuntimeError("pandas required for CDP SDK fetch")
    try:
        from coinbase.rest import RESTClient
    except Exception as e:
        raise RuntimeError("coinbase-advanced-py not installed") from e

    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    if not api_key or not api_secret:
        raise RuntimeError("CDP credentials not set in env")

    client = RESTClient(api_key=api_key, api_secret=api_secret)
    # Map minutes to SDK enum granularity string and seconds for alignment
    def map_granularity(minutes: int) -> str:
        table = {
            1: "ONE_MINUTE",
            5: "FIVE_MINUTE",
            15: "FIFTEEN_MINUTE",
            60: "ONE_HOUR",
            360: "SIX_HOUR",
            1440: "ONE_DAY",
        }
        if minutes in table:
            return table[minutes]
        # pick closest supported
        choices = sorted(table.keys(), key=lambda k: abs(k - minutes))
        return table[choices[0]]

    def closest_minutes(minutes: int) -> int:
        supported = [1, 5, 15, 60, 360, 1440]
        return min(supported, key=lambda k: abs(k - minutes))

    gran_minutes = closest_minutes(int(interval_minutes))
    gran_str = map_granularity(gran_minutes)
    gran_sec = gran_minutes * 60

    # Chunked paging to avoid large-window 400s
    end_dt = datetime.utcnow().replace(tzinfo=timezone.utc, microsecond=0)
    caps = {60: 300, 300: 600, 900: 1000, 3600: 2000}
    per_chunk = caps.get(gran_sec, 300)

    # Align to bar boundary
    def align_dt(dt: datetime) -> datetime:
        epoch = int(dt.timestamp())
        floored = epoch - (epoch % gran_sec)
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    # Probe last ~10 bars to validate parameters
    probe_end = align_dt(end_dt)
    probe_start = align_dt(probe_end - timedelta(seconds=gran_sec * 10))
    logger.info("CDP probe product=%s granularity=%s start=%s end=%s", product_id, gran_str, probe_start, probe_end)
    probe = client.get_candles(
        product_id=product_id,
        granularity=gran_str,
        start=probe_start.isoformat(),
        end=probe_end.isoformat()
    )
    probe_data = probe.get('candles') if isinstance(probe, dict) else probe
    if not probe_data:
        raise ValueError('Empty CDP probe response')
    try:
        rows = []
        remaining = num_points
        chunk_end = end_dt
        max_chunks = 200
        chunks = 0
        # Start chunking from the validated probe_end
        chunk_end = probe_end
        while remaining > 0 and chunks < max_chunks:
            span_pts = min(per_chunk, remaining)
            chunk_start = align_dt(chunk_end - timedelta(seconds=gran_sec * span_pts))
            if chunk_start >= chunk_end:
                break
            logger.info("CDP chunk product=%s granularity=%s start=%s end=%s remaining=%s", product_id, gran_str, chunk_start, chunk_end, remaining)
            resp = client.get_candles(
                product_id=product_id,
                granularity=gran_str,
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat()
            )
            data = resp.get('candles') if isinstance(resp, dict) else resp
            if not data:
                break
            for c in data:
                if isinstance(c, dict):
                    t = c.get('start') or c.get('time')
                    close = c.get('close')
                else:
                    t = c[0]
                    close = c[4]
                if t is None or close is None:
                    continue
                if isinstance(t, (int, float)):
                    dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
                else:
                    dt = pd.to_datetime(t, utc=True)
                rows.append({'date': dt, 'price': float(close)})
            chunk_end = chunk_start
            remaining -= span_pts
            chunks += 1
        if not rows:
            raise ValueError('Empty CDP response across chunks')
        df = pd.DataFrame(rows).dropna().drop_duplicates(subset=['date']).sort_values('date')
        if len(df) > num_points:
            df = df.tail(num_points)
        if df.empty:
            raise ValueError('No rows after normalization')
        logger.info("CDP fetch complete product=%s rows=%s window=%s->%s", product_id, len(df), df['date'].min(), df['date'].max())
        print(f"data_source=cdp product={product_id}")
        return df[['date','price']]
    except Exception as e:
        logger.exception("CDP fetch failed for %s", product_id)
        raise

def get_training_data(product_id, num_points=36000, interval_minutes=1):
    """Unified data provider with priority: FIX -> Coinbase -> Simulated.
    If env FIX_DATA_URL is set, tries to fetch JSON time-series from there first.
    Expected FIX_DATA_URL response format (array of objects): [{"date": ISO8601, "price": number}, ...]
    """
    url = os.getenv('FIX_DATA_URL')
    if url:
        try:
            if requests is None or pd is None:
                raise RuntimeError("requests/pandas not available for FIX_DATA_URL fetch")
            logger.info("Attempting FIX provider url=%s product=%s", url, product_id)
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            js = r.json()
            if not isinstance(js, list) or not js:
                raise ValueError("FIX_DATA_URL returned empty or invalid payload")
            df = pd.DataFrame(js)
            # Normalize columns
            if 'timestamp' in df.columns and 'date' not in df.columns:
                df['date'] = df['timestamp']
            if 'close' in df.columns and 'price' not in df.columns:
                df['price'] = df['close']
            if 'time' in df.columns and 'date' not in df.columns:
                df['date'] = df['time']
            if 'price' not in df.columns or 'date' not in df.columns:
                raise ValueError("FIX_DATA_URL missing required fields 'date' and 'price'")
            df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['date','price']).sort_values('date')
            # If too long, take the last num_points at the requested granularity hint
            if len(df) > num_points:
                df = df.tail(num_points)
            logger.info("FIX provider success product=%s rows=%s max_date=%s", product_id, len(df), df['date'].max())
            print(f"data_source=fix product={product_id}")
            return df[['date','price']]
        except Exception as e:
            logger.warning("FIX provider failed (%s); falling back to CDP", e)
    # Try CDP SDK next
    try:
        return fetch_product_data_cdp(product_id, num_points=num_points, interval_minutes=interval_minutes)
    except Exception as e:
        logger.warning("CDP SDK failed (%s); falling back to REST", e)
    # Coinbase as default
    return fetch_product_data_rest(product_id, num_points=num_points, interval_minutes=interval_minutes)

def main():
    """Enhanced market prediction using scikit-learn ensemble on minute-level Coinbase data."""
    product_id = resolve_product_id()
    logger.info("Resolved product_id=%s", product_id)
    print(f"🚀 Enhanced {product_id} Prediction with Scikit-Learn")
    print("=" * 70)

    # Input bounds derived from feature windows to avoid under/over-fetching
    intervals_in_day = 24 * 60
    longest_window = intervals_in_day * 7  # 1-week rolling window in features
    buffer_rows = 500  # extra rows to survive dropna and train/test split
    min_required = longest_window + buffer_rows
    max_points_per_dataset = 50000  # cap API pull size
    max_datasets = 10  # prevent excessive paging

    # Manual entry: number of datasets
    while True:
        try:
            num_datasets = int(input("\nEnter number of datasets to fetch via REST: "))
            if not (1 <= num_datasets <= max_datasets):
                print(f"❌ Must be between 1 and {max_datasets}. Try again.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Enter a number.")

    # Manual entry: data points per dataset
    while True:
        try:
            points_env = int(input("Enter data points per dataset: "))
            minimum_per_set = longest_window // max(1, num_datasets)
            if not (minimum_per_set <= points_env <= max_points_per_dataset):
                print(f"❌ Must be ≥{minimum_per_set} (supports feature windows) and ≤{max_points_per_dataset}. Try again.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Enter a number.")

    # Manual entry: interval in minutes
    while True:
        try:
            interval_env = int(input("Enter interval in minutes: "))
            if interval_env <= 0 or interval_env > 60:
                print("❌ Must be >0 and ≤60 minutes. Try again.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Enter a number.")

    total_rows = num_datasets * points_env
    if total_rows < min_required:
        print(f"⚠️ Warning: ~{min_required} rows recommended for 1-week feature windows; you requested {total_rows}.")

    logger.info("Manual entry: datasets=%s points=%s interval=%s", num_datasets, points_env, interval_env)
    print(f"\n📊 Fetching {num_datasets} dataset(s) ({points_env} points each, {interval_env}-minute intervals)")
    
    # Fetch multiple datasets and combine
    all_data = []
    for i in range(num_datasets):
        logger.info("Fetching dataset %d/%d", i+1, num_datasets)
        print(f"  ⏳ Fetching dataset {i+1}/{num_datasets}...")
        df_chunk = get_training_data(product_id, num_points=points_env, interval_minutes=interval_env)
        all_data.append(df_chunk)
    
    df = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=['date']).sort_values('date')
    logger.info("Combined %d dataset(s) into %d total rows", num_datasets, len(df))
    print(f"✅ Combined into {len(df)} total rows\n")
    
    # For testing: use 30-minute horizon and 1% threshold (0.01)
    # Compute how many steps ahead that corresponds to given interval_env
    future_minutes = 30
    future_steps = max(1, int(round(future_minutes / max(1, interval_env))))
    test_pct_threshold = 0.005
    df = create_enhanced_features(df, pct_threshold=test_pct_threshold, future_steps=future_steps)
    print(f"⚙️  Created {len(df.columns)-3} technical features")  # -3 for date, price, target
    
    # Select features (exclude non-predictive columns)
    feature_cols = [col for col in df.columns if col not in ['date', 'price', 'future_price', 'next_return', 'target']]
    X = df[feature_cols]

    # Drop zero-variance / constant features which can break scalers or lead to
    # degenerate models that always predict the same class.
    col_std = X.std(axis=0, skipna=True)
    zero_var = col_std[col_std <= 1e-12].index.tolist()
    if zero_var:
        print(f"⚠️ Dropping {len(zero_var)} zero-variance feature(s): {zero_var[:10]}")
        feature_cols = [c for c in feature_cols if c not in set(zero_var)]
        X = X[feature_cols]
    y_raw = df['target']

    # Encode labels to non-negative integers for models that require it (e.g., XGBoost)
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = y_raw.map(label_map)
    
    print(f"🎯 Using {len(feature_cols)} features for prediction")
    
    # Class distribution (reported with original labels)
    class_dist = y_raw.value_counts().sort_index()
    print(f"\n📈 Class Distribution:")
    print(f"   Decrease ≥0.2%: {class_dist.get(-1, 0)} ({class_dist.get(-1, 0)/len(y_raw)*100:.1f}%)")
    print(f"   No Change: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(y_raw)*100:.1f}%)")
    print(f"   Increase ≥0.2%: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(y_raw)*100:.1f}%)")
    
    # Time series split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"\n🤖 Training model zoo with multi-scaler pipelines...")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier

    env_gpu = os.getenv('USE_GPU')
    cuda_available = detect_cuda_support()
    if env_gpu is None:
        use_gpu = cuda_available
        logger.info("USE_GPU env not set; auto-detected CUDA availability=%s", cuda_available)
    else:
        requested_gpu = env_gpu.lower() in ['1', 'true', 'yes', 'on']
        if requested_gpu and not cuda_available:
            print("⚠️ CUDA not detected; falling back to CPU execution despite USE_GPU request.")
            logger.warning("USE_GPU requested but CUDA unavailable; forcing CPU mode.")
            use_gpu = False
        else:
            use_gpu = requested_gpu
            logger.info("USE_GPU env set to %s -> use_gpu=%s", env_gpu, use_gpu)

    if use_gpu:
        print("⚡ GPU acceleration enabled; CUDA-optimized learners active.")
    else:
        print("🖥️  Running all learners on CPU.")

    # Deterministic CV strategy: choose the largest n_splits (up to 5) such that
    # every training fold contains at least 2 classes. If none satisfies that,
    # fall back to min_splits and enable in-fold oversampling when available.
    def choose_timeseries_splits(y, max_splits=5, min_splits=2):
        for s in range(max_splits, min_splits - 1, -1):
            try:
                tscv_try = TimeSeriesSplit(n_splits=s)
                ok = True
                for train_idx, _ in tscv_try.split(np.zeros(len(y)), y):
                    y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                    if len(np.unique(y_train_fold)) < 2:
                        ok = False
                        break
                if ok:
                    return s, False
            except Exception:
                continue
        return min_splits, True

    chosen_splits, need_oversample_flag = choose_timeseries_splits(y_train, max_splits=5, min_splits=2)
    tscv = TimeSeriesSplit(n_splits=chosen_splits)
    use_imb_global = (ImbPipeline is not None and RandomOverSampler is not None)
    if need_oversample_flag and not use_imb_global:
        print("⚠️ TimeSeriesSplit folds lack class diversity and 'imbalanced-learn' is not installed; consider installing it or increasing data/event rate")

    # Base tree models (no scaling needed)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=140, learning_rate=0.2, max_depth=6, random_state=42
    )

    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=18, min_samples_split=4,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    models = [
        ('rf', rf),
        ('gb', gb),
        ('et', et)
    ]

    if lgb is not None:
        lgb_params = dict(
            objective='multiclass',
            num_class=3,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        if use_gpu:
            lgb_params.update({
                'device': 'gpu',
                'gpu_platform_id': int(os.getenv('LGBM_GPU_PLATFORM', 0)),
                'gpu_device_id': int(os.getenv('LGBM_GPU_DEVICE', 0))
            })
        lgbm = lgb.LGBMClassifier(**lgb_params)
        models.append(('lgbm_gpu' if use_gpu else 'lgbm', lgbm))
    else:
        print("⚠️ LightGBM not installed; skipping LGBM model.")

    if xgb is not None:
        xgb_params = dict(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=3,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        if use_gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
        else:
            xgb_params.update({
                'tree_method': 'hist',
                'predictor': 'auto'
            })
        xgb_model = xgb.XGBClassifier(**xgb_params)
        models.append(('xgb_gpu' if use_gpu else 'xgb', xgb_model))
    else:
        print("⚠️ XGBoost not installed; skipping GPU-accelerated gradient boosting.")
    # Scaler definitions. For QuantileTransformer set n_quantiles relative to
    # the available training samples to avoid warnings or degenerate behaviour
    # when n_quantiles > n_samples.
    qt_n_quantiles = min(1000, max(10, X_train.shape[0]))
    scaler_defs = [
        ('standard', StandardScaler()),
        ('robust', RobustScaler()),
        ('minmax', MinMaxScaler()),
        ('maxabs', MaxAbsScaler()),
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=qt_n_quantiles))
    ]
    for sc_name, sc in scaler_defs:
        # Use imblearn Pipeline with RandomOverSampler if available so resampling
        # happens inside CV folds (no leakage). Fall back to sklearn Pipeline if not.
        use_imb = use_imb_global or need_oversample_flag

        # SGD pipeline
        if use_imb:
            sgd_pipe = ImbPipeline([
                ('scaler', sc),
                ('oversample', RandomOverSampler(random_state=42)),
                ('sgd', SGDClassifier(
                    loss='log_loss',
                    class_weight='balanced',
                    random_state=42,
                    max_iter=2000,
                    tol=1e-3
                ))
            ])
        else:
            sgd_pipe = Pipeline([
                ('scaler', sc),
                ('sgd', SGDClassifier(
                    loss='log_loss',
                    class_weight='balanced',
                    random_state=42,
                    max_iter=2000,
                    tol=1e-3
                ))
            ])
        models.append((f'sgd_{sc_name}', sgd_pipe))

        # SVM pipeline (calibrated)
        svm_linear = LinearSVC(class_weight='balanced', random_state=42, max_iter=4500)
        if use_imb:
            svm_pipe = ImbPipeline([
                ('scaler', sc),
                ('oversample', RandomOverSampler(random_state=42)),
                ('svm', CalibratedClassifierCV(svm_linear, method='sigmoid', cv=3))
            ])
        else:
            svm_pipe = Pipeline([
                ('scaler', sc),
                ('svm', CalibratedClassifierCV(svm_linear, method='sigmoid', cv=3))
            ])
        models.append((f'svm_{sc_name}', svm_pipe))

        # KNN pipeline
        if use_imb:
            knn_pipe = ImbPipeline([
                ('scaler', sc),
                ('oversample', RandomOverSampler(random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance'))
            ])
        else:
            knn_pipe = Pipeline([
                ('scaler', sc),
                ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance'))
            ])
        models.append((f'knn_{sc_name}', knn_pipe))

    print("\n🔍 Evaluating candidate models with TimeSeriesSplit...")
    model_scores = []
    trained_models = {}
    for name, mdl in models:
        print(f"   {name:<18} CV training...", end=' ')
        try:
            # Run cross_val_score (pipelines may include in-fold oversampling when enabled).
            scores = cross_val_score(mdl, X_train, y_train, cv=tscv, scoring='accuracy')
            # If cross_val_score returns NaNs for any fold, treat as failed
            if np.isnan(scores).any():
                print("failed (NaN in CV scores)")
                continue
            mean_score, std_score = float(scores.mean()), float(scores.std())
        except Exception as err:
            print(f"failed ({err})")
            continue

        # Basic sanity checks on CV
        if std_score > max(0.5, abs(mean_score)):
            print(f"warning: high CV variance {mean_score:.3f} ± {std_score:.3f}")

        # Fit model and collect diagnostics to help trace scaler/prediction issues
        try:
            mdl.fit(X_train, y_train)
        except Exception as err:
            print(f"fit failed ({err})")
            continue

        # Diagnostic: inspect scaler behaviour inside Pipeline (if present)
        try:
            scaler = None
            if hasattr(mdl, 'named_steps') and 'scaler' in mdl.named_steps:
                scaler = mdl.named_steps['scaler']
            elif hasattr(mdl, 'steps'):
                steps = dict(mdl.steps)
                scaler = steps.get('scaler')

            if scaler is not None:
                try:
                    # Build a diag input that respects the scaler fit contract but
                    # still falls back to numpy arrays if columns mismatch.
                    def _diag_input(scaler_obj, X_ref):
                        if hasattr(scaler_obj, 'feature_names_in_') and hasattr(X_ref, 'reindex'):
                            missing = [c for c in scaler_obj.feature_names_in_ if c not in X_ref.columns]
                            if not missing:
                                return X_ref.reindex(columns=scaler_obj.feature_names_in_)
                        return X_ref.values if hasattr(X_ref, 'values') else np.asarray(X_ref)

                    X_diag = _diag_input(scaler, X_test)
                    try:
                        X_test_scaled = scaler.transform(X_diag)
                    except Exception:
                        # Last-resort: strip column labels to avoid feature-name checks
                        X_test_scaled = scaler.transform(np.asarray(X_test))
                    col_std = np.nanstd(X_test_scaled, axis=0)
                    zero_var_cols = np.where(col_std < 1e-12)[0]
                    if len(zero_var_cols) > 0:
                        print(f"   scaler={type(scaler).__name__} zero-variance cols={len(zero_var_cols)}")
                except Exception as err_inner:
                    print(f"   scaler diagnostic failed ({err_inner})")

            # Diagnostic: check whether the model predicts a constant label on X_train/X_test
            try:
                preds_train = mdl.predict(X_train)
                unique_train = np.unique(preds_train)
                if len(unique_train) == 1:
                    print(f"   skipping: model predicts single class {unique_train[0]} on X_train")
                    continue
            except Exception:
                pass
            try:
                preds = mdl.predict(X_test)
                unique_preds = np.unique(preds)
                if len(unique_preds) == 1:
                    print(f"   warning: model predicts single class {unique_preds[0]} on X_test")
            except Exception:
                # Non-fatal; some estimators may not implement predict yet
                pass

            trained_models[name] = mdl
            model_scores.append((name, mean_score, std_score))
            print(f"{mean_score:.3f} ± {std_score:.3f}")
        except Exception as err:
            print(f"post-fit diagnostics failed ({err})")
            continue

    if not model_scores:
        print("\n⚠️ No models passed the per-fold class check; attempting relaxed evaluation (secondary pass) to salvage any trainable models...")
        # Secondary, relaxed pass: try evaluating models without pre-skip but still handle exceptions per-model
        for name, mdl in models:
            print(f"   secondary {name:<18} CV training...", end=' ')
            try:
                scores = cross_val_score(mdl, X_train, y_train, cv=tscv, scoring='accuracy')
                if np.isnan(scores).any():
                    print("failed (NaN in CV scores)")
                    continue
                mean_score, std_score = float(scores.mean()), float(scores.std())
            except Exception as err:
                print(f"failed ({err})")
                continue

            try:
                mdl.fit(X_train, y_train)
            except Exception as err:
                print(f"fit failed ({err})")
                continue

            # Minimal diagnostics (no deep scaler checks here)
            try:
                preds_train = mdl.predict(X_train)
                if len(np.unique(preds_train)) == 1:
                    print(f"   skipping: model predicts single class {np.unique(preds_train)[0]} on X_train")
                    continue
            except Exception:
                pass

            trained_models[name] = mdl
            model_scores.append((name, mean_score, std_score))
            print(f"{mean_score:.3f} ± {std_score:.3f}")

        if not model_scores:
            raise RuntimeError("No candidate models were successfully trained.")

    top_k = min(6, len(model_scores))
    top_models = sorted(model_scores, key=lambda x: x[1], reverse=True)[:top_k]
    print(f"\n✨ Top {top_k} models selected for the ensemble:")
    for name, mean_score, std_score in top_models:
        print(f"   {name:<18} CV {mean_score:.3f} ± {std_score:.3f}")

    ensemble_estimators = [(name, trained_models[name]) for name, _, _ in top_models]

    # Simple class rebalancing for final ensemble training: upsample minority
    # classes in the training set to reduce degenerate single-class models.
    try:
        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        label_col = y_train.name if hasattr(y_train, 'name') and y_train.name else 'target_label'
        if label_col not in train_df.columns:
            train_df[label_col] = y_train.reset_index(drop=True)
        counts = train_df[label_col].value_counts()
        max_count = int(counts.max())
        resampled_parts = [
            train_df[train_df[label_col] == cls].sample(max_count, replace=True, random_state=42)
            for cls in counts.index
        ]
        train_resampled = pd.concat(resampled_parts).sample(frac=1, random_state=42).reset_index(drop=True)
        X_train_res = train_resampled[feature_cols]
        y_train_res = train_resampled[label_col]

        # Refit selected estimators on the balanced training set
        for nm, estimator in ensemble_estimators:
            try:
                estimator.fit(X_train_res, y_train_res)
            except Exception as err:
                print(f"   warning: refit for ensemble member {nm} failed ({err}); using original fit")

        ensemble = VotingClassifier(ensemble_estimators, voting='soft')
        ensemble.fit(X_train_res, y_train_res)
    except Exception as err:
        print(f"   warning: ensemble rebalancing failed ({err}), falling back to original training data")
        ensemble = VotingClassifier(ensemble_estimators, voting='soft')
        ensemble.fit(X_train, y_train)
    
    # Predictions
    y_pred_enc = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    confidence = np.max(y_proba, axis=1)
    # Decode predictions back to original labels for reporting
    y_pred = pd.Series(y_pred_enc).map(inv_label_map).to_numpy()
    y_test_decoded = pd.Series(y_test).map(inv_label_map).to_numpy()
    
    # Performance evaluation
    accuracy = accuracy_score(y_test_decoded, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test_decoded, y_pred, 
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
    test_df['next_return'] = test_df['next_return']

    # === Diagnostic: check label semantics and estimator class ordering ===
    print("\n🔎 Prediction -> Return diagnostics:")
    print(f"   inv_label_map: {inv_label_map}")
    try:
        # Show classes_ for each trained estimator used in the ensemble
        for nm, est in ensemble_estimators:
            classes_attr = getattr(est, 'classes_', None)
            print(f"   estimator={nm} classes_={classes_attr}")
    except Exception:
        pass

    # Build DataFrame with actual and predicted labels (decoded) for analysis
    try:
        test_df['predicted_decoded'] = test_df['predicted']
        test_df['actual_decoded'] = pd.Series(y_test_decoded).values

        # Mean next_return by predicted label
        pred_stats = test_df.groupby('predicted_decoded')['next_return'].agg(['mean','count']).to_dict()
        actual_stats = test_df.groupby('actual_decoded')['next_return'].agg(['mean','count']).to_dict()

        print("   Mean next_return by predicted label (label: mean, count):")
        for lbl, row in test_df.groupby('predicted_decoded')['next_return'].agg(['mean','count']).iterrows():
            print(f"      {lbl}: mean={row['mean']:.6f} count={int(row['count'])}")

        print("   Mean next_return by actual label (label: mean, count):")
        for lbl, row in test_df.groupby('actual_decoded')['next_return'].agg(['mean','count']).iterrows():
            print(f"      {lbl}: mean={row['mean']:.6f} count={int(row['count'])}")

        # Quick sanity checks: sign agreement
        sign_mismatch = []
        for lbl in [-1, 0, 1]:
            pred_mean = test_df[test_df['predicted_decoded'] == lbl]['next_return'].mean()
            if pd.isna(pred_mean):
                continue
            # For downside label (-1) expect negative mean; for upside (1) expect positive mean
            if lbl == -1 and pred_mean > 0:
                sign_mismatch.append((lbl, pred_mean))
            if lbl == 1 and pred_mean < 0:
                sign_mismatch.append((lbl, pred_mean))
        if sign_mismatch:
            print("   ⚠️ Warning: sign mismatch detected for predicted labels:")
            for lbl, meanv in sign_mismatch:
                print(f"      label={lbl} mean_next_return={meanv:.6f}")
        else:
            print("   ✅ Predicted label signs align with mean returns (basic check)")
    except Exception as err:
        print(f"   diagnostic failed: {err}")
    # Bollinger heuristic signal for interpretability
    test_df['bb_signal'] = np.where(
        test_df['bbp'] <= 0.2, 'LONG bias',
        np.where(test_df['bbp'] >= 0.8, 'SHORT bias', 'Neutral')
    )

    # 12-hour outlook using 30-minute windows
    horizon_hours = 12
    window_minutes = 30
    outlook_start = test_df['date'].max() - timedelta(hours=horizon_hours)
    horizon_df = test_df[test_df['date'] >= outlook_start].copy()
    if horizon_df.empty:
        print(f"\n⚠️ Not enough recent data for {horizon_hours}-hour outlook.")
    else:
        horizon_df = horizon_df.set_index('date')
        grouped = horizon_df.groupby(pd.Grouper(freq=f'{window_minutes}min'))
        direction_labels = {1: "Bullish", 0: "Sideways", -1: "Bearish"}
        print(f"\n🕒 {horizon_hours}-Hour Outlook ({window_minutes}-minute windows):")
        window_counter = 0
        for _, window_slice in grouped:
            if window_slice.empty:
                continue
            window_counter += 1
            direction_value = window_slice['predicted'].value_counts().idxmax()
            direction = direction_labels.get(direction_value, "Unknown")
            avg_conf = window_slice['confidence'].mean()
            start_ts = window_slice.index.min()
            end_ts = window_slice.index.max()
            entry_price = window_slice['price'].iloc[0]
            target_price = window_slice['future_price'].iloc[-1]
            if pd.isna(target_price):
                target_price = window_slice['price'].iloc[-1]
            pct_move = ((target_price - entry_price) / entry_price * 100) if entry_price else 0.0
            print(
                f"   Window {window_counter:02d} "
                f"[{start_ts:%H:%M} - {end_ts:%H:%M}] "
                f"{direction:<8} target≈${target_price:.2f} "
                f"Δ{pct_move:+.2f}% conf {avg_conf:.1%}"
            )
        if window_counter == 0:
            print("   (No qualifying 30-minute windows found)")
    
    # Forward simulation projecting beyond latest timestamp
    sim_hours = 12
    sim_window = 30
    sim_steps = (sim_hours * 60) // sim_window
    last_timestamp = test_df['date'].max()
    last_price = test_df['future_price'].iloc[-1]
    if pd.isna(last_price):
        last_price = test_df['price'].iloc[-1]
    # Use median (robust) aggregation for class returns to reduce small-sample noise
    grp = test_df.groupby('predicted')['next_return']
    class_return_map = grp.median().to_dict()
    default_return = test_df['next_return'].median()
    recent_signals = test_df.sort_values('date')[['predicted', 'confidence']].tail(sim_steps)
    signal_seq = recent_signals['predicted'].tolist()
    conf_seq = recent_signals['confidence'].tolist()
    direction_labels = {1: "Bullish", 0: "Sideways", -1: "Bearish"}
    print(f"\n🔮 Forward Simulation (next {sim_hours}h in {sim_window}-minute steps):")
    logger.info("Forward simulation start horizon=%sh window=%sm steps=%s", sim_hours, sim_window, sim_steps)
    if not signal_seq:
        print("   (Insufficient signal history to project forward.)")
        logger.warning("Forward simulation skipped due to missing signals.")
    else:
        projected_price = last_price
        for step in range(1, sim_steps + 1):
            idx = step - 1
            direction = signal_seq[idx] if idx < len(signal_seq) else signal_seq[-1]
            confidence_est = conf_seq[idx] if idx < len(conf_seq) else conf_seq[-1]
            expected_return = class_return_map.get(direction, default_return)
            minute_factor = max(window_minutes / max(interval_env, 1), 1)
            projected_return = expected_return * minute_factor
            projected_price = projected_price * (1 + projected_return)
            window_start = last_timestamp + timedelta(minutes=(idx * sim_window) + interval_env)
            window_end = window_start + timedelta(minutes=sim_window)
            msg = (
                f"   Step {step:02d} [{window_start:%Y-%m-%d %H:%M} - {window_end:%H:%M}] "
                f"{direction_labels.get(direction, 'Unknown'):<8} "
                f"target≈${projected_price:.2f} Δ{projected_return*100:+.2f}% "
                f"conf {confidence_est:.1%}"
            )
            print(msg)
            logger.info("ForwardSim %s", msg.strip())
        logger.info("Forward simulation completed with final price %.2f", projected_price)

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
    else:
        print("⚠️ No high-confidence trade signals found.")
        return

    # Output target price and direction
    row = best_trade.iloc[0]
    print(f"\n🚩 Target Trade Signal:")
    print(f"   Direction: {direction}")
    print(f"   Entry Price: ${row['price']:.2f}")
    print(f"   Target Price (next interval): ${row['future_price']:.2f}")
    print(f"   Confidence: {row['confidence']:.1%}")
    print(f"   BB%: {row['bbp']:.2f} ({'LOW' if row['bbp']<=0.2 else 'HIGH' if row['bbp']>=0.8 else 'MID'})")
    print(f"   Bollinger Heuristic: { 'LONG bias' if row['bbp']<=0.2 else 'SHORT bias' if row['bbp']>=0.8 else 'Neutral'}")
    print(f"   Timestamp: {row['date'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Expected Return: {(row['future_price'] - row['price']) / row['price'] * 100:.2f}%")

    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n🔄 Cross-Validation:")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    print(f"\n✅ Analysis Complete!")
    print(f"   📊 Dataset: {len(X_train)} train + {len(X_test)} test samples")
    print(f"   🤖 Ensemble: {len(ensemble_estimators)} models (multi-scaler mix incl. tree & linear pipelines)")
    print(f"   📈 Features: {len(feature_cols)} technical indicators")

if __name__ == "__main__":
    main()
