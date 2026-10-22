# Optional heavy dependencies are imported lazily to allow module import without them.
import os
try:
    import numpy as np
except Exception:
    np = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone
except Exception:
    RandomForestClassifier = GradientBoostingClassifier = VotingClassifier = ExtraTreesClassifier = None
    train_test_split = cross_val_score = TimeSeriesSplit = None
    CalibratedClassifierCV = None
    classification_report = confusion_matrix = accuracy_score = None
    LogisticRegression = LinearRegression = SVC = None

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.svm._base')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.calibration')

try:
    from config.prediction_spec import (
        HORIZON_MINUTES,
        PCT_THRESHOLD,
        BAR_MINUTES,
        NUM_DATA_POINTS,
        TIMEFRAME_LABEL,
        HORIZON_GRID_MINUTES,
        THRESHOLD_GRID,
        HORIZON_HOURS,
        spec_summary,
    )
except Exception:  # pragma: no cover
    HORIZON_MINUTES = 15
    PCT_THRESHOLD = 0.0015   # match config.prediction_spec.py (per SOP D5/F4)
    BAR_MINUTES = 5
    NUM_DATA_POINTS = 11000
    TIMEFRAME_LABEL = '15 minutes'
    HORIZON_GRID_MINUTES = (3, 6, 12, 24)
    THRESHOLD_GRID = (0.010, 0.015, 0.020)
    HORIZON_HOURS = 12
    spec_summary = lambda: f'horizon={HORIZON_MINUTES}m, threshold=±{PCT_THRESHOLD*100:.1f}%'
    # NOTE: do NOT reassign the sklearn names (StandardScaler, Pipeline, clone, etc.)
    # here. They were successfully imported earlier at module scope and must remain
    # bound to their real classes. The earlier chain assignment
    # 'StandardScaler = MinMaxScaler = ... = None' would silently overwrite the good
    # imports with NoneType on any path where this except branch runs (e.g. a missing
    # 'config.prediction_spec' module) and break every downstream scaler call.

# Defensive defaults: ensure the spec names referenced by function
# default-argument expressions always exist, regardless of which branch
# of the spec import above ran (success or fallback).
PCT_THRESHOLD = globals().get('PCT_THRESHOLD', 0.0015)
TIMEFRAME_LABEL = globals().get('TIMEFRAME_LABEL', f'{HORIZON_MINUTES} minutes')

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


def make_time_series_split(n_samples, desired_splits=5):
    """Build a valid walk-forward splitter for time-ordered model validation."""
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for time-series cross-validation.")
    n_splits = min(desired_splits, n_samples - 1)
    n_splits = max(2, n_splits)
    return TimeSeriesSplit(n_splits=n_splits)


def walk_forward_scores(estimator, X, y, desired_splits=5, scoring='accuracy'):
    """
    Walk-forward CV that skips folds whose training slice has only one class.

    Time-series splits on small windows frequently produce early folds where
    the entire training slice is monotonic, leaving GradientBoostingClassifier
    (and others) unable to encode y. We drop those folds instead of letting
    them poison the mean with NaNs. Each surviving fold contributes a single
    score (fit on the train slice, evaluated on the test slice) -- no nested
    CV, so a small rare class never trips StratifiedKFold n_splits>=2 check.
    """
    import numpy as _np
    splitter = make_time_series_split(len(X), desired_splits=desired_splits)
    y_arr = _np.asarray(y)
    scores = []
    for train_idx, test_idx in splitter.split(X):
        y_train_fold = y_arr[train_idx]
        y_test_fold  = y_arr[test_idx]
        if len(_np.unique(y_train_fold)) < 2:
            continue  # train slice is single-class; drop the fold
        if len(y_test_fold) == 0:
            continue  # nothing to score against
        est = clone(estimator)
        est = _fit_with_balanced_sample_weight(est, X.iloc[train_idx], y_train_fold)
        y_pred = est.predict(X.iloc[test_idx])
        if scoring == 'accuracy':
            from sklearn.metrics import accuracy_score as _acc
            scores.append(float(_acc(y_test_fold, y_pred)))
        else:
            from sklearn.metrics import get_scorer as _get_scorer
            scorer = _get_scorer(scoring)
            scores.append(float(scorer(est, X.iloc[test_idx], y_test_fold)))
    return _np.array(scores) if scores else _np.array([_np.nan])


def fit_calibrated_classifier(estimator, X_train, y_train, desired_splits=3):
    """Fit calibrated probabilities, falling back to the raw model when folds are sparse."""
    if len(np.unique(y_train)) < 2:
        estimator = _fit_with_balanced_sample_weight(estimator, X_train, y_train)
        return estimator

    cv = make_time_series_split(len(X_train), desired_splits=desired_splits)
    try:
        calibrated = CalibratedClassifierCV(estimator=clone(estimator), method='sigmoid', cv=cv)
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=clone(estimator), method='sigmoid', cv=cv)

    from sklearn.utils.class_weight import compute_sample_weight as _csw_calib
    _sw_calib = _csw_calib('balanced', y_train)

    # CalibratedClassifierCV.fit() emits a UserWarning from sklearn/calibration.py
    # when the wrapped estimator's .fit() doesn't propagate sample_weight to its
    # sub-estimators (a long-standing sklearn limitation -- see scikit-learn #21134).
    # We accept that limitation deliberately: balancing only on the calibration step
    # is better than no balancing, and the warning is misleading ("result is likely
    # to be incorrect" is too strong). Suppress it for this scope only.
    import warnings as _warnings_calib
    with _warnings_calib.catch_warnings():
        _warnings_calib.filterwarnings(
            'ignore',
            category=UserWarning,
            module='sklearn.calibration',
            message='Since VotingClassifier does not appear to accept sample_weight*',
        )
        try:
            calibrated.fit(X_train, y_train, sample_weight=_sw_calib)
            return calibrated
        except TypeError:
            # sample_weight was rejected by VotingClassifier or CalibratedClassifierCV.
            # Retry without it -- calibration is still useful even if balancing only
            # applies to the raw ensemble path.
            try:
                calibrated.fit(X_train, y_train)
                return calibrated
            except Exception as exc:
                print(f"   ⚠  Calibration skipped: {exc}")
                estimator.fit(X_train, y_train)
                return estimator
        except Exception as exc:
            print(f"   ⚠  Calibration skipped: {exc}")
            estimator.fit(X_train, y_train)
            return estimator


def print_baseline_report(y_train, y_test):
    majority_class = y_train.value_counts().idxmax()
    baseline_pred = np.full(len(y_test), majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    print(f"   Baseline majority-class accuracy: {baseline_accuracy:.3f} (class {majority_class:+.0f})")
    return baseline_accuracy


def print_confidence_report(y_true, y_pred, confidence):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)

    print("\n🎚️  Confidence Calibration Check:")
    for low, high in [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() == 0:
            continue
        bucket_accuracy = accuracy_score(y_true[mask], y_pred[mask])
        coverage = mask.mean()
        label = f"{low:.0%}-{min(high, 1.0):.0%}"
        print(f"   {label:<8} | Accuracy: {bucket_accuracy:.3f} | Coverage: {coverage:.1%} ({mask.sum()} samples)")

    high_conf_mask = confidence >= 0.70
    if high_conf_mask.any():
        high_conf_accuracy = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
        print(f"   High-confidence gate >=70%: {high_conf_accuracy:.3f} accuracy on {high_conf_mask.mean():.1%} of test rows")
    else:
        print("   High-confidence gate >=70%: no qualifying test rows")


def format_display_time(timestamp):
    """Convert timestamps to America/New_York for user-facing output."""
    ts = pd.to_datetime(timestamp, utc=True, errors='coerce')
    if pd.isna(ts):
        return "Invalid timestamp"
    return ts.tz_convert(DISPLAY_TZ).strftime('%Y-%m-%d %I:%M:%S %p %Z')


def display_forecast_windows(price_df, horizon_hours=HORIZON_HOURS):
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
        print(f"   Predicted Range: ${forecast_stats['predicted_min']:.2f} to ${forecast_stats['predicted_max']:.2f}")
        print(f"   Forecast Range (with current): ${forecast_stats['forecast_min']:.2f} to ${forecast_stats['forecast_max']:.2f}")
        if bool(forecast_df.get('baseline_adjusted', pd.Series([False])).iloc[0]):
            gap_pct = float(forecast_df.get('baseline_gap_pct', pd.Series([0.0])).iloc[0])
            print(f"   Baseline Alignment: Enabled (raw first-step gap {gap_pct:+.2f}%)")
        print(f"   Forecast Volatility: {forecast_stats['forecast_volatility']:.4f}")
        print(f"   Forecast Choppiness: {forecast_stats['forecast_choppiness']:.2f}")
        return True

    except Exception as e:
        print(f"\n⚠️ Failed to generate forecast windows: {e}")
        return False

def detect_bar_minutes(df, default=BAR_MINUTES):
    """Infer the candle size (in minutes) from the 'date' column.

    Uses the median spacing between consecutive timestamps, which is robust to
    missing bars and market closures (unlike mean or the first gap). Falls back
    to the spec default when the index is too short or unparseable.
    """
    try:
        if df is None or len(df) < 3 or 'date' not in df.columns:
            return int(default)
        deltas = pd.to_datetime(df['date']).diff().dropna()
        deltas = deltas[deltas > pd.Timedelta(0)]
        if deltas.empty:
            return int(default)
        minutes = float(deltas.dt.total_seconds().median() / 60.0)
        # Snap to a sane candle size; Coinbase supports 1/5/15/60-minute bars.
        allowed = (1, 5, 15, 60)
        return int(min(allowed, key=lambda a: abs(a - minutes)))
    except Exception:
        return int(default)


def horizon_to_steps(horizon_minutes, bar_minutes=BAR_MINUTES):
    """Convert a horizon expressed in MINUTES into a whole number of BARS.

    The feature/target pipeline shifts prices by a bar count, so every
    minute-denominated horizon must be converted before it is passed in.
    """
    return max(int(round(float(horizon_minutes) / float(bar_minutes))), 1)


def create_enhanced_features(df, pct_threshold=PCT_THRESHOLD, horizon_steps=HORIZON_MINUTES):
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

    # Classification target. Keep rows without a known future price out of training.
    df['future_price'] = df['price'].shift(-horizon_steps)
    df['next_return'] = (df['future_price'] - df['price']) / (df['price'] + 1e-8)
    df['target'] = np.nan
    valid_future = df['future_price'].notna()
    df.loc[valid_future, 'target'] = 0
    df.loc[valid_future & (df['next_return'] >= pct_threshold), 'target'] = 1
    df.loc[valid_future & (df['next_return'] <= -pct_threshold), 'target'] = -1
    
    # Replace inf and nan values
    df = df.replace([np.inf, -np.inf], np.nan)
    df['next_return'] = (df['future_price'] - df['price']) / df['price']
    
    return df.dropna()


def evaluate_threshold_horizon_grid(raw_df, thresholds=None, horizons=None):
    """Search label definitions with walk-forward CV on pre-holdout data only.

    Defaults derive from the shared prediction spec (config.prediction_spec)
    so the search grid stays centered on the chosen product horizon/threshold.
    """
    thresholds = thresholds or list(THRESHOLD_GRID)
    horizons = horizons or list(HORIZON_GRID_MINUTES)
    search_cutoff = max(int(len(raw_df) * 0.8), 1)
    search_df = raw_df.iloc[:search_cutoff].copy()
    results = []

    print("\n🧪 Label Search: threshold × horizon walk-forward CV")
    for horizon in horizons:
        for threshold in thresholds:
            try:
                candidate = create_enhanced_features(
            search_df,
            pct_threshold=threshold,
            horizon_steps=horizon_to_steps(horizon),
        )
                feature_cols = [
                    col for col in candidate.columns
                    if col not in ['date', 'price', 'high', 'low', 'future_price', 'next_return', 'target', 'signal', 'bb_pct', 'macd', 'macd_signal']
                ]
                X_candidate = candidate[feature_cols]
                y_candidate = candidate['target'].astype(int)

                if len(y_candidate) < 200 or y_candidate.nunique() < 2:
                    continue

                search_model = RandomForestClassifier(
                    n_estimators=60,
                    max_depth=10,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_jobs=-1
                )
                scores = walk_forward_scores(search_model, X_candidate, y_candidate, desired_splits=3, scoring='accuracy')
                class_balance = y_candidate.value_counts(normalize=True).to_dict()
                majority_baseline = float(max(class_balance.values()))
                results.append({
                    'threshold': threshold,
                    'horizon': horizon,
                    'cv_mean': float(scores.mean()),
                    'cv_spread': float(scores.std() * 2),
                    'baseline': majority_baseline,
                    'edge': float(scores.mean() - majority_baseline),
                    'samples': len(y_candidate),
                    'class_balance': class_balance
                })
            except Exception as exc:
                print(f"   ⚠  skipped threshold={threshold:.4f}, horizon={horizon}m: {exc}")

    if not results:
        print("   ⚠  Label search produced no valid candidates; using shared spec threshold=${PCT_THRESHOLD:.4f}, horizon=${HORIZON_MINUTES}m")
        return {'threshold': PCT_THRESHOLD, 'horizon': HORIZON_MINUTES, 'cv_mean': 0.0, 'cv_spread': 0.0}

    results.sort(key=lambda row: (row['edge'], row['cv_mean'], -row['cv_spread']), reverse=True)
    for i, row in enumerate(results[:5], 1):
        print(
            f"   {i}. threshold={row['threshold']:.4f}, horizon={row['horizon']:>2}m | "
            f"CV={row['cv_mean']:.3f} (±{row['cv_spread']:.3f}) | "
            f"baseline={row['baseline']:.3f} | edge={row['edge']:+.3f} | samples={row['samples']}"
        )

    stable = [row for row in results if row['cv_spread'] <= 0.20]
    best = stable[0] if stable else results[0]
    print(
        f"   Selected label: threshold={best['threshold']:.4f}, "
        f"horizon={best['horizon']}m, CV={best['cv_mean']:.3f} (±{best['cv_spread']:.3f}), "
        f"baseline={best.get('baseline', 0.0):.3f}, edge={best.get('edge', 0.0):+.3f}"
    )
    return best

def fetch_bitcoin_data(num_points=NUM_DATA_POINTS, interval_minutes=BAR_MINUTES):
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


def _detect_estimator_backend(estimator):
    """Return 'cuML' if `estimator` is a cuml class, else 'sklearn'.

    Used by _fit_with_balanced_sample_weight to give a more actionable warning
    when sample_weight is rejected (cuML RF on RAPIDS < 25.x is the common case).
    """
    mod = type(estimator).__module__ or ''
    if mod.startswith('cuml'):
        return 'cuML'
    if mod.startswith('sklearn'):
        return 'sklearn'
    return mod or 'unknown'


def _fit_with_balanced_sample_weight(estimator, X, y, **fit_kwargs):
    """Fit `estimator` with sklearn's 'balanced' class_weight expressed as sample_weight.

    Replaces the constructor-level `class_weight='balanced'` kwarg that cuML does
    not accept on __init__. Computes per-sample weights so each class contributes
    equally to the loss, then forwards them via `sample_weight` to .fit().

    Strategy (most permissive first):
      1. Inspect the estimator type:
         - sklearn Pipeline           -> route sample_weight to its final step via
                                       <last_step_name>__sample_weight= (sklearn
                                       >=1.9 rejects top-level sample_weight on
                                       Pipeline.fit with ValueError, so we cannot
                                       fall back to it there).
         - VotingClassifier           -> pass weights to all adapted members.
         - everything else            -> direct .fit(X, y, sample_weight=sw).
      2. On TypeError or ValueError, retry without sample_weight (cuML builds that
         reject it entirely -- pre-25.x cuML RF; sklearn Pipelines in 1.9 that we
         failed to recognize; etc.) and warn once per (estimator class, backend).

    cuML's signature introspection is unreliable: cuML 24.x added sample_weight
    to LogisticRegression.fit but not RandomForestClassifier.fit until later
    releases. So we never trust inspect.signature(); we always try the call
    first and fall back on failure.

    Returns the fitted estimator, which may be a weighted CPU forest fallback.
    """
    try:
        from sklearn.utils.class_weight import compute_sample_weight as _csw
        sw = _csw('balanced', y)
    except Exception:
        sw = None
    if sw is None:
        return estimator.fit(X, y, **fit_kwargs)

    cls = type(estimator).__name__
    backend = _detect_estimator_backend(estimator)

    # Route 1: sklearn Pipeline -- sklearn >=1.9 raises ValueError when
    # sample_weight is passed top-level. We rewrite it to the final step's
    # sample_weight via the stepname__param convention.
    try:
        from sklearn.pipeline import Pipeline as _Pipeline
    except Exception:
        _Pipeline = None
    if _Pipeline is not None and isinstance(estimator, _Pipeline):
        last_name, last_est = estimator.steps[-1]
        try:
            return estimator.fit(X, y, **{f"{last_name}__sample_weight": sw}, **fit_kwargs)
        except (TypeError, ValueError) as exc:
            # The final step rejected sample_weight (e.g. cuML classifier wrapped
            # in a sklearn Pipeline). Fall back to plain fit, applying sample_weight
            # to the wrapped sklearn estimator upstream if possible.
            _emit_sample_weight_skip_warning(cls, backend, exc)
            try:
                _fit_with_balanced_sample_weight(last_est, X, y)
                estimator.steps[-1] = (last_name, last_est)
                return estimator
            except Exception:
                return estimator.fit(X, y, **fit_kwargs)

    # Voting members support weights, including the adapted pipelines. Let
    # routing errors surface instead of silently training an unbalanced model.
    if isinstance(estimator, VotingClassifier):
        return estimator.fit(X, y, sample_weight=sw, **fit_kwargs)

    # Route 3: everything else -- direct pass-through.
    try:
        return estimator.fit(X, y, sample_weight=sw, **fit_kwargs)
    except (TypeError, ValueError) as exc:
        if backend == 'cuML' and cls == 'RandomForestClassifier':
            if 'sample_weight' not in str(exc):
                raise
            # Preserve the requested sklearn settings when available. Cloned
            # cuML estimators may only retain their supported constructor params.
            params = getattr(estimator, '_cpu_fallback_kwargs', None)
            if params is None:
                supported = RandomForestClassifier().get_params()
                params = {k: v for k, v in estimator.get_params().items()
                          if k in supported}
            params = dict(params)
            params['class_weight'] = None  # sw already balances the classes
            params.setdefault('n_jobs', -1)
            fallback = RandomForestClassifier(**params)
            if not getattr(_fit_with_balanced_sample_weight, '_rf_fallback_reported', False):
                print("   [training] cuML RandomForest does not support sample_weight; "
                      "using a CPU RandomForest with balanced sample weights.")
                _fit_with_balanced_sample_weight._rf_fallback_reported = True
            return fallback.fit(X, y, sample_weight=sw, **fit_kwargs)
        _emit_sample_weight_skip_warning(cls, backend, exc)
        return estimator.fit(X, y, **fit_kwargs)


def _emit_sample_weight_skip_warning(cls, backend, exc):
    """Print the 'sample_weight rejected' warning at most once per (class, backend).

    Factored out of _fit_with_balanced_sample_weight so all three routes share the
    same dedup logic. The message clearly distinguishes cuML from sklearn and from
    sklearn Pipeline / heterogeneous-ensemble edge cases so the root cause shows up
    without ambiguity.
    """
    _key = (cls, backend)
    _seen = getattr(_emit_sample_weight_skip_warning, '_seen', set())
    if _key in _seen:
        return
    _seen.add(_key)
    _emit_sample_weight_skip_warning._seen = _seen
    if backend == 'cuML':
        print(f"   ⚠  cuML {cls}.fit() rejected sample_weight ({exc}); "
              f"this is a known cuML limitation (cuML estimators don't expose "
              f"sample_weight as a fit kwarg). Class balancing is being skipped on "
              f"the GPU side for this estimator type. For ensemble sub-estimators, "
              f"the construction site should swap cuML for sklearn equivalents.")
    elif cls == 'Pipeline':
        print(f"   ⚠  sklearn Pipeline.fit() rejected sample_weight ({exc}); "
              f"class balancing is being skipped on the final step.")
    elif 'Heterogeneous' in cls:
        print(f"   ⚠  sklearn {cls}.fit() can't propagate sample_weight; "
              f"class balancing is being skipped. Sub-estimators should expose "
              f"class_weight='balanced' on the constructor instead.")
    else:
        print(f"   ⚠  sklearn {cls}.fit() rejected sample_weight ({exc}); "
              f"class balancing is being skipped for this estimator.")

def _gpu_rf(**kwargs):
    """Return a RandomForestClassifier, preferring cuML when CUDA is available.

    cuML RF does not accept sklearn's `n_jobs` (GPU parallelism is implicit) or
    `class_weight` (cuML has no per-class weighting on the constructor; balance
    the training set with compute_sample_weight('balanced') at fit time instead).
    All other sklearn-compatible kwargs are forwarded as-is.
    """
    if CUDA_AVAILABLE and _cuml_RF is not None:
        gpu_kw = {k: v for k, v in kwargs.items() if k not in _CUM_RF_UNSUPPORTED}
        try:
            estimator = _cuml_RF(**gpu_kw)
            estimator._cpu_fallback_kwargs = dict(kwargs)
            return estimator
        except TypeError as exc:
            # Defensive: if a future cuML version supports these, fall back gracefully.
            print(f"   ⚠  cuML RF rejected kwargs ({exc}); falling back to sklearn.")
    return RandomForestClassifier(**kwargs)


def _gpu_lr(**kwargs):
    """Return a LogisticRegression, preferring cuML when CUDA is available.

    cuML's LogisticRegression signature varies across RAPIDS versions:
      * `class_weight`: not accepted in any version (apply at fit time instead).
      * `random_state`: not accepted in many versions (cuML seeds differ).
      * `solver`, `multi_class`, `n_jobs`, etc.: not accepted in many.

    We hard-strip `class_weight` and `random_state` (the two that matter for
    this codebase) and probe cuML's __init__ signature for anything else. Any
    kwarg not in cuML's signature is silently dropped before forwarding. If the
    underlying call still TypeErrors, we fall back to sklearn and warn once.
    """
    if CUDA_AVAILABLE and _cuml_LR is not None:
        import inspect as _inspect_lr
        try:
            _ctor_params = set(_inspect_lr.signature(_cuml_LR.__init__).parameters.keys())
        except (TypeError, ValueError):
            _ctor_params = set()
        # Always strip the known-bad kwargs, plus anything not in cuML's signature.
        gpu_kw = {
            k: v for k, v in kwargs.items()
            if k not in _CUM_LR_UNSUPPORTED and (not _ctor_params or k in _ctor_params or k == 'self')
        }
        try:
            return _cuml_LR(**gpu_kw)
        except TypeError as exc:
            _gpu_lr._warned = getattr(_gpu_lr, '_warned', False)
            if not _gpu_lr._warned:
                _gpu_lr._warned = True
                print(f"   ⚠  cuML LR rejected kwargs ({exc}); falling back to sklearn.")
        except Exception:
            pass
    return LogisticRegression(**kwargs)


def _gpu_scaler_by_name(name):
    """Return a scaler instance, preferring cuML GPU variants when CUDA is available.

    Mirrors _cpu_scaler_by_name() but routes StandardScaler, MinMaxScaler, and
    RobustScaler through cuML when those bindings were imported successfully.
    PowerTransformer and QuantileTransformer have no cuML equivalent and always
    run on sklearn CPU.
    """
    if CUDA_AVAILABLE:
        if name == 'StandardScaler' and _cuml_StandardScaler is not None:
            return _cuml_StandardScaler()
        if name == 'MinMaxScaler' and _cuml_MinMaxScaler is not None:
            return _cuml_MinMaxScaler()
        if name == 'RobustScaler' and _cuml_RobustScaler is not None:
            return _cuml_RobustScaler()
    return _cpu_scaler_by_name(name)



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
        results = []
        splitter = make_time_series_split(len(X_train), desired_splits=cv)

        # Use sklearn pipelines for evaluation so scaling is fitted inside each CV fold.
        scaler_names = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer', 'QuantileTransformer']
        for name in scaler_names:
            scored = None  # (avg_score, fitted_scaler_instance) on success
            # Primary attempt: GPU-aware scaler + GPU-aware RF pipeline.
            try:
                scaler = _gpu_scaler_by_name(name)
                rf = _gpu_rf(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                candidate = Pipeline([
                    ('scaler', scaler),
                    ('rf', rf)
                ])
                scores = cross_val_score(candidate, X_train, y_train, cv=splitter, scoring='accuracy')
                avg_score = float(scores.mean())
                if np.isnan(avg_score):
                    raise ValueError("cross_val_score returned NaN")
                # Re-use a fresh, unfitted scaler instance so the booster can fit+train later.
                scored = (avg_score, _gpu_scaler_by_name(name))
            except Exception as e_primary:
                # Fallback attempt: CPU-only sklearn scaler + sklearn RF. cuML bindings
                # occasionally change signatures between RAPIDS releases (e.g. moved out
                # of `cuml.preprocessing`), which surfaces as "NoneType not callable".
                try:
                    scaler = _cpu_scaler_by_name(name)
                    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                    candidate = Pipeline([
                        ('scaler', scaler),
                        ('rf', rf),
                    ])
                    scores = cross_val_score(candidate, X_train, y_train, cv=splitter, scoring='accuracy')
                    avg_score = float(scores.mean())
                    if np.isnan(avg_score):
                        raise ValueError("cross_val_score returned NaN")
                    scored = (avg_score, _cpu_scaler_by_name(name))
                    print(f"    ↪ {name} GPU path failed ({type(e_primary).__name__}); CPU fallback succeeded (CV={avg_score:.4f}).")
                except Exception as e_cpu:
                    print(f"    ⚠  {name} scaler eval failed: {type(e_primary).__name__}: {e_primary}; CPU fallback: {type(e_cpu).__name__}: {e_cpu}")
            if scored is not None:
                results.append((name, scored[0], scored[1]))
        
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
        self.feature_columns = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Defensive: if scaler evaluation produced no usable scalers (all 5
        # failed on this platform/dependency combination), the booster will
        # vote "neutral" later rather than crash with ZeroDivisionError.
        if not scalers_list:
            print('    ⚠️  ScalerEvaluator returned no results; confidence booster will be neutral.')
            return
        
        for scaler_name, _, scaler in scalers_list[:3]:
            try:
                X_scaled = scaler.fit_transform(X_train)
            except Exception as e:
                print(f"    ⚠  {scaler_name} scaler failed in booster: {e}. Falling back to CPU scaler.")
                scaler = _cpu_scaler_by_name(scaler_name)
                X_scaled = scaler.fit_transform(X_train)
            self.models[scaler_name] = {}
            
            # Random Forest
            # _gpu_rf strips n_jobs (implicit on GPU); all other kwargs forwarded.
            # class_weight is applied via sample_weight below (cuML-friendly).
            try:
                rf = _gpu_rf(
                    n_estimators=100, max_depth=12, min_samples_split=5,
                    random_state=random_state, n_jobs=-1
                )
            except Exception:
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=12, min_samples_split=5,
                    class_weight='balanced', random_state=random_state, n_jobs=-1
                )
            # Apply balanced class_weight via sample_weight (works on cuML RF and sklearn RF).
            rf = _fit_with_balanced_sample_weight(rf, X_scaled, y_train)
            self.models[scaler_name]['rf'] = (rf, scaler)
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=80, learning_rate=0.1, max_depth=6,
                random_state=random_state
            )
            gb.fit(X_scaled, y_train)
            self.models[scaler_name]['gb'] = (gb, scaler)

    def _as_feature_frame(self, X_test_latest):
        """Return one prediction row with the same feature names used during scaler fit."""
        if self.feature_columns is None:
            return X_test_latest.reshape(1, -1) if hasattr(X_test_latest, 'reshape') else np.asarray(X_test_latest).reshape(1, -1)

        if hasattr(X_test_latest, 'columns'):
            return X_test_latest.reindex(columns=self.feature_columns)

        if hasattr(X_test_latest, 'to_frame'):
            return X_test_latest.to_frame().T.reindex(columns=self.feature_columns)

        values = np.asarray(X_test_latest).reshape(1, -1)
        return pd.DataFrame(values, columns=self.feature_columns)
    
    def predict_with_boost(self, X_test_latest):
        """Predict on single row, returning (direction, boosted_confidence).
        
        Args:
            X_test_latest: single row feature vector (1D array or Series)
        
        Returns:
            (direction, boosted_confidence, breakdown_str)
        """
        X_row = self._as_feature_frame(X_test_latest)
        
        votes = []
        confidences = []
        breakdown = []
        
        for scaler_name in sorted(self.models.keys()):
            for model_type in ['rf', 'gb']:
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
        
        # Voting: majority direction. Guard against empty votes (e.g. when the
        # upstream scaler evaluation returned an empty list and no models were
        # trained into `self.models`). Without this guard we previously hit
        # `ZeroDivisionError: division by zero` at `vote_agreement`.
        if not votes:
            breakdown_str = '(no scaler/models trained; ensemble unavailable)'
            return 0, 0.0, breakdown_str

        vote_sum = sum(votes)
        final_direction = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else 0)

        # Confidence boost: average confidences, weighted by vote agreement.
        # `np.mean(confidences)` is also guarded implicitly because confidences
        # is only appended alongside votes; we add an explicit safe-mean to
        # silence `RuntimeWarning: Mean of empty slice`.
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        vote_agreement = abs(vote_sum) / len(votes)  # How unanimous? (safe: votes non-empty)
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
    raw_df = fetch_bitcoin_data(num_points=NUM_DATA_POINTS, interval_minutes=BAR_MINUTES)
    print(f"📊 Fetched {len(raw_df)} data points of Bitcoin data at {BAR_MINUTES}-minute intervals")

    # Hardcoded label spec -- the auto-search (evaluate_threshold_horizon_grid)
    # is bypassed. Use the values from config.prediction_spec.PCT_THRESHOLD and
    # HORIZON_MINUTES directly. To re-enable auto-search, uncomment the next two
    # lines and delete the 'selected_threshold' / 'selected_horizon' assignments below.
    # best_label = evaluate_threshold_horizon_grid(raw_df)
    # selected_threshold = best_label['threshold']; selected_horizon = best_label['horizon']
    selected_threshold = PCT_THRESHOLD
    selected_horizon = HORIZON_MINUTES
    move_label = f"{selected_threshold * 100:.3f}%"

    df = create_enhanced_features(
        raw_df,
        pct_threshold=selected_threshold,
        horizon_steps=horizon_to_steps(selected_horizon),
    )
    print(f"⚙️  Created {len(df.columns)-3} technical features")  # -3 for date, price, target
    print(f"🎯 Training label: next {selected_horizon} minute(s), move threshold ±{move_label}")

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

    # --- Economic & Game Theory Voting Engine ---
    # SignalVotingEngine (Adam Smith / BdM Expected Utility / Nash Equilibrium) was
    # removed on 2026-10-21; see MODEL_ACCURACY_DIAGNOSIS_AND_SOP.md for context. The
    # classifier ensemble downstream is unchanged.


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
    feature_cols = [col for col in df.columns if col not in ['date', 'price', 'high', 'low', 'future_price', 'next_return', 'target', 'signal', 'bb_pct', 'macd', 'macd_signal']]
    X = df[feature_cols]
    y = df['target'].astype(int)
    
    print(f"\n🎯 Using {len(feature_cols)} features for prediction")
    
    # Class distribution
    class_dist = y.value_counts().sort_index()
    print(f"\n📈 Class Distribution:")
    print(f"   Decrease ≥{move_label}: {class_dist.get(-1, 0)} ({class_dist.get(-1, 0)/len(y)*100:.1f}%)")
    print(f"   No Change: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
    print(f"   Increase ≥{move_label}: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
    
    # Time series split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print_baseline_report(y_train, y_test)
    
    # --- Fast path: serve a cached state from a previous run ---------------------
    # To enable: set the environment variable FAST_PATH_MODEL_STATE=1. The cached
    # state is honoured only when it is younger than MAX_STATE_AGE_HOURS (default 0.5 h).
    # On any failure (missing file, schema mismatch, too-old state, decode error, or
    # feature-set mismatch with the current dataset) the fast path is bypassed silently
    # and the normal scaler-evaluation path runs -- it never blocks a prediction.
    if os.environ.get("FAST_PATH_MODEL_STATE", "").lower() in ("1", "true", "yes"):
        try:
            import datetime as _dt_fp
            from services.model_state import load_state as _load_state, ModelStateError as _fp_ModelStateError
            max_age_hours = float(os.environ.get("MAX_STATE_AGE_HOURS", "0.5"))
            _state = _load_state()
            _age_s = (_dt_fp.datetime.now(_dt_fp.timezone.utc) - _state.saved_at).total_seconds()
            if _age_s > max_age_hours * 3600:
                raise RuntimeError(f"cached state is {_age_s/3600:.2f}h old (> {max_age_hours}h)")
            _feature_cols = list(_state.feature_cols)
            _missing = [c for c in _feature_cols if c not in df.columns]
            if _missing:
                raise RuntimeError(f"cached state expects {len(_missing)} feature(s) missing from current data: {_missing[:5]}")
            _X_latest = df[_feature_cols].iloc[[-1]]
            _y_proba = _state.ensemble.predict_proba(_X_latest)[0]
            _y_pred = int(np.argmax(_y_proba)) - 1   # classes were [-1, 0, 1] -> probs order matches
            _conf = float(np.max(_y_proba))
            print(f"   ⚡ Fast-path predict (cache {_age_s:.1f}s old, schema v{_state.schema_version}): direction={_y_pred:+d}  confidence={_conf:.2%}")
            print(f"   📦 Reused ensemble: {_state.ensemble.__class__.__name__}, booster: {_state.booster.__class__.__name__}, scalers: {len(list(_state.scaler_results))}")
            # Note: we deliberately do NOT save again here (no fresh training happened).
            # The cached state remains valid; the next normal run will refresh it.
            return
        except (FileNotFoundError, _fp_ModelStateError, RuntimeError) as _fp_exc:
            print(f"   ⏩  Fast path unavailable ({_fp_exc}); falling back to fresh training.")
        except Exception as _fp_exc:
            print(f"   ⏩  Fast path error ({type(_fp_exc).__name__}: {_fp_exc}); falling back to fresh training.")

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
    
    # SignalVotingEngine was removed on 2026-10-21 (Adam Smith /
    # BdM Expected Utility / Nash Equilibrium frameworks). Without
    # a voter to verify the ensemble against, the voter-first /
    # ensemble-verified fusion step is disabled; the printed ensemble
    # verdict above stands on its own, and the downstream classifier
    # pipeline picks up from this row.
    print("   ⚠️  Voter Engine removed -- fusion skipped; final action defers to ensemble/booster pipeline.")
    print()
    
    # Create ensemble of classifiers
    print(f"\n🤖 Training Scikit-Learn Ensemble...")
    
    # Random Forest -- balancing applied at fit time via sample_weight
    # (cuML doesn't accept class_weight in __init__; the explicit fit() below wins
    # for both GPU and CPU paths).
    rf = _gpu_rf(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
    )
    
    # MinMaxScaler + Gradient Boosting (balanced learner)
    gb_minmax_pipe = Pipeline([
        ('scaler', _gpu_scaler_by_name('MinMaxScaler')),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
    ])
    
    # Logistic Regression with StandardScaler
    lr_scaled_pipe = Pipeline([
        ('scaler', _gpu_scaler_by_name('StandardScaler')),
        ('lr', _gpu_lr(max_iter=2000, random_state=42))  # balancing via sample_weight at fit time
    ])
    
    # SVM with scaling
    svm_pipe = Pipeline([
        ('scaler', _gpu_scaler_by_name('StandardScaler')),
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
    ])

    # Extra Trees adds a more randomized tree ensemble that can reduce RF correlation.
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=14, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Use CPU ensemble members for consistent sample-weight support. The factory
    # adapts pipelines and removes constructor class weights so balanced sample
    # weights apply exactly once during cross-validation and calibration.
    _ensemble_rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    _ensemble_lr_scaled_pipe = Pipeline([
        ('scaler', _cpu_scaler_by_name('StandardScaler')),
        ('lr', LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42,
        )),
    ])
    from weighted_voting import make_weighted_voting_classifier
    ensemble = make_weighted_voting_classifier([
        ('rf', _ensemble_rf),
        ('gb', gb),
        ('gb_minmax', gb_minmax_pipe),
        ('lr_scaled', _ensemble_lr_scaled_pipe),
        ('svm', svm_pipe),
        ('et', et)
    ], voting='soft')
    
    cv_scores = walk_forward_scores(clone(ensemble), X_train, y_train, desired_splits=5, scoring='accuracy')
    print(f"\n🔄 Walk-Forward Cross-Validation:")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")

    # Train calibrated ensemble
    ensemble = fit_calibrated_classifier(ensemble, X_train, y_train, desired_splits=3)
    
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
                              target_names=[f"Decrease ≥{move_label}", "No Change", f"Increase ≥{move_label}"],
                              labels=[-1, 0, 1], zero_division=0))
    print_confidence_report(y_test, y_pred, confidence)
    
    # Feature importance from Random Forest
    # Apply balanced class_weight via sample_weight (works on cuML RF and sklearn RF)
    rf = _fit_with_balanced_sample_weight(rf, X_train, y_train)
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
        display_forecast_windows(raw_df, horizon_hours=HORIZON_HOURS)
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

    print(f"\n✅ Analysis Complete!")
    print(f"   📊 Dataset: {len(X_train)} train + {len(X_test)} test samples")
    print(f"   🤖 Ensemble: 6 algorithms (RF, GB, GB-MinMax, LR-Scaled, SVM, ExtraTrees)")
    print(f"   📈 Features: {len(feature_cols)} technical indicators")

    # --- Persist fitted state to disk so the next run can short-circuit predict. ---
    # Errors here must NEVER break the prediction run -- the state cache is a nice-to-have,
    # not a requirement. The whole block is wrapped in a try/except that logs and continues.
    try:
        from services.model_state import save_state as _save_state
        _modstate_path = _save_state(
            ensemble=ensemble,
            booster=booster,
            scaler_results=scaler_results,
            feature_cols=feature_cols,
            cv_mean=float(cv_scores.mean()),
            accuracy=float(accuracy),
            metadata={"trigger": "main-end"},
        )
        print(f"   💾 Model state saved to {_modstate_path}")
    except Exception as _modstate_exc:
        # Never let a state-cache failure block a prediction. One-line log only.
        print(f"   ⚠️  Model state save skipped: {type(_modstate_exc).__name__}: {_modstate_exc}")


if __name__ == "__main__":
    main()
