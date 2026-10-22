#!/usr/bin/env python3
"""
Enhanced Bitcoin Price Forecasting System
Revamped signal logic with multi-horizon technical indicators and 12-hour forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

# Optional GPU backends (cuML, XGBoost). Detected at import time.
CUDA_AVAILABLE = False
_gpu_count = 0
_cuml_RF_reg = None
_cuml_GB_reg = None
_cuml_StandardScaler = None
_cuml_MinMaxScaler = None
_cuml_RobustScaler = None
_cuml_LinearRegression = None
XGBOOST_AVAILABLE = False
# CUDA availability is independent of which optional cuML components exist.
try:
    import cupy as cp_reg
    _gpu_count = cp_reg.cuda.runtime.getDeviceCount()
    CUDA_AVAILABLE = _gpu_count > 0
except Exception:
    pass

if CUDA_AVAILABLE:
    try:
        import cuml
        cuml.set_global_output_type('numpy')
    except Exception:
        pass

    def _optional_cuml_component(module, name):
        from importlib import import_module
        try:
            return getattr(import_module(module), name)
        except Exception:
            return None

    _cuml_RF_reg = _optional_cuml_component('cuml.ensemble', 'RandomForestRegressor')
    _cuml_GB_reg = _optional_cuml_component('cuml.ensemble', 'GradientBoostingRegressor')
    _cuml_StandardScaler = _optional_cuml_component('cuml.preprocessing', 'StandardScaler')
    _cuml_MinMaxScaler = _optional_cuml_component('cuml.preprocessing', 'MinMaxScaler')
    _cuml_RobustScaler = _optional_cuml_component('cuml.preprocessing', 'RobustScaler')
    _cuml_LinearRegression = _optional_cuml_component('cuml.linear_model', 'LinearRegression')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

_CUM_RF_REG_UNSUPPORTED = frozenset({'n_jobs'})
_CUM_GB_REG_UNSUPPORTED = frozenset({'n_jobs'})
_CUM_RF_REG_FIT_UNSUPPORTED = set()
_CUM_GB_REG_FIT_UNSUPPORTED = {'sample_weight'}

try:
    from config.prediction_spec import HORIZON_HOURS, spec_summary
except Exception:  # pragma: no cover
    HORIZON_HOURS = 12
    spec_summary = lambda: f"regression_horizon={HORIZON_HOURS}h"

class EnhancedBitcoinForecaster:
    """Enhanced forecasting system with multi-horizon signal detection and 12-hour forecasts."""

    def __init__(self, data_update_interval=300):
        self.data_update_interval = data_update_interval
        self.models = {}
        self.trained_models = {}
        self.last_update = None
        self.history_identity = {}
        self.forecast_reference_price = None

        # Ensemble base models. Each entry is a thunk (zero-arg callable)
        # that returns the estimator instance -- this defers backend selection
        # (CPU sklearn vs GPU cuML/XGBoost) to call time so missing deps
        # never break import. The thunks reference module-level helpers
        # (_gpu_rf_reg, _gpu_gb_reg, _gpu_scaler_reg, _make_xgb_regressor).
        self.models = {
            'RandomForest':     lambda: MultiOutputRegressor(EnhancedBitcoinForecaster._gpu_rf_reg(
                n_estimators=250, max_depth=16, min_samples_split=4,
                random_state=42, n_jobs=-1
            )),
            'GradientBoosting': lambda: MultiOutputRegressor(EnhancedBitcoinForecaster._gpu_gb_reg(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                random_state=42
            )),
            'ExtraTrees':       lambda: MultiOutputRegressor(
                ExtraTreesRegressor(
                    n_estimators=250, max_depth=14, min_samples_split=3,
                    random_state=42, n_jobs=-1
                )
            ),
            'XGBoost':          lambda: MultiOutputRegressor(EnhancedBitcoinForecaster._make_xgb_regressor()),
            'Pipeline_RF':      lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('rf', MultiOutputRegressor(RandomForestRegressor(
                    n_estimators=150, random_state=7, n_jobs=-1)))
            ]),
            'Pipeline_RF_Scaled': lambda: Pipeline([
                ('scaler', EnhancedBitcoinForecaster._gpu_scaler_reg('StandardScaler')),
                ('rf', MultiOutputRegressor(EnhancedBitcoinForecaster._gpu_rf_reg(
                    n_estimators=200, max_depth=14, random_state=11, n_jobs=-1)))
            ]),
        }

    @staticmethod
    def _drop_normalized_columns(df):
        """Drop normalized helper columns if present to prevent leakage."""
        drop_cols = [c for c in df.columns if 'normalized' in c.lower()]
        return df.drop(columns=drop_cols, errors='ignore')

    @staticmethod
    def _linear_regression_slope(series):
        x = np.arange(len(series), dtype=float)
        y = np.asarray(series, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom <= 1e-12:
            return 0.0
        numer = np.sum((x - x_mean) * (y - y_mean))
        return numer / denom

    @staticmethod
    def _dmi(df, period=14):
        close = df['price']
        high = close.rolling(2, min_periods=1).max()
        low = close.rolling(2, min_periods=1).min()

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_components = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1)
        tr = tr_components.max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean() + 1e-8

        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=1).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=1).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(period, min_periods=1).mean()

        return plus_di, minus_di, adx, atr

    def preprocess_data(self, df):
        """Ensure datetime handling, sorting and stable resampling without dropping feature windows."""
        if 'date' not in df.columns or 'price' not in df.columns:
            raise ValueError("Input data must include 'date' and 'price' columns")

        out = df.copy()
        out['date'] = pd.to_datetime(out['date'], utc=True, errors='coerce')
        out['price'] = pd.to_numeric(out['price'], errors='coerce')
        out = out.dropna(subset=['date', 'price']).sort_values('date')
        out = out.drop_duplicates(subset='date', keep='last')

        # Only completed 15-minute windows are valid training/inference inputs.
        spacing = out['date'].diff().dropna()
        spacing = spacing[spacing > pd.Timedelta(0)]
        bar_size = spacing.median() if not spacing.empty else pd.Timedelta(minutes=5)
        complete_until = min(out['date'].max() + bar_size, pd.Timestamp.now(tz='UTC'))
        # Snap to 15-minute windows and preserve past missing windows via forward fill.
        out = out.set_index('date').resample('15min').last()
        # No backward fill, no time interpolation across the full grid -- both leak future observations into the past.
        # Forward-fill only so each missing bar carries the most recent past observed price.
        out['price'] = out['price'].ffill()
        out = out.loc[out.index + pd.Timedelta(minutes=15) <= complete_until]
        out = out.reset_index()
        return out

    def create_prediction_features(self, df):
        """Create revamped multi-horizon signal features including DPO, LR slope, DMI, MACD and SMA relations."""
        df = self.preprocess_data(df)
        p = df['price']

        # Multi-horizon lags and returns (15m bars)
        lags = [1, 2, 4, 8, 12, 24, 48]
        for lag in lags:
            df[f'price_lag_{lag}'] = p.shift(lag)
            df[f'return_lag_{lag}'] = p.pct_change(lag)

        horizons = {
            '1h': 4,
            '3h': 12,
            '6h': 24,
            '12h': 48
        }

        for name, window in horizons.items():
            sma = p.rolling(window, min_periods=1).mean()
            rolling_min = p.rolling(window, min_periods=1).min()
            rolling_max = p.rolling(window, min_periods=1).max()
            rolling_std = p.rolling(window, min_periods=1).std().fillna(0.0)

            df[f'sma_{name}'] = sma
            df[f'price_vs_sma_{name}'] = (p - sma) / (sma + 1e-8)
            df[f'sma_slope_{name}'] = sma.diff()
            df[f'volatility_{name}'] = rolling_std
            df[f'price_position_{name}'] = (p - rolling_min) / (rolling_max - rolling_min + 1e-8)
            df[f'linreg_slope_{name}'] = p.rolling(window, min_periods=2).apply(self._linear_regression_slope, raw=False)

            # DPO by horizon
            shift_periods = (window // 2) + 1
            df[f'dpo_{name}'] = p - sma.shift(shift_periods)

        # MACD signals
        ema_fast = p.ewm(span=12, adjust=False).mean()
        ema_slow = p.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Directional movement indicators
        plus_di, minus_di, adx, atr = self._dmi(df, period=14)
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx'] = adx
        df['atr'] = atr

        # Signal direction + intensity score
        df['signal_direction'] = np.sign(
            0.30 * df['macd_hist'] +
            0.30 * df['price_vs_sma_3h'] +
            0.20 * (df['plus_di'] - df['minus_di']) +
            0.20 * df['linreg_slope_3h']
        )
        df['signal_intensity'] = (
            df['macd_hist'].abs() * 0.35 +
            df['price_vs_sma_6h'].abs() * 0.25 +
            df['adx'].fillna(0.0) * 0.20 +
            df['dpo_3h'].abs() * 0.20
        )

        # Choppiness index over 12h window
        tr = p.diff().abs().fillna(0.0)
        chop_window = 48
        tr_sum = tr.rolling(chop_window, min_periods=2).sum()
        hh = p.rolling(chop_window, min_periods=2).max()
        ll = p.rolling(chop_window, min_periods=2).min()
        df['choppiness'] = 100 * (np.log10((tr_sum + 1e-8) / (hh - ll + 1e-8)) / np.log10(chop_window))

        # Time features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Stabilize NaN/inf introduced by rolling/shift so windows are not dropped
        df = df.replace([np.inf, -np.inf], np.nan)
        feature_cols = [c for c in df.columns if c not in ['date', 'price']]
        # Forward-fill only within each segment; the caller is responsible for not bridging the train/test boundary.
        df[feature_cols] = df[feature_cols].ffill().fillna(0.0)
        return df

    def prepare_multistep_data(self, df, forecast_horizon=48):
        """Prepare supervised data for direct multi-output forecasting (default 12h = 48 15-min steps)."""
        df = self.create_prediction_features(df)

        for step in range(1, forecast_horizon + 1):
            df[f'target_{step}'] = df['price'].shift(-step)

        feature_cols = [c for c in df.columns if c not in ['date', 'price'] and not c.startswith('target_')]
        target_cols = [f'target_{step}' for step in range(1, forecast_horizon + 1)]

        # Only rows without future targets are removed; feature windows are preserved.
        prepared = df.dropna(subset=target_cols).copy()
        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        # Forward-fill only within each segment; the caller is responsible for not bridging the train/test boundary.
        prepared[feature_cols] = prepared[feature_cols].ffill().fillna(0.0)

        return prepared, feature_cols, target_cols

    def prepare_models(self, df, horizon_hours=HORIZON_HOURS):
        """Reuse fitted forecast models in fast mode; train on a cache miss."""
        import os
        import time
        from pathlib import Path
        from services.model_state import _atomic_write_joblib
        import joblib

        fast = os.environ.get("FAST_PATH_MODEL_STATE", "").lower() in ("1", "true", "yes")
        cache_path = Path(__file__).resolve().parent / "services" / "_state" / "forecast_state.joblib"
        if fast:
            try:
                state = joblib.load(cache_path)
                age = time.time() - state["saved_at"]
                max_age = float(os.environ.get("MAX_STATE_AGE_HOURS", "72")) * 3600
                if (state["schema_version"] == 1 and state["horizon_hours"] == horizon_hours
                        and 0 <= age <= max_age and state["trained_models"]):
                    self.trained_models = state["trained_models"]
                    from services.history_runtime import model_identity
                    self.history_identity = state.get('history_identity') or model_identity(
                        'forecast', state['saved_at'], cached=True)
                    self.history_identity = dict(self.history_identity, cached=True)
                    print(f"   Reusing cached forecast models ({age / 3600:.2f}h old)")
                    return True
                print("   Forecast cache expired or incompatible; training forecast models.")
            except Exception as exc:
                print(f"   Forecast cache unavailable ({exc}); training forecast models.")

        if not self.train_models(df, horizon_hours=horizon_hours):
            return False
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_joblib(cache_path, {
                "schema_version": 1,
                "saved_at": time.time(),
                "horizon_hours": horizon_hours,
                "trained_models": self.trained_models,
                "history_identity": self.history_identity,
            }, compress=3, overwrite=True)
        except Exception as exc:
            print(f"   Forecast cache save skipped: {exc}")
        return True

    def train_models(self, df, horizon_hours=HORIZON_HOURS):
        """Train ensemble models for requested horizon."""
        forecast_horizon = horizon_hours * 4
        print(f"🤖 Training enhanced forecasting models for {horizon_hours}h ({forecast_horizon} steps)...")

        prepared_df, feature_cols, target_cols = self.prepare_multistep_data(df, forecast_horizon=forecast_horizon)

        if len(prepared_df) < 120:
            print("   ⚠️  Insufficient data for stable multi-horizon training")
            return False

        X = prepared_df[feature_cols]
        y = prepared_df[target_cols]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.trained_models = {}
        print(f"   📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Iterate over thunks (zero-arg callables). Invoke each to materialize
        # the actual estimator instance. Thunks allow deferred backend
        # selection (cuML/XGBoost/sklearn) without breaking import if a dep is missing.
        for name, model_factory in self.models.items():
            try:
                print(f"   🔄 Training {name}...", end=' ')
                model = model_factory()  # invoke thunk -> estimator
                model.fit(X_train, y_train)
                backend = EnhancedBitcoinForecaster._detect_estimator_backend(model)
                backend_marker = "GPU" if backend in ('cuML', 'xgboost') else "CPU"
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                self.trained_models[name] = model
                print(f"[{backend_marker}] MSE: {mse:.2f}, MAE: {mae:.2f}")
            except Exception as e:
                print(f"Failed: {e}")

        print(f"   ✅ Successfully trained {len(self.trained_models)} models")
        self.last_update = datetime.now(timezone.utc)
        from services.history_runtime import model_identity
        # Last training label is the furthest target of the last training row.
        cutoff = pd.to_datetime(prepared_df.iloc[len(X_train) - 1]['date'], utc=True) + timedelta(minutes=15 * (forecast_horizon + 1))
        self.history_identity = model_identity('forecast', self.last_update, cutoff)
        return len(self.trained_models) > 0

    def generate_forecast(self, latest_data, horizon_hours=HORIZON_HOURS):
        """Generate direct 12-hour (or requested horizon) forecast with 15-minute windows."""
        if not self.trained_models:
            print("   ❌ No trained models available")
            return None

        steps = horizon_hours * 4
        print(f"🔮 Generating {horizon_hours}-hour forecast ({steps} steps)...")

        latest_data = self._drop_normalized_columns(latest_data)
        df_feat = self.create_prediction_features(latest_data)
        feature_cols = [c for c in df_feat.columns if c not in ['date', 'price']]

        X_latest = df_feat[feature_cols].tail(1)
        ensemble_predictions = []

        for name, model in self.trained_models.items():
            try:
                pred_path = model.predict(X_latest)[0][:steps]
                ensemble_predictions.append(pred_path)
                print(f"   ✅ {name}: Generated {steps} predictions")
            except Exception as e:
                print(f"   ❌ {name}: Failed - {e}")

        if not ensemble_predictions:
            print("   ❌ No successful predictions generated")
            return None

        ensemble_avg = np.mean(ensemble_predictions, axis=0)
        ensemble_std = np.std(ensemble_predictions, axis=0) if len(ensemble_predictions) > 1 else np.zeros(steps)
        current_price = float(pd.to_numeric(df_feat['price'], errors='coerce').dropna().iloc[-1])
        self.forecast_reference_price = current_price
        raw_first_prediction = float(ensemble_avg[0]) if len(ensemble_avg) > 0 else current_price
        baseline_gap_pct = 0.0
        baseline_adjusted = False
        baseline_scale = 1.0

        # Re-anchor forecast levels when the first-step prediction drifts too far from the latest price.
        if current_price > 0 and np.isfinite(raw_first_prediction) and raw_first_prediction > 0:
            baseline_gap_pct = ((raw_first_prediction - current_price) / current_price) * 100.0
            if abs(baseline_gap_pct) > 2.0:
                baseline_scale = current_price / raw_first_prediction
                ensemble_avg = ensemble_avg * baseline_scale
                ensemble_std = ensemble_std * abs(baseline_scale)
                baseline_adjusted = True
                print(f"   Applied baseline alignment ({baseline_gap_pct:+.2f}% first-step gap)")

        start_time = pd.to_datetime(df_feat['date'].iloc[-1], utc=True) + timedelta(minutes=15)
        timestamps = [start_time + timedelta(minutes=15 * i) for i in range(steps)]

        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_price': ensemble_avg,
            'prediction_std': ensemble_std,
            'interval_minutes': [15 * (i + 1) for i in range(steps)],
            'baseline_adjusted': baseline_adjusted,
            'baseline_gap_pct': baseline_gap_pct,
            'baseline_scale': baseline_scale
        })
        return forecast_df

    def calculate_forecast_statistics(self, forecast_df, current_price, horizon_hours=HORIZON_HOURS):
        """Compute required summary stats including choppiness and volatility."""
        prices = forecast_df['predicted_price']
        returns = prices.pct_change().fillna(0.0)

        mean_price = float(prices.mean())
        raw_min = float(prices.min())
        raw_max = float(prices.max())
        inclusive_min = float(min(raw_min, current_price))
        inclusive_max = float(max(raw_max, current_price))

        # Adaptive BB-percentile bias: derive upper/lower bounds from the predicted
        # price series itself, using a blend of Bollinger Bands (mean +/- k*std)
        # and percentile bands (e.g. 95th / 5th). This makes the predicted range
        # adapt to the model-forecast volatility instead of a hardcoded fraction
        # of the directional move.
        #
        #   forecast_upper = alpha * bb_upper + (1 - alpha) * p95
        #   forecast_lower = alpha * bb_lower + (1 - alpha) * p05
        #
        # For a bullish forecast we extend max up to forecast_upper (BB-percentile
        # ceiling) and allow min to dip down to forecast_lower (handle-bar wick).
        # Bearish is symmetric. BB k=1.5σ + percentile blend = adaptive tolerance.
        BB_K = 1.5               # Bollinger Band sigma multiplier (~87% coverage)
        PCT_BLEND = 0.6          # weight on BB vs percentile (1-PCT_BLEND for pct)
        bullish = mean_price > current_price
        # Defensive std/percentile computation (guard against degenerate series)
        price_std = float(prices.std()) if len(prices) > 1 else 0.0
        try:
            p05 = float(np.percentile(prices, 5))
            p95 = float(np.percentile(prices, 95))
        except Exception:
            p05, p95 = raw_min, raw_max
        bb_upper = mean_price + BB_K * price_std
        bb_lower = mean_price - BB_K * price_std
        # Blend BB (smooth) with percentile (robust to tails)
        adaptive_upper = PCT_BLEND * bb_upper + (1.0 - PCT_BLEND) * p95
        adaptive_lower = PCT_BLEND * bb_lower + (1.0 - PCT_BLEND) * p05
        if bullish:
            # Upside bias: extend max up to the adaptive (BB-percentile) ceiling.
            predicted_max = max(raw_max, adaptive_upper)
            # Allow a handle-bar dip down to the adaptive floor (BB-percentile).
            predicted_min = min(raw_min, adaptive_lower)
        else:
            # Downside bias: extend min down to the adaptive (BB-percentile) floor.
            predicted_min = min(raw_min, adaptive_lower)
            # Allow a handle-bar spike up to the adaptive ceiling (BB-percentile).
            predicted_max = max(raw_max, adaptive_upper)
        stats = {
            'current_price': float(current_price),
            'horizon_hours': int(horizon_hours),
            'forecast_min': inclusive_min,
            'forecast_max': inclusive_max,
            'predicted_min': predicted_min,
            'predicted_max': predicted_max,
            'forecast_mean': mean_price,
            'forecast_median': float(prices.median()),
            'forecast_average': mean_price,
            'forecast_volatility': float(returns.std()),
            'forecast_std': float(prices.std()),
            'forecast_range_includes_current': bool(inclusive_min <= float(current_price) <= inclusive_max)
        }

        tr = prices.diff().abs().fillna(0.0)
        tr_sum = tr.sum()
        price_span = max(float(prices.max() - prices.min()), 1e-8)
        n = len(prices) if len(prices) > 1 else 2
        stats['forecast_choppiness'] = float(100 * np.log10((tr_sum + 1e-8) / price_span) / np.log10(n))

        return stats

    # ----- GPU dispatch helpers (static methods so lambdas inside self.models can call them by name) -----
    @staticmethod
    def _gpu_rf_reg(**kwargs):
        """Return a RandomForestRegressor, preferring cuML when CUDA is available."""
        if CUDA_AVAILABLE and _cuml_RF_reg is not None:
            gpu_kw = {k: v for k, v in kwargs.items() if k not in _CUM_RF_REG_UNSUPPORTED}
            try:
                return _cuml_RF_reg(**gpu_kw)
            except TypeError as exc:
                print(f"   ⚠  cuML RF regressor rejected kwargs ({exc}); falling back to sklearn.")
        return RandomForestRegressor(**kwargs)

    @staticmethod
    def _gpu_gb_reg(**kwargs):
        """Return a GradientBoostingRegressor, preferring cuML when CUDA is available."""
        if CUDA_AVAILABLE and _cuml_GB_reg is not None:
            gpu_kw = {k: v for k, v in kwargs.items() if k not in _CUM_GB_REG_UNSUPPORTED}
            try:
                return _cuml_GB_reg(**gpu_kw)
            except TypeError as exc:
                print(f"   ⚠  cuML GB regressor rejected kwargs ({exc}); falling back to sklearn.")
        return GradientBoostingRegressor(**kwargs)

    @staticmethod
    def _gpu_scaler_reg(name):
        """Return a scaler, preferring cuML GPU variants when CUDA is available."""
        if CUDA_AVAILABLE:
            if name == "StandardScaler" and _cuml_StandardScaler is not None:
                return _cuml_StandardScaler()
            if name == "MinMaxScaler" and _cuml_MinMaxScaler is not None:
                return _cuml_MinMaxScaler()
            if name == "RobustScaler" and _cuml_RobustScaler is not None:
                return _cuml_RobustScaler()
        if name == "StandardScaler":
            return StandardScaler()
        if name == "MinMaxScaler":
            return MinMaxScaler()
        if name == "RobustScaler":
            return RobustScaler()
        raise ValueError(f"Unknown scaler name: {name}")

    @staticmethod
    def _make_xgb_regressor():
        """Return an XGBRegressor, GPU-accelerated when possible."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed; pip install xgboost")
        try:
            if CUDA_AVAILABLE:
                return xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    tree_method="hist", device="cuda",
                    random_state=42, n_jobs=-1,
                )
        except Exception:
            pass
        return xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            tree_method="hist",
            random_state=42, n_jobs=-1,
        )

    @staticmethod
    def _detect_estimator_backend(estimator):
        """Identify the prediction backend, looking through sklearn wrappers.

        The xgboost marker denotes CUDA execution; CPU XGBoost uses the CPU
        marker (sklearn) to preserve the existing backend-label interface.
        """
        if isinstance(estimator, Pipeline):
            return EnhancedBitcoinForecaster._detect_estimator_backend(estimator.steps[-1][1])
        if isinstance(estimator, MultiOutputRegressor):
            fitted = getattr(estimator, 'estimators_', None)
            inner = fitted[0] if fitted else estimator.estimator
            return EnhancedBitcoinForecaster._detect_estimator_backend(inner)
        mod = type(estimator).__module__ or ""
        if mod.startswith("cuml"):
            return "cuML"
        if mod.startswith("xgboost"):
            params = estimator.get_params()
            device = str(params.get('device') or 'cpu')
            # A fitted booster may have fallen back to CPU despite device=cuda.
            try:
                import json
                config = json.loads(estimator.get_booster().save_config())
                device = config['learner']['generic_param']['device']
            except (AttributeError, KeyError, ValueError):
                pass
            gpu = device.startswith(('cuda', 'gpu')) or (
                device == 'cpu' and params.get('device') is None
                and params.get('tree_method') == 'gpu_hist'
            )
            return "xgboost" if gpu else "sklearn"
        if mod.startswith("sklearn"):
            return "sklearn"
        return mod or "unknown"


def run_enhanced_forecasting(df_combined):
    """Main function to run revamped forecasting with 12-hour output at 15-minute windows."""
    print("\n🚀 STARTING ENHANCED FORECASTING SYSTEM")
    print("=" * 80)

    forecaster = EnhancedBitcoinForecaster()

    success = forecaster.train_models(df_combined, horizon_hours=HORIZON_HOURS)
    if not success:
        print("❌ Failed to train models")
        return None

    forecast_12h = forecaster.generate_forecast(df_combined, horizon_hours=HORIZON_HOURS)
    if forecast_12h is None:
        print("❌ Failed to generate 12-hour forecast")
        return None

    current_price = pd.to_numeric(df_combined['price'], errors='coerce').dropna().iloc[-1]
    stats_12h = forecaster.calculate_forecast_statistics(forecast_12h, current_price, horizon_hours=HORIZON_HOURS)

    return {
        'forecast_12h': forecast_12h,
        'stats_12h': stats_12h,
        'forecaster': forecaster
    }


def _fetch_btc_data(hours=72, samples=None):
    """Fetch recent BTC-USD 15-minute candles from Coinbase API. Falls back to synthetic data."""
    total_points = int(samples if samples is not None else hours * 4)
    if total_points <= 0:
        raise ValueError("Data sample count must be positive")

    try:
        import requests
        import time as _time
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        granularity = 900  # 15-minute bars
        points_per_request = 300
        all_data = []
        end_time = datetime.utcnow()

        request_count = max(1, -(-total_points // points_per_request))
        for _ in range(request_count):
            start_time = end_time - timedelta(seconds=granularity * points_per_request)
            params = {
                "granularity": granularity,
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            end_time = datetime.fromtimestamp(data[-1][0])
            _time.sleep(0.3)

        if not all_data:
            raise ValueError("No data received from API")

        candles = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
        candles = candles.sort_values("time").drop_duplicates(subset=["time"])
        candles["date"] = pd.to_datetime(candles["time"], unit="s", utc=True)
        candles["price"] = pd.to_numeric(candles["close"], errors="coerce")
        df = candles[["date", "price"]].dropna().tail(total_points)
        print(f"📡 Fetched {len(df)} BTC-USD 15-minute candles from Coinbase API")
        return df

    except Exception as e:
        print(f"⚠️  API fetch failed ({e}). Using synthetic data for demonstration.")
        periods = total_points
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq="15min", tz="UTC")
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(periods) * 150) + 65000
        prices = np.clip(prices, 1000, None)
        return pd.DataFrame({"date": dates, "price": prices})


if __name__ == "__main__":
    print("\n🚀 ENHANCED BITCOIN FORECASTING SYSTEM")
    print("=" * 80)

    data_samples = 12000
    df_input = _fetch_btc_data(samples=data_samples)

    results = run_enhanced_forecasting(df_input)

    if results is None:
        print("\n❌ Forecasting failed. Exiting.")
    else:
        forecast_12h = results["forecast_12h"]
        stats = results["stats_12h"]

        print("\n" + "=" * 80)
        print("📊 12-HOUR PRICE FORECAST (15-minute intervals)")
        print("=" * 80)
        display_df = forecast_12h.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        display_df["predicted_price"] = display_df["predicted_price"].map("${:,.2f}".format)
        display_df["prediction_std"] = display_df["prediction_std"].map("±${:,.2f}".format)
        print(display_df[["timestamp", "predicted_price", "prediction_std", "interval_minutes"]].to_string(index=False))

        print("\n" + "=" * 80)
        print("📈 FORECAST SUMMARY STATISTICS")
        print("=" * 80)
        current = stats["current_price"]
        forecast_mean = stats["forecast_mean"]
        change_pct = (forecast_mean - current) / current * 100

        print(f"   💰 Current Price:        ${current:,.2f}")
        print(f"   📊 Forecast Mean:        ${forecast_mean:,.2f}  ({change_pct:+.2f}%)")
        print(f"   📉 Predicted Min:        ${stats['predicted_min']:,.2f}")
        print(f"   📈 Predicted Max:        ${stats['predicted_max']:,.2f}")
        print(f"   📍 Range w/ Current:     ${stats['forecast_min']:,.2f} to ${stats['forecast_max']:,.2f}")
        print(f"   📐 Forecast Median:      ${stats['forecast_median']:,.2f}")
        print(f"   📏 Forecast Std Dev:     ${stats['forecast_std']:,.2f}")
        print(f"   🌊 Forecast Volatility:  {stats['forecast_volatility']:.4f}")
        print(f"   🔀 Choppiness Index:     {stats['forecast_choppiness']:.2f}")
        print(f"   ⏱️  Horizon:              {stats['horizon_hours']} hours")
        print("=" * 80)
