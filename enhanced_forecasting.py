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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EnhancedBitcoinForecaster:
    """Enhanced forecasting system with multi-horizon signal detection and 12-hour forecasts."""

    def __init__(self, data_update_interval=300):
        self.data_update_interval = data_update_interval
        self.models = {}
        self.trained_models = {}
        self.last_update = None

        # Ensemble base models
        self.models = {
            'RandomForest': MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=250, max_depth=16, min_samples_split=4,
                    random_state=42, n_jobs=-1
                )
            ),
            'GradientBoosting': MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=5,
                    random_state=42
                )
            ),
            'ExtraTrees': MultiOutputRegressor(
                ExtraTreesRegressor(
                    n_estimators=250, max_depth=14, min_samples_split=3,
                    random_state=42, n_jobs=-1
                )
            ),
            'Pipeline_RF': Pipeline([
                ('scaler', StandardScaler()),
                ('rf', MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=7, n_jobs=-1)))
            ])
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

        # Snap to 15-minute windows and preserve all windows via interpolation/fill.
        out = out.set_index('date').resample('15min').last()
        out['price'] = out['price'].interpolate(method='time').ffill().bfill()
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
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
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
        prepared[feature_cols] = prepared[feature_cols].ffill().bfill().fillna(0.0)

        return prepared, feature_cols, target_cols

    def train_models(self, df, horizon_hours=12):
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

        for name, model in self.models.items():
            try:
                print(f"   🔄 Training {name}...", end=' ')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                self.trained_models[name] = model
                print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")
            except Exception as e:
                print(f"Failed: {e}")

        print(f"   ✅ Successfully trained {len(self.trained_models)} models")
        self.last_update = datetime.now()
        return len(self.trained_models) > 0

    def generate_forecast(self, latest_data, horizon_hours=12):
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

    def calculate_forecast_statistics(self, forecast_df, current_price, horizon_hours=12):
        """Compute required summary stats including choppiness and volatility."""
        prices = forecast_df['predicted_price']
        returns = prices.pct_change().fillna(0.0)

        mean_price = float(prices.mean())
        predicted_min = float(prices.min())
        predicted_max = float(prices.max())
        inclusive_min = float(min(predicted_min, current_price))
        inclusive_max = float(max(predicted_max, current_price))
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


def run_enhanced_forecasting(df_combined):
    """Main function to run revamped forecasting with 12-hour output at 15-minute windows."""
    print("\n🚀 STARTING ENHANCED FORECASTING SYSTEM")
    print("=" * 80)

    forecaster = EnhancedBitcoinForecaster()

    success = forecaster.train_models(df_combined, horizon_hours=12)
    if not success:
        print("❌ Failed to train models")
        return None

    forecast_12h = forecaster.generate_forecast(df_combined, horizon_hours=12)
    if forecast_12h is None:
        print("❌ Failed to generate 12-hour forecast")
        return None

    current_price = pd.to_numeric(df_combined['price'], errors='coerce').dropna().iloc[-1]
    stats_12h = forecaster.calculate_forecast_statistics(forecast_12h, current_price, horizon_hours=12)

    return {
        'forecast_12h': forecast_12h,
        'stats_12h': stats_12h,
        'forecaster': forecaster
    }


def _fetch_btc_data(hours=72):
    """Fetch recent BTC-USD 15-minute candles from Coinbase API. Falls back to synthetic data."""
    try:
        import requests
        import time as _time
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        granularity = 900  # 15-minute bars
        points_per_request = 300
        total_points = hours * 4  # 4 bars per hour
        all_data = []
        end_time = datetime.utcnow()

        for _ in range(max(1, total_points // points_per_request)):
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
        df = candles[["date", "price"]].dropna()
        print(f"📡 Fetched {len(df)} BTC-USD 15-minute candles from Coinbase API")
        return df

    except Exception as e:
        print(f"⚠️  API fetch failed ({e}). Using synthetic data for demonstration.")
        periods = hours * 4
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq="15min", tz="UTC")
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(periods) * 150) + 65000
        prices = np.clip(prices, 1000, None)
        return pd.DataFrame({"date": dates, "price": prices})


if __name__ == "__main__":
    print("\n🚀 ENHANCED BITCOIN FORECASTING SYSTEM")
    print("=" * 80)

    df_input = _fetch_btc_data(hours=72)

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
