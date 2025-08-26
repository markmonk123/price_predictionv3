import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def compute_adx(df: pd.DataFrame, period: int = 14):
    """Compute ADX, DI+, and DI- indicators."""
    high = df['high']
    low = df['low']
    close = df['close']
    up_move = high.diff().to_numpy()
    down_move = (-low.diff()).to_numpy()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=1).sum() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=1).sum() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=1).mean()
    return plus_di, minus_di, adx
from sklearn.metrics import mean_absolute_percentage_error


def fetch_btc_data(num_points: int = 26000, interval_minutes: int = 5) -> pd.DataFrame:
    """Fetch historical BTC-USD candles using Coinbase public API (5-minute intervals).

    Returns a DataFrame with columns: date, open, high, low, close.
    Falls back to synthetic data on failure.
    """
    all_data = []
    points_per_request = 300  # Coinbase API limit per request
    num_requests = (num_points + points_per_request - 1) // points_per_request

    # Coinbase API expects ISO8601 times and granularity in seconds
    granularity = interval_minutes * 60
    end_time = datetime.utcnow()
    end_time -= timedelta(
        minutes=end_time.minute % interval_minutes,
        seconds=end_time.second,
        microseconds=end_time.microsecond,
    )
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

    try:
        for _ in range(num_requests):
            start_time = end_time - timedelta(minutes=points_per_request * interval_minutes)
            params = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "granularity": granularity,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            # Coinbase returns [time, low, high, open, close, volume]
            oldest = data[-1][0]
            end_time = datetime.fromtimestamp(oldest)
            time.sleep(0.2)
        if not all_data:
            raise ValueError("no data fetched")
        df = pd.DataFrame(all_data, columns=["start", "low", "high", "open", "close", "volume"])
        df = df.drop_duplicates("start").sort_values("start")
        df["date"] = pd.to_datetime(df["start"].astype(int), unit="s")
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        return df[["date", "open", "high", "low", "close"]].reset_index(drop=True)
    except Exception:
        # fallback synthetic data
        dates = pd.date_range(
            end=datetime.utcnow(), periods=num_points, freq=f"{interval_minutes}T"
        )
        prices = np.cumsum(np.random.randn(num_points)) + 60000
        df = pd.DataFrame(
            {"date": dates, "open": prices, "high": prices, "low": prices, "close": prices}
        )
        return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD, linear regression slope, Detrended Price Oscillator, ADX, DI+, DI-."""
    df = df.copy()
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    def lr_slope(x):
        x_idx = np.arange(len(x))
        slope, _, _, _, _ = stats.linregress(x_idx, x)
        return slope

    df["slope"] = close.rolling(window=20).apply(lr_slope, raw=True)

    period = 20
    sma = close.rolling(window=period).mean()
    shift = period // 2 + 1
    df["dpo"] = close - sma.shift(shift)

    # Add ADX, DI+, DI-
    plus_di, minus_di, adx = compute_adx(df, period=14)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx

    return df.dropna().reset_index(drop=True)


def classify_market(df: pd.DataFrame) -> str:
    """Classify the current market state as bullish, bearish, or sideways using MACD, slope, DPO, ADX, DI+ and DI-."""
    latest = df.iloc[-1]
    strong_trend = latest["adx"] > 25
    bullish = (
        latest["macd"] > latest["macd_signal"]
        and latest["slope"] > 0
        and latest["dpo"] > 0
        and latest["plus_di"] > latest["minus_di"]
        and strong_trend
    )
    bearish = (
        latest["macd"] < latest["macd_signal"]
        and latest["slope"] < 0
        and latest["dpo"] < 0
        and latest["minus_di"] > latest["plus_di"]
        and strong_trend
    )
    if bullish:
        return "bullish market (strong ADX)"
    if bearish:
        return "bearish market (strong ADX)"
    return "sideways/weak trend market"


def train_and_predict(df: pd.DataFrame):
    """Train and compare multiple models, ensemble predictions for price, high, and low 12 hours ahead."""
    horizon = 48  # 12 hours with 5-min intervals
    df["future_price"] = df["close"].shift(-horizon)
    df["future_high"] = df["high"].rolling(window=horizon).max().shift(-horizon + 1)
    df["future_low"] = df["low"].rolling(window=horizon).min().shift(-horizon + 1)
    df = df.dropna().reset_index(drop=True)

    features = ["macd", "macd_signal", "macd_hist", "slope", "dpo"]
    X = df[features]
    y_price = df["future_price"]
    y_high = df["future_high"]
    y_low = df["future_low"]

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_price_train, y_price_test = y_price[:split], y_price[split:]
    y_high_train, y_high_test = y_high[:split], y_high[split:]
    y_low_train, y_low_test = y_low[:split], y_low[split:]

    regressors = [
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ("LinearRegression", LinearRegression()),
    ]

    results = {}
    latest_features = X.iloc[[-1]]

    for target, y_train, y_test in [
        ("price", y_price_train, y_price_test),
        ("high", y_high_train, y_high_test),
        ("low", y_low_train, y_low_test),
    ]:
        preds = []
        test_preds = []
        scores = []
        for name, reg in regressors:
            reg.fit(X_train, y_train)
            pred = reg.predict(latest_features)[0]
            preds.append(pred)
            test_pred = reg.predict(X_test)
            test_preds.append(test_pred)
            mape = mean_absolute_percentage_error(y_test, test_pred) * 100
            scores.append((name, 100 - mape))
        # Ensemble: average predictions
        ensemble_pred = np.mean(preds)
        # Best model by accuracy
        best_model_idx = np.argmax([score for _, score in scores])
        best_pred = preds[best_model_idx]
        results[target] = {
            "ensemble": ensemble_pred,
            "best_model": regressors[best_model_idx][0],
            "best_pred": best_pred,
            "accuracies": dict(scores),
        }

    # Return ensemble predictions and best model info for price, high, low
    return (
        results["price"]["ensemble"],
        results["high"]["ensemble"],
        results["low"]["ensemble"],
        results["price"]["accuracies"],
        results["high"]["accuracies"],
        results["low"]["accuracies"],
        results["price"]["best_model"],
        results["high"]["best_model"],
        results["low"]["best_model"],
    )


def main():
    df = fetch_btc_data()
    df = compute_indicators(df)
    market = classify_market(df)
    (
        pred_price, pred_high, pred_low,
        price_accuracies, high_accuracies, low_accuracies,
        price_best, high_best, low_best
    ) = train_and_predict(df)
    print(f"Market state: {market}")
    print(f"Predicted price in 12h: ${pred_price:.2f} (ensemble)")
    print(f"Expected high: ${pred_high:.2f} | Expected low: ${pred_low:.2f} (ensemble)")
    print(f"Best model for price: {price_best} | Accuracies: {price_accuracies}")
    print(f"Best model for high: {high_best} | Accuracies: {high_accuracies}")
    print(f"Best model for low: {low_best} | Accuracies: {low_accuracies}")


if __name__ == "__main__":
    main()
