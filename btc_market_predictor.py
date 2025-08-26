import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


def fetch_btc_data(num_points: int = 26000, interval_minutes: int = 15) -> pd.DataFrame:
    """Fetch historical BTC-USD candles using Coinbase public API.

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
    """Compute MACD, linear regression slope, and Detrended Price Oscillator."""
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

    return df.dropna().reset_index(drop=True)


def classify_market(df: pd.DataFrame) -> str:
    """Classify the current market state as bullish or bearish."""
    latest = df.iloc[-1]
    bullish = (
        latest["macd"] > latest["macd_signal"]
        and latest["slope"] > 0
        and latest["dpo"] > 0
    )
    bearish = (
        latest["macd"] < latest["macd_signal"]
        and latest["slope"] < 0
        and latest["dpo"] < 0
    )
    if bullish:
        return "bullish market"
    if bearish:
        return "bearish market"
    return "sideways market"


def train_and_predict(df: pd.DataFrame):
    """Train models and predict price, high and low 12 hours ahead."""
    horizon = 48  # 12 hours with 15-min intervals
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

    model_price = RandomForestRegressor(n_estimators=200, random_state=42)
    model_price.fit(X_train, y_price_train)
    price_pred_test = model_price.predict(X_test)
    mape = mean_absolute_percentage_error(y_price_test, price_pred_test) * 100
    accuracy = 100 - mape

    model_high = RandomForestRegressor(n_estimators=200, random_state=42)
    model_low = RandomForestRegressor(n_estimators=200, random_state=42)
    model_high.fit(X_train, y_high[:split])
    model_low.fit(X_train, y_low[:split])

    latest_features = X.iloc[[-1]]
    pred_price = model_price.predict(latest_features)[0]
    pred_high = model_high.predict(latest_features)[0]
    pred_low = model_low.predict(latest_features)[0]

    return pred_price, pred_high, pred_low, accuracy


def main():
    df = fetch_btc_data()
    df = compute_indicators(df)
    market = classify_market(df)
    pred_price, pred_high, pred_low, accuracy = train_and_predict(df)
    print(f"Market state: {market}")
    print(
        f"Predicted price in 12h: ${pred_price:.2f} (accuracy: {accuracy:.2f}%)\n"
        f"Expected high: ${pred_high:.2f} | Expected low: ${pred_low:.2f}"
    )


if __name__ == "__main__":
    main()
