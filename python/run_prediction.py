#!/usr/bin/env python

"""
Bitcoin Prediction Model Runner
Runs the enhanced prediction model from Python and returns results as JSON
"""

import argparse
import contextlib
import json
import sys
import os
import datetime
from dateutil.parser import parse

# Add parent directory to path to import from enhanced_prediction.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import numpy as np
    import pandas as pd
    import requests
    from enhanced_prediction import create_enhanced_features, horizon_to_steps, main
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    print(json.dumps({"error": f"Import error: {str(e)}", "errorType": "ImportError"}))
    sys.exit(1)

try:
    from config.prediction_spec import (
        HORIZON_MINUTES,
        PCT_THRESHOLD,
        BAR_MINUTES,
        NUM_DATA_POINTS,
        TIMEFRAME_LABEL,
        spec_summary,
    )
except Exception as _spec_err:
    print(json.dumps({"error": f"prediction_spec import failed: {_spec_err}", "errorType": "ImportError"}))
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Bitcoin prediction model')
    parser.add_argument('--price', type=float, required=True, help='Current Bitcoin price')
    parser.add_argument('--volume', type=float, required=False, default=100.0, help='Current trading volume')
    parser.add_argument('--time', type=str, required=False, default=None, help='Current timestamp (ISO format)')
    parser.add_argument('--window', type=int, required=False, default=NUM_DATA_POINTS,
                        help='Historical candle count to fetch (5-minute bars)')
    parser.add_argument('--allow-synthetic', action='store_true',
                        help='Allow synthetic fallback if live market history fetch fails')
    return parser.parse_args()


def to_naive_utc(timestamp):
    """Normalize None / ISO string / naive / aware datetimes to naive UTC.

    Node passes `new Date().toISOString()` (aware, 'Z'), while candle times are
    built from epoch seconds; mixing aware and naive datetimes raises TypeError
    on comparison, so everything is kept as naive UTC internally.
    """
    if timestamp is None:
        return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    if isinstance(timestamp, str):
        timestamp = parse(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return timestamp


def utc_from_epoch(seconds):
    """Epoch seconds -> naive UTC datetime."""
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).replace(tzinfo=None)


def fetch_recent_market_data(current_price, volume, timestamp, window_size=NUM_DATA_POINTS):
    """
    Fetch recent BTC-USD candles from Coinbase at the spec bar size.

    Coinbase caps every /candles response at 300 rows, so reaching the full
    window requires paginating backwards in 300-bar pages until we either hit
    the requested window size or the API runs out of older data.
    """
    timestamp = to_naive_utc(timestamp)

    # Coinbase candle endpoint returns [time, low, high, open, close, volume]
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    points_per_request = 300
    num_requests = (window_size + points_per_request - 1) // points_per_request
    end_time = timestamp
    all_candles = []

    for _ in range(num_requests):
        start_time = end_time - datetime.timedelta(
            minutes=points_per_request * BAR_MINUTES)

        params = {
            "granularity": BAR_MINUTES * 60,
            "start": start_time.isoformat() + "Z",
            "end": end_time.isoformat() + "Z"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        page = response.json()

        if not page:
            break

        all_candles.extend(page)

        # Coinbase returns each page newest-first, so the LAST row is the
        # oldest candle on the page: step the window back past it.
        oldest_timestamp = min(c[0] for c in page)
        next_end = utc_from_epoch(oldest_timestamp)
        if next_end >= end_time:
            # No forward progress (API clamped the range); stop to avoid looping.
            break
        end_time = next_end - datetime.timedelta(seconds=1)

        if len(all_candles) >= window_size:
            break

    if not all_candles:
        raise ValueError("No candles returned from Coinbase API")

    # De-duplicate overlapping boundary candles, then order oldest -> newest.
    by_time = {c[0]: c for c in all_candles}
    candles = sorted(by_time.values(), key=lambda x: x[0])[-window_size:]

    if len(candles) < max(10, window_size // 3):
        raise ValueError(f"Insufficient candle history returned ({len(candles)} rows)")

    # Untouched candles (with high/low) for scoring earlier predictions: the
    # ledger must never see the live ticker price spliced into a bar's close.
    candles_df = pd.DataFrame({
        "date": [utc_from_epoch(c[0]) for c in candles],
        "low": [float(c[1]) for c in candles],
        "high": [float(c[2]) for c in candles],
        "price": [float(c[4]) for c in candles],   # close
    })

    df = pd.DataFrame({
        "date": candles_df["date"],
        "price": candles_df["price"].copy(),
        "volume": [float(c[5]) for c in candles]
    })

    # Use latest values from the upstream market-data source for consistency.
    df.loc[df.index[-1], "price"] = float(current_price)
    df.loc[df.index[-1], "volume"] = float(volume)

    return df, candles_df

def generate_synthetic_data(current_price, volume, timestamp, window_size=NUM_DATA_POINTS):
    """
    Generate synthetic price data to feed into the model
    """
    timestamp = to_naive_utc(timestamp)

    # Generate timestamps going back from current time at the spec bar size
    timestamps = [timestamp - datetime.timedelta(minutes=i * BAR_MINUTES) for i in range(window_size)]
    timestamps.reverse()  # Oldest first

    # Generate price data with some randomness but trending to current price
    price_volatility = current_price * 0.001  # 0.1% volatility per step
    prices = []
    price = current_price * 0.98  # Start 2% below current price

    for i in range(window_size):
        # Random walk with drift toward current price
        drift = (current_price - price) * 0.1
        random_step = np.random.normal(0, price_volatility)
        price = price + drift + random_step
        prices.append(price)

    # Generate volume data
    volumes = [volume * (0.5 + np.random.random()) for _ in range(window_size)]

    # Create DataFrame
    df = pd.DataFrame({
        'date': timestamps,
        'price': prices,
        'volume': volumes
    })

    return df

MODEL_VERSION = "run_prediction-rf-v1"
SIGNAL = "live_rf"


def history_disabled():
    return os.environ.get("PREDICTION_HISTORY", "").strip().lower() in ("0", "false", "no", "off")


def record_prediction(*, candles_df, generated_at, reference_price, direction, confidence,
                      training_cutoff, runtime_factory=None):
    """Write this run's direction call to the durable ledger and score older ones.

    Best effort: a ledger failure never fails the prediction, it is reported in
    the returned dict instead. The prediction is made from a live ticker price
    at `generated_at`, usually mid-candle, so the target is the candle close
    HORIZON_MINUTES after the current candle closes (aligned to 5-minute bars,
    which the ledger requires for exact scoring).
    """
    if history_disabled():
        return {"recorded": False, "reason": "PREDICTION_HISTORY disabled"}
    from services.history_runtime import HistoryRuntime, model_identity
    from services.prediction_history import epoch, iso

    now = epoch(generated_at)
    bar_seconds = BAR_MINUTES * 60
    current_bar_close = -(-now // bar_seconds) * bar_seconds
    target_end = current_bar_close + int(HORIZON_MINUTES) * 60

    factory = runtime_factory or HistoryRuntime
    runtime = factory("python/run_prediction.py", candles_df, now=now,
                      reference_price=reference_price, data_cutoff=now)
    if runtime.history is None:
        return {"recorded": False, "reason": "prediction ledger unavailable (see stderr)"}
    try:
        identity = model_identity("run_prediction:rf", now, training_cutoff, version=MODEL_VERSION)
        runtime.history.record(
            runtime.run_id, kind="direction", signal=SIGNAL,
            target_start=target_end - bar_seconds, target_end=target_end,
            horizon_minutes=int(HORIZON_MINUTES), reference_price=float(reference_price),
            predicted_direction=int(direction), confidence=float(confidence),
            threshold=float(PCT_THRESHOLD), generated_at=now, **identity)
        return {"recorded": True, "run_id": runtime.run_id, "target_end": iso(target_end),
                "newly_scored": runtime.last_scored, "expired": runtime.last_expired}
    except Exception as exc:
        return {"recorded": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        runtime.finish()


def run_prediction(price, volume, timestamp=None, window_size=NUM_DATA_POINTS, allow_synthetic=False):
    """
    Run the Bitcoin prediction model and return results
    """
    try:
        data_source = "live"
        candles_df = None
        try:
            df, candles_df = fetch_recent_market_data(price, volume, timestamp, window_size)
        except Exception:
            if not allow_synthetic:
                raise
            # Explicit demo fallback
            df = generate_synthetic_data(price, volume, timestamp, window_size)
            data_source = "synthetic"

        # Create enhanced features. Keep the trailing rows whose future price is
        # not known yet: the last one carries the live price we predict from.
        enhanced_df = create_enhanced_features(
            df,
            pct_threshold=PCT_THRESHOLD,
            horizon_steps=horizon_to_steps(HORIZON_MINUTES),
            include_unlabeled=True,
        )
        if enhanced_df.empty or enhanced_df['date'].iloc[-1] != df['date'].iloc[-1]:
            raise ValueError("Latest candle does not have valid inference features")
        labeled_df = enhanced_df.dropna(subset=['future_price', 'next_return', 'target'])

        # Select features (excluding target variables and non-predictive columns)
        feature_cols = [col for col in enhanced_df.columns
                       if col not in ['date', 'price', 'future_price', 'next_return', 'target']]
        X = enhanced_df[feature_cols].iloc[-1:]

        # Create a simple random forest classifier
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )

        # Fit only on rows whose outcome is already known
        y_train = labeled_df['target'].astype(int)
        X_train = labeled_df[feature_cols]
        model.fit(X_train, y_train)

        # Get prediction probabilities for the current price
        probs = model.predict_proba(X)[0]

        # Get prediction class
        prediction = model.predict(X)[0]

        # Map probabilities to classes (-1, 0, 1) via model.classes_ so that
        # classes absent from the training window (e.g. only {-1, 1} observed)
        # don't get mislabeled by positional indexing. probs[i] corresponds
        # to model.classes_[i], NOT to a fixed {-1, 0, 1} slot.
        class_to_prob = dict(zip(model.classes_, probs))
        decrease_prob = float(class_to_prob.get(-1, 0.0))
        no_change_prob = float(class_to_prob.get(0, 0.0))
        increase_prob = float(class_to_prob.get(1, 0.0))

        # Return result as JSON
        result = {
            "price": float(price),
            "timestamp": to_naive_utc(timestamp).isoformat() + "Z",
            "predicted_direction": int(prediction),
            "increase_probability": float(increase_prob),
            "decrease_probability": float(decrease_prob),
            "no_change_probability": float(no_change_prob),
            "confidence": float(max(increase_prob, decrease_prob, no_change_prob)),
            "timeframe": TIMEFRAME_LABEL,
            "threshold": float(PCT_THRESHOLD),
            "horizon_minutes": int(HORIZON_MINUTES),
            "spec": spec_summary(),
            "simulated": data_source == "synthetic",
            "data_source": data_source
        }

        if data_source == "live":
            try:
                result["history"] = record_prediction(
                    candles_df=candles_df,
                    generated_at=to_naive_utc(timestamp),
                    reference_price=price,
                    direction=prediction,
                    confidence=result["confidence"],
                    training_cutoff=labeled_df['date'].iloc[-1],
                )
            except Exception as exc:
                result["history"] = {"recorded": False, "reason": f"{type(exc).__name__}: {exc}"}
        else:
            # Synthetic prices scored against real Coinbase candles would be noise.
            result["history"] = {"recorded": False, "reason": "synthetic data is not recorded"}

        return result

    except Exception as e:
        return {"error": str(e), "errorType": type(e).__name__}

if __name__ == "__main__":
    args = parse_arguments()
    timestamp = to_naive_utc(args.time)

    # Node reads stdout in JSON mode: one non-JSON line (ledger progress, model
    # chatter) fails the whole run, so everything except the result goes to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        result = run_prediction(
            price=args.price,
            volume=args.volume,
            timestamp=timestamp,
            window_size=args.window,
            allow_synthetic=args.allow_synthetic
        )

    # Output as JSON
    print(json.dumps(result))
