#!/usr/bin/env python

"""
Bitcoin Prediction Model Runner
Runs the enhanced prediction model from Python and returns results as JSON
"""

import argparse
import json
import sys
import os
import datetime
from typing import Optional

from dateutil.parser import parse

# Add parent directory to path to import from enhanced_prediction.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import numpy as np
    import pandas as pd
    from enhanced_prediction import create_enhanced_features
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    print(json.dumps({"error": f"Import error: {str(e)}", "errorType": "ImportError"}))
    sys.exit(1)

PCT_THRESHOLD = 0.002
MIN_FEATURE_WINDOW = 24 * 60 * 7  # Minimum rows required for the full enhanced feature set (7 days)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Bitcoin prediction model')
    parser.add_argument('--price', type=float, required=True, help='Current Bitcoin price')
    parser.add_argument('--volume', type=float, required=False, default=100.0, help='Current trading volume')
    parser.add_argument('--time', type=str, required=False, default=None, help='Current timestamp (ISO format)')
    parser.add_argument('--window', type=int, required=False, default=60, help='Historical data window size')
    return parser.parse_args()

def _normalize_timestamp(timestamp: Optional[datetime.datetime]) -> datetime.datetime:
    """Ensure we always operate on a timezone-aware datetime instance."""

    if timestamp is None:
        return datetime.datetime.now(datetime.timezone.utc)

    if isinstance(timestamp, str):
        return parse(timestamp)

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=datetime.timezone.utc)

    return timestamp


def generate_synthetic_data(current_price, volume, timestamp, window_size=60, random_state=None):
    """Generate lightweight synthetic price data that preserves local trends."""

    timestamp = _normalize_timestamp(timestamp)
    window_size = max(int(window_size), 2)

    if random_state is None:
        seed_source = int(timestamp.timestamp()) % (2 ** 32 - 1)
        random_state = seed_source

    rng = np.random.default_rng(random_state)

    timestamps = pd.date_range(end=timestamp, periods=window_size, freq='T', tz=timestamp.tzinfo)

    baseline_price = max(current_price * 0.98, 1.0)
    drift_per_step = (current_price - baseline_price) / max(window_size - 1, 1)
    volatility = max(current_price * 0.001, 0.01)

    noise = rng.normal(0, volatility, window_size - 1)
    price_steps = drift_per_step + noise
    prices = baseline_price + np.cumsum(np.concatenate(([0.0], price_steps)))

    # Ensure the final price is close to the provided current price
    final_adjustment = current_price - prices[-1]
    prices += np.linspace(0.0, final_adjustment, window_size)
    prices = np.maximum(prices, 0.01)

    volumes = volume * rng.uniform(0.6, 1.4, window_size)

    df = pd.DataFrame({
        'date': timestamps.tz_convert(None),
        'price': prices.astype(float),
        'volume': volumes.astype(float)
    })

    return df


def _format_response(price, timestamp, predicted_direction, increase_prob, decrease_prob, no_change_prob, confidence, threshold=PCT_THRESHOLD, method="random_forest"):
    timestamp = _normalize_timestamp(timestamp)

    return {
        "price": float(price),
        "timestamp": timestamp.isoformat(),
        "predicted_direction": int(predicted_direction),
        "increase_probability": float(increase_prob),
        "decrease_probability": float(decrease_prob),
        "no_change_probability": float(no_change_prob),
        "confidence": float(confidence),
        "timeframe": "1 minute",
        "threshold": float(threshold),
        "model": method
    }


def _statistical_fallback(df, price, timestamp):
    """A lightweight statistical heuristic used when there is not enough data for ML."""

    returns = df['price'].pct_change().dropna()

    if returns.empty:
        probs = np.array([0.34, 0.33, 0.33], dtype=float)
    else:
        recent_returns = returns.tail(min(30, len(returns)))
        momentum_short = recent_returns.mean()
        momentum_long = returns.mean()
        volatility = returns.std() + 1e-6

        directional_score = (momentum_short + momentum_long) / 2.0
        stability_score = max(1.0 - min(volatility * 50.0, 0.99), 0.01)

        increase_score = max(directional_score, 0.0) + 1e-6
        decrease_score = max(-directional_score, 0.0) + 1e-6
        no_change_score = stability_score + 1e-6

        scores = np.array([increase_score, decrease_score, no_change_score], dtype=float)
        probs = scores / scores.sum()

    increase_prob, decrease_prob, no_change_prob = probs

    probability_map = {
        1: float(increase_prob),
        -1: float(decrease_prob),
        0: float(no_change_prob)
    }
    predicted_direction = max(probability_map, key=probability_map.get)
    confidence = probability_map[predicted_direction]

    return _format_response(
        price=price,
        timestamp=timestamp,
        predicted_direction=predicted_direction,
        increase_prob=probability_map[1],
        decrease_prob=probability_map[-1],
        no_change_prob=probability_map[0],
        confidence=confidence,
        method="statistical"
    )

def run_prediction(price, volume, timestamp=None, window_size=60):
    """Run the Bitcoin prediction model and return results."""

    timestamp = _normalize_timestamp(timestamp)

    try:
        df = generate_synthetic_data(price, volume, timestamp, window_size)

        if window_size < MIN_FEATURE_WINDOW:
            return _statistical_fallback(df, price, timestamp)

        enhanced_df = create_enhanced_features(df, pct_threshold=PCT_THRESHOLD)

        if enhanced_df.empty:
            return _statistical_fallback(df, price, timestamp)

        feature_cols = [
            col for col in enhanced_df.columns
            if col not in ['date', 'price', 'future_price', 'next_return', 'target']
        ]

        X = enhanced_df[feature_cols].iloc[-1:]
        X_train = enhanced_df[feature_cols][:-1]
        y_train = enhanced_df['target'][:-1]

        if X_train.empty or y_train.empty or len(np.unique(y_train)) < 2:
            return _statistical_fallback(df, price, timestamp)

        model = RandomForestClassifier(
            n_estimators=75,
            max_depth=8,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        probs = model.predict_proba(X)[0]
        class_probabilities = dict(zip(model.classes_, probs))

        increase_prob = class_probabilities.get(1, 0.0)
        decrease_prob = class_probabilities.get(-1, 0.0)
        no_change_prob = class_probabilities.get(0, 0.0)

        # Ensure probabilities sum to 1 even if some classes are missing
        prob_sum = increase_prob + decrease_prob + no_change_prob
        if prob_sum == 0:
            return _statistical_fallback(df, price, timestamp)

        if prob_sum != 1:
            increase_prob /= prob_sum
            decrease_prob /= prob_sum
            no_change_prob /= prob_sum

        prediction = model.predict(X)[0]
        confidence = max(increase_prob, decrease_prob, no_change_prob)

        return _format_response(
            price=price,
            timestamp=timestamp,
            predicted_direction=prediction,
            increase_prob=increase_prob,
            decrease_prob=decrease_prob,
            no_change_prob=no_change_prob,
            confidence=confidence
        )

    except Exception as exc:  # pylint: disable=broad-except
        try:
            df = locals().get('df') or generate_synthetic_data(price, volume, timestamp, window_size)
            fallback = _statistical_fallback(df, price, timestamp)
            fallback['error'] = str(exc)
            fallback['errorType'] = type(exc).__name__
            fallback['model'] = 'statistical-fallback'
            return fallback
        except Exception as inner_exc:  # pylint: disable=broad-except
            return {"error": str(inner_exc), "errorType": type(inner_exc).__name__}

if __name__ == "__main__":
    args = parse_arguments()
    timestamp = datetime.datetime.now() if args.time is None else parse(args.time)

    result = run_prediction(
        price=args.price,
        volume=args.volume,
        timestamp=timestamp,
        window_size=args.window
    )

    # Output as JSON
    print(json.dumps(result))
