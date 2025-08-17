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
from dateutil.parser import parse

# Add parent directory to path to import from enhanced_prediction.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import numpy as np
    import pandas as pd
    from enhanced_prediction import create_enhanced_features, main
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    print(json.dumps({"error": f"Import error: {str(e)}", "errorType": "ImportError"}))
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Bitcoin prediction model')
    parser.add_argument('--price', type=float, required=True, help='Current Bitcoin price')
    parser.add_argument('--volume', type=float, required=False, default=100.0, help='Current trading volume')
    parser.add_argument('--time', type=str, required=False, default=None, help='Current timestamp (ISO format)')
    parser.add_argument('--window', type=int, required=False, default=60, help='Historical data window size')
    return parser.parse_args()

def generate_synthetic_data(current_price, volume, timestamp, window_size=60):
    """
    Generate synthetic price data to feed into the model
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    elif isinstance(timestamp, str):
        timestamp = parse(timestamp)

    # Generate timestamps going back from current time
    timestamps = [timestamp - datetime.timedelta(minutes=i) for i in range(window_size)]
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

def run_prediction(price, volume, timestamp=None, window_size=60):
    """
    Run the Bitcoin prediction model and return results
    """
    try:
        # Generate synthetic historical data based on current price
        df = generate_synthetic_data(price, volume, timestamp, window_size)

        # Create enhanced features
        enhanced_df = create_enhanced_features(df, pct_threshold=0.002)

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

        # Fit on all data except the last row
        y_train = enhanced_df['target'][:-1]
        X_train = enhanced_df[feature_cols][:-1]
        model.fit(X_train, y_train)

        # Get prediction probabilities for the current price
        probs = model.predict_proba(X)[0]

        # Get prediction class
        prediction = model.predict(X)[0]

        # Map probabilities to classes (-1, 0, 1)
        if len(probs) == 2:  # Binary classification
            decrease_prob = probs[0]
            increase_prob = probs[1]
            no_change_prob = 0
        else:  # Multi-class classification
            decrease_prob = probs[0]
            no_change_prob = probs[1]
            increase_prob = probs[2]

        # Return result as JSON
        result = {
            "price": float(price),
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime.datetime) else \
                        datetime.datetime.now().isoformat(),
            "predicted_direction": int(prediction),
            "increase_probability": float(increase_prob),
            "decrease_probability": float(decrease_prob),
            "no_change_probability": float(no_change_prob),
            "confidence": float(max(increase_prob, decrease_prob, no_change_prob)),
            "timeframe": "1 minute",
            "threshold": 0.002
        }

        return result

    except Exception as e:
        return {"error": str(e), "errorType": type(e).__name__}

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
