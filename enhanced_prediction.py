import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import time

def create_enhanced_features(df, pct_threshold=0.002):
    """Create comprehensive technical indicators for Bitcoin 1-minute interval prediction."""
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
        df[f'price_sma_ratio_{window}'] = df['price'] / df[f'sma_{window}']
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
        rs = gain / loss
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
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
    df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Price momentum and slopes (scaled for 1-min)
    for period in [30, 60, 120]: # 30m, 1h, 2h
        df[f'momentum_{period}'] = (df['price'] / df['price'].shift(period) - 1) * 100
        
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
        df[f'price_position_{window}'] = (df['price'] - df[f'support_{window}']) / (df[f'resistance_{window}'] - df[f'support_{window}'])
    
    # Trend indicators (scaled for 1-min)
    df['price_trend_30'] = np.where(df['price'] > df['sma_30'], 1, 0)
    df['price_trend_120'] = np.where(df['price'] > df['sma_120'], 1, 0)
    df['trend_strength'] = df['price_trend_30'] + df['price_trend_120']
    
    # Classification target
    df['future_price'] = df['price'].shift(-1)
    df['next_return'] = (df['future_price'] - df['price']) / df['price']
    df['target'] = 0
    df.loc[df['next_return'] >= pct_threshold, 'target'] = 1
    df.loc[df['next_return'] <= -pct_threshold, 'target'] = -1
    
    return df.dropna()

def fetch_bitcoin_data(num_points=36000, interval_minutes=1):
    """Fetch a specific number of Bitcoin price data points from Coinbase API."""
    print(f"Fetching last {num_points} {interval_minutes}-minute data points...")
    all_data = []
    points_per_request = 300  # Coinbase API limit per request
    num_requests = (num_points + points_per_request - 1) // points_per_request
    granularity = interval_minutes * 60

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    end_time = datetime.utcnow()

    try:
        for i in range(num_requests):
            start_time = end_time - timedelta(minutes=points_per_request * interval_minutes)
            
            params = {
                "granularity": granularity,
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
            
            print(f"Request {i+1}/{num_requests}: Fetching data from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print("   No more data available from API.")
                break
            
            all_data.extend(data)
            
            oldest_timestamp = data[-1][0]
            end_time = datetime.fromtimestamp(oldest_timestamp)

            time.sleep(0.5)

        if not all_data:
            raise ValueError("No data received from Coinbase API")

        df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
        df = df.sort_values("time", ascending=False).drop_duplicates(subset=['time']).sort_values("time")
        
        df["date"] = pd.to_datetime(df["time"], unit="s")
        df["price"] = pd.to_numeric(df["close"], errors='coerce')
        
        print(f"Successfully fetched {len(df)} unique {interval_minutes}-minute data points.")
        return df[["date", "price"]].dropna()
        
    except Exception as e:
        print(f"API Error: {e}. Using simulated data...")
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=num_points), periods=num_points, freq=f'{interval_minutes}T')
        prices = np.cumsum(np.random.randn(num_points) * 2) + 60000
        return pd.DataFrame({'date': dates, 'price': prices})

def main():
    """Enhanced Bitcoin prediction using scikit-learn ensemble on 1-minute data."""
    print("🚀 Enhanced Bitcoin 1-Minute Interval Prediction with Scikit-Learn")
    print("=" * 70)
    
    # Get data and create features
    df = fetch_bitcoin_data(num_points=36000, interval_minutes=1)
    print(f"📊 Fetched {len(df)} data points of Bitcoin data at 1-minute intervals")
    
    df = create_enhanced_features(df, pct_threshold=0.002) # Adjusted threshold for smaller timeframe
    print(f"⚙️  Created {len(df.columns)-3} technical features")  # -3 for date, price, target
    
    # Select features (exclude non-predictive columns)
    feature_cols = [col for col in df.columns if col not in ['date', 'price', 'future_price', 'next_return', 'target']]
    X = df[feature_cols]
    y = df['target']
    
    print(f"🎯 Using {len(feature_cols)} features for prediction")
    
    # Class distribution
    class_dist = y.value_counts().sort_index()
    print(f"\n📈 Class Distribution:")
    print(f"   Decrease ≥0.2%: {class_dist.get(-1, 0)} ({class_dist.get(-1, 0)/len(y)*100:.1f}%)")
    print(f"   No Change: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
    print(f"   Increase ≥0.2%: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
    
    # Time series split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
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
    
    print(f"\n🎯 Bitcoin 0.2% Shift Predictions:")
    print("=" * 80)
    
    # High confidence predictions
    high_conf_predictions = test_df[
        (test_df['predicted'] != 0) & (test_df['confidence'] >= 0.7)
    ]
    
    if len(high_conf_predictions) > 0:
        print(f"🔥 High Confidence Predictions (≥70%):")
        for _, row in high_conf_predictions.iterrows():
            direction = "📈 INCREASE" if row['predicted'] == 1 else "📉 DECREASE"
            timestamp = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"   Price ${row['price']:.2f} | {direction} by 0.2% | Confidence: {row['confidence']:.1%} | {timestamp}")
    else:
        print("⚠️  No high-confidence significant moves predicted")
    
    # All predictions
    print(f"\n📋 All Test Predictions:")
    for _, row in test_df.iterrows():
        direction_map = {1: "Increase", -1: "Decrease", 0: "No Change"}
        direction = direction_map[row['predicted']]
        timestamp = row['date'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"Price ${row['price']:.2f} | Confidence: {row['confidence']:.3f} | {direction} by 0.2% {timestamp}")
    
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
