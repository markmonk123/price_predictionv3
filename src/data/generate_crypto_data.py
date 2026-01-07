"""
Generate realistic crypto historical data for demonstration purposes.

This module creates synthetic data that mimics real cryptocurrency price patterns
when actual historical data cannot be fetched (e.g., in restricted network environments).
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_realistic_ohlcv(days=365, initial_price=45000, volatility=0.03, trend=0.0002):
    """
    Generate realistic OHLCV (Open, High, Low, Close, Volume) data for crypto.
    
    Uses geometric Brownian motion with realistic intraday patterns.
    
    Args:
        days: Number of days to generate
        initial_price: Starting price (default: $45,000 for BTC)
        volatility: Daily volatility (default: 3% per day)
        trend: Daily trend/drift (default: 0.02% per day)
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    logger.info(f"Generating {days} days of realistic crypto OHLCV data...")
    
    np.random.seed(42)  # For reproducibility
    
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    
    # Generate close prices using geometric Brownian motion
    returns = np.random.normal(trend, volatility, days)
    price_series = initial_price * np.exp(np.cumsum(returns))
    
    # Generate realistic OHLC from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, price_series)):
        # Intraday volatility (smaller than daily)
        intraday_vol = volatility * 0.3
        
        # Generate open (previous close + gap)
        if i == 0:
            open_price = close * (1 + np.random.normal(0, intraday_vol/2))
        else:
            open_price = price_series[i-1] * (1 + np.random.normal(0, intraday_vol/2))
        
        # Generate high and low around the range
        high_low_range = abs(close - open_price) * np.random.uniform(1.2, 2.0)
        high = max(open_price, close) + high_low_range * np.random.uniform(0.3, 0.8)
        low = min(open_price, close) - high_low_range * np.random.uniform(0.3, 0.8)
        
        # Ensure OHLC logic
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (log-normal distribution, typical for crypto)
        base_volume = 1e8  # 100M base volume
        volume = np.random.lognormal(np.log(base_volume), 0.5)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    logger.info(f"Generated data from {df.index[0]} to {df.index[-1]}")
    logger.info(f"Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    
    return df


def engineer_features(df):
    """
    Create features from OHLCV data for model training.
    
    Same feature engineering as the real data fetcher to maintain compatibility.
    """
    logger.info("Engineering features from OHLCV data...")
    
    feature_df = pd.DataFrame(index=df.index)
    
    # Price-based features
    feature_df['close'] = df['Close']
    feature_df['open'] = df['Open']
    feature_df['high'] = df['High']
    feature_df['low'] = df['Low']
    feature_df['volume'] = df['Volume']
    
    # Returns (percentage change)
    feature_df['return_1d'] = df['Close'].pct_change(1)
    feature_df['return_7d'] = df['Close'].pct_change(7)
    feature_df['return_30d'] = df['Close'].pct_change(30)
    
    # Moving averages
    feature_df['ma_7'] = df['Close'].rolling(window=7).mean()
    feature_df['ma_30'] = df['Close'].rolling(window=30).mean()
    feature_df['ma_ratio'] = feature_df['ma_7'] / feature_df['ma_30']
    
    # Volatility (rolling standard deviation)
    feature_df['volatility_7d'] = df['Close'].pct_change().rolling(window=7).std()
    feature_df['volatility_30d'] = df['Close'].pct_change().rolling(window=30).std()
    
    # Price range features
    feature_df['high_low_ratio'] = df['High'] / df['Low']
    feature_df['close_open_ratio'] = df['Close'] / df['Open']
    
    # Volume features
    feature_df['volume_ma_7'] = df['Volume'].rolling(window=7).mean()
    feature_df['volume_ratio'] = df['Volume'] / feature_df['volume_ma_7']
    
    # Momentum indicators
    # RSI (Relative Strength Index) - simplified
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    feature_df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    feature_df['macd'] = exp1 - exp2
    feature_df['macd_signal'] = feature_df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    bb_ma = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    feature_df['bb_upper'] = bb_ma + (bb_std * 2)
    feature_df['bb_lower'] = bb_ma - (bb_std * 2)
    feature_df['bb_width'] = (feature_df['bb_upper'] - feature_df['bb_lower']) / bb_ma
    
    logger.info(f"Created {feature_df.shape[1]} features")
    
    return feature_df


def create_target(df, threshold=0.0, forward_days=1):
    """
    Create binary target variable based on future price movement.
    
    Args:
        df: DataFrame with 'Close' column
        threshold: Percentage threshold for positive class (default: 0.0 = any increase)
        forward_days: Number of days to look forward (default: 1)
        
    Returns:
        pd.Series: Binary target (1 = price increase, 0 = price decrease/flat)
    """
    logger.info(f"Creating target: {forward_days}-day forward return > {threshold*100}%")
    
    # Calculate forward returns
    future_close = df['Close'].shift(-forward_days)
    forward_return = (future_close - df['Close']) / df['Close']
    
    # Binary classification: 1 if price increases above threshold, 0 otherwise
    target = (forward_return > threshold).astype(int)
    
    # Log class distribution
    value_counts = target.value_counts()
    logger.info(f"Target distribution: Class 0: {value_counts.get(0, 0)}, Class 1: {value_counts.get(1, 0)}")
    
    return target


def generate_crypto_dataset(days=365, initial_price=45000, forward_days=1, threshold=0.0):
    """
    Generate realistic crypto dataset for ensemble training.
    
    Args:
        days: Number of days to generate
        initial_price: Starting price
        forward_days: Days to look forward for target
        threshold: Return threshold for positive class
        
    Returns:
        tuple: (X, y) where X is feature DataFrame and y is target Series
    """
    # Generate OHLCV data
    df = generate_realistic_ohlcv(days, initial_price)
    
    # Engineer features
    features = engineer_features(df)
    
    # Create target
    target = create_target(df, threshold, forward_days)
    
    # Align features and target (drop NaN values from feature engineering)
    combined = features.join(target.rename('target'))
    combined = combined.dropna()
    
    logger.info(f"Final dataset shape: {combined.shape}")
    logger.info(f"Date range: {combined.index[0]} to {combined.index[-1]}")
    
    # Split features and target
    X = combined.drop(columns=['target'])
    y = combined['target']
    
    return X, y


def save_crypto_dataset(filename='crypto_historical_data.csv', days=365, initial_price=45000):
    """
    Generate crypto data and save to CSV for training.
    
    Args:
        filename: Output CSV filename
        days: Number of days to generate
        initial_price: Starting price
    """
    logger.info(f"Generating crypto dataset: {days} days, starting at ${initial_price}")
    
    X, y = generate_crypto_dataset(days, initial_price)
    
    # Combine for saving
    df = X.copy()
    df['target'] = y
    
    # Save to CSV
    df.to_csv(filename)
    logger.info(f"Saved {len(df)} records to {filename}")
    
    return filename


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate realistic crypto historical data')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data to generate')
    parser.add_argument('--price', type=float, default=45000, help='Initial price (e.g., 45000 for BTC)')
    parser.add_argument('--output', type=str, default='crypto_historical_data.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    try:
        filename = save_crypto_dataset(args.output, args.days, args.price)
        print(f"\n✓ Successfully created {filename}")
        print(f"  Use this file with: python -m src.models.ensemble_zoo --input {filename} --target target")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)
