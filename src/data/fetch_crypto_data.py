"""
Historical Cryptocurrency Data Fetcher

Fetches historical OHLCV data from public cryptocurrency APIs and prepares it
for ensemble model training. Uses free APIs that don't require authentication.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_crypto_historical_data(symbol='BTC-USD', days=365):
    """
    Fetch historical cryptocurrency data using yfinance as a reliable source.
    
    Args:
        symbol: Trading pair symbol (default: 'BTC-USD')
        days: Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        
        logger.info(f"Fetching {days} days of historical data for {symbol}...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        logger.info(f"Fetched {len(df)} records from {df.index[0]} to {df.index[-1]}")
        
        return df
        
    except ImportError:
        logger.error("yfinance not installed. Install with: pip install yfinance")
        raise
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise


def engineer_features(df):
    """
    Create features from OHLCV data for model training.
    
    Features created:
    - Price changes (returns)
    - Moving averages
    - Volatility measures
    - Volume indicators
    - Technical indicators
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
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


def prepare_crypto_dataset(symbol='BTC-USD', days=365, forward_days=1, threshold=0.0):
    """
    Fetch and prepare cryptocurrency data for ensemble training.
    
    Args:
        symbol: Trading pair symbol
        days: Number of days of historical data
        forward_days: Days to look forward for target
        threshold: Return threshold for positive class
        
    Returns:
        tuple: (X, y) where X is feature DataFrame and y is target Series
    """
    # Fetch data
    df = fetch_crypto_historical_data(symbol, days)
    
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


def save_crypto_dataset(filename='crypto_historical_data.csv', symbol='BTC-USD', days=365):
    """
    Fetch crypto data and save to CSV for training.
    
    Args:
        filename: Output CSV filename
        symbol: Trading pair symbol
        days: Number of days of historical data
    """
    logger.info(f"Preparing crypto dataset: {symbol}, {days} days")
    
    X, y = prepare_crypto_dataset(symbol, days)
    
    # Combine for saving
    df = X.copy()
    df['target'] = y
    
    # Save to CSV
    df.to_csv(filename)
    logger.info(f"Saved {len(df)} records to {filename}")
    
    return filename


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch historical crypto data')
    parser.add_argument('--symbol', type=str, default='BTC-USD', help='Trading pair symbol')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--output', type=str, default='crypto_historical_data.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    try:
        filename = save_crypto_dataset(args.output, args.symbol, args.days)
        print(f"\n✓ Successfully created {filename}")
        print(f"  Use this file with: python -m src.models.ensemble_zoo --input {filename} --target target")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)
