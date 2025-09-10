#!/usr/bin/env python3
"""
Enhanced Bitcoin Price Forecasting System
Provides 12-hour predictions with 30-minute intervals, market sentiment analysis,
multi-model training, and comprehensive statistical analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

class EnhancedBitcoinForecaster:
    """Enhanced forecasting system with multi-model training and 12-hour predictions."""
    
    def __init__(self, data_update_interval=300):  # 5 minutes = 300 seconds
        self.data_update_interval = data_update_interval
        self.models = {}
        self.trained_models = {}
        self.is_training = False
        self.last_update = None
        self.latest_data = None
        
        # Initialize multiple models for ensemble
        self.models = {
            'RandomForest_1': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'RandomForest_2': RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_split=3,
                random_state=123, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                random_state=42
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=180, max_depth=14, min_samples_split=4,
                random_state=42, n_jobs=-1
            ),
            'Pipeline_RF': Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
        }
    
    def create_prediction_features(self, df):
        """Create features specifically designed for multi-step ahead prediction."""
        df = df.copy()
        
        # Lag features for sequence prediction
        for lag in range(1, 25):  # 24 lags for 12-hour prediction (30-min intervals)
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            if lag <= 12:  # Only short-term returns
                df[f'return_lag_{lag}'] = df['price'].pct_change(lag)
        
        # Rolling statistics for different horizons
        for window in [6, 12, 24, 48]:  # 3h, 6h, 12h, 24h in 30-min intervals
            df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'std_{window}'] = df['price'].rolling(window=window).std()
            df[f'min_{window}'] = df['price'].rolling(window=window).min()
            df[f'max_{window}'] = df['price'].rolling(window=window).max()
            df[f'median_{window}'] = df['price'].rolling(window=window).median()
            
            # Price position within range
            df[f'price_position_{window}'] = (
                (df['price'] - df[f'min_{window}']) / 
                (df[f'max_{window}'] - df[f'min_{window}'] + 1e-8)
            )
        
        # Momentum and acceleration features
        df['momentum_short'] = df['price'].pct_change(6)   # 3-hour momentum
        df['momentum_medium'] = df['price'].pct_change(12)  # 6-hour momentum
        df['momentum_long'] = df['price'].pct_change(24)   # 12-hour momentum
        
        # Volatility features
        df['volatility_6h'] = df['price'].pct_change().rolling(window=12).std()
        df['volatility_12h'] = df['price'].pct_change().rolling(window=24).std()
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def prepare_multistep_data(self, df, forecast_horizon=24):
        """Prepare data for multi-step ahead prediction (24 steps = 12 hours)."""
        # Create features
        df = self.create_prediction_features(df)
        
        # Create target variables for each step ahead (1 to 24 steps)
        for step in range(1, forecast_horizon + 1):
            df[f'target_{step}'] = df['price'].shift(-step)
        
        # Remove rows with NaN values
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col not in ['date', 'price']]
        target_cols = [f'target_{step}' for step in range(1, forecast_horizon + 1)]
        
        # Drop rows with NaN in features or targets
        valid_rows = df[feature_cols + target_cols].dropna()
        
        return df.loc[valid_rows.index], feature_cols, target_cols
    
    def train_models(self, df):
        """Train all models on the provided data."""
        print("🤖 Training enhanced forecasting models...")
        
        # Prepare data for multi-step prediction
        prepared_df, feature_cols, target_cols = self.prepare_multistep_data(df)
        
        if len(prepared_df) < 50:  # Need minimum data for training
            print("   ⚠️  Insufficient data for multi-step training")
            return False
        
        X = prepared_df[feature_cols]
        # For now, train on 1-step ahead target, but we'll use recursive prediction
        y = prepared_df['target_1'] 
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        self.trained_models = {}
        model_scores = {}
        
        print(f"   📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train each model
        for name, model in self.models.items():
            try:
                print(f"   🔄 Training {name}...", end=' ')
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_scores[name] = {'mse': mse, 'mae': mae}
                self.trained_models[name] = model
                
                print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")
                
            except Exception as e:
                print(f"Failed: {e}")
        
        print(f"   ✅ Successfully trained {len(self.trained_models)} models")
        self.last_update = datetime.now()
        
        return True
    
    def recursive_predict(self, model, X_initial, steps=24):
        """Use recursive prediction to forecast multiple steps ahead."""
        predictions = []
        current_X = X_initial.copy()
        
        for step in range(steps):
            # Predict next value
            next_pred = model.predict(current_X.tail(1))[0]
            predictions.append(next_pred)
            
            # Update features for next prediction
            # Shift lag features
            new_row = current_X.iloc[-1].copy()
            
            # Update price lags
            for lag in range(24, 1, -1):  # Start from highest lag
                if f'price_lag_{lag}' in new_row.index:
                    if f'price_lag_{lag-1}' in new_row.index:
                        new_row[f'price_lag_{lag}'] = new_row[f'price_lag_{lag-1}']
            
            # Set the first lag to the predicted price
            if 'price_lag_1' in new_row.index:
                new_row['price_lag_1'] = next_pred
            
            # Update rolling statistics (simplified)
            # For a full implementation, we'd need to maintain a rolling window
            
            # Append new row
            current_X = pd.concat([current_X, pd.DataFrame([new_row])], ignore_index=True)
        
        return predictions
    
    def generate_12_hour_forecast(self, latest_data):
        """Generate 12-hour forecast with 30-minute intervals."""
        if not self.trained_models:
            print("   ❌ No trained models available")
            return None
        
        print("🔮 Generating 12-hour forecast...")
        
        # Prepare the latest data
        df_prep, feature_cols, _ = self.prepare_multistep_data(latest_data)
        
        if len(df_prep) == 0:
            print("   ❌ Unable to prepare data for prediction")
            return None
        
        X_latest = df_prep[feature_cols].tail(50)  # Use last 50 points for context
        
        # Generate predictions from each model
        ensemble_predictions = []
        
        for name, model in self.trained_models.items():
            try:
                predictions = self.recursive_predict(model, X_latest, steps=24)
                ensemble_predictions.append(predictions)
                print(f"   ✅ {name}: Generated 24 predictions")
            except Exception as e:
                print(f"   ❌ {name}: Failed - {e}")
        
        if not ensemble_predictions:
            print("   ❌ No successful predictions generated")
            return None
        
        # Average predictions across models
        ensemble_avg = np.mean(ensemble_predictions, axis=0)
        ensemble_std = np.std(ensemble_predictions, axis=0) if len(ensemble_predictions) > 1 else np.zeros(24)
        
        # Create timestamps for 12 hours ahead (30-minute intervals)
        start_time = pd.to_datetime(latest_data['date'].iloc[-1]) + timedelta(minutes=30)
        timestamps = [start_time + timedelta(minutes=30*i) for i in range(24)]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_price': ensemble_avg,
            'prediction_std': ensemble_std,
            'interval_minutes': [30 * (i+1) for i in range(24)]
        })
        
        return forecast_df
    
    def analyze_market_sentiment(self, df, window_hours=6):
        """Analyze market sentiment using variance over specified window."""
        print(f"📈 Analyzing market sentiment over {window_hours}-hour window...")
        
        # Convert window to 30-minute intervals
        window_periods = window_hours * 2
        
        if len(df) < window_periods:
            print(f"   ⚠️  Insufficient data for {window_hours}-hour analysis")
            return "INSUFFICIENT_DATA", 0.0, {}
        
        # Calculate price variance over the window
        recent_prices = df['price'].tail(window_periods)
        price_change_pct = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
        
        # Calculate variance
        returns = recent_prices.pct_change().dropna()
        variance_pct = returns.std() * 100
        
        # Determine sentiment based on 1% threshold
        sentiment = "NEUTRAL"
        if abs(price_change_pct) >= 1.0:
            if price_change_pct > 0:
                sentiment = "BULLISH"
            else:
                sentiment = "BEARISH"
        
        # Additional metrics
        metrics = {
            'price_change_pct': price_change_pct,
            'variance_pct': variance_pct,
            'max_price': recent_prices.max(),
            'min_price': recent_prices.min(),
            'avg_price': recent_prices.mean(),
            'median_price': recent_prices.median(),
            'volatility': returns.std() * np.sqrt(window_periods * 24 * 365),  # Annualized volatility
            'trend_strength': abs(price_change_pct) / (variance_pct + 1e-8)
        }
        
        return sentiment, price_change_pct, metrics
    
    def calculate_forecast_statistics(self, forecast_df, current_price):
        """Calculate comprehensive statistics for the 12-hour forecast."""
        print("📊 Calculating forecast statistics...")
        
        predicted_prices = forecast_df['predicted_price']
        
        # Basic statistics
        stats = {
            'current_price': current_price,
            'forecast_high': predicted_prices.max(),
            'forecast_low': predicted_prices.min(),
            'forecast_median': predicted_prices.median(),
            'forecast_average': predicted_prices.mean(),
            'forecast_std': predicted_prices.std(),
            'price_range': predicted_prices.max() - predicted_prices.min(),
            'price_range_pct': ((predicted_prices.max() - predicted_prices.min()) / current_price) * 100
        }
        
        # Delta values (change from current price)
        stats['delta_high'] = stats['forecast_high'] - current_price
        stats['delta_low'] = stats['forecast_low'] - current_price
        stats['delta_median'] = stats['forecast_median'] - current_price
        stats['delta_average'] = stats['forecast_average'] - current_price
        
        # Delta percentages
        stats['delta_high_pct'] = (stats['delta_high'] / current_price) * 100
        stats['delta_low_pct'] = (stats['delta_low'] / current_price) * 100
        stats['delta_median_pct'] = (stats['delta_median'] / current_price) * 100
        stats['delta_average_pct'] = (stats['delta_average'] / current_price) * 100
        
        # Find when high and low occur
        high_idx = predicted_prices.idxmax()
        low_idx = predicted_prices.idxmin()
        
        stats['high_time'] = forecast_df.loc[high_idx, 'timestamp']
        stats['low_time'] = forecast_df.loc[low_idx, 'timestamp']
        stats['high_interval'] = forecast_df.loc[high_idx, 'interval_minutes']
        stats['low_interval'] = forecast_df.loc[low_idx, 'interval_minutes']
        
        return stats
    
    def format_enhanced_output(self, forecast_df, sentiment, sentiment_change, sentiment_metrics, stats):
        """Format the enhanced output with all required information."""
        
        print("\n" + "="*80)
        print("🔮 ENHANCED 12-HOUR BITCOIN PRICE FORECAST")
        print("="*80)
        
        # Current status
        print(f"\n📊 CURRENT STATUS:")
        print(f"   Current Price: ${stats['current_price']:,.2f}")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Market sentiment
        sentiment_emoji = {"BULLISH": "🐂", "BEARISH": "🐻", "NEUTRAL": "➡️"}
        print(f"\n📈 MARKET SENTIMENT (6-hour analysis):")
        print(f"   {sentiment_emoji.get(sentiment, '❓')} Market Direction: {sentiment}")
        print(f"   💹 Price Change: {sentiment_change:+.2f}% (6-hour)")
        print(f"   📊 Variance: {sentiment_metrics.get('variance_pct', 0):.2f}%")
        print(f"   🎯 Trend Strength: {sentiment_metrics.get('trend_strength', 0):.2f}")
        
        # 12-hour forecast summary
        print(f"\n🔮 12-HOUR FORECAST SUMMARY:")
        print(f"   📈 Predicted HIGH: ${stats['forecast_high']:,.2f} at {stats['high_time'].strftime('%H:%M')} ({stats['high_interval']} min)")
        print(f"   📉 Predicted LOW:  ${stats['forecast_low']:,.2f} at {stats['low_time'].strftime('%H:%M')} ({stats['low_interval']} min)")
        print(f"   📊 Average Price:  ${stats['forecast_average']:,.2f}")
        print(f"   📊 Median Price:   ${stats['forecast_median']:,.2f}")
        print(f"   📏 Price Range:    ${stats['price_range']:,.2f} ({stats['price_range_pct']:.2f}%)")
        
        # Delta analysis
        print(f"\n📈 DELTA ANALYSIS (vs Current Price):")
        print(f"   📈 HIGH Delta:     ${stats['delta_high']:+,.2f} ({stats['delta_high_pct']:+.2f}%)")
        print(f"   📉 LOW Delta:      ${stats['delta_low']:+,.2f} ({stats['delta_low_pct']:+.2f}%)")
        print(f"   📊 AVERAGE Delta:  ${stats['delta_average']:+,.2f} ({stats['delta_average_pct']:+.2f}%)")
        print(f"   📊 MEDIAN Delta:   ${stats['delta_median']:+,.2f} ({stats['delta_median_pct']:+.2f}%)")
        
        # Detailed 30-minute predictions
        print(f"\n⏰ DETAILED 30-MINUTE PREDICTIONS:")
        print("-" * 80)
        print(f"{'Time':<8} {'Price':<12} {'Change':<10} {'Change%':<8} {'Std Dev':<8}")
        print("-" * 80)
        
        for i, row in forecast_df.iterrows():
            change = row['predicted_price'] - stats['current_price']
            change_pct = (change / stats['current_price']) * 100
            
            time_str = row['timestamp'].strftime('%H:%M')
            price_str = f"${row['predicted_price']:,.2f}"
            change_str = f"${change:+.2f}"
            change_pct_str = f"{change_pct:+.2f}%"
            std_str = f"±{row['prediction_std']:.2f}"
            
            print(f"{time_str:<8} {price_str:<12} {change_str:<10} {change_pct_str:<8} {std_str:<8}")
        
        # Model information
        print(f"\n🤖 MODEL INFORMATION:")
        print(f"   📊 Models Used: {len(self.trained_models)} ensemble models")
        print(f"   🔄 Last Update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never'}")
        print(f"   ⏰ Update Interval: {self.data_update_interval // 60} minutes")
        print(f"   🎯 Prediction Method: Recursive multi-step forecasting")
        
        print("\n" + "="*80)


def run_enhanced_forecasting(df_combined):
    """Main function to run the enhanced forecasting system."""
    
    print("\n🚀 STARTING ENHANCED FORECASTING SYSTEM")
    print("="*80)
    
    # Initialize forecaster
    forecaster = EnhancedBitcoinForecaster()
    
    # Train models
    success = forecaster.train_models(df_combined)
    if not success:
        print("❌ Failed to train models")
        return None
    
    # Generate 12-hour forecast
    forecast_df = forecaster.generate_12_hour_forecast(df_combined)
    if forecast_df is None:
        print("❌ Failed to generate forecast")
        return None
    
    # Analyze market sentiment
    sentiment, sentiment_change, sentiment_metrics = forecaster.analyze_market_sentiment(df_combined)
    
    # Calculate statistics
    current_price = df_combined['price'].iloc[-1]
    stats = forecaster.calculate_forecast_statistics(forecast_df, current_price)
    
    # Format and display output
    forecaster.format_enhanced_output(forecast_df, sentiment, sentiment_change, sentiment_metrics, stats)
    
    return {
        'forecast': forecast_df,
        'sentiment': sentiment,
        'sentiment_metrics': sentiment_metrics,
        'stats': stats,
        'forecaster': forecaster
    }


if __name__ == "__main__":
    # This will be called from the main prediction script
    pass