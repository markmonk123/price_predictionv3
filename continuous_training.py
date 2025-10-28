#!/usr/bin/env python3
"""
Continuous Model Training and Data Update System
Implements automatic data updates every 5 minutes and model retraining
with at least 3 models running for data integrity.
"""

import threading
import time
import queue
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import data normalizer
from data_normalizer import DataNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContinuousTrainingSystem:
    """System for continuous model training and data updates."""
    
    def __init__(self, update_interval=300, min_models=3):  # 5 minutes default
        self.update_interval = update_interval  # seconds
        self.min_models = min_models
        self.is_running = False
        self.data_queue = queue.Queue()
        self.model_queue = queue.Queue()
        self.latest_data = None
        self.model_pool = {}
        self.training_pool = {}
        self.data_integrity_checks = []
        self.last_prediction = None  # Store last predicted price for retraining

        # Thread pools
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.training_threads = []
        
    def start_continuous_system(self, initial_data):
        """Start the continuous training and data update system."""
        logger.info("🚀 Starting Continuous Training System")
        logger.info(f"   ⏰ Update interval: {self.update_interval} seconds")
        logger.info(f"   🤖 Minimum models: {self.min_models}")
        
        self.is_running = True
        self.latest_data = initial_data
        
        # Start data update thread
        data_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        data_thread.start()
        
        # Start model training threads
        for i in range(self.min_models):
            training_thread = threading.Thread(
                target=self._model_training_loop,
                args=(f"model_{i+1}",),
                daemon=True
            )
            training_thread.start()
            self.training_threads.append(training_thread)
        
        # Start integrity monitoring thread
        integrity_thread = threading.Thread(target=self._integrity_monitor_loop, daemon=True)
        integrity_thread.start()
        
        logger.info("✅ All systems started successfully")
        
    def stop_continuous_system(self):
        """Stop the continuous training system."""
        logger.info("🛑 Stopping Continuous Training System")
        self.is_running = False
        self.executor.shutdown(wait=True)

    def _drop_normalized_fields(self, df):
        """Remove columns that appear to be normalized or scaled."""
        if df is None:
            return df

        drop_cols = []
        for col in df.columns:
            if col in {"date", "price"}:
                continue

            lower = col.lower()
            if "norm" in lower or "scale" in lower:
                drop_cols.append(col)
                continue

            series = df[col].dropna()
            if not series.empty and series.between(0, 1).all():
                drop_cols.append(col)

        if drop_cols:
            logger.warning(f"Removing normalized columns: {drop_cols}")
            df = df.drop(columns=drop_cols)

        return df

    def _data_update_loop(self):
        """Continuously update data every 5 minutes."""
        logger.info("📊 Data update loop started")

        while self.is_running:
            try:
                # Simulate new data arrival
                new_data = self._generate_new_data_point()

                if new_data is not None:
                    # Drop unexpected normalized fields
                    new_data = self._drop_normalized_fields(new_data)
                    self.latest_data = self._drop_normalized_fields(self.latest_data)

                    # Add to existing data and apply smoothing
                    self.latest_data = pd.concat([self.latest_data, new_data], ignore_index=True)
                    self.latest_data['price'] = (
                        self.latest_data['price'].rolling(window=5, min_periods=1).mean()
                    )

                    # Keep only last 30000 points of 5-minute data
                    if len(self.latest_data) > 30000:
                        self.latest_data = self.latest_data.tail(30000)
                    
                    # Queue for model retraining
                    self.data_queue.put(self.latest_data.copy())
                    
                    logger.info(f"📈 Data updated - {len(self.latest_data)} points available")
                    
                    # Perform data integrity check
                    self._perform_data_integrity_check()
                
            except Exception as e:
                logger.error(f"❌ Error in data update loop: {e}")
            
            # Wait for next update
            time.sleep(self.update_interval)
    
    def _generate_new_data_point(self):
        """Generate a new data point (simulating real-time data)."""
        if self.latest_data is None or len(self.latest_data) == 0:
            return None
        
        # Get last data point
        last_point = self.latest_data.iloc[-1]
        last_time = pd.to_datetime(last_point['date'])
        last_price = last_point['price']
        
        # Generate new timestamp (5 minutes later)
        new_time = last_time + timedelta(minutes=5)
        
        # Generate new price with random walk
        price_change = np.random.normal(0, 0.01)  # 1% volatility
        new_price = last_price * (1 + price_change)
        
        # Create new data point
        new_data = pd.DataFrame({
            'date': [new_time],
            'price': [new_price]
        })
        
        return new_data
    
    def _model_training_loop(self, model_id):
        """Continuously retrain models when new data arrives."""
        logger.info(f"🤖 Model training loop started for {model_id}")
        
        from enhanced_forecasting import EnhancedBitcoinForecaster
        
        forecaster = EnhancedBitcoinForecaster()
        
        while self.is_running:
            try:
                # Wait for new data
                if not self.data_queue.empty():
                    new_data = self.data_queue.get(timeout=1)
                    new_data = self._drop_normalized_fields(new_data)

                    # Incorporate last prediction and current price
                    if self.last_prediction is not None and len(new_data) > 30000:
                        new_data = new_data.tail(30000)
                    if self.last_prediction is not None and len(new_data) >= 29999:
                        actual_price = new_data.iloc[-1]['price']
                        actual_time = pd.to_datetime(new_data.iloc[-1]['date'])
                        predicted_time = actual_time + timedelta(minutes=5)
                        historical = new_data.iloc[-(29998+1):-1]
                        head_rows = pd.DataFrame({
                            'date': [actual_time, predicted_time],
                            'price': [actual_price, self.last_prediction]
                        })
                        new_data = pd.concat([head_rows, historical], ignore_index=True)

                    logger.info(f"🔄 {model_id}: Retraining with {len(new_data)} data points")

                    # Resample to 30-minute intervals for training
                    train_data = new_data.copy()
                    train_data['date'] = pd.to_datetime(train_data['date'])
                    train_data = (
                        train_data.resample('30T', on='date').last().dropna().reset_index()
                    )

                    # Retrain models (normalization happens inside train_models)
                    start_time = time.time()
                    success = forecaster.train_models(train_data)
                    training_time = time.time() - start_time

                    if success:
                        # Store trained model with its normalizer
                        self.model_pool[model_id] = {
                            'forecaster': forecaster,
                            'training_time': training_time,
                            'data_size': len(train_data),
                            'timestamp': datetime.now(),
                            'model_id': model_id
                        }

                        logger.info(f"✅ {model_id}: Training completed in {training_time:.2f}s")
                        logger.info(f"   📊 Normalized features using forecaster's normalizer")

                        # Generate and log 12-hour forecast (predictions are denormalized inside generate_12_hour_forecast)
                        forecast = forecaster.generate_12_hour_forecast(train_data)
                        if forecast is not None:
                            self.last_prediction = forecast['predicted_price'].iloc[0]
                            logger.info(f"   🔮 Next prediction (denormalized): ${self.last_prediction:.2f}")
                            for _, row in forecast.head(3).iterrows():  # Log first 3 predictions
                                logger.info(
                                    f"   🕒 {row['timestamp']}: ${row['predicted_price']:.2f}"
                                )
                    else:
                        logger.warning(f"⚠️ {model_id}: Training failed")
                    
                else:
                    time.sleep(1)  # Wait a bit if no new data
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error in {model_id} training loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _perform_data_integrity_check(self):
        """Perform data integrity checks."""
        if self.latest_data is None or len(self.latest_data) < 10:
            return
        
        integrity_report = {
            'timestamp': datetime.now(),
            'data_points': len(self.latest_data),
            'price_range': {
                'min': self.latest_data['price'].min(),
                'max': self.latest_data['price'].max(),
                'current': self.latest_data['price'].iloc[-1]
            },
            'checks': {}
        }
        
        # Check for missing values
        missing_values = self.latest_data.isnull().sum().sum()
        integrity_report['checks']['missing_values'] = missing_values == 0
        
        # Check for reasonable price changes
        price_changes = self.latest_data['price'].pct_change().abs()
        extreme_changes = (price_changes > 0.1).sum()  # More than 10% change
        integrity_report['checks']['reasonable_changes'] = extreme_changes == 0
        
        # Check for timestamp continuity
        time_diffs = pd.to_datetime(self.latest_data['date']).diff()
        expected_diff = timedelta(minutes=5)
        irregular_intervals = (time_diffs != expected_diff).sum() - 1  # Exclude first NaT
        integrity_report['checks']['regular_intervals'] = irregular_intervals <= 1  # Allow some tolerance
        
        # Store integrity check
        self.data_integrity_checks.append(integrity_report)
        
        # Keep only last 20 checks
        if len(self.data_integrity_checks) > 20:
            self.data_integrity_checks = self.data_integrity_checks[-20:]
        
        # Log integrity status
        all_checks_passed = all(integrity_report['checks'].values())
        if all_checks_passed:
            logger.info("✅ Data integrity check passed")
        else:
            failed_checks = [k for k, v in integrity_report['checks'].items() if not v]
            logger.warning(f"⚠️ Data integrity issues: {failed_checks}")
    
    def _integrity_monitor_loop(self):
        """Monitor system integrity and model performance."""
        logger.info("🔍 Integrity monitor started")
        
        while self.is_running:
            try:
                # Check model pool health
                active_models = len(self.model_pool)
                
                if active_models < self.min_models:
                    logger.warning(f"⚠️ Only {active_models}/{self.min_models} models active")
                else:
                    logger.info(f"✅ {active_models} models active and healthy")
                
                # Check model performance variance
                if len(self.model_pool) >= 2:
                    self._check_model_agreement()
                
                # Report system status
                self._report_system_status()
                
            except Exception as e:
                logger.error(f"❌ Error in integrity monitor: {e}")
            
            time.sleep(60)  # Check every minute
    
    def _check_model_agreement(self):
        """Check agreement between different models."""
        if len(self.model_pool) < 2:
            return
        
        # Generate a simple test prediction to check model agreement
        try:
            test_data = self.latest_data.tail(50) if self.latest_data is not None else None
            if test_data is None or len(test_data) < 10:
                return
            
            predictions = []
            for model_id, model_info in self.model_pool.items():
                try:
                    forecaster = model_info['forecaster']
                    forecast = forecaster.generate_12_hour_forecast(test_data)
                    if forecast is not None:
                        predictions.append(forecast['predicted_price'].iloc[0])  # First prediction
                except Exception as e:
                    logger.warning(f"⚠️ Model {model_id} prediction failed: {e}")
            
            if len(predictions) >= 2:
                # Calculate coefficient of variation
                pred_std = np.std(predictions)
                pred_mean = np.mean(predictions)
                cv = pred_std / pred_mean if pred_mean != 0 else 0
                
                if cv > 0.05:  # More than 5% variation
                    logger.warning(f"⚠️ High model disagreement: CV = {cv:.3f}")
                else:
                    logger.info(f"✅ Good model agreement: CV = {cv:.3f}")
        
        except Exception as e:
            logger.warning(f"⚠️ Error checking model agreement: {e}")
    
    def _report_system_status(self):
        """Report comprehensive system status."""
        status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'active_models': len(self.model_pool),
            'data_points': len(self.latest_data) if self.latest_data is not None else 0,
            'last_update': None,
            'integrity_checks': len(self.data_integrity_checks)
        }
        
        if self.latest_data is not None and len(self.latest_data) > 0:
            status['last_update'] = str(self.latest_data['date'].iloc[-1])
            status['current_price'] = f"${self.latest_data['price'].iloc[-1]:.2f}"
        
        # Log status every 10 minutes
        current_time = datetime.now()
        if not hasattr(self, '_last_status_report') or \
           (current_time - self._last_status_report).seconds >= 600:
            
            logger.info("📊 SYSTEM STATUS REPORT:")
            logger.info(f"   🤖 Active Models: {status['active_models']}")
            logger.info(f"   📈 Data Points: {status['data_points']}")
            logger.info(f"   💰 Current Price: {status.get('current_price', 'N/A')}")
            logger.info(f"   ✅ Integrity Checks: {status['integrity_checks']}")
            
            self._last_status_report = current_time
    
    def get_latest_prediction(self):
        """Get the latest ensemble prediction from all active models."""
        if len(self.model_pool) == 0:
            return None
        
        try:
            all_forecasts = []
            
            for model_id, model_info in self.model_pool.items():
                try:
                    forecaster = model_info['forecaster']
                    forecast = forecaster.generate_12_hour_forecast(self.latest_data)
                    if forecast is not None:
                        all_forecasts.append(forecast)
                except Exception as e:
                    logger.warning(f"⚠️ {model_id} prediction failed: {e}")
            
            if not all_forecasts:
                return None
            
            # Ensemble the predictions
            ensemble_forecast = all_forecasts[0].copy()
            
            if len(all_forecasts) > 1:
                # Average predictions
                price_matrix = np.array([f['predicted_price'].values for f in all_forecasts])
                ensemble_forecast['predicted_price'] = np.mean(price_matrix, axis=0)
                ensemble_forecast['prediction_std'] = np.std(price_matrix, axis=0)
            
            return ensemble_forecast
            
        except Exception as e:
            logger.error(f"❌ Error generating ensemble prediction: {e}")
            return None
    
    def get_system_metrics(self):
        """Get comprehensive system metrics."""
        return {
            'active_models': len(self.model_pool),
            'data_points': len(self.latest_data) if self.latest_data is not None else 0,
            'integrity_checks': len(self.data_integrity_checks),
            'last_integrity_check': self.data_integrity_checks[-1] if self.data_integrity_checks else None,
            'model_details': {
                model_id: {
                    'training_time': info['training_time'],
                    'data_size': info['data_size'],
                    'timestamp': info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                }
                for model_id, info in self.model_pool.items()
            }
        }


def run_continuous_training_demo(initial_data, duration_minutes=15):
    """Run a demonstration of the continuous training system."""
    
    print("\n" + "="*80)
    print("🔄 CONTINUOUS TRAINING SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize system
    system = ContinuousTrainingSystem(update_interval=30, min_models=3)  # 30 seconds for demo
    
    # Start system
    system.start_continuous_system(initial_data)
    
    try:
        print(f"🏃 Running for {duration_minutes} minutes...")
        print("📊 Watch the logs for real-time updates...")
        
        # Run for specified duration
        start_time = time.time()
        while time.time() - start_time < duration_minutes * 60:
            time.sleep(10)  # Check every 10 seconds
            
            # Get latest prediction
            prediction = system.get_latest_prediction()
            if prediction is not None:
                current_price = system.latest_data['price'].iloc[-1] if system.latest_data is not None else 0
                next_price = prediction['predicted_price'].iloc[0]
                change_pct = ((next_price - current_price) / current_price) * 100
                
                print(f"🔮 Latest Prediction: ${current_price:.2f} → ${next_price:.2f} ({change_pct:+.2f}%)")
        
        # Final metrics
        metrics = system.get_system_metrics()
        print("\n📊 FINAL SYSTEM METRICS:")
        print(f"   🤖 Active Models: {metrics['active_models']}")
        print(f"   📈 Data Points: {metrics['data_points']}")
        print(f"   ✅ Integrity Checks: {metrics['integrity_checks']}")
        
    finally:
        # Stop system
        system.stop_continuous_system()
        print("🛑 System stopped")
    
    return system


if __name__ == "__main__":
    # This will be called from the main script
    pass