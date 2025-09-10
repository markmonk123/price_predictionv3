#!/usr/bin/env python3
"""
Enhanced Continuous Model Training and Data Update System
Implements automatic data updates every 5 minutes and progressive model training
with historical model results inclusion and denormalized data processing.
"""

import threading
import time
import queue
import logging
import pickle
import json
import copy
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_preprocessor import DataPreprocessor, preprocess_data_for_model
from gpu_optimizer import GPUOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressiveTrainingSystem:
    """Enhanced system for continuous model training with progressive learning and denormalized data processing."""
    
    def __init__(self, update_interval=300, min_models=3, enable_gpu=True):  # 5 minutes default
        self.update_interval = update_interval  # seconds
        self.min_models = min_models
        self.is_running = False
        self.data_queue = queue.Queue()
        self.model_queue = queue.Queue()
        self.latest_data = None
        self.model_pool = {}
        self.training_pool = {}
        self.data_integrity_checks = []
        self.prediction_history = []  # Store prediction results for progressive learning
        self.model_performance_history = {}  # Track model performance over time
        
        # Initialize preprocessor and GPU optimizer
        self.preprocessor = DataPreprocessor()
        self.gpu_optimizer = GPUOptimizer() if enable_gpu else None
        
        # Progressive learning components
        self.heuristic_features = {}  # Store learned patterns
        self.ensemble_weights = {}  # Dynamic ensemble weights
        self.model_confidence_scores = {}  # Track model reliability
        
        # Data storage for progressive training
        self.historical_training_data = []
        self.max_historical_samples = 10000  # Keep last 10k samples
        
        # Thread pools
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.training_threads = []
        
        # Performance tracking
        self.training_metrics = {
            'total_retrains': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'data_updates': 0,
            'last_update_time': None,
            'average_training_time': 0
        }
        
    def start_progressive_system(self, initial_data):
        """Start the progressive training and data update system."""
        logger.info("🚀 Starting Progressive Training System")
        logger.info(f"   ⏰ Update interval: {self.update_interval} seconds")
        logger.info(f"   🤖 Minimum models: {self.min_models}")
        logger.info(f"   🧠 Progressive learning: ENABLED")
        logger.info(f"   🎮 GPU optimization: {'ENABLED' if self.gpu_optimizer else 'DISABLED'}")
        
        # Initialize GPU optimization if available
        if self.gpu_optimizer:
            gpu_results = self.gpu_optimizer.run_full_optimization()
            logger.info(f"   🎮 GPU setup: {len(gpu_results['gpu_info']['gpus'])} GPU(s) found")
        
        self.is_running = True
        
        # Process initial data with preprocessor
        processed_data, metadata = self.preprocessor.prepare_denormalized_data(initial_data)
        self.latest_data = processed_data
        self._add_to_historical_training_data(processed_data)
        
        logger.info(f"   📊 Initial data: {processed_data.shape[0]} samples, {len(metadata['feature_columns'])} features")
        
        # Start data update thread
        data_thread = threading.Thread(target=self._progressive_data_update_loop, daemon=True)
        data_thread.start()
        
        # Start progressive model training threads
        for i in range(self.min_models):
            training_thread = threading.Thread(
                target=self._progressive_model_training_loop,
                args=(f"progressive_model_{i+1}",),
                daemon=True
            )
            training_thread.start()
            self.training_threads.append(training_thread)
        
        # Start heuristic learning thread
        heuristic_thread = threading.Thread(target=self._heuristic_learning_loop, daemon=True)
        heuristic_thread.start()
        
        # Start integrity monitoring thread
        integrity_thread = threading.Thread(target=self._integrity_monitor_loop, daemon=True)
        integrity_thread.start()
        
        logger.info("✅ All progressive systems started successfully")
        
    def stop_progressive_system(self):
        """Stop the progressive training system."""
        logger.info("🛑 Stopping Progressive Training System")
        self.is_running = False
        self.executor.shutdown(wait=True)
        
        # Save current state
        self._save_progressive_state()
        
    def _add_to_historical_training_data(self, processed_data):
        """Add new data to historical training set for progressive learning."""
        # Only keep data with valid targets
        valid_data = processed_data.dropna(subset=['target']).copy()
        
        if len(valid_data) > 0:
            self.historical_training_data.append(valid_data)
            
            # Maintain maximum size
            total_samples = sum(len(df) for df in self.historical_training_data)
            while total_samples > self.max_historical_samples and len(self.historical_training_data) > 1:
                removed = self.historical_training_data.pop(0)
                total_samples -= len(removed)
            
            logger.info(f"📚 Historical training data: {total_samples} samples across {len(self.historical_training_data)} batches")
    
    def _get_combined_training_data(self) -> pd.DataFrame:
        """Combine historical training data with prediction results for progressive learning."""
        if not self.historical_training_data:
            return pd.DataFrame()
        
        # Combine all historical data
        combined_data = pd.concat(self.historical_training_data, ignore_index=True)
        
        # Add prediction results as additional features for progressive learning
        if self.prediction_history:
            prediction_features = self._create_prediction_features(combined_data)
            combined_data = pd.concat([combined_data, prediction_features], axis=1)
        
        return combined_data
    
    def _create_prediction_features(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from historical predictions for progressive learning."""
        prediction_features = pd.DataFrame(index=base_data.index)
        
        if len(self.prediction_history) >= 3:
            # Recent prediction trends
            recent_predictions = [p['prediction'] for p in self.prediction_history[-5:]]
            prediction_features['recent_pred_trend'] = np.mean(recent_predictions) if recent_predictions else 0
            prediction_features['pred_volatility'] = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
            
            # Model confidence patterns
            recent_confidence = [p.get('confidence', 0.5) for p in self.prediction_history[-5:]]
            prediction_features['recent_confidence'] = np.mean(recent_confidence)
            
            # Success rate features
            if len(self.prediction_history) >= 10:
                recent_success = [p.get('was_correct', False) for p in self.prediction_history[-10:]]
                prediction_features['recent_success_rate'] = np.mean(recent_success)
        
        # Fill any missing values
        prediction_features = prediction_features.fillna(0)
        
        return prediction_features
    
    def _progressive_data_update_loop(self):
        """Enhanced data update loop with progressive learning integration."""
        logger.info("🔄 Progressive data update loop started")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Fetch new data (this would typically come from an API)
                new_data = self._fetch_new_market_data()
                
                if new_data is not None and len(new_data) > 0:
                    # Process new data with preprocessor (denormalized)
                    processed_data, metadata = self.preprocessor.prepare_denormalized_data(new_data)
                    
                    # Validate data integrity
                    integrity_check = self.preprocessor.validate_data_integrity(processed_data)
                    
                    if integrity_check['validation_passed']:
                        # Update latest data
                        old_data = self.latest_data
                        self.latest_data = processed_data
                        
                        # Add to historical training data
                        self._add_to_historical_training_data(processed_data)
                        
                        # Store prediction result if we have a previous prediction
                        if old_data is not None and len(self.prediction_history) > 0:
                            self._update_prediction_accuracy(old_data, processed_data)
                        
                        # Notify models of new data
                        try:
                            self.data_queue.put({
                                'data': processed_data,
                                'metadata': metadata,
                                'timestamp': datetime.now(),
                                'integrity_check': integrity_check
                            }, timeout=1)
                        except queue.Full:
                            logger.warning("⚠️ Data queue full - dropping oldest update")
                            try:
                                self.data_queue.get_nowait()  # Remove oldest
                                self.data_queue.put({
                                    'data': processed_data,
                                    'metadata': metadata,
                                    'timestamp': datetime.now(),
                                    'integrity_check': integrity_check
                                })
                            except queue.Empty:
                                pass
                        
                        self.training_metrics['data_updates'] += 1
                        self.training_metrics['last_update_time'] = datetime.now()
                        
                        logger.info(f"📊 Data updated: {processed_data.shape[0]} samples, {len(metadata['feature_columns'])} features")
                    else:
                        logger.warning(f"⚠️ Data integrity check failed: {integrity_check['issues']}")
                
                # Wait for next update
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in data update loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _update_prediction_accuracy(self, old_data: pd.DataFrame, new_data: pd.DataFrame):
        """Update the accuracy of previous predictions for progressive learning."""
        if len(self.prediction_history) == 0:
            return
        
        # Get the most recent prediction
        last_prediction = self.prediction_history[-1]
        
        # Check if the prediction was correct
        if len(old_data) > 0 and len(new_data) > 0:
            actual_price_change = (new_data['price'].iloc[-1] - old_data['price'].iloc[-1]) / old_data['price'].iloc[-1]
            predicted_direction = last_prediction['prediction']
            
            # Determine if prediction was correct
            threshold = 0.01  # 1% threshold
            was_correct = False
            
            if predicted_direction == 1 and actual_price_change >= threshold:
                was_correct = True
            elif predicted_direction == -1 and actual_price_change <= -threshold:
                was_correct = True
            elif predicted_direction == 0 and abs(actual_price_change) < threshold:
                was_correct = True
            
            # Update prediction history
            last_prediction['was_correct'] = was_correct
            last_prediction['actual_change'] = actual_price_change
            
            # Update metrics
            if was_correct:
                self.training_metrics['successful_predictions'] += 1
            else:
                self.training_metrics['failed_predictions'] += 1
            
            logger.info(f"📈 Prediction accuracy updated: {'✅ Correct' if was_correct else '❌ Incorrect'} "
                       f"(Predicted: {predicted_direction}, Actual change: {actual_price_change:.4f})")
    
    def _progressive_model_training_loop(self, model_id):
        """Enhanced model training loop with progressive learning."""
        logger.info(f"🧠 Progressive model training loop started for {model_id}")
        
        try:
            # Import ML libraries
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            import joblib
        except ImportError as e:
            logger.error(f"❌ Required ML libraries not available: {e}")
            return
        
        # Initialize progressive model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model_performance = {'accuracy_history': [], 'training_times': []}
        
        while self.is_running:
            try:
                # Wait for new data
                if not self.data_queue.empty():
                    data_update = self.data_queue.get(timeout=1)
                    new_data = data_update['data']
                    metadata = data_update['metadata']
                    
                    # Get combined training data (historical + progressive features)
                    combined_data = self._get_combined_training_data()
                    
                    if len(combined_data) < 100:  # Need minimum data for training
                        logger.info(f"🔄 {model_id}: Insufficient data ({len(combined_data)} samples), waiting...")
                        time.sleep(5)
                        continue
                    
                    logger.info(f"🔄 {model_id}: Progressive training with {len(combined_data)} samples")
                    
                    # Prepare training data
                    feature_columns = metadata['feature_columns']
                    X = combined_data[feature_columns].fillna(0)
                    y = combined_data['target'].dropna()
                    
                    # Align X and y
                    min_len = min(len(X), len(y))
                    X = X.iloc[:min_len]
                    y = y.iloc[:min_len]
                    
                    if len(X) < 50:
                        continue
                    
                    # Split data for validation
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=42
                    )
                    
                    # Progressive training with warm start
                    start_time = time.time()
                    
                    # If this is not the first training, use warm start for progressive learning
                    if hasattr(model, 'n_estimators_') and model.n_estimators_ > 0:
                        # Add more estimators for progressive learning
                        model.n_estimators += 20
                        model.warm_start = True
                    
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Evaluate model
                    val_predictions = model.predict(X_val)
                    accuracy = accuracy_score(y_val, val_predictions)
                    
                    # Update performance tracking
                    model_performance['accuracy_history'].append(accuracy)
                    model_performance['training_times'].append(training_time)
                    self.model_confidence_scores[model_id] = accuracy
                    
                    # Store progressive model
                    self.model_pool[model_id] = {
                        'model': model,
                        'accuracy': accuracy,
                        'training_time': training_time,
                        'data_size': len(combined_data),
                        'timestamp': datetime.now(),
                        'model_id': model_id,
                        'performance_history': model_performance.copy(),
                        'progressive_features': True
                    }
                    
                    self.training_metrics['total_retrains'] += 1
                    avg_time = np.mean(model_performance['training_times'])
                    self.training_metrics['average_training_time'] = avg_time
                    
                    logger.info(f"✅ {model_id}: Progressive training completed - "
                               f"Accuracy: {accuracy:.3f}, Time: {training_time:.2f}s")
                    
                    # Generate prediction on latest data
                    if len(new_data) > 0:
                        latest_features = new_data[feature_columns].fillna(0).iloc[-1:] 
                        try:
                            prediction = model.predict(latest_features)[0]
                            prediction_proba = model.predict_proba(latest_features)[0]
                            confidence = np.max(prediction_proba)
                            
                            # Store prediction for progressive learning
                            prediction_record = {
                                'model_id': model_id,
                                'prediction': prediction,
                                'confidence': confidence,
                                'timestamp': datetime.now(),
                                'features_used': len(feature_columns),
                                'was_correct': None  # Will be updated later
                            }
                            
                            self.prediction_history.append(prediction_record)
                            
                            # Keep only recent predictions
                            if len(self.prediction_history) > 100:
                                self.prediction_history = self.prediction_history[-100:]
                            
                            logger.info(f"🔮 {model_id}: Prediction = {prediction}, Confidence = {confidence:.3f}")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ {model_id}: Prediction generation failed: {e}")
                
                else:
                    time.sleep(1)  # Wait if no new data
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error in {model_id} progressive training loop: {e}")
                time.sleep(5)
    
    def _heuristic_learning_loop(self):
        """Background loop for learning heuristic patterns from model performance."""
        logger.info("🧠 Heuristic learning loop started")
        
        while self.is_running:
            try:
                if len(self.prediction_history) >= 10 and len(self.model_confidence_scores) >= 2:
                    # Learn from prediction patterns
                    self._update_ensemble_weights()
                    self._extract_heuristic_patterns()
                    
                time.sleep(120)  # Run every 2 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in heuristic learning loop: {e}")
                time.sleep(60)
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on model performance."""
        if len(self.model_confidence_scores) < 2:
            return
        
        # Calculate weights based on recent accuracy
        total_confidence = sum(self.model_confidence_scores.values())
        
        if total_confidence > 0:
            for model_id, confidence in self.model_confidence_scores.items():
                self.ensemble_weights[model_id] = confidence / total_confidence
        else:
            # Equal weights if no confidence info
            equal_weight = 1.0 / len(self.model_confidence_scores)
            for model_id in self.model_confidence_scores.keys():
                self.ensemble_weights[model_id] = equal_weight
        
        logger.info(f"🎯 Updated ensemble weights: {self.ensemble_weights}")
    
    def _extract_heuristic_patterns(self):
        """Extract heuristic patterns from prediction history."""
        if len(self.prediction_history) < 20:
            return
        
        # Analyze prediction success patterns
        recent_predictions = self.prediction_history[-20:]
        correct_predictions = [p for p in recent_predictions if p.get('was_correct') is True]
        
        if len(correct_predictions) >= 5:
            # Extract patterns from successful predictions
            avg_confidence = np.mean([p['confidence'] for p in correct_predictions])
            successful_models = [p['model_id'] for p in correct_predictions]
            
            self.heuristic_features['min_confidence_threshold'] = avg_confidence * 0.8
            self.heuristic_features['preferred_models'] = list(set(successful_models))
            
            logger.info(f"🎓 Learned heuristics: Min confidence = {avg_confidence*0.8:.3f}, "
                       f"Preferred models = {len(set(successful_models))}")
    
    def _fetch_new_market_data(self):
        """Fetch new market data (simulation for demo)."""
        if self.latest_data is None or len(self.latest_data) == 0:
            return None
        
        # Simulate new data point
        last_point = self.latest_data.iloc[-1]
        last_time = pd.to_datetime(last_point['date'])
        last_price = last_point['price']
        
        # Generate new timestamp (5 minutes later)
        new_time = last_time + timedelta(minutes=5)
        
        # Generate new price with market-like behavior
        # Include some trend and volatility
        trend = np.random.normal(0, 0.001)  # Small trend
        volatility = np.random.normal(0, 0.01)  # 1% volatility
        price_change = trend + volatility
        new_price = last_price * (1 + price_change)
        
        # Create new data point
        new_data = pd.DataFrame({
            'date': [new_time],
            'price': [new_price]
        })
        
        return new_data
    
    def _save_progressive_state(self):
        """Save the current progressive learning state."""
        try:
            state = {
                'ensemble_weights': self.ensemble_weights,
                'heuristic_features': self.heuristic_features,
                'model_confidence_scores': self.model_confidence_scores,
                'training_metrics': self.training_metrics,
                'prediction_history': self.prediction_history[-50:],  # Save last 50
            }
            
            # Save to file
            state_file = Path('progressive_training_state.json')
            with open(state_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                import json
                json.dump(state, f, default=str, indent=2)
            
            logger.info(f"💾 Progressive training state saved to {state_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save progressive state: {e}")
    
    def load_progressive_state(self, state_file='progressive_training_state.json'):
        """Load previous progressive learning state."""
        try:
            with open(state_file, 'r') as f:
                import json
                state = json.load(f)
            
            self.ensemble_weights = state.get('ensemble_weights', {})
            self.heuristic_features = state.get('heuristic_features', {})
            self.model_confidence_scores = state.get('model_confidence_scores', {})
            self.training_metrics = state.get('training_metrics', self.training_metrics)
            self.prediction_history = state.get('prediction_history', [])
            
            logger.info(f"📚 Progressive training state loaded from {state_file}")
            
        except FileNotFoundError:
            logger.info("ℹ️ No previous progressive state found, starting fresh")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load progressive state: {e}")
    
    def get_progressive_prediction(self):
        """Get ensemble prediction using progressive learning weights."""
        if len(self.model_pool) == 0:
            return None
        
        try:
            all_predictions = []
            model_weights = []
            
            for model_id, model_info in self.model_pool.items():
                try:
                    model = model_info['model']
                    
                    # Prepare latest data for prediction
                    if self.latest_data is not None and len(self.latest_data) > 0:
                        # Get preprocessed features
                        processed_data, metadata = self.preprocessor.prepare_denormalized_data(self.latest_data)
                        feature_columns = metadata['feature_columns']
                        latest_features = processed_data[feature_columns].fillna(0).iloc[-1:]
                        
                        prediction = model.predict(latest_features)[0]
                        prediction_proba = model.predict_proba(latest_features)[0]
                        confidence = np.max(prediction_proba)
                        
                        # Apply heuristic filtering
                        min_confidence = self.heuristic_features.get('min_confidence_threshold', 0.4)
                        if confidence >= min_confidence:
                            all_predictions.append(prediction)
                            weight = self.ensemble_weights.get(model_id, 1.0) * confidence
                            model_weights.append(weight)
                        
                except Exception as e:
                    logger.warning(f"⚠️ {model_id} prediction failed: {e}")
            
            if not all_predictions:
                return None
            
            # Weighted ensemble prediction
            if len(all_predictions) == 1:
                final_prediction = all_predictions[0]
            else:
                # Weighted voting
                weighted_sum = sum(pred * weight for pred, weight in zip(all_predictions, model_weights))
                total_weight = sum(model_weights)
                final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(all_predictions)
                final_prediction = int(np.round(final_prediction))  # Ensure integer class
            
            result = {
                'prediction': final_prediction,
                'confidence': np.mean(model_weights) / len(all_predictions) if model_weights else 0.5,
                'models_used': len(all_predictions),
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating progressive prediction: {e}")
            return None

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
                            'date': [predicted_time, actual_time],
                            'price': [self.last_prediction, actual_price]
                        })
                        new_data = pd.concat([head_rows, historical], ignore_index=True)

                    logger.info(f"🔄 {model_id}: Retraining with {len(new_data)} data points")

                    # Resample to 30-minute intervals for training
                    train_data = new_data.copy()
                    train_data['date'] = pd.to_datetime(train_data['date'])
                    train_data = (
                        train_data.resample('30T', on='date').last().dropna().reset_index()
                    )

                    # Retrain models
                    start_time = time.time()
                    success = forecaster.train_models(train_data)
                    training_time = time.time() - start_time

                    if success:
                        # Store trained model
                        self.model_pool[model_id] = {
                            'forecaster': forecaster,
                            'training_time': training_time,
                            'data_size': len(train_data),
                            'timestamp': datetime.now(),
                            'model_id': model_id
                        }

                        logger.info(f"✅ {model_id}: Training completed in {training_time:.2f}s")

                        # Generate and log 12-hour forecast
                        forecast = forecaster.generate_12_hour_forecast(train_data)
                        if forecast is not None:
                            self.last_prediction = forecast['predicted_price'].iloc[0]
                            for _, row in forecast.iterrows():
                                logger.info(
                                    f"🕒 {row['timestamp']}: ${row['predicted_price']:.2f}"
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