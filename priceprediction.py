import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import matplotlib.pyplot as plt
import requests
import time
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from data_preprocessor import DataPreprocessor, preprocess_data_for_model
from gpu_optimizer import GPUOptimizer

def enhanced_bitcoin_prediction_with_denormalized_data():
    """
    Enhanced Bitcoin price prediction using denormalized data processing
    and the new decentralized preprocessing system.
    """
    print("\n🚀 ENHANCED BITCOIN PRICE PREDICTION WITH DENORMALIZED DATA")
    print("="*80)
    
    # Initialize GPU optimization
    print("🎮 Setting up GPU optimization...")
    gpu_optimizer = GPUOptimizer()
    gpu_results = gpu_optimizer.run_full_optimization()
    
    # Initialize the decentralized data preprocessor
    print("🔧 Initializing decentralized data preprocessor...")
    preprocessor = DataPreprocessor()
    
    # Fetch and process data
    print("📊 Fetching Bitcoin data...")
    df = fetch_bitcoin_futures_data()
    
    if df is None or len(df) < 100:
        print("❌ Failed to fetch sufficient data")
        return None
    
    print(f"✅ Fetched {len(df)} data points")
    
    # Process data with denormalized preprocessing
    print("🔄 Processing data with denormalized preprocessing...")
    processed_df, metadata = preprocessor.prepare_denormalized_data(df)
    
    # Validate data integrity
    integrity_check = preprocessor.validate_data_integrity(processed_df)
    print(f"🔍 Data integrity check: {'✅ Passed' if integrity_check['validation_passed'] else '❌ Failed'}")
    
    if not integrity_check['validation_passed']:
        print(f"⚠️ Data integrity issues: {integrity_check['issues']}")
    
    # Remove rows with NaN targets for training
    training_data = processed_df.dropna(subset=['target']).copy()
    latest_data = processed_df.tail(1).copy()  # Keep latest for prediction
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   📈 Total processed samples: {len(processed_df)}")
    print(f"   🎯 Training samples: {len(training_data)}")
    print(f"   🔮 Prediction samples: {len(latest_data)}")
    print(f"   📋 Features: {len(metadata['feature_columns'])}")
    print(f"   📊 Memory usage: {integrity_check['memory_usage_mb']:.2f} MB")
    print(f"   🔧 Denormalized: {metadata['denormalized']}")
    print(f"   🏗️ Structure preserved: {metadata['structure_preserved']}")
    
    if len(training_data) < 50:
        print("❌ Insufficient training data")
        return None
    
    # Prepare features for model training (already denormalized)
    feature_columns = metadata['feature_columns']
    print(f"🎯 Using {len(feature_columns)} denormalized features")
    
    X_training = training_data[feature_columns]
    y_training = training_data['target']
    X_latest = latest_data[feature_columns]
    
    # Handle any remaining missing values
    print("🔧 Final data cleaning...")
    X_training = X_training.fillna(0)
    X_latest = X_latest.fillna(0)
    
    # Check class distribution
    class_counts = y_training.value_counts().sort_index()
    print(f"\n📈 Target Class Distribution:")
    print(f"   📉 Decrease ≥1.0%: {class_counts.get(-1, 0):3d} ({class_counts.get(-1, 0)/len(y_training)*100:5.1f}%)")
    print(f"   ➡️  No Change:     {class_counts.get(0, 0):3d} ({class_counts.get(0, 0)/len(y_training)*100:5.1f}%)")
    print(f"   📈 Increase ≥1.0%: {class_counts.get(1, 0):3d} ({class_counts.get(1, 0)/len(y_training)*100:5.1f}%)")
    
    # Time series split for proper validation
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, shuffle=False)
    print(f"🔄 Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    
    # Feature selection (working with denormalized data)
    print("🎯 Performing feature selection on denormalized data...")
    k_best = min(50, len(feature_columns))
    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    X_latest_selected = selector.transform(X_latest)
    
    selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
    print(f"✅ Selected {len(selected_features)} most informative denormalized features")
    
    # Initialize optimized ML models
    print("🤖 Initializing ML models with GPU optimization...")
    
    # Use GPU-optimized batch sizes if available
    batch_sizes = gpu_results.get('recommended_batch_sizes', {})
    n_estimators = batch_sizes.get('medium_model_batch_size', 200)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=min(n_estimators, 200),
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        ),
        
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            C=0.1,
            random_state=42
        ),
        
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            n_jobs=-1
        )
    }
    
    # Train and evaluate models with denormalized data
    print("\n🏋️ Training models with denormalized data...")
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"   🔄 Training {name}...")
        start_time = time.time()
        
        try:
            model.fit(X_train_selected, y_train)
            training_time = time.time() - start_time
            
            # Evaluate on test set
            test_predictions = model.predict(X_test_selected)
            test_accuracy = accuracy_score(y_test, test_predictions)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            model_results[name] = {
                'test_accuracy': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time
            }
            
            trained_models[name] = model
            
            print(f"   ✅ {name}: Accuracy = {test_accuracy:.3f}, CV = {cv_mean:.3f}±{cv_std:.3f}, Time = {training_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ {name}: Training failed - {e}")
    
    # Create ensemble with best models
    print("\n🎯 Creating optimized ensemble...")
    best_models = sorted(model_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
    ensemble_models = [(name, trained_models[name]) for name, _ in best_models]
    
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='soft'
    )
    
    print(f"   🤖 Ensemble models: {[name for name, _ in ensemble_models]}")
    ensemble.fit(X_train_selected, y_train)
    
    # Final ensemble evaluation
    ensemble_predictions = ensemble.predict(X_test_selected)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    print(f"   ✅ Ensemble accuracy: {ensemble_accuracy:.3f}")
    
    # Generate final prediction on latest data (denormalized)
    print("\n🔮 Generating prediction with denormalized data...")
    latest_prediction = ensemble.predict(X_latest_selected)[0]
    latest_probabilities = ensemble.predict_proba(X_latest_selected)[0]
    confidence = np.max(latest_probabilities)
    
    # Interpret prediction
    prediction_labels = {-1: "📉 DECREASE ≥1.0%", 0: "➡️ NO SIGNIFICANT CHANGE", 1: "📈 INCREASE ≥1.0%"}
    prediction_label = prediction_labels[latest_prediction]
    
    current_price = latest_data['price'].iloc[0]
    
    print(f"\n🎯 FINAL PREDICTION RESULTS:")
    print(f"   💰 Current Price: ${current_price:.2f}")
    print(f"   🔮 Prediction: {prediction_label}")
    print(f"   🎯 Confidence: {confidence:.1%}")
    print(f"   🤖 Models Used: {len(ensemble_models)}")
    print(f"   📊 Features Used: {len(selected_features)} (denormalized)")
    print(f"   ⚡ GPU Optimized: {gpu_results['gpu_info']['cuda_available']}")
    
    # Return comprehensive results
    results = {
        'prediction': latest_prediction,
        'prediction_label': prediction_label,
        'confidence': confidence,
        'current_price': current_price,
        'model_results': model_results,
        'ensemble_accuracy': ensemble_accuracy,
        'features_used': len(selected_features),
        'data_denormalized': True,
        'gpu_optimized': gpu_results['gpu_info']['cuda_available'],
        'data_integrity': integrity_check['validation_passed'],
        'preprocessor_metadata': metadata
    }
    
    return results
    """Create comprehensive technical indicators and ML features for Bitcoin price prediction."""
    df = df.copy()  # Avoid modifying original DataFrame
    
    # Basic time features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_end'] = (df['date'].dt.day >= 28).astype(int)
    
    # Price lag features
    for lag in [1, 2, 3, 5, 7, 10, 14, 21]:
        df[f'price_lag{lag}'] = df['price'].shift(lag)
        df[f'return_lag{lag}'] = df['price'].pct_change(lag)
    
    # Rolling statistics
    for window in [5, 10, 15, 20, 30, 50]:
        df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['price'].ewm(span=window).mean()
        df[f'std_{window}'] = df['price'].rolling(window=window).std()
        df[f'min_{window}'] = df['price'].rolling(window=window).min()
        df[f'max_{window}'] = df['price'].rolling(window=window).max()
        df[f'median_{window}'] = df['price'].rolling(window=window).median()
        
        # Price position within range
        df[f'price_position_{window}'] = (df['price'] - df[f'min_{window}']) / (df[f'max_{window}'] - df[f'min_{window}'])
        
        # Price relative to moving averages
        df[f'price_to_sma_{window}'] = df['price'] / df[f'sma_{window}']
        df[f'price_to_ema_{window}'] = df['price'] / df[f'ema_{window}']
    
    # Volatility features
    for window in [5, 10, 20, 30]:
        returns = df['price'].pct_change()
        df[f'volatility_{window}'] = returns.rolling(window=window).std()
        df[f'realized_vol_{window}'] = np.sqrt(252) * returns.rolling(window=window).std()
    
    # MACD indicators (multiple timeframes)
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
        ema_fast = df['price'].ewm(span=fast).mean()
        ema_slow = df['price'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        df[f'macd_{fast}_{slow}'] = macd
        df[f'macd_signal_{fast}_{slow}'] = macd_signal
        df[f'macd_histogram_{fast}_{slow}'] = macd - macd_signal
        df[f'macd_crossover_{fast}_{slow}'] = (macd > macd_signal).astype(int)
    
    # RSI (Relative Strength Index) - multiple periods
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    for period in [7, 14, 21, 30]:
        df[f'rsi_{period}'] = calculate_rsi(df['price'], period)
        df[f'rsi_oversold_{period}'] = (df[f'rsi_{period}'] < 30).astype(int)
        df[f'rsi_overbought_{period}'] = (df[f'rsi_{period}'] > 70).astype(int)
    
    # Bollinger Bands (multiple periods)
    for period in [10, 20, 30]:
        for std_dev in [1.5, 2.0, 2.5]:
            bb_ma = df['price'].rolling(window=period).mean()
            bb_std = df['price'].rolling(window=period).std()
            
            df[f'bb_upper_{period}_{std_dev}'] = bb_ma + (bb_std * std_dev)
            df[f'bb_lower_{period}_{std_dev}'] = bb_ma - (bb_std * std_dev)
            df[f'bb_width_{period}_{std_dev}'] = (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}']) / bb_ma
            df[f'bb_position_{period}_{std_dev}'] = (df['price'] - df[f'bb_lower_{period}_{std_dev}']) / (df[f'bb_upper_{period}_{std_dev}'] - df[f'bb_lower_{period}_{std_dev}'])
    
    # Linear Regression Slope (momentum indicators)
    def calc_slope(window):
        if len(window) < len(window):
            return np.nan
        x = np.arange(len(window))
        slope, _, r_value, _, _ = stats.linregress(x, window)
        return slope
    
    def calc_r_squared(window):
        if len(window) < len(window):
            return np.nan
        x = np.arange(len(window))
        _, _, r_value, _, _ = stats.linregress(x, window)
        return r_value ** 2
    
    for period in [5, 10, 14, 21, 30]:
        df[f'lr_slope_{period}'] = df['price'].rolling(window=period).apply(calc_slope, raw=True)
        df[f'lr_r2_{period}'] = df['price'].rolling(window=period).apply(calc_r_squared, raw=True)
    
    # Momentum indicators
    for period in [3, 7, 14, 21, 30]:
        df[f'momentum_{period}'] = (df['price'] / df['price'].shift(period) - 1) * 100
        df[f'roc_{period}'] = ((df['price'] - df['price'].shift(period)) / df['price'].shift(period)) * 100
    
    # Price acceleration (use periods that exist in momentum)
    for period in [7, 14, 21]:
        df[f'acceleration_{period}'] = df[f'momentum_{period}'].diff()
    
    # Stochastic Oscillator
    for period in [14, 21]:
        lowest_low = df['price'].rolling(window=period).min()
        highest_high = df['price'].rolling(window=period).max()
        df[f'stoch_k_{period}'] = 100 * (df['price'] - lowest_low) / (highest_high - lowest_low)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
    
    # Williams %R
    for period in [14, 21]:
        highest_high = df['price'].rolling(window=period).max()
        lowest_low = df['price'].rolling(window=period).min()
        df[f'williams_r_{period}'] = -100 * (highest_high - df['price']) / (highest_high - lowest_low)
    
    # Detrended Price Oscillator (DPO)
    for period in [10, 20, 30]:
        sma = df['price'].rolling(window=period).mean()
        shift_periods = (period // 2) + 1
        df[f'dpo_{period}'] = df['price'] - sma.shift(shift_periods)
    
    # Average True Range (using high-low as proxy)
    df['high_low_spread'] = (df['price'].rolling(window=5).max() - df['price'].rolling(window=5).min())
    for period in [7, 14, 21]:
        df[f'atr_{period}'] = df['high_low_spread'].rolling(window=period).mean()
    
    # Price channels and support/resistance
    for window in [10, 20, 50]:
        df[f'resistance_{window}'] = df['price'].rolling(window=window).max()
        df[f'support_{window}'] = df['price'].rolling(window=window).min()
        df[f'channel_position_{window}'] = (df['price'] - df[f'support_{window}']) / (df[f'resistance_{window}'] - df[f'support_{window}'])
    
    # Trend strength indicators
    for window in [10, 20, 50]:
        df[f'uptrend_{window}'] = (df['price'] > df[f'sma_{window}']).astype(int)
        df[f'strong_uptrend_{window}'] = ((df['price'] > df[f'sma_{window}']) & (df[f'sma_{window}'] > df[f'sma_{window}'].shift(1))).astype(int)
    
    # Volume proxy indicators (using price volatility)
    df['volume_proxy'] = df['price'].rolling(window=20).std()
    df['price_volume_trend'] = df['price'] * df['volume_proxy']
    df['volume_sma'] = df['volume_proxy'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume_proxy'] / df['volume_sma']
    
    # Fractal indicators
    def is_fractal_high(series, period=5):
        return series == series.rolling(window=period*2+1, center=True).max()
    
    def is_fractal_low(series, period=5):
        return series == series.rolling(window=period*2+1, center=True).min()
    
    df['fractal_high'] = is_fractal_high(df['price']).astype(int)
    df['fractal_low'] = is_fractal_low(df['price']).astype(int)
    
    # BLOCKCHAIN AND TRANSACTION FEATURES
    # Add blockchain-specific features if available
    blockchain_features = ['transaction_count', 'tx_variance', 'mempool_congestion', 
                          'estimated_conf_time', 'network_stress', 'tx_momentum', 
                          'tx_acceleration', 'price_tx_correlation', 'network_activity_score']
    
    for feature in blockchain_features:
        if feature in df.columns:
            # Transaction count features
            if feature == 'transaction_count':
                df['tx_count_ma_7'] = df[feature].rolling(window=7).mean()
                df['tx_count_ma_30'] = df[feature].rolling(window=30).mean()
                df['tx_count_trend'] = (df['tx_count_ma_7'] / df['tx_count_ma_30'] - 1) * 100
                df['tx_count_zscore'] = (df[feature] - df['tx_count_ma_30']) / df[feature].rolling(window=30).std()
                
            # Mempool stress indicators
            elif feature == 'mempool_congestion':
                df['mempool_stress_ma'] = df[feature].rolling(window=7).mean()
                df['mempool_spike'] = (df[feature] > df['mempool_stress_ma'] * 1.5).astype(int)
                
            # Confirmation time analysis
            elif feature == 'estimated_conf_time':
                df['conf_time_acceptable'] = (df[feature] <= 240).astype(int)  # <= 4 hours
                df['conf_time_ma'] = df[feature].rolling(window=7).mean()
                df['conf_time_volatility'] = df[feature].rolling(window=7).std()
                
            # Network activity patterns
            elif feature == 'network_activity_score':
                df['network_activity_ma'] = df[feature].rolling(window=7).mean()
                df['network_activity_trend'] = df[feature].diff()
                
    # Price-Transaction relationship features
    if 'transaction_count' in df.columns:
        # Transaction volume vs price movement correlation
        price_returns = df['price'].pct_change()
        tx_returns = df['transaction_count'].pct_change()
        
        df['price_tx_rolling_corr'] = price_returns.rolling(window=14).corr(tx_returns)
        df['tx_price_divergence'] = (tx_returns - price_returns).abs()
        
        # Transaction efficiency relative to price
        df['tx_per_dollar'] = df['transaction_count'] / df['price']
        df['tx_efficiency'] = df['tx_per_dollar'] / df['tx_per_dollar'].rolling(window=30).mean()
    
    # Classification target: 1 if price increases >=1.0% next day, -1 if decreases <=-1.0%, 0 otherwise
    df['future_price'] = df['price'].shift(-1)
    df['pct_change'] = (df['future_price'] - df['price']) / df['price']
    df['target'] = 0
    df.loc[df['pct_change'] >= pct_threshold, 'target'] = 1
    df.loc[df['pct_change'] <= -pct_threshold, 'target'] = -1
    
    # Keep the last row (most recent) for prediction even if target is NaN
    # Only drop NaN from training data, keep latest for prediction
    return df

# Fetch historical BTC-USD daily prices from Coinbase API
def fetch_bitcoin_futures_data():
    """
    Fetch the last 60,000 1-minute BTC-USD data points from Coinbase API,
    then resample to daily closing prices to match the rest of the analysis pipeline.
    """
    print("Fetching last 60,000 1-minute data points...")
    all_data = []
    total_points_to_fetch = 60000
    points_per_request = 300  # Coinbase API limit per request
    num_requests = total_points_to_fetch // points_per_request
    
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    # Start from the current time and go backwards
    end_time = datetime.utcnow()

    try:
        for i in range(num_requests):
            # Calculate the start time for this chunk
            # The API takes times in ISO 8601 format
            start_time = end_time - timedelta(minutes=points_per_request)
            
            params = {
                "granularity": 60,  # 60 seconds = 1 minute
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
            
            # The timestamp of the oldest candle becomes the end_time for the next request
            # This allows us to paginate backwards in time
            oldest_timestamp = data[-1][0]
            end_time = datetime.fromtimestamp(oldest_timestamp)

            time.sleep(0.5) # Add a delay to respect API rate limits

        if not all_data:
            raise ValueError("No data received from Coinbase API")

        # Convert raw list of lists to a DataFrame
        candles = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
        # Sort by time, remove any duplicates, and sort again
        candles = candles.sort_values("time", ascending=False).drop_duplicates(subset=['time']).sort_values("time")
        
        candles["date"] = pd.to_datetime(candles["time"], unit="s")
        candles["price"] = pd.to_numeric(candles["close"], errors='coerce')
        
        print(f"Successfully fetched {len(candles)} unique 1-minute data points.")
        
        # --- RESAMPLING TO DAILY DATA ---
        # The rest of the script (features, blockchain data) is based on a daily timeframe.
        # We resample the 1-minute data to daily data to ensure compatibility.
        print("Resampling 1-minute data to daily closing prices...")
        candles.set_index('date', inplace=True)
        daily_df = candles['price'].resample('D').last().dropna().to_frame().reset_index()
        
        print(f"Resampled to {len(daily_df)} days of BTC price data.")
        return daily_df

    except Exception as e:
        print(f"Error fetching data: {e}")
        # Fallback to simulated data for testing
        print("Using simulated data for testing...")
        dates = pd.date_range(start="2024-01-01", periods=300, freq='D')
        prices = np.cumsum(np.random.randn(300) * 1000) + 45000
        return pd.DataFrame({'date': dates, 'price': prices})


def fetch_blockchain_data():
    """Fetch Bitcoin blockchain and mempool data from multiple APIs."""
    blockchain_data = []
    
    try:
        print("🔗 Fetching blockchain transaction data...")
        
        # Fetch current mempool statistics
        mempool_url = "https://mempool.space/api/mempool"
        mempool_response = requests.get(mempool_url, timeout=10)
        
        if mempool_response.status_code == 200:
            mempool_data = mempool_response.json()
            print(f"   📊 Current mempool: {mempool_data.get('count', 0)} transactions")
            print(f"   ⏱️  Average fee: {mempool_data.get('vsize', 0)/1000:.1f} sat/vB")
        else:
            mempool_data = {}
            
        # Fetch recent blockchain statistics
        stats_url = "https://mempool.space/api/statistics"
        stats_response = requests.get(stats_url, timeout=10)
        
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            print(f"   🔗 Recent network activity retrieved")
        else:
            stats_data = {}
            
        # Fetch historical transaction count data (simplified approach)
        # Using blockchain.info API for historical data
        blockchain_info_url = "https://api.blockchain.info/charts/n-transactions?timespan=180days&format=json"
        blockchain_response = requests.get(blockchain_info_url, timeout=15)
        
        historical_txns = []
        if blockchain_response.status_code == 200:
            blockchain_data_raw = blockchain_response.json()
            print(f"   📈 Retrieved {len(blockchain_data_raw.get('values', []))} days of transaction data")
            
            for point in blockchain_data_raw.get('values', []):
                date = datetime.fromtimestamp(point['x'])
                tx_count = point['y']
                historical_txns.append({
                    'date': date,
                    'transaction_count': tx_count
                })
        
        # Create comprehensive blockchain dataset
        if historical_txns:
            df_blockchain = pd.DataFrame(historical_txns)
            df_blockchain['date'] = pd.to_datetime(df_blockchain['date']).dt.date
            
            # Add current mempool data as latest point
            if mempool_data:
                current_data = {
                    'date': datetime.now().date(),
                    'transaction_count': mempool_data.get('count', 0),
                    'mempool_size': mempool_data.get('vsize', 0),
                    'avg_fee': mempool_data.get('vsize', 0) / max(mempool_data.get('count', 1), 1)
                }
                
                # Add mempool analysis
                df_blockchain = add_mempool_analysis(df_blockchain, current_data)
                
        else:
            # Fallback: Generate simulated blockchain data
            print("   ⚠️  Using simulated blockchain data")
            dates = pd.date_range(start=datetime.now() - timedelta(days=180), periods=180, freq='D')
            tx_counts = np.random.normal(300000, 50000, 180).astype(int)  # Typical BTC daily tx count
            
            df_blockchain = pd.DataFrame({
                'date': dates.date,
                'transaction_count': tx_counts
            })
            
            # Add simulated mempool data
            current_data = {
                'date': datetime.now().date(),
                'transaction_count': np.random.randint(5000, 15000),  # Current mempool
                'mempool_size': np.random.randint(50000000, 150000000),  # vBytes
                'avg_fee': np.random.uniform(10, 50)  # sat/vB
            }
            
            df_blockchain = add_mempool_analysis(df_blockchain, current_data)
            
        return df_blockchain
        
    except Exception as e:
        print(f"   ❌ Error fetching blockchain data: {e}")
        # Return minimal simulated data with all required columns
        dates = pd.date_range(start=datetime.now() - timedelta(days=180), periods=180, freq='D')
        tx_counts = np.random.normal(300000, 50000, 180).astype(int)
        
        df_blockchain = pd.DataFrame({
            'date': dates.date,
            'transaction_count': tx_counts,
            'tx_variance': np.random.uniform(-0.1, 0.1, 180),
            'mempool_congestion': np.random.uniform(0, 1, 180),
            'estimated_conf_time': np.random.uniform(10, 240, 180)  # minutes
        })
        
        # Add mempool analysis to ensure all columns are present
        current_data = {
            'mempool_size': 100000000,  # Default 100MB
            'avg_fee': 20  # Default 20 sat/vB
        }
        df_blockchain = add_mempool_analysis(df_blockchain, current_data)
        
        return df_blockchain


def add_mempool_analysis(df_blockchain, current_mempool):
    """Add advanced mempool and transaction analysis."""
    
    # Calculate transaction count variance
    df_blockchain['tx_count_change'] = df_blockchain['transaction_count'].pct_change()
    df_blockchain['tx_variance'] = df_blockchain['transaction_count'].rolling(window=7).std() / df_blockchain['transaction_count'].rolling(window=7).mean()
    
    # Add moving averages for transaction trends
    df_blockchain['tx_ma_7'] = df_blockchain['transaction_count'].rolling(window=7).mean()
    df_blockchain['tx_ma_14'] = df_blockchain['transaction_count'].rolling(window=14).mean()
    df_blockchain['tx_trend'] = (df_blockchain['tx_ma_7'] / df_blockchain['tx_ma_14'] - 1) * 100
    
    # Mempool congestion analysis
    mempool_size = current_mempool.get('mempool_size', 100000000)  # Default 100MB
    avg_fee = current_mempool.get('avg_fee', 20)  # Default 20 sat/vB
    
    # Estimate confirmation time based on mempool size and average fee
    # Simplified model: higher mempool = longer confirmation time
    base_time = 10  # minutes for next block
    congestion_factor = min(mempool_size / 50000000, 5)  # Scale factor
    fee_factor = max(1 / (avg_fee / 10), 0.1)  # Lower fees = longer time
    
    estimated_conf_time = base_time * congestion_factor * fee_factor
    
    # Add mempool metrics to latest data point
    df_blockchain.loc[df_blockchain.index[-1], 'mempool_size'] = mempool_size
    df_blockchain.loc[df_blockchain.index[-1], 'avg_fee'] = avg_fee
    df_blockchain.loc[df_blockchain.index[-1], 'estimated_conf_time'] = estimated_conf_time
    
    # Fill missing mempool data with interpolation/simulation
    df_blockchain['mempool_size'] = df_blockchain['mempool_size'].fillna(method='bfill').fillna(mempool_size)
    df_blockchain['avg_fee'] = df_blockchain['avg_fee'].fillna(method='bfill').fillna(avg_fee)
    df_blockchain['estimated_conf_time'] = df_blockchain['estimated_conf_time'].fillna(method='bfill').fillna(estimated_conf_time)
    
    # Congestion level (0-1 scale)
    df_blockchain['mempool_congestion'] = np.clip(df_blockchain['mempool_size'] / 200000000, 0, 1)
    
    # Transaction efficiency metrics
    df_blockchain['tx_per_block'] = df_blockchain['transaction_count'] / 144  # Assuming ~144 blocks per day
    df_blockchain['network_activity_score'] = (
        df_blockchain['transaction_count'] / df_blockchain['transaction_count'].mean() * 0.5 +
        (1 - df_blockchain['mempool_congestion']) * 0.3 +
        (1 / (df_blockchain['avg_fee'] / df_blockchain['avg_fee'].mean())) * 0.2
    )
    
    print(f"   ⏱️  Current estimated confirmation time: {estimated_conf_time:.1f} minutes")
    print(f"   🚦 Mempool congestion level: {df_blockchain['mempool_congestion'].iloc[-1]:.1%}")
    
    # Check if confirmation time exceeds 4 hours (240 minutes)
    if estimated_conf_time > 240:
        print(f"   🚨 WARNING: Estimated confirmation time ({estimated_conf_time:.1f} min) exceeds 4 hours!")
    else:
        print(f"   ✅ Confirmation time within acceptable range")
    
    return df_blockchain


def merge_price_and_blockchain_data(price_df, blockchain_df):
    """Merge price and blockchain data, calculate correlations."""
    
    # Convert date columns to ensure compatibility
    price_df = price_df.copy()
    blockchain_df = blockchain_df.copy()
    
    # Ensure dates are datetime objects for merging
    price_df['date'] = pd.to_datetime(price_df['date'])
    blockchain_df['date'] = pd.to_datetime(blockchain_df['date'])
    
    # Merge dataframes - use outer join if inner join fails
    merged_df = pd.merge(price_df, blockchain_df, on='date', how='inner')
    
    # If no matching dates, use outer join and fill missing values
    if len(merged_df) == 0:
        print("   ⚠️  No matching dates found, using outer join with forward fill...")
        merged_df = pd.merge(price_df, blockchain_df, on='date', how='outer')
        merged_df = merged_df.sort_values('date')
        
        # Forward fill blockchain data for dates with price data only
        for col in blockchain_df.columns:
            if col != 'date':
                merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Keep only rows where we have price data
        merged_df = merged_df[merged_df['price'].notna()]
    
    print(f"   🔄 Merged dataset: {len(merged_df)} days of combined data")
    
    if len(merged_df) > 1:
        # Calculate price-transaction correlations
        price_change = merged_df['price'].pct_change()
        tx_change = merged_df['transaction_count'].pct_change()
        
        correlation = price_change.corr(tx_change)
        
        # Transaction volume vs price variance
        price_volatility = price_change.rolling(window=7).std()
        tx_volatility = merged_df['tx_variance']
        
        vol_correlation = price_volatility.corr(tx_volatility)
        
        print(f"   📊 Price-Transaction Correlation: {correlation:.3f}")
        print(f"   📈 Price Volatility vs Transaction Variance: {vol_correlation:.3f}")
        
        # Add correlation features to dataframe
        merged_df['price_tx_correlation'] = price_change.rolling(window=7).corr(tx_change)
        merged_df['price_volatility'] = price_volatility
        merged_df['price_change'] = price_change
        
        # Network stress indicators
        merged_df['network_stress'] = (
            merged_df['mempool_congestion'] * 0.4 +
            (merged_df['estimated_conf_time'] / 240) * 0.3 +  # Normalize to 4-hour scale
            merged_df['tx_variance'] * 0.3
        )
        
        # Transaction momentum indicators - check if columns exist first
        if 'tx_ma_7' in merged_df.columns and 'transaction_count' in merged_df.columns:
            merged_df['tx_momentum'] = merged_df['transaction_count'] / merged_df['tx_ma_7']
        else:
            merged_df['tx_momentum'] = 1.0  # Default neutral momentum
            
        if 'tx_trend' in merged_df.columns:
            merged_df['tx_acceleration'] = merged_df['tx_trend'].diff()
        else:
            merged_df['tx_acceleration'] = 0.0  # Default no acceleration
    
    return merged_df



def create_30min_data(df_daily):
    """Convert daily data to 30-minute intervals using interpolation and noise."""
    print("🔄 Converting daily data to 30-minute intervals...")
    
    # Ensure we have a datetime column
    df_daily = df_daily.copy()
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    # Create 30-minute intervals for the last 7 days to have enough data for 12-hour forecast
    end_date = df_daily['date'].max()
    start_date = end_date - timedelta(days=7)
    
    # Create 30-minute timestamps
    timestamps_30min = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq='30min'
    )
    
    # Get the last 7 days of daily data
    recent_daily = df_daily[df_daily['date'] >= start_date].sort_values('date')
    
    if len(recent_daily) == 0:
        print("   ⚠️  No recent data available, using last available data")
        recent_daily = df_daily.tail(7)
    
    # Create 30-minute dataframe
    df_30min = pd.DataFrame({'date': timestamps_30min})
    
    # Interpolate prices with realistic intraday variation
    base_prices = np.interp(
        timestamps_30min.astype(np.int64),
        recent_daily['date'].astype(np.int64),
        recent_daily['price']
    )
    
    # Add realistic intraday volatility (smaller than daily volatility)
    daily_volatility = recent_daily['price'].pct_change().std()
    intraday_volatility = daily_volatility * 0.3  # 30% of daily volatility for 30-min intervals
    
    # Add random walk with intraday patterns
    random_walk = np.random.normal(0, intraday_volatility, len(timestamps_30min))
    
    # Add time-of-day effects (higher volatility during trading hours)
    hours = timestamps_30min.hour
    time_effect = 1 + 0.5 * np.sin(2 * np.pi * (hours - 12) / 24)  # Peak around noon
    
    # Combine effects
    price_variations = random_walk * time_effect
    df_30min['price'] = base_prices * (1 + price_variations)
    
    # Ensure prices are positive and reasonable
    df_30min['price'] = np.maximum(df_30min['price'], base_prices * 0.95)
    
    print(f"   ✅ Created {len(df_30min)} 30-minute data points")
    print(f"   📊 Price range: ${df_30min['price'].min():.2f} - ${df_30min['price'].max():.2f}")
    
    return df_30min


def main():
    """Advanced ML-powered Bitcoin 1.0% price prediction system with blockchain analysis."""
    print("--- SCRIPT EXECUTION STARTED ---")
    print("🚀 Advanced ML Bitcoin 1.0% Price Prediction System")
    print("🔗 Including Blockchain Transaction & Mempool Analysis")
    print("=" * 70)
    
    print("--- FETCHING PRICE DATA ---")
    # Fetch price data
    df_price = fetch_bitcoin_futures_data()
    print(f"📊 Raw price data: {len(df_price)} days of Bitcoin prices")
    
    print("--- FETCHING BLOCKCHAIN DATA ---")
    # Fetch blockchain data
    df_blockchain = fetch_blockchain_data()
    print(f"🔗 Blockchain data: {len(df_blockchain)} days of transaction data")
    
    print("--- MERGING DATA ---")
    # Merge price and blockchain data
    df = merge_price_and_blockchain_data(df_price, df_blockchain)
    print(f"🔄 Combined dataset: {len(df)} days")
    
    # Analyze transaction time compliance (4-hour threshold)
    print(f"\n⏱️  TRANSACTION TIME ANALYSIS:")
    print("-" * 50)
    
    if 'estimated_conf_time' in df.columns and len(df) > 0:
        current_conf_time = df['estimated_conf_time'].iloc[-1]
        avg_conf_time = df['estimated_conf_time'].mean()
        max_conf_time = df['estimated_conf_time'].max()
        
        # Calculate percentage of transactions exceeding 4 hours
        exceeding_4h = (df['estimated_conf_time'] > 240).sum()
        total_days = len(df)
        exceeding_pct = (exceeding_4h / total_days) * 100
        
        print(f"   📈 Current confirmation time: {current_conf_time:.1f} minutes")
        print(f"   📊 Average confirmation time: {avg_conf_time:.1f} minutes")
        print(f"   ⚡ Maximum confirmation time: {max_conf_time:.1f} minutes")
        print(f"   🚨 Days exceeding 4 hours: {exceeding_4h}/{total_days} ({exceeding_pct:.1f}%)")
        
        if current_conf_time > 240:
            print(f"   ⚠️  WARNING: Current confirmation time exceeds 4-hour threshold!")
        else:
            print(f"   ✅ Current confirmation time is acceptable")
    
    # Analyze transaction variance vs price correlation
    print(f"\n📈 TRANSACTION VARIANCE vs PRICE ANALYSIS:")
    print("-" * 50)
    
    if 'transaction_count' in df.columns and len(df) > 7:
        # Calculate transaction variance
        df['tx_variance_pct'] = df['transaction_count'].pct_change().rolling(window=7).std() * 100
        df['price_variance_pct'] = df['price'].pct_change().rolling(window=7).std() * 100
        
        # Calculate correlation
        tx_price_corr = df['tx_variance_pct'].corr(df['price_variance_pct'])
        
        # Current metrics
        current_tx_variance = df['tx_variance_pct'].iloc[-1]
        current_price_variance = df['price_variance_pct'].iloc[-1]
        
        # Recent trend analysis
        recent_tx_trend = df['transaction_count'].tail(7).pct_change().mean() * 100
        recent_price_trend = df['price'].tail(7).pct_change().mean() * 100
        
        print(f"   📊 Transaction-Price Variance Correlation: {tx_price_corr:.3f}")
        print(f"   📈 Current Transaction Variance: {current_tx_variance:.2f}%")
        print(f"   💰 Current Price Variance: {current_price_variance:.2f}%")
        print(f"   📊 7-day Transaction Trend: {recent_tx_trend:+.2f}%")
        print(f"   💹 7-day Price Trend: {recent_price_trend:+.2f}%")
        
        # Analyze relationship strength
        if abs(tx_price_corr) > 0.5:
            print(f"   🔥 STRONG correlation between transaction variance and price")
        elif abs(tx_price_corr) > 0.3:
            print(f"   ⚡ MODERATE correlation between transaction variance and price")
        else:
            print(f"   ➡️  WEAK correlation between transaction variance and price")
    
    print("--- CREATING FEATURES ---")
    # Create enhanced features including blockchain data
    df = create_features(df)
    print(f"⚙️  Enhanced dataset: {len(df)} samples with {len(df.columns)-4} ML features")
    
    # Separate training data (with targets) from latest prediction data
    training_data = df.dropna(subset=['target']).copy()
    latest_data = df.iloc[-1:].copy()  # Most recent day for prediction
    
    print(f"📈 Training data: {len(training_data)} samples")
    print(f"🔮 Latest data for prediction: {latest_data['date'].iloc[0].strftime('%Y-%m-%d')} at ${latest_data['price'].iloc[0]:.2f}")
    
    # Feature selection - get all non-target columns
    feature_columns = [col for col in training_data.columns if col not in ['date', 'price', 'future_price', 'pct_change', 'target']]
    print(f"🎯 Total features available: {len(feature_columns)}")
    
    # Count blockchain-specific features
    blockchain_features = [col for col in feature_columns if any(keyword in col.lower() 
                          for keyword in ['tx_', 'mempool', 'conf_time', 'network_', 'blockchain'])]
    print(f"🔗 Blockchain features: {len(blockchain_features)}")
    
    X_training = training_data[feature_columns]
    y_training = training_data['target']
    X_latest = latest_data[feature_columns]
    
    # Handle any remaining NaN values more comprehensively
    print(f"   🔧 Handling missing values...")
    
    # Check for NaN values
    nan_columns = X_training.columns[X_training.isna().any()].tolist()
    if nan_columns:
        print(f"   ⚠️  Found NaN values in {len(nan_columns)} columns")
        
        # For blockchain features with insufficient history, use forward fill then backward fill
        for col in nan_columns:
            if any(keyword in col.lower() for keyword in ['tx_', 'mempool', 'conf_time', 'network_', 'blockchain']):
                X_training[col] = X_training[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                X_latest[col] = X_latest[col].fillna(X_training[col].mean())
            else:
                # For other features, use mean imputation
                X_training[col] = X_training[col].fillna(X_training[col].mean())
                X_latest[col] = X_latest[col].fillna(X_training[col].mean())
    
    # Final check and cleanup
    X_training = X_training.fillna(0)  # Fill any remaining NaN with 0
    X_latest = X_latest.fillna(0)
    
    print("--- CHECKING CLASS DISTRIBUTION ---")
    # Check class distribution
    class_counts = y_training.value_counts().sort_index()
    print(f"\n📈 Target Class Distribution:")
    print(f"   📉 Decrease ≥1.0%: {class_counts.get(-1, 0):3d} ({class_counts.get(-1, 0)/len(y_training)*100:5.1f}%)")
    print(f"   ➡️  No Change:     {class_counts.get(0, 0):3d} ({class_counts.get(0, 0)/len(y_training)*100:5.1f}%)")
    print(f"   📈 Increase ≥1.0%: {class_counts.get(1, 0):3d} ({class_counts.get(1, 0)/len(y_training)*100:5.1f}%)")
    
    # Time series split for proper backtesting
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.25, shuffle=False)
    
    print(f"\n🔄 Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    
    print("--- PERFORMING FEATURE SELECTION ---")
    # Feature selection using statistical tests
    print("\n🎯 Performing feature selection...")
    k_best = min(75, len(feature_columns))  # Increased to accommodate blockchain features
    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    X_latest_selected = selector.transform(X_latest)
    
    selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
    selected_blockchain_features = [f for f in selected_features if any(keyword in f.lower() 
                                  for keyword in ['tx_', 'mempool', 'conf_time', 'network_', 'blockchain'])]
    
    print(f"   ✅ Selected {len(selected_features)} most informative features")
    print(f"   🔗 Including {len(selected_blockchain_features)} blockchain features")
    
    # Initialize ML models with optimized parameters
    print("\n🤖 Initializing Advanced ML Models...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        ),
        
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            C=0.1,
            random_state=42
        ),

        'Support Vector Machine': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            C=1.0,
            gamma='scale',
            random_state=42
        ),

        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=15,
            weights='distance',
            metric='minkowski'
        ),

        'Naive Bayes': GaussianNB()
    }
    
    # Train and evaluate individual models on full training data
    model_scores = {}
    trained_models = {}
    
    print("\n--- TRAINING INDIVIDUAL MODELS ---")
    print("\n📊 Training Individual Models:")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"   Training {name}...", end=' ')
        
        # Train model on full selected training data
        model.fit(X_train_selected, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='accuracy')
        
        # Test predictions (for validation)
        y_pred = model.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        model_scores[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy
        }
        
        trained_models[name] = model
        
        print(f"CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f} | Test: {test_accuracy:.3f}")
    
    # Create ensemble model
    print(f"\n--- CREATING ENSEMBLE MODEL ---")
    print(f"\n🎭 Creating Ensemble Model...")
    
    # Select best performing models for ensemble
    best_models = sorted(model_scores.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:5]
    ensemble_estimators = [(name, trained_models[name]) for name, _ in best_models]
    
    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft'  # Use probability predictions
    )
    
    # Train ensemble on full training data
    ensemble.fit(X_train_selected, y_train)
    
    # Validation predictions
    y_pred_ensemble = ensemble.predict(X_test_selected)
    y_proba_ensemble = ensemble.predict_proba(X_test_selected)
    confidence_scores = np.max(y_proba_ensemble, axis=1)
    
    # Performance evaluation
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    
    print(f"   ✅ Ensemble accuracy: {ensemble_accuracy:.3f}")
    print(f"   📊 Using top {len(ensemble_estimators)} models: {[name for name, _ in ensemble_estimators]}")
    
    # NEXT MOVE PREDICTION - The main goal!
    print(f"\n--- PREDICTING NEXT MOVE ---")
    print(f"\n🔮 PREDICTING NEXT 1.0% BITCOIN MOVE:")
    print("=" * 80)
    
    # Make prediction on latest data
    next_prediction = ensemble.predict(X_latest_selected)[0]
    next_probabilities = ensemble.predict_proba(X_latest_selected)[0]
    next_confidence = np.max(next_probabilities)
    
    # Get current price and date
    current_price = latest_data['price'].iloc[0]
    current_date = latest_data['date'].iloc[0]
    
    # Prediction mapping
    direction_map = {1: "INCREASE", -1: "DECREASE", 0: "NO SIGNIFICANT CHANGE"}
    direction_emoji = {"INCREASE": "📈", "DECREASE": "📉", "NO SIGNIFICANT CHANGE": "➡️"}
    direction = direction_map[next_prediction]
    emoji = direction_emoji[direction]
    
    # Calculate target prices for 1.0% moves
    target_price_up = current_price * 1.01
    target_price_down = current_price * 0.99
    
    print(f"🎯 NEXT MOVE PREDICTION:")
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   Current Date:  {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   {emoji} Predicted Direction: {direction}")
    print(f"   🎯 Confidence Score: {next_confidence:.1%}")
    
    if next_prediction == 1:
        print(f"   📈 Target Price (1.0% up): ${target_price_up:,.2f}")
        print(f"   💰 Expected Gain: ${target_price_up - current_price:,.2f}")
    elif next_prediction == -1:
        print(f"   📉 Target Price (1.0% down): ${target_price_down:,.2f}")
        print(f"   ⚠️  Expected Loss: ${current_price - target_price_down:,.2f}")
    else:
        print(f"   ➡️  Price expected to stay between ${target_price_down:,.2f} and ${target_price_up:,.2f}")
    
    # Blockchain context for prediction
    if 'estimated_conf_time' in latest_data.columns:
        current_conf_time = latest_data['estimated_conf_time'].iloc[0]
        print(f"   ⏱️  Current confirmation time: {current_conf_time:.1f} minutes")
        
        if current_conf_time > 240:
            print(f"   🚨 Network congestion may impact trading activity!")
        else:
            print(f"   ✅ Network operating normally")
    
    if 'mempool_congestion' in latest_data.columns:
        current_congestion = latest_data['mempool_congestion'].iloc[0]
        print(f"   🚦 Network congestion: {current_congestion:.1%}")
    
    # Show individual model predictions for transparency
    print(f"\n🤖 Individual Model Predictions:")
    print("-" * 50)
    
    for name, model in trained_models.items():
        individual_pred = model.predict(X_latest_selected)[0]
        if hasattr(model, 'predict_proba'):
            individual_proba = model.predict_proba(X_latest_selected)[0]
            individual_conf = np.max(individual_proba)
        else:
            individual_conf = 0.5  # Default for models without probability
        
        ind_direction = direction_map[individual_pred]
        ind_emoji = direction_emoji[ind_direction]
        print(f"   {ind_emoji} {name:<20}: {ind_direction:<20} (Conf: {individual_conf:.1%})")
    
    # Feature importance analysis including blockchain features
    best_tree_model = None
    for name, model in trained_models.items():
        if 'Random Forest' in name or 'Gradient Boosting' in name or 'Extra Trees' in name:
            best_tree_model = model
            break
    
    if best_tree_model:
        if hasattr(best_tree_model, 'feature_importances_'):
            importances = best_tree_model.feature_importances_
        else:
            importances = best_tree_model.named_steps[list(best_tree_model.named_steps.keys())[-1]].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔥 Top 15 Most Important Features for Prediction:")
        blockchain_count = 0
        for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
            is_blockchain = any(keyword in row['feature'].lower() 
                              for keyword in ['tx_', 'mempool', 'conf_time', 'network_', 'blockchain'])
            marker = "🔗" if is_blockchain else "📊"
            if is_blockchain:
                blockchain_count += 1
            print(f"   {i:2d}. {marker} {row['feature']:<25}: {row['importance']:.4f}")
        
        print(f"\n   🔗 Blockchain features in top 15: {blockchain_count}")
    
    # Final prediction summary in requested format
    print(f"\n🎯 FINAL PREDICTION SUMMARY:")
    print("=" * 80)
    next_date = current_date + pd.Timedelta(days=1)
    timestamp = next_date.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Price ${current_price:.2f} | Predicted Price: TBD | {direction} by 1.0% {timestamp}")
    print(f"Confidence: {next_confidence:.1%} | Ensemble of {len(ensemble_estimators)} models")
    print(f"Blockchain-Enhanced Prediction | {len(selected_blockchain_features)} network features")
    
    if next_confidence >= 0.7:
        print("🔥 HIGH CONFIDENCE PREDICTION!")
    elif next_confidence >= 0.6:
        print("⚡ MODERATE CONFIDENCE PREDICTION")
    else:
        print("⚠️  LOW CONFIDENCE - Monitor closely")
    
    print(f"\n✨ Blockchain-Enhanced Prediction Complete!")
    print(f"   📊 Trained on: {len(X_training)} historical samples")
    print(f"   🤖 Models: {len(models)} individual + 1 ensemble")
    print(f"   🎯 Features: {len(selected_features)} selected from {len(feature_columns)} total")
    print(f"   🔗 Blockchain: {len(selected_blockchain_features)} network activity features")
    print(f"   🏆 Best Validation Accuracy: {max(score['test_accuracy'] for score in model_scores.values()):.3f}")
    
    print("--- SCRIPT EXECUTION FINISHED ---")
    
    # Import and run enhanced forecasting system
    try:
        from enhanced_forecasting import run_enhanced_forecasting
        
        print("\n" + "="*80)
        print("🔮 RUNNING ENHANCED FORECASTING SYSTEM")
        print("="*80)
        
        # Convert daily data to 30-minute intervals by interpolation
        # This simulates having 30-minute data
        df_30min = create_30min_data(df)
        
        # Run enhanced forecasting
        enhanced_results = run_enhanced_forecasting(df_30min)
        
        if enhanced_results:
            print("\n✅ Enhanced forecasting completed successfully!")
            
            # Demonstrate continuous training system
            try:
                from continuous_training import run_continuous_training_demo
                
                print("\n" + "="*80)
                print("🔄 STARTING CONTINUOUS TRAINING DEMONSTRATION")
                print("="*80)
                print("⏰ Running 3-minute demonstration of continuous model updates...")
                print("💡 In production, this would run indefinitely with 5-minute intervals")
                
                # Run a short demo (3 minutes)
                continuous_system = run_continuous_training_demo(df_30min, duration_minutes=3)
                
                print("\n✅ Continuous training demonstration completed!")
                
            except Exception as e:
                print(f"\n❌ Error running continuous training demo: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("\n❌ Enhanced forecasting failed")
            
    except Exception as e:
        print(f"\n❌ Error running enhanced forecasting: {e}")
        import traceback
        traceback.print_exc()
    
    return ensemble, selected_features, next_prediction, next_confidence, current_price


if __name__ == "__main__":
    main()