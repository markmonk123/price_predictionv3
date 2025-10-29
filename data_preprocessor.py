#!/usr/bin/env python3
"""
Decentralized Data Preprocessing Module
Handles all data preprocessing independently from model training,
ensuring data integrity and avoiding mutable reference issues.
"""

import numpy as np
import pandas as pd
import copy
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Decentralized data preprocessing system that handles feature creation,
    normalization/denormalization, and data integrity checks independently
    from the model training pipeline.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the preprocessor with configuration."""
        self.config = config or self._get_default_config()
        self.feature_stats = {}  # Store statistics for denormalization
        self.feature_scaler_params = {}  # Store scaling parameters
        self.processed_features = []
        self.is_fitted = False
        
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            'pct_threshold': 0.01,
            'lag_periods': [1, 2, 3, 5, 7, 10, 14, 21],
            'rolling_windows': [5, 10, 15, 20, 30, 50],
            'min_data_points': 100,
            'denormalize_features': True,
            'preserve_structure': True,
            'handle_missing': 'fill',
            'blockchain_features': ['transaction_count', 'tx_variance', 'mempool_congestion', 
                                  'estimated_conf_time', 'network_stress', 'tx_momentum', 
                                  'tx_acceleration', 'price_tx_correlation', 'network_activity_score']
        }
    
    def create_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create raw technical indicators and features without normalization.
        Returns structured data that maintains original data types and ranges.
        """
        logger.info("🔧 Creating raw features without normalization...")
        
        # Work with a deep copy to avoid mutable reference issues
        df_processed = copy.deepcopy(df)
        
        # Ensure date column is datetime
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        # Basic time features (structural, not normalized)
        df_processed['dayofweek'] = df_processed['date'].dt.dayofweek
        df_processed['month'] = df_processed['date'].dt.month
        df_processed['quarter'] = df_processed['date'].dt.quarter
        df_processed['is_weekend'] = (df_processed['dayofweek'] >= 5).astype(int)
        df_processed['is_month_end'] = (df_processed['date'].dt.day >= 28).astype(int)
        df_processed['hour'] = df_processed['date'].dt.hour if hasattr(df_processed['date'].dt, 'hour') else 0
        
        # Price lag features (preserve original price scale)
        for lag in self.config['lag_periods']:
            df_processed[f'price_lag{lag}'] = df_processed['price'].shift(lag)
            df_processed[f'return_lag{lag}'] = df_processed['price'].pct_change(lag)
        
        # Rolling statistics (preserve original scale)
        for window in self.config['rolling_windows']:
            df_processed[f'sma_{window}'] = df_processed['price'].rolling(window=window).mean()
            df_processed[f'ema_{window}'] = df_processed['price'].ewm(span=window).mean()
            df_processed[f'std_{window}'] = df_processed['price'].rolling(window=window).std()
            df_processed[f'min_{window}'] = df_processed['price'].rolling(window=window).min()
            df_processed[f'max_{window}'] = df_processed['price'].rolling(window=window).max()
            df_processed[f'median_{window}'] = df_processed['price'].rolling(window=window).median()
            
            # Ratio features (naturally bounded)
            range_val = df_processed[f'max_{window}'] - df_processed[f'min_{window}']
            df_processed[f'price_position_{window}'] = np.where(
                range_val > 0,
                (df_processed['price'] - df_processed[f'min_{window}']) / range_val,
                0.5  # Default to middle position if no range
            )
            
            # Relative price features
            df_processed[f'price_to_sma_{window}'] = np.where(
                df_processed[f'sma_{window}'] > 0,
                df_processed['price'] / df_processed[f'sma_{window}'],
                1.0
            )
            df_processed[f'price_to_ema_{window}'] = np.where(
                df_processed[f'ema_{window}'] > 0,
                df_processed['price'] / df_processed[f'ema_{window}'],
                1.0
            )
        
        # Technical indicators
        self._add_technical_indicators(df_processed)
        
        # Blockchain features if available
        self._add_blockchain_features(df_processed)
        
        # Create target variable
        df_processed = self._create_target_variable(df_processed)
        
        logger.info(f"✅ Created {len(df_processed.columns)} features maintaining original structure")
        return df_processed
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """Add technical indicators while preserving structure."""
        
        # RSI-like indicator (naturally 0-100 range)
        for window in [14, 21]:
            price_changes = df['price'].diff()
            gains = price_changes.where(price_changes > 0, 0)
            losses = -price_changes.where(price_changes < 0, 0)
            
            avg_gains = gains.rolling(window=window).mean()
            avg_losses = losses.rolling(window=window).mean()
            
            rs = np.where(avg_losses > 0, avg_gains / avg_losses, 100)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Band position (naturally 0-1 range)
        for window in [20, 50]:
            sma = df['price'].rolling(window=window).mean()
            std = df['price'].rolling(window=window).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            band_width = upper_band - lower_band
            df[f'bb_position_{window}'] = np.where(
                band_width > 0,
                (df['price'] - lower_band) / band_width,
                0.5
            )
        
        # MACD (preserve original scale)
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume proxy indicators
        df['volume_proxy'] = df['price'].rolling(window=20).std()
        df['volume_sma'] = df['volume_proxy'].rolling(window=20).mean()
        df['volume_ratio'] = np.where(
            df['volume_sma'] > 0,
            df['volume_proxy'] / df['volume_sma'],
            1.0
        )
    
    def _add_blockchain_features(self, df: pd.DataFrame) -> None:
        """Add blockchain-specific features if available."""
        
        for feature in self.config['blockchain_features']:
            if feature in df.columns:
                if feature == 'transaction_count':
                    df['tx_count_ma_7'] = df[feature].rolling(window=7).mean()
                    df['tx_count_ma_30'] = df[feature].rolling(window=30).mean()
                    df['tx_count_trend'] = np.where(
                        df['tx_count_ma_30'] > 0,
                        (df['tx_count_ma_7'] / df['tx_count_ma_30'] - 1) * 100,
                        0
                    )
                    
                elif feature == 'mempool_congestion':
                    df['mempool_stress_ma'] = df[feature].rolling(window=7).mean()
                    df['mempool_spike'] = (df[feature] > df['mempool_stress_ma'] * 1.5).astype(int)
                
                elif feature == 'estimated_conf_time':
                    df['conf_time_acceptable'] = (df[feature] <= 240).astype(int)
                    df['conf_time_ma'] = df[feature].rolling(window=7).mean()
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for classification."""
        df['future_price'] = df['price'].shift(-1)
        df['pct_change'] = (df['future_price'] - df['price']) / df['price']
        
        # Use more granular thresholds to ensure balanced classes
        threshold = self.config['pct_threshold']
        
        # Create initial target
        df['target'] = 0  # No significant change
        df.loc[df['pct_change'] >= threshold, 'target'] = 1  # Increase
        df.loc[df['pct_change'] <= -threshold, 'target'] = -1  # Decrease
        
        # Ensure we have balanced classes by adjusting threshold if needed
        valid_targets = df.dropna(subset=['target'])
        if len(valid_targets) > 10:
            class_counts = valid_targets['target'].value_counts()
            min_class_size = len(valid_targets) * 0.15  # At least 15% per class
            
            # Fix the issue - use .values directly not as function call
            if any(count < min_class_size for count in class_counts.values):
                # Use quantile-based approach for better balance
                pct_changes = df['pct_change'].dropna()
                if len(pct_changes) > 0:
                    # Use 33rd and 67th percentiles as thresholds
                    try:
                        lower_threshold = pct_changes.quantile(0.33)
                        upper_threshold = pct_changes.quantile(0.67)
                        
                        df['target'] = 0  # Reset
                        df.loc[df['pct_change'] <= lower_threshold, 'target'] = -1
                        df.loc[df['pct_change'] >= upper_threshold, 'target'] = 1
                    except Exception as e:
                        # Fallback to simple percentile-based approach
                        sorted_changes = np.sort(pct_changes.values)
                        n = len(sorted_changes)
                        lower_idx = int(n * 0.33)
                        upper_idx = int(n * 0.67)
                        
                        if lower_idx < n and upper_idx < n:
                            lower_threshold = sorted_changes[lower_idx]
                            upper_threshold = sorted_changes[upper_idx]
                            
                            df['target'] = 0  # Reset
                            df.loc[df['pct_change'] <= lower_threshold, 'target'] = -1
                            df.loc[df['pct_change'] >= upper_threshold, 'target'] = 1
        
        return df
    
    def prepare_denormalized_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare data for model input with denormalized features.
        Returns structured data and metadata about transformations.
        """
        logger.info("🔄 Preparing denormalized data for model input...")
        
        # Create features first
        df_features = self.create_raw_features(df)
        
        # Identify feature columns (exclude metadata)
        exclude_cols = ['date', 'price', 'future_price', 'pct_change', 'target']
        feature_columns = [col for col in df_features.columns if col not in exclude_cols]
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_features, feature_columns)
        
        # Store statistics for potential renormalization if needed
        self._compute_feature_statistics(df_clean, feature_columns)
        
        # Prepare final dataset
        final_data = df_clean.copy()
        
        # Create metadata
        metadata = {
            'feature_columns': feature_columns,
            'data_shape': final_data.shape,
            'missing_values_handled': True,
            'denormalized': True,
            'structure_preserved': True,
            'feature_stats': self.feature_stats
        }
        
        logger.info(f"✅ Prepared denormalized data: {final_data.shape[0]} samples, {len(feature_columns)} features")
        return final_data, metadata
    
    def _handle_missing_values(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Handle missing values while preserving data structure."""
        df_clean = df.copy()
        
        for col in feature_columns:
            if df_clean[col].isna().any():
                # For ratio features, use 1.0 as default
                if any(keyword in col.lower() for keyword in ['ratio', '_to_', 'position']):
                    df_clean[col] = df_clean[col].fillna(1.0)
                # For indicator features, use 0
                elif any(keyword in col.lower() for keyword in ['rsi', 'bb_', 'is_', 'spike']):
                    df_clean[col] = df_clean[col].fillna(0.0)
                # For price-based features, use forward fill then mean
                else:
                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(df_clean[col].mean())
        
        # Final cleanup
        df_clean = df_clean.fillna(0)
        return df_clean
    
    def _compute_feature_statistics(self, df: pd.DataFrame, feature_columns: List[str]) -> None:
        """Compute and store feature statistics for potential denormalization."""
        self.feature_stats = {}
        
        for col in feature_columns:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        
        self.is_fitted = True
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data integrity and return diagnostic information."""
        logger.info("🔍 Validating data integrity...")
        
        diagnostics = {
            'shape': df.shape,
            'missing_values': df.isna().sum().sum(),
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_types': df.dtypes.to_dict(),
            'validation_passed': True
        }
        
        # Check for issues
        issues = []
        if diagnostics['missing_values'] > 0:
            issues.append(f"Missing values: {diagnostics['missing_values']}")
        if diagnostics['infinite_values'] > 0:
            issues.append(f"Infinite values: {diagnostics['infinite_values']}")
        if diagnostics['duplicate_rows'] > 0:
            issues.append(f"Duplicate rows: {diagnostics['duplicate_rows']}")
        
        diagnostics['issues'] = issues
        diagnostics['validation_passed'] = len(issues) == 0
        
        if not diagnostics['validation_passed']:
            logger.warning(f"⚠️ Data integrity issues found: {issues}")
        else:
            logger.info("✅ Data integrity validation passed")
        
        return diagnostics
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data specifically for feature importance analysis."""
        processed_df, metadata = self.prepare_denormalized_data(df)
        
        # Remove rows with NaN targets
        clean_df = processed_df.dropna(subset=['target'])
        
        return clean_df, metadata
    
    def export_preprocessing_config(self) -> Dict:
        """Export current preprocessing configuration for reproducibility."""
        return {
            'config': self.config,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted,
            'processed_features': self.processed_features
        }
    
    def import_preprocessing_config(self, config_dict: Dict) -> None:
        """Import preprocessing configuration for consistency."""
        self.config = config_dict.get('config', self._get_default_config())
        self.feature_stats = config_dict.get('feature_stats', {})
        self.is_fitted = config_dict.get('is_fitted', False)
        self.processed_features = config_dict.get('processed_features', [])


# Convenience functions for backward compatibility
def create_enhanced_features(df: pd.DataFrame, pct_threshold: float = 0.002) -> pd.DataFrame:
    """
    Backward compatibility function that creates enhanced features
    using the new decentralized preprocessing system.
    """
    config = {'pct_threshold': pct_threshold}
    preprocessor = DataPreprocessor(config)
    processed_df, _ = preprocessor.prepare_denormalized_data(df)
    return processed_df


def preprocess_data_for_model(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to preprocess data for model training with denormalized features.
    
    Args:
        df: Input DataFrame with price and date columns
        config: Optional configuration dictionary
    
    Returns:
        Tuple of (processed_dataframe, metadata)
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.prepare_denormalized_data(df)


if __name__ == "__main__":
    # Test the preprocessor
    import requests
    from datetime import datetime, timedelta
    
    print("🧪 Testing DataPreprocessor...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 50000
    
    test_df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    processed_df, metadata = preprocessor.prepare_denormalized_data(test_df)
    
    print(f"✅ Processed {processed_df.shape[0]} samples with {len(metadata['feature_columns'])} features")
    print(f"✅ Data integrity: {preprocessor.validate_data_integrity(processed_df)['validation_passed']}")
    print("🎉 DataPreprocessor test completed successfully!")