#!/usr/bin/env python3
"""
Data Normalization Module
Handles normalization and denormalization of features and predictions.
Ensures smooth data flow: raw data → features → normalization → model → denormalization → display
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Tuple, Optional


class DataNormalizer:
    """
    Handles normalization and denormalization of data for ML models.
    Stores normalization parameters to ensure consistency across training and prediction.
    """
    
    def __init__(self, method='standard'):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('standard' for z-score, 'minmax' for 0-1 scaling)
        """
        self.method = method
        self.feature_stats = {}  # Stores mean/std or min/max for each feature
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, exclude_cols=None):
        """
        Fit the normalizer to the data by computing normalization parameters.
        
        Args:
            df: DataFrame with features to normalize
            exclude_cols: List of columns to exclude from normalization (e.g., 'date', 'price', 'target')
        
        Returns:
            self
        """
        if exclude_cols is None:
            exclude_cols = ['date', 'price', 'target', 'future_price', 'pct_change', 'next_return']
        
        # Get columns to normalize
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        
        print(f"   📊 Fitting normalizer on {len(cols_to_normalize)} features using {self.method} normalization")
        
        for col in cols_to_normalize:
            if col not in df.columns:
                continue
                
            values = df[col].dropna()
            
            if len(values) == 0:
                continue
            
            if self.method == 'standard':
                # Z-score normalization: (x - mean) / std
                mean = values.mean()
                std = values.std()
                
                # Avoid division by zero
                if std < 1e-8:
                    std = 1.0
                
                self.feature_stats[col] = {'mean': mean, 'std': std}
                
            elif self.method == 'minmax':
                # Min-max normalization: (x - min) / (max - min)
                min_val = values.min()
                max_val = values.max()
                
                # Avoid division by zero
                if abs(max_val - min_val) < 1e-8:
                    max_val = min_val + 1.0
                
                self.feature_stats[col] = {'min': min_val, 'max': max_val}
        
        self.is_fitted = True
        print(f"   ✅ Normalizer fitted on {len(self.feature_stats)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
        """
        Transform data using fitted normalization parameters.
        
        Args:
            df: DataFrame to normalize
            exclude_cols: List of columns to exclude from normalization
        
        Returns:
            Normalized DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform. Call fit() first.")
        
        if exclude_cols is None:
            exclude_cols = ['date', 'price', 'target', 'future_price', 'pct_change', 'next_return']
        
        df_normalized = df.copy()
        
        normalized_count = 0
        for col in df.columns:
            if col in exclude_cols or col not in self.feature_stats:
                continue
            
            stats = self.feature_stats[col]
            
            if self.method == 'standard':
                # Z-score normalization
                mean = stats['mean']
                std = stats['std']
                df_normalized[col] = (df[col] - mean) / std
                
            elif self.method == 'minmax':
                # Min-max normalization
                min_val = stats['min']
                max_val = stats['max']
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            
            normalized_count += 1
        
        # Replace any inf or nan values that may have resulted from normalization
        df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with 0 for normalized features only
        for col in df_normalized.columns:
            if col not in exclude_cols and col in self.feature_stats:
                df_normalized[col] = df_normalized[col].fillna(0)
        
        return df_normalized
    
    def fit_transform(self, df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
        """
        Fit normalizer and transform data in one step.
        
        Args:
            df: DataFrame to fit and normalize
            exclude_cols: List of columns to exclude from normalization
        
        Returns:
            Normalized DataFrame
        """
        self.fit(df, exclude_cols)
        return self.transform(df, exclude_cols)
    
    def denormalize_predictions(self, predictions: np.ndarray, feature_name='price') -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
            feature_name: Name of the feature being predicted (default: 'price')
        
        Returns:
            Denormalized predictions
        """
        if not self.is_fitted:
            print("   ⚠️  Normalizer not fitted, returning predictions unchanged")
            return predictions
        
        # If predicting raw price, no denormalization needed (price is never normalized)
        if feature_name == 'price':
            return predictions
        
        # If predicting a normalized feature, denormalize it
        if feature_name in self.feature_stats:
            stats = self.feature_stats[feature_name]
            
            if self.method == 'standard':
                mean = stats['mean']
                std = stats['std']
                return predictions * std + mean
                
            elif self.method == 'minmax':
                min_val = stats['min']
                max_val = stats['max']
                return predictions * (max_val - min_val) + min_val
        
        # Feature not found in stats, return unchanged
        return predictions
    
    def denormalize_dataframe(self, df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
        """
        Denormalize an entire DataFrame back to original scale.
        
        Args:
            df: Normalized DataFrame
            exclude_cols: List of columns to exclude from denormalization
        
        Returns:
            Denormalized DataFrame
        """
        if not self.is_fitted:
            print("   ⚠️  Normalizer not fitted, returning DataFrame unchanged")
            return df
        
        if exclude_cols is None:
            exclude_cols = ['date', 'price', 'target', 'future_price', 'pct_change', 'next_return']
        
        df_denormalized = df.copy()
        
        for col in df.columns:
            if col in exclude_cols or col not in self.feature_stats:
                continue
            
            stats = self.feature_stats[col]
            
            if self.method == 'standard':
                mean = stats['mean']
                std = stats['std']
                df_denormalized[col] = df[col] * std + mean
                
            elif self.method == 'minmax':
                min_val = stats['min']
                max_val = stats['max']
                df_denormalized[col] = df[col] * (max_val - min_val) + min_val
        
        return df_denormalized
    
    def save(self, filepath: str):
        """
        Save normalizer parameters to disk.
        
        Args:
            filepath: Path to save the normalizer
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'feature_stats': self.feature_stats,
                'is_fitted': self.is_fitted
            }, f)
        print(f"   💾 Normalizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataNormalizer':
        """
        Load normalizer parameters from disk.
        
        Args:
            filepath: Path to load the normalizer from
        
        Returns:
            Loaded DataNormalizer instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        normalizer = cls(method=data['method'])
        normalizer.feature_stats = data['feature_stats']
        normalizer.is_fitted = data['is_fitted']
        
        print(f"   📂 Normalizer loaded from {filepath}")
        return normalizer
    
    def get_stats_summary(self) -> Dict:
        """
        Get a summary of normalization statistics.
        
        Returns:
            Dictionary with normalization statistics summary
        """
        if not self.is_fitted:
            return {'status': 'not fitted'}
        
        return {
            'method': self.method,
            'num_features': len(self.feature_stats),
            'features': list(self.feature_stats.keys())[:10],  # First 10 features
            'fitted': self.is_fitted
        }


def create_normalizer_for_pipeline(method='standard') -> DataNormalizer:
    """
    Factory function to create a normalizer for the prediction pipeline.
    
    Args:
        method: Normalization method ('standard' or 'minmax')
    
    Returns:
        DataNormalizer instance
    """
    return DataNormalizer(method=method)


if __name__ == "__main__":
    # Test the normalizer
    print("🧪 Testing DataNormalizer")
    
    # Create sample data
    test_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'price': np.random.randn(100) * 1000 + 50000,
        'feature1': np.random.randn(100) * 100,
        'feature2': np.random.randn(100) * 50,
        'target': np.random.choice([-1, 0, 1], 100)
    })
    
    print("\n📊 Original data (first 5 rows):")
    print(test_data.head())
    
    # Test standard normalization
    normalizer = DataNormalizer(method='standard')
    normalized_data = normalizer.fit_transform(test_data)
    
    print("\n📈 Normalized data (first 5 rows):")
    print(normalized_data.head())
    
    # Test denormalization
    denormalized_data = normalizer.denormalize_dataframe(normalized_data)
    
    print("\n📉 Denormalized data (first 5 rows):")
    print(denormalized_data.head())
    
    # Check if denormalization worked
    for col in ['feature1', 'feature2']:
        original = test_data[col].values
        denorm = denormalized_data[col].values
        diff = np.abs(original - denorm).max()
        print(f"\n✅ {col}: Max difference = {diff:.6f}")
    
    print("\n✅ DataNormalizer test complete!")
