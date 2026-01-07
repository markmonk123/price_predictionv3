"""
Model utilities for data preprocessing and scaler evaluation.

This module provides helper functions for converting data to numpy arrays,
generating candidate scalers, and ranking scalers based on cross-validation performance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
)
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def to_numpy_contiguous(X):
    """
    Convert input data to contiguous numpy float64 array.
    
    Contiguous arrays enable efficient memory access patterns and pointer-style operations,
    which can significantly improve performance for numerical computations.
    
    Args:
        X: Input data (numpy array, pandas DataFrame, or list)
        
    Returns:
        numpy.ndarray: Contiguous float64 array
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # Convert to float64 and ensure C-contiguous memory layout
    X = np.ascontiguousarray(X, dtype=np.float64)
    
    logger.debug(f"Converted to contiguous array with shape {X.shape}")
    return X


def get_candidate_scalers():
    """
    Get a list of candidate scaler instances for evaluation.
    
    Returns a diverse set of scalers suitable for different data distributions:
    - StandardScaler: assumes Gaussian distribution
    - RobustScaler: robust to outliers using median/IQR
    - MinMaxScaler: scales to [0, 1] range
    - MaxAbsScaler: scales by maximum absolute value
    
    Returns:
        list: List of tuples (scaler_name, scaler_instance)
    """
    scalers = [
        ('standard', StandardScaler()),
        ('robust', RobustScaler()),
        ('minmax', MinMaxScaler()),
        ('maxabs', MaxAbsScaler()),
    ]
    logger.info(f"Generated {len(scalers)} candidate scalers")
    return scalers


def rank_scalers_by_estimator(X, y, estimators, candidate_scalers, cv=5, scoring='roc_auc', n_jobs=1):
    """
    Rank scalers by their mean cross-validation score across multiple estimators.
    
    For each scaler, evaluates multiple estimators using cross-validation and
    computes the mean score across all estimators. Returns scalers sorted by
    descending mean score.
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target vector (numpy array or pandas Series)
        estimators: List of tuples (name, estimator_instance)
        candidate_scalers: List of tuples (name, scaler_instance)
        cv: Number of cross-validation folds (default: 5)
        scoring: Scoring metric (default: 'roc_auc')
        n_jobs: Number of parallel jobs. Set to 1 to avoid nested parallelism (default: 1)
        
    Returns:
        list: List of tuples (scaler_name, scaler_instance, mean_score) sorted by score
    """
    logger.info(f"Ranking {len(candidate_scalers)} scalers using {len(estimators)} estimators")
    logger.info(f"Using {cv}-fold CV with scoring={scoring}")
    
    # Convert to contiguous numpy arrays for efficiency
    X = to_numpy_contiguous(X)
    if isinstance(y, pd.Series):
        y = y.values
    y = np.ascontiguousarray(y)
    
    def evaluate_scaler(scaler_name, scaler):
        """Evaluate a single scaler across all estimators."""
        try:
            # Fit scaler and transform data
            X_scaled = scaler.fit_transform(X)
            
            # Evaluate each estimator
            scores = []
            for est_name, estimator in estimators:
                try:
                    cv_scores = cross_val_score(
                        estimator, X_scaled, y, 
                        cv=cv, scoring=scoring, n_jobs=1  # Avoid nested parallelism
                    )
                    mean_score = np.mean(cv_scores)
                    scores.append(mean_score)
                    logger.debug(f"Scaler={scaler_name}, Estimator={est_name}, Score={mean_score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to evaluate {est_name} with {scaler_name}: {e}")
                    continue
            
            if scores:
                mean_score = np.mean(scores)
                logger.info(f"Scaler {scaler_name}: mean score = {mean_score:.4f}")
                return (scaler_name, scaler, mean_score)
            else:
                logger.warning(f"No valid scores for scaler {scaler_name}")
                return (scaler_name, scaler, -np.inf)
                
        except Exception as e:
            logger.error(f"Error evaluating scaler {scaler_name}: {e}")
            return (scaler_name, scaler, -np.inf)
    
    # Parallel evaluation of scalers
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_scaler)(name, scaler) 
        for name, scaler in candidate_scalers
    )
    
    # Sort by score (descending)
    results = sorted(results, key=lambda x: x[2], reverse=True)
    
    logger.info("Scaler ranking complete:")
    for i, (name, _, score) in enumerate(results[:5], 1):
        logger.info(f"  {i}. {name}: {score:.4f}")
    
    return results
