"""
Utility functions for machine learning pipelines with imbalanced-learn.

This module provides helper functions for:
- Converting data to contiguous numpy arrays (memory-efficient)
- Ranking scalers by performance
- Parallel evaluation using joblib

Security & robustness:
- All functions use deterministic random_state
- Logging is configurable via environment variables
- Memory-efficient operations with contiguous arrays
"""

import logging
import os
from typing import List, Tuple, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    MaxAbsScaler, Normalizer
)

# Configure logging level from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def to_numpy_contiguous(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], 
    dtype: type = np.float64
) -> np.ndarray:
    """
    Convert input data to a contiguous numpy array.
    
    This function ensures memory-efficient representation by:
    - Converting pandas DataFrames/Series to numpy
    - Ensuring C-contiguous memory layout
    - Casting to specified dtype
    
    Args:
        data: Input data (pandas DataFrame, Series, or numpy array)
        dtype: Target dtype for the output array (default: np.float64)
    
    Returns:
        Contiguous numpy array with specified dtype
        
    Security notes:
        - Validates input types to prevent injection
        - Uses safe numpy operations only
    """
    if data is None:
        raise ValueError("Input data cannot be None")
    
    # Convert pandas to numpy
    if isinstance(data, (pd.DataFrame, pd.Series)):
        logger.debug(f"Converting pandas {type(data).__name__} to numpy array")
        arr = data.values
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
    # Ensure contiguous and correct dtype
    if not arr.flags['C_CONTIGUOUS'] or arr.dtype != dtype:
        logger.debug("Creating contiguous array copy with dtype conversion")
        arr = np.ascontiguousarray(arr, dtype=dtype)
    
    return arr


def get_candidate_scalers() -> List[Tuple[str, BaseEstimator]]:
    """
    Return a list of candidate scalers for evaluation.
    
    Returns:
        List of (name, scaler_instance) tuples
        
    Scalers included:
        - StandardScaler: standardize features by removing mean and scaling to unit variance
        - RobustScaler: scale using statistics that are robust to outliers
        - MinMaxScaler: scale features to a given range (default [0, 1])
        - MaxAbsScaler: scale by maximum absolute value
        - Normalizer: normalize samples individually to unit norm
    """
    scalers = [
        ("standard", StandardScaler()),
        ("robust", RobustScaler()),
        ("minmax", MinMaxScaler()),
        ("maxabs", MaxAbsScaler()),
        ("normalizer", Normalizer()),
    ]
    logger.info(f"Generated {len(scalers)} candidate scalers")
    return scalers


def rank_scalers_by_estimator(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    estimator: BaseEstimator,
    scalers: Optional[List[Tuple[str, BaseEstimator]]] = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1
) -> List[Tuple[str, float]]:
    """
    Rank scalers by cross-validation performance with a given estimator.
    
    Uses joblib for parallel evaluation of different scalers. This is safe
    because each job gets a cloned estimator and scaler.
    
    Args:
        X: Feature matrix
        y: Target vector
        estimator: Base estimator to evaluate with each scaler
        scalers: List of (name, scaler) tuples. If None, uses get_candidate_scalers()
        cv: Number of cross-validation folds
        scoring: Scoring metric for evaluation
        random_state: Random state for reproducibility
        n_jobs: Number of parallel jobs (-1 uses all processors)
    
    Returns:
        List of (scaler_name, mean_score) tuples, sorted by score (descending)
        
    Security notes:
        - Uses deterministic random_state for reproducibility
        - Clones estimators to prevent state leakage
        - Parallel jobs are isolated
    """
    if scalers is None:
        scalers = get_candidate_scalers()
    
    # Convert to contiguous arrays for efficiency
    X_arr = to_numpy_contiguous(X)
    y_arr = to_numpy_contiguous(y)
    
    logger.info(f"Ranking {len(scalers)} scalers using {cv}-fold CV with {scoring}")
    
    def evaluate_scaler(name: str, scaler: BaseEstimator) -> Tuple[str, float]:
        """Evaluate a single scaler in parallel."""
        try:
            # Clone to avoid state sharing across parallel jobs
            scaler_clone = clone(scaler)
            estimator_clone = clone(estimator)
            
            # Create a simple pipeline: scale then predict
            X_scaled = scaler_clone.fit_transform(X_arr, y_arr)
            
            # Cross-validate
            cv_splitter = StratifiedKFold(
                n_splits=cv, 
                shuffle=True, 
                random_state=random_state
            )
            scores = cross_val_score(
                estimator_clone, 
                X_scaled, 
                y_arr, 
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=1  # Each parallel job should use 1 core
            )
            mean_score = scores.mean()
            
            logger.debug(f"Scaler '{name}': {mean_score:.4f} ± {scores.std():.4f}")
            return (name, mean_score)
        except Exception as e:
            logger.warning(f"Scaler '{name}' failed: {e}")
            return (name, -np.inf)
    
    # Parallel evaluation with joblib
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(evaluate_scaler)(name, scaler) for name, scaler in scalers
    )
    
    # Sort by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Top scaler: {results[0][0]} with score {results[0][1]:.4f}")
    return results


def validate_features(
    X: Union[pd.DataFrame, np.ndarray],
    max_features: int = 10000,
    max_samples: int = 1000000
) -> None:
    """
    Validate feature matrix for safety and size constraints.
    
    Args:
        X: Feature matrix to validate
        max_features: Maximum allowed number of features
        max_samples: Maximum allowed number of samples
        
    Raises:
        ValueError: If validation fails
        
    Security notes:
        - Prevents memory exhaustion attacks via size limits
        - Validates data types
    """
    if isinstance(X, pd.DataFrame):
        n_samples, n_features = X.shape
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            n_samples, n_features = X.shape[0], 1
        else:
            n_samples, n_features = X.shape
    else:
        raise TypeError(f"Unsupported data type: {type(X)}")
    
    if n_samples > max_samples:
        raise ValueError(
            f"Too many samples: {n_samples} > {max_samples}. "
            "This may be a memory exhaustion attack."
        )
    
    if n_features > max_features:
        raise ValueError(
            f"Too many features: {n_features} > {max_features}. "
            "This may be a memory exhaustion attack."
        )
    
    logger.debug(f"Validated features: {n_samples} samples, {n_features} features")


def get_random_state(seed: Optional[int] = None) -> int:
    """
    Get random state from environment or use provided seed.
    
    Args:
        seed: Optional seed value. If None, reads from RANDOM_STATE env var
    
    Returns:
        Random state integer for reproducibility
    """
    if seed is not None:
        return seed
    
    env_seed = os.getenv("RANDOM_STATE", "42")
    try:
        return int(env_seed)
    except ValueError:
        logger.warning(f"Invalid RANDOM_STATE env var: {env_seed}, using 42")
        return 42
