"""
Utility functions for ML pipeline preprocessing and evaluation.

Provides memory-efficient numpy array handling, scaler evaluation,
and parallelized model ranking using joblib.
"""

import logging
import warnings
from typing import List, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
)
from sklearn.model_selection import cross_val_score

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def to_numpy_contiguous(
    data: Union[np.ndarray, pd.DataFrame, pd.Series],
    dtype: np.dtype = np.float64
) -> np.ndarray:
    """
    Convert input data to contiguous numpy array for memory efficiency.
    
    Ensures C-contiguous memory layout to reduce fragmentation and
    improve cache performance.
    
    Args:
        data: Input data (numpy array, pandas DataFrame, or Series)
        dtype: Target numpy dtype (default: float64)
    
    Returns:
        Contiguous numpy array with specified dtype
    
    Raises:
        ValueError: If input cannot be converted to numeric array
    """
    try:
        if isinstance(data, pd.DataFrame):
            arr = data.values
        elif isinstance(data, pd.Series):
            arr = data.values
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.array(data)
        
        # Ensure contiguous memory layout and correct dtype
        if not arr.flags['C_CONTIGUOUS'] or arr.dtype != dtype:
            arr = np.ascontiguousarray(arr, dtype=dtype)
        
        logger.debug(
            f"Converted to contiguous array: shape={arr.shape}, "
            f"dtype={arr.dtype}, contiguous={arr.flags['C_CONTIGUOUS']}"
        )
        
        return arr
    
    except Exception as e:
        logger.error(f"Failed to convert data to numpy array: {e}")
        raise ValueError(f"Cannot convert input to numpy array: {e}")


def get_candidate_scalers() -> List[Tuple[str, BaseEstimator]]:
    """
    Get list of candidate scalers for evaluation.
    
    Returns top-5 commonly used scalers that work well with
    different data distributions.
    
    Returns:
        List of (scaler_name, scaler_instance) tuples
    """
    scalers = [
        ('standard', StandardScaler()),
        ('minmax', MinMaxScaler()),
        ('robust', RobustScaler()),
        ('maxabs', MaxAbsScaler()),
        ('quantile', QuantileTransformer(n_quantiles=100, output_distribution='normal')),
    ]
    
    logger.info(f"Initialized {len(scalers)} candidate scalers for evaluation")
    return scalers


def _evaluate_single_scaler(
    scaler_name: str,
    scaler: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    estimator: BaseEstimator,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    random_state: int = 42
) -> Tuple[str, float]:
    """
    Evaluate a single scaler with cross-validation (internal helper).
    
    Args:
        scaler_name: Name identifier for the scaler
        scaler: Scaler instance to evaluate
        X: Feature matrix
        y: Target vector
        estimator: Base estimator for evaluation
        cv: Number of CV folds
        scoring: Scoring metric
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (scaler_name, mean_cv_score)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold
    
    try:
        # Create pipeline with scaler and estimator
        pipeline = Pipeline([
            ('scaler', scaler),
            ('estimator', estimator)
        ])
        
        # Use stratified K-fold for imbalanced datasets
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state
        )
        
        # Perform cross-validation
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=1  # Parallelism handled at higher level
        )
        
        mean_score = np.mean(scores)
        logger.debug(
            f"Scaler '{scaler_name}': mean_score={mean_score:.4f}, "
            f"std={np.std(scores):.4f}"
        )
        
        return (scaler_name, mean_score)
    
    except Exception as e:
        logger.warning(f"Failed to evaluate scaler '{scaler_name}': {e}")
        return (scaler_name, -np.inf)  # Return very low score on failure


def rank_scalers_by_estimator(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    estimator: BaseEstimator,
    candidate_scalers: Optional[List[Tuple[str, BaseEstimator]]] = None,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    n_jobs: int = -1,
    random_state: int = 42
) -> List[Tuple[str, float]]:
    """
    Rank scalers by cross-validated performance with a given estimator.
    
    Uses joblib parallelism to evaluate multiple scalers concurrently.
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target vector (numpy array or pandas Series)
        estimator: Base estimator for evaluation
        candidate_scalers: List of (name, scaler) tuples. If None, uses default set.
        cv: Number of cross-validation folds
        scoring: Sklearn scoring metric
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        random_state: Random seed for reproducibility
    
    Returns:
        List of (scaler_name, mean_score) tuples sorted by descending score
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rankings = rank_scalers_by_estimator(
        ...     X_train, y_train,
        ...     estimator=RandomForestClassifier(random_state=42),
        ...     n_jobs=-1
        ... )
        >>> best_scaler_name = rankings[0][0]
    """
    # Convert inputs to numpy arrays
    X_array = to_numpy_contiguous(X, dtype=np.float64)
    y_array = to_numpy_contiguous(y, dtype=np.int64)
    
    # Get candidate scalers if not provided
    if candidate_scalers is None:
        candidate_scalers = get_candidate_scalers()
    
    logger.info(
        f"Ranking {len(candidate_scalers)} scalers using {cv}-fold CV "
        f"with scoring='{scoring}'"
    )
    
    # Parallel evaluation of scalers
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_evaluate_single_scaler)(
            scaler_name,
            scaler,
            X_array,
            y_array,
            estimator,
            cv=cv,
            scoring=scoring,
            random_state=random_state
        )
        for scaler_name, scaler in candidate_scalers
    )
    
    # Sort by score (descending)
    ranked_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    logger.info("Scaler ranking completed:")
    for rank, (name, score) in enumerate(ranked_results, 1):
        logger.info(f"  {rank}. {name}: {score:.4f}")
    
    return ranked_results


def validate_input_array(
    X: np.ndarray,
    max_features: int = 1000,
    max_samples: int = 1000000
) -> None:
    """
    Validate input array dimensions for security and memory constraints.
    
    Args:
        X: Input feature array
        max_features: Maximum allowed number of features
        max_samples: Maximum allowed number of samples
    
    Raises:
        ValueError: If input exceeds size limits
    """
    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {X.ndim}D array with shape {X.shape}"
        )
    
    n_samples, n_features = X.shape
    
    if n_features > max_features:
        raise ValueError(
            f"Too many features: {n_features} > {max_features}"
        )
    
    if n_samples > max_samples:
        raise ValueError(
            f"Too many samples: {n_samples} > {max_samples}"
        )
    
    logger.debug(f"Input validation passed: shape={X.shape}")


def safe_dtype_conversion(
    arr: np.ndarray,
    target_dtype: np.dtype = np.float64
) -> np.ndarray:
    """
    Safely convert array dtype with overflow checking.
    
    Args:
        arr: Input array
        target_dtype: Target numpy dtype
    
    Returns:
        Converted array
    
    Raises:
        ValueError: If conversion would cause overflow
    """
    try:
        # Check for potential overflow
        if np.issubdtype(arr.dtype, np.integer) and np.issubdtype(target_dtype, np.floating):
            if arr.size > 0:
                arr_min, arr_max = arr.min(), arr.max()
                logger.debug(f"Converting integer array: min={arr_min}, max={arr_max}")
        
        converted = arr.astype(target_dtype)
        return converted
    
    except (ValueError, OverflowError) as e:
        logger.error(f"Dtype conversion failed: {e}")
        raise ValueError(f"Cannot safely convert array dtype: {e}")
