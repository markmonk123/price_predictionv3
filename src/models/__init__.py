"""
Models package for imbalanced-learn based ensemble training.

This package provides:
- Utility functions for data preparation and scaler evaluation
- Ensemble zoo with three balanced ensemble strategies
- CLI tools for model training

Usage:
    from src.models import EnsembleZoo, get_candidate_scalers
    
    zoo = EnsembleZoo(output_dir='models')
    results = zoo.build_and_train_all(X, y)
"""

from .ensemble_zoo import EnsembleZoo
from .utils import (
    to_numpy_contiguous,
    get_candidate_scalers,
    rank_scalers_by_estimator,
    validate_features,
    get_random_state
)

__all__ = [
    'EnsembleZoo',
    'to_numpy_contiguous',
    'get_candidate_scalers',
    'rank_scalers_by_estimator',
    'validate_features',
    'get_random_state'
]
