"""
Models module for imbalanced-learn ensemble zoo.

Exports main functions for building and training ensemble classifiers.
"""

from .ensemble_zoo import (
    build_and_evaluate_ensembles,
    create_balanced_stacking_ensemble,
    create_balanced_voting_ensemble,
    create_hybrid_ensemble,
    evaluate_ensemble,
)
from .utils import (
    to_numpy_contiguous,
    get_candidate_scalers,
    rank_scalers_by_estimator,
    validate_input_array,
    safe_dtype_conversion,
)

__all__ = [
    # Ensemble builders
    'build_and_evaluate_ensembles',
    'create_balanced_stacking_ensemble',
    'create_balanced_voting_ensemble',
    'create_hybrid_ensemble',
    'evaluate_ensemble',
    # Utilities
    'to_numpy_contiguous',
    'get_candidate_scalers',
    'rank_scalers_by_estimator',
    'validate_input_array',
    'safe_dtype_conversion',
]
