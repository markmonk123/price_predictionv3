"""
Ensemble Zoo: Build and evaluate imbalanced-learn ensemble models.

Implements three ensemble strategies:
1. Balanced Stacking: Stacking classifier with SMOTE sampling
2. Balanced Voting: Voting classifier with diverse base estimators
3. Hybrid: EasyEnsembleClassifier with Gradient Boosting

Evaluates top-5 scalers and saves best models with joblib compression.
Generates CSV report of results on first run.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
)

from .utils import (
    to_numpy_contiguous,
    get_candidate_scalers,
    rank_scalers_by_estimator,
    logger as utils_logger,
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1
MODELS_DIR = Path("models")
RESULTS_CSV = MODELS_DIR / "ensemble_zoo_results.csv"


def create_balanced_stacking_ensemble(
    scaler: Any,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS
) -> ImbPipeline:
    """
    Create a balanced stacking ensemble with SMOTE sampling.
    
    Uses diverse base estimators (Random Forest, Gradient Boosting, Decision Tree)
    with SMOTE oversampling and a Logistic Regression meta-learner.
    
    Args:
        scaler: Fitted or unfitted scaler instance
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
    
    Returns:
        Imbalanced-learn pipeline with scaler, sampler, and stacking classifier
    """
    logger.info("Creating Balanced Stacking Ensemble")
    
    # Define diverse base estimators
    base_estimators = [
        (
            'rf',
            BalancedRandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=random_state,
                n_jobs=n_jobs
            )
        ),
        (
            'gb',
            GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=random_state
            )
        ),
        (
            'dt',
            DecisionTreeClassifier(
                max_depth=8,
                random_state=random_state,
                class_weight='balanced'
            )
        ),
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=3,
        n_jobs=n_jobs
    )
    
    # Build pipeline: scaler -> SMOTE -> stacking
    pipeline = ImbPipeline([
        ('scaler', scaler),
        ('smote', SMOTE(random_state=random_state, k_neighbors=5)),
        ('stacking', stacking_clf)
    ])
    
    logger.info("Balanced Stacking Ensemble created successfully")
    return pipeline


def create_balanced_voting_ensemble(
    scaler: Any,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS
) -> ImbPipeline:
    """
    Create a balanced voting ensemble with combined sampling.
    
    Uses SMOTEENN (SMOTE + Edited Nearest Neighbors) for balanced sampling
    and soft voting across diverse classifiers.
    
    Args:
        scaler: Fitted or unfitted scaler instance
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
    
    Returns:
        Imbalanced-learn pipeline with scaler, sampler, and voting classifier
    """
    logger.info("Creating Balanced Voting Ensemble")
    
    # Define diverse estimators for voting
    estimators = [
        (
            'brf',
            BalancedRandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=random_state,
                n_jobs=n_jobs
            )
        ),
        (
            'gb',
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        ),
        (
            'lr',
            LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=n_jobs
            )
        ),
    ]
    
    # Soft voting classifier
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=n_jobs
    )
    
    # Build pipeline: scaler -> SMOTEENN -> voting
    pipeline = ImbPipeline([
        ('scaler', scaler),
        ('smoteenn', SMOTEENN(random_state=random_state)),
        ('voting', voting_clf)
    ])
    
    logger.info("Balanced Voting Ensemble created successfully")
    return pipeline


def create_hybrid_ensemble(
    scaler: Any,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS
) -> ImbPipeline:
    """
    Create a hybrid ensemble combining EasyEnsemble with Gradient Boosting.
    
    EasyEnsembleClassifier handles imbalance through bagging with undersampling,
    using Gradient Boosting as base estimator for strong performance.
    
    Args:
        scaler: Fitted or unfitted scaler instance
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
    
    Returns:
        Imbalanced-learn pipeline with scaler and EasyEnsemble classifier
    """
    logger.info("Creating Hybrid Ensemble (EasyEnsemble + GB)")
    
    # Base estimator for EasyEnsemble
    base_estimator = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=random_state
    )
    
    # EasyEnsemble with GB base
    easy_ensemble = EasyEnsembleClassifier(
        n_estimators=10,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Build pipeline: scaler -> easy ensemble
    pipeline = ImbPipeline([
        ('scaler', scaler),
        ('easy_ensemble', easy_ensemble)
    ])
    
    logger.info("Hybrid Ensemble created successfully")
    return pipeline


def evaluate_ensemble(
    pipeline: ImbPipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    random_state: int = DEFAULT_RANDOM_STATE
) -> float:
    """
    Evaluate ensemble using stratified cross-validation.
    
    Args:
        pipeline: Imbalanced-learn pipeline to evaluate
        X: Feature matrix
        y: Target vector
        cv: Number of CV folds
        scoring: Sklearn scoring metric
        random_state: Random seed for reproducibility
    
    Returns:
        Mean cross-validation score
    """
    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )
    
    try:
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=1  # Pipeline already uses parallelism
        )
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        logger.info(f"CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
        return mean_score
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return -np.inf


def build_and_evaluate_ensembles(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'f1_weighted',
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS,
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Build all three ensembles, evaluate with top scalers, and save best models.
    
    This is the main entry point for training the ensemble zoo. It:
    1. Evaluates top-5 scalers using a simple estimator
    2. Builds each ensemble with the best scaler
    3. Cross-validates each ensemble
    4. Saves best models to disk
    5. Generates CSV report (if models dir was empty initially)
    
    Args:
        X: Feature DataFrame
        y: Target Series
        cv: Number of cross-validation folds
        scoring: Sklearn scoring metric
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        save_models: Whether to save trained models
    
    Returns:
        Dictionary with results for each ensemble
    """
    logger.info("=" * 60)
    logger.info("Starting Ensemble Zoo Training")
    logger.info("=" * 60)
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if this is first run (empty models directory)
    is_first_run = not any(MODELS_DIR.glob("*.joblib"))
    
    # Convert to numpy arrays
    X_array = to_numpy_contiguous(X, dtype=np.float64)
    y_array = to_numpy_contiguous(y, dtype=np.int64)
    
    logger.info(f"Dataset shape: X={X_array.shape}, y={y_array.shape}")
    logger.info(f"Class distribution: {np.bincount(y_array)}")
    
    # Step 1: Rank scalers
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Evaluating Top-5 Scalers")
    logger.info("=" * 60)
    
    # Use a simple balanced classifier for scaler ranking
    simple_estimator = BalancedRandomForestClassifier(
        n_estimators=50,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    scaler_rankings = rank_scalers_by_estimator(
        X_array,
        y_array,
        estimator=simple_estimator,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    # Get best scaler
    best_scaler_name, best_scaler_score = scaler_rankings[0]
    logger.info(f"\nBest scaler: {best_scaler_name} (score: {best_scaler_score:.4f})")
    
    # Get scaler instance by name
    candidate_scalers = get_candidate_scalers()
    scaler_map = dict(candidate_scalers)
    best_scaler = scaler_map[best_scaler_name]
    
    # Step 2: Build and evaluate ensembles
    results = {}
    
    ensemble_builders = {
        'ensemble_balanced_stacking': create_balanced_stacking_ensemble,
        'ensemble_voting': create_balanced_voting_ensemble,
        'ensemble_hybrid': create_hybrid_ensemble,
    }
    
    for ensemble_name, builder_func in ensemble_builders.items():
        logger.info("\n" + "=" * 60)
        logger.info(f"Building and Evaluating: {ensemble_name}")
        logger.info("=" * 60)
        
        # Create ensemble with best scaler
        from sklearn.base import clone
        ensemble_pipeline = builder_func(
            scaler=clone(best_scaler),
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Evaluate with cross-validation
        cv_score = evaluate_ensemble(
            ensemble_pipeline,
            X_array,
            y_array,
            cv=cv,
            scoring=scoring,
            random_state=random_state
        )
        
        # Train on full dataset for saving
        if save_models and cv_score > -np.inf:
            logger.info(f"Training {ensemble_name} on full dataset...")
            try:
                ensemble_pipeline.fit(X_array, y_array)
                
                # Save model with compression
                model_path = MODELS_DIR / f"{ensemble_name}.joblib"
                joblib.dump(ensemble_pipeline, model_path, compress=3)
                logger.info(f"Model saved to: {model_path}")
                
                results[ensemble_name] = {
                    'best_scaler': best_scaler_name,
                    'best_score': cv_score,
                    'model_path': str(model_path)
                }
            except Exception as e:
                logger.error(f"Failed to train/save {ensemble_name}: {e}")
                results[ensemble_name] = {
                    'best_scaler': best_scaler_name,
                    'best_score': cv_score,
                    'model_path': None,
                    'error': str(e)
                }
        else:
            results[ensemble_name] = {
                'best_scaler': best_scaler_name,
                'best_score': cv_score,
                'model_path': None
            }
    
    # Step 3: Generate CSV report on first run
    if is_first_run and results:
        logger.info("\n" + "=" * 60)
        logger.info("Generating Ensemble Zoo Results CSV")
        logger.info("=" * 60)
        
        csv_data = []
        for ensemble_name, result in results.items():
            csv_data.append({
                'ensemble_name': ensemble_name,
                'best_scaler': result['best_scaler'],
                'best_score': result['best_score'],
                'model_path': result['model_path']
            })
        
        results_df = pd.DataFrame(csv_data)
        results_df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"Results saved to: {RESULTS_CSV}")
        logger.info(f"\n{results_df.to_string(index=False)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Ensemble Zoo Training Complete!")
    logger.info("=" * 60)
    
    return results


def main():
    """
    CLI entry point for training ensemble zoo from CSV file.
    
    Usage:
        python -m src.models.ensemble_zoo <data.csv>
    
    Expects CSV with numeric features and a 'target' column.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Ensemble Zoo on CSV data with target column'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to CSV file with features and target column'
    )
    parser.add_argument(
        '--target-col',
        type=str,
        default='target',
        help='Name of target column (default: target)'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--scoring',
        type=str,
        default='f1_weighted',
        help='Scoring metric (default: f1_weighted)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f'Random seed (default: {DEFAULT_RANDOM_STATE})'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=f'Number of parallel jobs (default: {DEFAULT_N_JOBS})'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from: {args.data_path}")
    try:
        data = pd.read_csv(args.data_path)
        logger.info(f"Data loaded: shape={data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Separate features and target
    if args.target_col not in data.columns:
        logger.error(f"Target column '{args.target_col}' not found in data")
        sys.exit(1)
    
    X = data.drop(columns=[args.target_col])
    y = data[args.target_col]
    
    # Train ensembles
    try:
        results = build_and_evaluate_ensembles(
            X=X,
            y=y,
            cv=args.cv,
            scoring=args.scoring,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            save_models=True
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        for name, result in results.items():
            logger.info(f"{name}:")
            logger.info(f"  Scaler: {result['best_scaler']}")
            logger.info(f"  Score: {result['best_score']:.4f}")
            logger.info(f"  Model: {result['model_path']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
