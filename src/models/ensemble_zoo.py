"""
Ensemble Zoo: Build and evaluate imbalanced-aware ensemble models.

This module implements multiple ensemble strategies combining resampling techniques
(SMOTE, RandomUnderSampler, SMOTEENN) with various classifiers to handle class imbalance.
Models are saved with joblib compression and results are tracked in a CSV report.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

from .utils import to_numpy_contiguous, get_candidate_scalers, rank_scalers_by_estimator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_ensembles(X, y, output_dir='models', cv_splits=5, scoring='roc_auc', n_jobs=1, random_state=42):
    """
    Build and evaluate three imbalanced-aware ensemble models.
    
    Creates three different ensemble strategies:
    1. balanced_stacking: Scaler -> SMOTE -> Stacking (RF, GB, BalancedRF) with LR final
    2. balanced_voting: Multiple pipelines with different resampling strategies in a voting ensemble
    3. hybrid_zoo: EasyEnsemble + GB pipeline combined via voting
    
    For each ensemble:
    - Evaluates top-5 scalers using cross-validation
    - Selects the best scaler
    - Fits the final model with best scaler
    - Saves model to output_dir with joblib compression
    
    On first run (if ensemble_zoo_results.csv doesn't exist), creates a CSV report with:
    ensemble_name, best_scaler, best_score, model_path
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target vector (numpy array or pandas Series)
        output_dir: Directory to save models (default: 'models')
        cv_splits: Number of cross-validation folds (default: 5)
        scoring: Scoring metric for evaluation (default: 'roc_auc')
        n_jobs: Number of parallel jobs (default: 1)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        dict: Dictionary with ensemble names as keys and fitted models as values
    """
    logger.info("=" * 80)
    logger.info("Starting Ensemble Zoo Model Building")
    logger.info("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")
    
    # Convert to contiguous numpy arrays
    X = to_numpy_contiguous(X)
    if isinstance(y, pd.Series):
        y = y.values
    y = np.ascontiguousarray(y)
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Get candidate scalers
    candidate_scalers = get_candidate_scalers()
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # Store results
    results = []
    fitted_models = {}
    
    # ============================================================================
    # 1. Balanced Stacking Ensemble
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Building Balanced Stacking Ensemble")
    logger.info("=" * 80)
    
    # Define base estimators for stacking
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=random_state, n_jobs=1)),
        ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=random_state)),
        ('brf', BalancedRandomForestClassifier(n_estimators=50, max_depth=10, random_state=random_state, n_jobs=1))
    ]
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=random_state, max_iter=1000),
        cv=3,
        n_jobs=1
    )
    
    # Evaluate scalers for stacking ensemble
    test_estimators = [('stacking', stacking_clf)]
    scaler_rankings = rank_scalers_by_estimator(
        X, y, test_estimators, candidate_scalers[:5], 
        cv=cv_splits, scoring=scoring, n_jobs=1
    )
    
    # Get best scaler
    best_scaler_name, best_scaler, best_score = scaler_rankings[0]
    logger.info(f"Best scaler for stacking: {best_scaler_name} (score={best_score:.4f})")
    
    # Build final pipeline
    stacking_pipeline = ImbPipeline([
        ('scaler', best_scaler),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', stacking_clf)
    ])
    
    # Fit and save
    logger.info("Fitting stacking ensemble...")
    stacking_pipeline.fit(X, y)
    model_path = output_path / 'ensemble_balanced_stacking.joblib'
    joblib.dump(stacking_pipeline, model_path, compress=3)
    logger.info(f"Saved to: {model_path}")
    
    fitted_models['balanced_stacking'] = stacking_pipeline
    results.append({
        'ensemble_name': 'balanced_stacking',
        'best_scaler': best_scaler_name,
        'best_score': best_score,
        'model_path': str(model_path)
    })
    
    # ============================================================================
    # 2. Balanced Voting Ensemble
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Building Balanced Voting Ensemble")
    logger.info("=" * 80)
    
    # Evaluate scalers for voting
    voting_test_estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1))
    ]
    scaler_rankings_voting = rank_scalers_by_estimator(
        X, y, voting_test_estimators, candidate_scalers[:5],
        cv=cv_splits, scoring=scoring, n_jobs=1
    )
    
    best_scaler_name_v, best_scaler_v, best_score_v = scaler_rankings_voting[0]
    logger.info(f"Best scaler for voting: {best_scaler_name_v} (score={best_score_v:.4f})")
    
    # Create multiple pipelines with different resampling strategies
    from sklearn.base import clone
    
    pipeline1 = ImbPipeline([
        ('scaler', clone(best_scaler_v)),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1))
    ])
    
    pipeline2 = ImbPipeline([
        ('scaler', clone(best_scaler_v)),
        ('rus', RandomUnderSampler(random_state=random_state)),
        ('classifier', GradientBoostingClassifier(n_estimators=50, random_state=random_state))
    ])
    
    pipeline3 = ImbPipeline([
        ('scaler', clone(best_scaler_v)),
        ('smoteenn', SMOTEENN(random_state=random_state)),
        ('classifier', BalancedRandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1))
    ])
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('smote_rf', pipeline1),
            ('rus_gb', pipeline2),
            ('smoteenn_brf', pipeline3)
        ],
        voting='soft',
        n_jobs=1
    )
    
    # Fit and save
    logger.info("Fitting voting ensemble...")
    voting_clf.fit(X, y)
    model_path_v = output_path / 'ensemble_voting.joblib'
    joblib.dump(voting_clf, model_path_v, compress=3)
    logger.info(f"Saved to: {model_path_v}")
    
    fitted_models['balanced_voting'] = voting_clf
    results.append({
        'ensemble_name': 'balanced_voting',
        'best_scaler': best_scaler_name_v,
        'best_score': best_score_v,
        'model_path': str(model_path_v)
    })
    
    # ============================================================================
    # 3. Hybrid Zoo Ensemble
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Building Hybrid Zoo Ensemble")
    logger.info("=" * 80)
    
    # Evaluate scalers for hybrid
    hybrid_test_estimators = [
        ('ee', EasyEnsembleClassifier(n_estimators=10, random_state=random_state, n_jobs=1))
    ]
    scaler_rankings_hybrid = rank_scalers_by_estimator(
        X, y, hybrid_test_estimators, candidate_scalers[:5],
        cv=cv_splits, scoring=scoring, n_jobs=1
    )
    
    best_scaler_name_h, best_scaler_h, best_score_h = scaler_rankings_hybrid[0]
    logger.info(f"Best scaler for hybrid: {best_scaler_name_h} (score={best_score_h:.4f})")
    
    # Create hybrid pipelines
    hybrid_pipeline1 = ImbPipeline([
        ('scaler', clone(best_scaler_h)),
        ('classifier', EasyEnsembleClassifier(n_estimators=10, random_state=random_state, n_jobs=1))
    ])
    
    hybrid_pipeline2 = ImbPipeline([
        ('scaler', clone(best_scaler_h)),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', GradientBoostingClassifier(n_estimators=50, random_state=random_state))
    ])
    
    # Create voting classifier
    hybrid_clf = VotingClassifier(
        estimators=[
            ('easy_ensemble', hybrid_pipeline1),
            ('smote_gb', hybrid_pipeline2)
        ],
        voting='soft',
        n_jobs=1
    )
    
    # Fit and save
    logger.info("Fitting hybrid ensemble...")
    hybrid_clf.fit(X, y)
    model_path_h = output_path / 'ensemble_hybrid.joblib'
    joblib.dump(hybrid_clf, model_path_h, compress=3)
    logger.info(f"Saved to: {model_path_h}")
    
    fitted_models['hybrid_zoo'] = hybrid_clf
    results.append({
        'ensemble_name': 'hybrid_zoo',
        'best_scaler': best_scaler_name_h,
        'best_score': best_score_h,
        'model_path': str(model_path_h)
    })
    
    # ============================================================================
    # Save Results CSV (on first run)
    # ============================================================================
    results_csv = output_path / 'ensemble_zoo_results.csv'
    if not results_csv.exists():
        logger.info(f"\nSaving results to: {results_csv}")
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_csv, index=False)
        logger.info("Results CSV created successfully")
    else:
        logger.info(f"\nResults CSV already exists: {results_csv}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Ensemble Zoo Building Complete")
    logger.info("=" * 80)
    
    return fitted_models


def main():
    """CLI entrypoint for building ensembles from CSV data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build ensemble models from CSV data')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--target', type=str, default='target', help='Name of target column')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--cv-splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--scoring', type=str, default='roc_auc', help='Scoring metric')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from: {args.input}")
        df = pd.read_csv(args.input, index_col=0, parse_dates=True)
        logger.info(f"Data loaded: {df.shape}")
        
        # Check target column
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in data")
        
        # Split features and target
        X = df.drop(columns=[args.target])
        y = df[args.target]
        
        # Build ensembles
        models = build_ensembles(
            X, y,
            output_dir=args.output_dir,
            cv_splits=args.cv_splits,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            random_state=args.random_state
        )
        
        logger.info(f"\nSuccessfully built {len(models)} ensemble models")
        return 0
        
    except Exception as e:
        logger.error(f"Error building ensembles: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
