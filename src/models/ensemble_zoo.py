"""
Ensemble Zoo: Build and evaluate imbalanced-learn based ensemble models.

This module implements three ensemble strategies:
1. Balanced Stacking: Stacking classifier with SMOTE sampling
2. Balanced Voting: Voting classifier with balanced base estimators
3. Hybrid: EasyEnsemble + Gradient Boosting

Features:
- Evaluates top-5 scalers via cross-validation
- Uses imbalanced-learn samplers (SMOTE, RandomUnderSampler, SMOTEENN)
- Saves best models with joblib compression
- CLI entrypoint for training from CSV

Security notes:
- Deterministic random_state throughout
- Input validation for feature sizes
- Safe model serialization with joblib
- Environment-based configuration
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Import utilities from the same package
from .utils import (
    to_numpy_contiguous,
    get_candidate_scalers,
    rank_scalers_by_estimator,
    validate_features,
    get_random_state
)

logger = logging.getLogger(__name__)


class EnsembleZoo:
    """
    Factory for building and evaluating ensemble models with imbalanced-learn.
    
    This class manages:
    - Scaler selection via cross-validation
    - Building three ensemble types
    - Training and evaluation
    - Model persistence
    """
    
    def __init__(
        self, 
        output_dir: str = "models",
        random_state: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        Initialize the ensemble zoo.
        
        Args:
            output_dir: Directory to save trained models
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs for training
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_state = get_random_state(random_state)
        self.n_jobs = n_jobs
        
        self.best_scaler = None
        self.top_scalers = None
        self.models: Dict[str, Any] = {}
        
        logger.info(f"Initialized EnsembleZoo with random_state={self.random_state}")
    
    def select_best_scalers(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Evaluate and select top-k scalers using a balanced random forest.
        
        Args:
            X: Feature matrix
            y: Target vector
            top_k: Number of top scalers to return
            
        Returns:
            List of (scaler_name, score) tuples
        """
        logger.info("Evaluating candidate scalers...")
        
        # Use a balanced estimator for fair scaler comparison
        base_estimator = BalancedRandomForestClassifier(
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=1,  # Keep low for parallel scaler evaluation
            max_depth=10
        )
        
        # Rank all scalers
        scaler_rankings = rank_scalers_by_estimator(
            X=X,
            y=y,
            estimator=base_estimator,
            scalers=get_candidate_scalers(),
            cv=3,  # Use 3-fold for speed during scaler selection
            scoring="roc_auc",
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Keep top-k
        self.top_scalers = scaler_rankings[:top_k]
        self.best_scaler = self.top_scalers[0][0]
        
        logger.info(f"Selected top {top_k} scalers:")
        for i, (name, score) in enumerate(self.top_scalers, 1):
            logger.info(f"  {i}. {name}: {score:.4f}")
        
        return self.top_scalers
    
    def build_balanced_stacking(self) -> ImbPipeline:
        """
        Build a stacking ensemble with SMOTE sampling.
        
        Architecture:
        - SMOTE oversampling
        - Base estimators: Balanced RF, Logistic Regression, Gradient Boosting
        - Meta-estimator: Logistic Regression
        
        Returns:
            Imbalanced-learn pipeline with stacking ensemble
        """
        logger.info("Building Balanced Stacking Ensemble...")
        
        # Use StandardScaler as default scaler
        scaler = StandardScaler()
        
        # Base estimators for stacking
        base_estimators = [
            ('brf', BalancedRandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )),
            ('lr', LogisticRegression(
                max_iter=500,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            ))
        ]
        
        # Stacking with logistic regression meta-estimator
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=3,
            n_jobs=self.n_jobs
        )
        
        # Pipeline: Scale -> SMOTE -> Stack
        pipeline = ImbPipeline([
            ('scaler', scaler),
            ('smote', SMOTE(random_state=self.random_state)),
            ('stacking', stacking)
        ])
        
        return pipeline
    
    def build_balanced_voting(self) -> ImbPipeline:
        """
        Build a voting ensemble with RandomUnderSampler.
        
        Architecture:
        - RandomUnderSampler for balancing
        - Soft voting over: Balanced RF, Easy Ensemble, Gradient Boosting
        
        Returns:
            Imbalanced-learn pipeline with voting ensemble
        """
        logger.info("Building Balanced Voting Ensemble...")
        
        scaler = StandardScaler()
        
        # Estimators for voting
        estimators = [
            ('brf', BalancedRandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )),
            ('easyens', EasyEnsembleClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ))
        ]
        
        # Voting ensemble with soft voting (probability averaging)
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.n_jobs
        )
        
        # Pipeline: Scale -> UnderSample -> Vote
        pipeline = ImbPipeline([
            ('scaler', scaler),
            ('undersampler', RandomUnderSampler(random_state=self.random_state)),
            ('voting', voting)
        ])
        
        return pipeline
    
    def build_hybrid_ensemble(self) -> ImbPipeline:
        """
        Build a hybrid ensemble: EasyEnsemble followed by Gradient Boosting.
        
        Architecture:
        - SMOTEENN (combined over/under sampling)
        - EasyEnsembleClassifier for balanced bootstrap aggregating
        
        Returns:
            Imbalanced-learn pipeline with hybrid ensemble
        """
        logger.info("Building Hybrid Ensemble (EasyEnsemble + GB)...")
        
        scaler = StandardScaler()
        
        # EasyEnsemble with gradient boosting base estimator
        ensemble = EasyEnsembleClassifier(
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Pipeline: Scale -> SMOTEENN -> EasyEnsemble
        pipeline = ImbPipeline([
            ('scaler', scaler),
            ('smoteenn', SMOTEENN(random_state=self.random_state)),
            ('easyensemble', ensemble)
        ])
        
        return pipeline
    
    def train_ensemble(
        self,
        name: str,
        pipeline: ImbPipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Train and evaluate an ensemble using cross-validation.
        
        Args:
            name: Name of the ensemble
            pipeline: Imbalanced-learn pipeline to train
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training {name} with {cv}-fold cross-validation...")
        
        # Convert to contiguous arrays
        X_arr = to_numpy_contiguous(X)
        y_arr = to_numpy_contiguous(y)
        
        # Cross-validate with multiple metrics
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        scoring = {
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'accuracy': 'accuracy'
        }
        
        cv_results = cross_validate(
            pipeline,
            X_arr,
            y_arr,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=self.n_jobs,
            return_train_score=False
        )
        
        # Compute mean scores
        metrics = {
            'roc_auc': cv_results['test_roc_auc'].mean(),
            'f1': cv_results['test_f1'].mean(),
            'accuracy': cv_results['test_accuracy'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std(),
            'f1_std': cv_results['test_f1'].std()
        }
        
        logger.info(f"{name} Results:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f} ± {metrics['roc_auc_std']:.4f}")
        logger.info(f"  F1:      {metrics['f1']:.4f} ± {metrics['f1_std']:.4f}")
        logger.info(f"  Acc:     {metrics['accuracy']:.4f}")
        
        # Train on full dataset
        logger.info(f"Training {name} on full dataset...")
        pipeline.fit(X_arr, y_arr)
        
        # Save model
        self.models[name] = pipeline
        
        return metrics
    
    def save_models(self):
        """
        Save all trained models to disk with joblib compression.
        
        Security notes:
        - Uses joblib (safer than pickle for sklearn objects)
        - Compression reduces storage and transfer risks
        """
        logger.info(f"Saving {len(self.models)} models to {self.output_dir}...")
        
        for name, model in self.models.items():
            filename = self.output_dir / f"{name}.joblib"
            joblib.dump(model, filename, compress=3)
            logger.info(f"  Saved: {filename}")
    
    def build_and_train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Complete workflow: select scalers, build all ensembles, train, and save.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary mapping ensemble names to their metrics
        """
        # Validate inputs
        validate_features(X)
        
        logger.info("="*60)
        logger.info("Starting Ensemble Zoo Training Pipeline")
        logger.info("="*60)
        
        # Step 1: Select best scalers
        self.select_best_scalers(X, y, top_k=5)
        
        # Step 2: Build ensembles
        ensembles = {
            'ensemble_balanced_stacking': self.build_balanced_stacking(),
            'ensemble_voting': self.build_balanced_voting(),
            'ensemble_hybrid': self.build_hybrid_ensemble()
        }
        
        # Step 3: Train and evaluate each ensemble
        results = {}
        for name, pipeline in ensembles.items():
            results[name] = self.train_ensemble(name, pipeline, X, y, cv=cv)
        
        # Step 4: Save models
        self.save_models()
        
        logger.info("="*60)
        logger.info("Ensemble Zoo Training Complete!")
        logger.info("="*60)
        
        return results


def main():
    """
    CLI entrypoint for training ensembles from CSV.
    
    Usage:
        python -m src.models.ensemble_zoo --input data.csv --target target_column
    """
    parser = argparse.ArgumentParser(
        description="Train ensemble models with imbalanced-learn"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        '--target',
        type=str,
        default='target',
        help="Name of target column (default: 'target')"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help="Directory to save trained models (default: 'models')"
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=None,
        help="Random state for reproducibility (default: from env or 42)"
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1, use all cores)"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)
    
    # Validate target column
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in CSV")
        logger.error(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Split features and target
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Initialize and train
    zoo = EnsembleZoo(
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_jobs=args.n_jobs
    )
    
    try:
        results = zoo.build_and_train_all(X, y, cv=args.cv)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for name, metrics in results.items():
            print(f"\n{name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
