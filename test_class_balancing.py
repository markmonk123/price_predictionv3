"""Check fold-local oversampling, calibration, class weights, and persistence."""
import contextlib
import importlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from class_balancing import balancing_mode, feature_importances, resampled_classifier
from weighted_voting import make_weighted_voting_classifier


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        self.seen_indices_ = np.asarray(X)[:, 0].copy()
        self.sample_weight_ = sample_weight
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1/len(self.classes_))

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


class BalancingTests(unittest.TestCase):
    def data(self):
        X = pd.DataFrame({'position': np.arange(180), 'value': np.sin(np.arange(180)/7)})
        y = pd.Series(np.where(np.arange(180) % 19 == 0, -1,
                               np.where(np.arange(180) % 7 == 0, 1, 0)))
        return X, y

    def test_resampling_is_fold_local_and_validation_unmodified(self):
        X, y = self.data()
        before = X.copy(deep=True)
        splitter = TimeSeriesSplit(n_splits=3)
        model = resampled_classifier(RecordingClassifier())
        result = cross_validate(model, X, y, cv=splitter, scoring='balanced_accuracy', return_estimator=True)
        for fitted, (train, validation) in zip(result['estimator'], splitter.split(X)):
            member = fitted.named_steps['classifier']
            self.assertEqual(len(set(member.counts_)), 1)
            self.assertLessEqual(member.seen_indices_.max(), train.max())
            self.assertTrue(set(member.seen_indices_).isdisjoint(validation))
            count = len(fitted.named_steps['class_sampler'].sample_indices_)
            self.assertEqual(len(fitted.predict(X.iloc[validation])), len(validation))
            self.assertEqual(len(fitted.named_steps['class_sampler'].sample_indices_), count)
            self.assertIsNone(member.sample_weight_)
        pd.testing.assert_frame_equal(X, before)

    def test_no_double_weights_nested_pipeline_and_voting(self):
        X, y = self.data()
        voter = make_weighted_voting_classifier([
            ('rf', RandomForestClassifier(n_estimators=3, class_weight='balanced', random_state=42)),
            ('lr', Pipeline([('scale', StandardScaler()),
                             ('lr', LogisticRegression(class_weight='balanced', max_iter=300))])),
        ], voting='soft')
        with patch.dict(os.environ, {'CLASS_BALANCING': 'random-over'}):
            for script in ('enhanced_prediction', 'priceprediction'):
                module = importlib.import_module(script)
                fitted = module._fit_with_balanced_sample_weight(voter, X, y)
                self.assertEqual(fitted.predict_proba(X.tail()).shape, (5, 3))
                params = fitted.named_steps['classifier'].get_params(deep=True)
                self.assertTrue(all(v is None for k, v in params.items()
                                    if k == 'class_weight' or k.endswith('__class_weight')))
                self.assertEqual(len(fitted.named_steps['class_sampler'].sample_indices_), 3*(y == 0).sum())
                calibrated = module.fit_calibrated_classifier(voter, X, y)
                self.assertEqual(calibrated.predict_proba(X.tail()).shape, (5, 3))

    def test_calibration_resamples_only_training_and_keeps_later_prior(self):
        X, y = self.data()
        with patch.dict(os.environ, {'CLASS_BALANCING': 'random-over'}):
            for script in ('enhanced_prediction', 'priceprediction'):
                module = importlib.import_module(script)
                model = module.fit_calibrated_classifier(RecordingClassifier(), X, y)
                folds = list(TimeSeriesSplit(n_splits=3).split(X))
                self.assertEqual(len(model.calibrated_classifiers_), 3)
                for calibrated, (train, validation) in zip(model.calibrated_classifiers_, folds):
                    fitted = calibrated.estimator
                    member = fitted.named_steps['classifier']
                    self.assertEqual(len(set(member.counts_)), 1)
                    self.assertTrue(set(member.seen_indices_).isdisjoint(validation))
                    self.assertLessEqual(member.seen_indices_.max(), train.max())
                self.assertTrue(np.isfinite(model.predict_proba(X.tail())).all())
                # A classifier with constant raw probabilities must retain the
                # majority prior after calibration, rather than become uniform.
                majority_index = int(np.flatnonzero(model.classes_ == 0)[0])
                self.assertGreater(model.predict_proba(X.tail())[:, majority_index].mean(), .6)

    def test_singleton_minority_and_single_class(self):
        X, _ = self.data()
        for y in (np.array([0]*179 + [1]), np.zeros(180)):
            model = resampled_classifier(RandomForestClassifier(n_estimators=2, random_state=42)).fit(X, y)
            self.assertEqual(len(model.predict(X.tail())), 5)
        with patch.dict(os.environ, {'CLASS_BALANCING': 'random-over'}):
            module = importlib.import_module('enhanced_prediction')
            model = module.fit_calibrated_classifier(RandomForestClassifier(n_estimators=2),
                                                     X.head(2), np.zeros(2))
            self.assertEqual(len(model.predict(X.tail())), 5)

    def test_clone_cache_round_trip_and_importances(self):
        X, y = self.data()
        fitted = clone(resampled_classifier(RandomForestClassifier(n_estimators=3, random_state=42))).fit(X, y)
        self.assertEqual(len(feature_importances(fitted)), 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.joblib'
            joblib.dump(fitted, path)
            loaded = joblib.load(path)
        np.testing.assert_array_equal(fitted.predict(X), loaded.predict(X))

    def test_previous_sample_weight_option_and_invalid_config(self):
        X, y = self.data()
        with patch.dict(os.environ, {'CLASS_BALANCING': 'sample-weight'}):
            for script in ('enhanced_prediction', 'priceprediction'):
                module = importlib.import_module(script)
                fitted = module._fit_with_balanced_sample_weight(RecordingClassifier(), X, y)
                self.assertFalse(hasattr(fitted, 'named_steps'))
                self.assertIsNotNone(fitted.sample_weight_)
        with patch.dict(os.environ, {'CLASS_BALANCING': 'unknown'}):
            with self.assertRaises(ValueError):
                balancing_mode()

    def test_gpu_forest_resampling_when_available(self):
        module = importlib.import_module('enhanced_prediction')
        if not module.CUDA_AVAILABLE:
            self.skipTest('CUDA/cuML not available')
        X, y = self.data()
        with patch.dict(os.environ, {'CLASS_BALANCING': 'random-over'}), \
                contextlib.redirect_stdout(io.StringIO()):
            model = module._fit_with_balanced_sample_weight(
                module._gpu_rf(n_estimators=3, max_depth=3, random_state=42), X, y)
        self.assertEqual(model.predict_proba(X.tail()).shape, (5, 3))


if __name__ == '__main__':
    unittest.main()
