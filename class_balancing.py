"""Fold-local imbalanced-learn resampling for the direction classifiers."""
import os

from sklearn.base import clone
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    # The previous sample-weight mode remains usable before installing imblearn.
    RandomOverSampler = object


class TrainingOverSampler(RandomOverSampler):
    """Allow a single-class early fold when its classifier supports that case."""

    def fit_resample(self, X, y, **params):
        import numpy as np
        if len(np.unique(y)) < 2:
            self.sampling_strategy_ = {}
            self.sample_indices_ = np.arange(len(y))
            return X, y
        return super().fit_resample(X, y, **params)


def balancing_mode():
    mode = os.environ.get('CLASS_BALANCING', 'random-over').strip().lower()
    if mode not in ('random-over', 'sample-weight'):
        raise ValueError('CLASS_BALANCING must be random-over or sample-weight')
    return mode


def resampling_enabled():
    return balancing_mode() == 'random-over'


def resampled_classifier(estimator):
    """Wrap a clone so every fit resamples only the data supplied to that fit."""
    try:
        from imblearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError('Install imbalanced-learn with python -m pip install -r requirements.txt '
                          'or set CLASS_BALANCING=sample-weight for the previous method.') from exc
    if isinstance(estimator, Pipeline) and 'class_sampler' in estimator.named_steps:
        return clone(estimator)
    model = clone(estimator)
    # Resampling supplies the balancing. Constructor weights would apply it twice.
    weights = {key: None for key in model.get_params(deep=True)
               if key == 'class_weight' or key.endswith('__class_weight')}
    model.set_params(**weights)
    return Pipeline([
        ('class_sampler', TrainingOverSampler(sampling_strategy='not majority', random_state=42)),
        ('classifier', model),
    ])


def feature_importances(estimator):
    """Unwrap the fitted resampling pipeline for the existing feature report."""
    if hasattr(estimator, 'named_steps') and 'class_sampler' in estimator.named_steps:
        estimator = estimator.named_steps['classifier']
    return estimator.feature_importances_


def fit_resampled_calibrated(estimator, X, y, cv):
    """Balance each model's training slice; calibrate on the untouched later slice."""
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    model = resampled_classifier(estimator)
    if len(np.unique(y)) < 2:
        return model.fit(X, y)
    calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=cv)
    try:
        # No sample weights: the sampler balances training, not calibration priors.
        return calibrated.fit(X, y)
    except ValueError as exc:
        print(f'   Calibration skipped: {exc}')
        return model.fit(X, y)
