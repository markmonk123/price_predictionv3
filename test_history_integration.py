"""CPU integration tests for both entrypoints; deterministic estimators, no network."""
import contextlib
import importlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import enhanced_forecasting as forecasting
from services.history_runtime import model_identity
from services.prediction_history import PredictionHistory, epoch


class Classifier:
    classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        return np.tile([.025, .025, .95], (len(X), 1))

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class Regressor:
    def predict(self, X):
        return np.tile(np.arange(48) + 82000., (len(X), 1))


class IntegrationTests(unittest.TestCase):
    def frame(self, count=300):
        end = pd.Timestamp.now(tz='UTC').floor('5min') - pd.Timedelta(minutes=5)
        dates = pd.date_range(end=end, periods=count, freq='5min')
        prices = 82000 + np.sin(np.arange(count)/9)*100 + np.arange(count)/10
        return pd.DataFrame(dict(date=dates, price=prices, high=prices+20, low=prices-20))

    def test_cached_and_fresh_entrypoints_save_all_forecast_windows(self):
        for script in ('enhanced_prediction', 'priceprediction'):
            module = importlib.import_module(script)
            for fast in (False, True):
                with self.subTest(script=script, fast=fast), tempfile.TemporaryDirectory() as temp:
                    path = str(Path(temp) / 'ledger.sqlite3')
                    raw = self.frame()
                    features = raw.assign(feature_a=np.arange(len(raw)), target=1,
                                          future_price=raw.price.shift(-3),
                                          next_return=raw.price.pct_change(3).shift(-3))

                    def feature_builder(df, include_unlabeled=False, **kwargs):
                        return features.copy() if include_unlabeled else features.dropna().copy()

                    trained = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1)
                    identity = model_identity('classifier', trained, raw.date.iloc[100])
                    state = SimpleNamespace(saved_at=trained.to_pydatetime(), ensemble=Classifier(),
                                            booster=object(), scaler_results=[], feature_cols=['feature_a'],
                                            schema_version=1, metadata={'history_identity': identity})

                    def prepare(owner, df, horizon_hours=12):
                        owner.trained_models = {'fixture': Regressor()}
                        owner.history_identity = model_identity('forecast', trained, raw.date.iloc[100], fast)
                        return True

                    booster = SimpleNamespace(predict_with_boost=lambda row: (1, .95, 'fixture'))
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        stack.enter_context(patch.dict(os.environ, {'PREDICTION_DB_PATH': path,
                                             'FAST_PATH_MODEL_STATE': '1' if fast else '0'}))
                        stack.enter_context(patch.object(module, 'CUDA_AVAILABLE', False))
                        stack.enter_context(patch.object(module, 'fetch_bitcoin_data', return_value=raw))
                        stack.enter_context(patch.object(module, 'create_enhanced_features', side_effect=feature_builder))
                        for name in ('identify_long_signal', 'identify_short_signal', 'identify_short_signal_lrs',
                                     'identify_long_signal_dpo', 'identify_long_signal_lrs'):
                            stack.enter_context(patch.object(module, name, return_value=features.iloc[:0]))
                        stack.enter_context(patch('services.model_state.load_state', return_value=state))
                        save = stack.enter_context(patch('services.model_state.save_state', return_value='fixture'))
                        stack.enter_context(patch.object(forecasting.EnhancedBitcoinForecaster, 'prepare_models', prepare))
                        stack.enter_context(patch.object(module.ScalerEvaluator, 'evaluate', return_value=[]))
                        stack.enter_context(patch.object(module, 'EnsembleConfidenceBooster', return_value=booster))
                        stack.enter_context(patch.object(module, 'clone', side_effect=lambda model: model))
                        stack.enter_context(patch('weighted_voting.make_weighted_voting_classifier', return_value=Classifier()))
                        stack.enter_context(patch.object(module, 'walk_forward_scores', return_value=np.array([.5, .5])))
                        fit = stack.enter_context(patch.object(module, 'fit_calibrated_classifier', return_value=Classifier()))
                        stack.enter_context(patch.object(module, '_fit_with_balanced_sample_weight',
                                                return_value=SimpleNamespace(feature_importances_=np.array([1.]))))
                        module.main()
                        self.assertEqual(fit.called, not fast)
                        self.assertEqual(save.called, not fast)
                    history = PredictionHistory(path)
                    try:
                        prices = history.rows(kind='price')
                        self.assertEqual(len(prices), 48)
                        self.assertEqual([r['horizon_minutes'] for r in prices], list(range(15, 721, 15)))
                        self.assertTrue(all(r['target_end']-r['target_start'] == 900 for r in prices))
                        self.assertTrue(all(r['cached'] == int(fast) for r in prices))
                        direction = [r for r in history.rows(kind='direction') if r['signal'] == 'classifier']
                        self.assertEqual(len(direction), 1)
                        self.assertEqual(direction[0]['historical'], 0)
                        self.assertEqual(direction[0]['reference_price'], raw.price.iloc[-1])
                        self.assertEqual(direction[0]['target_end'], epoch(raw.date.iloc[-1])+1200)
                        self.assertEqual(direction[0]['model_version'], identity['model_version'] if fast else
                                         save.call_args.kwargs['metadata']['history_identity']['model_version'])
                        if not fast:
                            heldout = [r for r in history.rows() if r['signal'] == 'heldout_classifier']
                            self.assertTrue(heldout)
                            self.assertTrue(all(r['historical'] for r in heldout))
                    finally:
                        history.close()

    def test_real_features_preserve_latest_unlabeled_row_without_future_leakage(self):
        raw = self.frame(12000)
        for name in ('enhanced_prediction', 'priceprediction'):
            module = importlib.import_module(name)
            with self.subTest(script=name):
                training = module.create_enhanced_features(raw, horizon_steps=3)
                live = module.create_enhanced_features(raw, horizon_steps=3, include_unlabeled=True)
                self.assertEqual(live.date.iloc[-1], raw.date.iloc[-1])
                self.assertEqual(training.date.iloc[-1], raw.date.iloc[-4])
                self.assertTrue(pd.isna(live.target.iloc[-1]))
                # Future labels never enter the feature matrix; the shared prefix is identical.
                columns = [c for c in live if c not in ('future_price', 'next_return', 'target')]
                pd.testing.assert_frame_equal(training[columns], live.loc[training.index, columns])
                for signal in ('long_signal', 'short_signal', 'short_signal_lrs',
                               'long_signal_dpo', 'long_signal_lrs'):
                    kwargs = {'adx_threshold': 25}
                    if signal in ('long_signal', 'short_signal_lrs', 'long_signal_lrs'):
                        kwargs['lr_period'] = 6 if name == 'enhanced_prediction' else 30
                    detected = getattr(module, 'identify_' + signal)(live, **kwargs)
                    self.assertIsInstance(detected, pd.DataFrame)

    def test_forecast_cache_reuse_and_expiry(self):
        trained_at = 1000000
        identity = model_identity('forecast', trained_at, trained_at-900)
        state = dict(schema_version=1, horizon_hours=12, saved_at=trained_at,
                     trained_models={'fixture': Regressor()}, history_identity=identity)
        for age in (1.07*3600, 72*3600, 72*3600+1):
            forecaster = forecasting.EnhancedBitcoinForecaster()
            with patch.dict(os.environ, {'FAST_PATH_MODEL_STATE': '1', 'MAX_STATE_AGE_HOURS': '72'}), \
                    patch('joblib.load', return_value=state), \
                    patch('time.time', return_value=trained_at+age), \
                    patch.object(forecaster, 'train_models', return_value=False) as train, \
                    contextlib.redirect_stdout(io.StringIO()):
                ready = forecaster.prepare_models(self.frame())
            self.assertEqual(train.called, age > 72*3600)
            self.assertEqual(ready, age <= 72*3600)
            if ready:
                self.assertEqual(forecaster.history_identity['model_version'], identity['model_version'])
                self.assertTrue(forecaster.history_identity['cached'])

    def test_real_forecast_training_cache_reload_and_displayed_prices(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = self.frame(1200)
            forecaster = forecasting.EnhancedBitcoinForecaster()
            forecaster.models = {'tiny': lambda: ExtraTreesRegressor(n_estimators=2, random_state=42)}
            with patch.object(forecasting, '__file__', str(Path(temp) / 'enhanced_forecasting.py')), \
                    patch.dict(os.environ, {'FAST_PATH_MODEL_STATE': '1'}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(forecaster.prepare_models(raw))
                identity = forecaster.history_identity.copy()
                forecast = forecaster.generate_forecast(raw)
                reloaded = forecasting.EnhancedBitcoinForecaster()
                with patch.object(reloaded, 'train_models') as train:
                    self.assertTrue(reloaded.prepare_models(raw))
                    train.assert_not_called()
                self.assertEqual(identity['model_version'], reloaded.history_identity['model_version'])
                pd.testing.assert_frame_equal(forecast, reloaded.generate_forecast(raw))
                path = str(Path(temp) / 'history.sqlite3')
                with patch.dict(os.environ, {'PREDICTION_DB_PATH': path}):
                    from services.history_runtime import HistoryRuntime
                    runtime = HistoryRuntime('integration', raw)
                    runtime.forecast(forecast, reloaded, reloaded.forecast_reference_price)
                    runtime.finish()
            history = PredictionHistory(path)
            try:
                rows = history.rows(kind='price')
                self.assertEqual(len(rows), 48)
                self.assertEqual([r['predicted_price'] for r in rows], forecast.predicted_price.tolist())
                self.assertTrue(all(r['trained_at'] and r['training_cutoff'] and r['cached'] for r in rows))
            finally:
                history.close()


if __name__ == '__main__':
    unittest.main()
