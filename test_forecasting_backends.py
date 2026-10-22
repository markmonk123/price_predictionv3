"""Focused tests for optional forecasting backends and wrapper labels."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import enhanced_forecasting as forecasting


class ForecastBackendTests(unittest.TestCase):
    def detect(self, estimator):
        return forecasting.EnhancedBitcoinForecaster._detect_estimator_backend(estimator)

    def optional_components(self, devices=1, missing=False):
        source = Path(forecasting.__file__).read_text(encoding="utf-8")
        source = source[source.index("CUDA_AVAILABLE = False"):source.index("\ntry:\n    import xgboost")]
        component = type("AvailableComponent", (), {})
        def load(module):
            if missing:
                raise ImportError("cuML absent")
            if module == "cuml.ensemble":
                return SimpleNamespace(RandomForestRegressor=component)
            return SimpleNamespace(StandardScaler=component, MinMaxScaler=component,
                                   RobustScaler=component, LinearRegression=component)
        cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: devices)))
        cuml = SimpleNamespace(set_global_output_type=lambda value: None)
        namespace = {}
        with patch.dict("sys.modules", {"cupy": cupy, "cuml": cuml}), patch("importlib.import_module", side_effect=load):
            exec(compile(source, "<optional backends>", "exec"), namespace)
        return namespace

    def test_missing_gradient_boosting_is_independent(self):
        state = self.optional_components()
        self.assertTrue(state["CUDA_AVAILABLE"])
        self.assertIsNotNone(state["_cuml_RF_reg"])
        self.assertIsNone(state["_cuml_GB_reg"])
        self.assertIsNotNone(state["_cuml_StandardScaler"])
        with patch.multiple(forecasting, **{key: state[key] for key in
                            ("CUDA_AVAILABLE", "_cuml_RF_reg", "_cuml_GB_reg")}):
            self.assertIsInstance(forecasting.EnhancedBitcoinForecaster._gpu_rf_reg(), state["_cuml_RF_reg"])
            self.assertIsInstance(forecasting.EnhancedBitcoinForecaster._gpu_gb_reg(), forecasting.GradientBoostingRegressor)

    def test_cpu_fallbacks(self):
        for state in (self.optional_components(devices=0), self.optional_components(missing=True)):
            with patch.multiple(forecasting, **{key: state[key] for key in
                                ("CUDA_AVAILABLE", "_cuml_RF_reg", "_cuml_GB_reg", "_cuml_StandardScaler")}):
                self.assertIsInstance(forecasting.EnhancedBitcoinForecaster._gpu_rf_reg(), RandomForestRegressor)
                self.assertIsInstance(forecasting.EnhancedBitcoinForecaster._gpu_scaler_reg("StandardScaler"), StandardScaler)

    def test_wrapped_and_fitted_backends(self):
        gpu = type("GPUForest", (), {"__module__": "cuml.ensemble"})()
        cpu = RandomForestRegressor()
        self.assertEqual(self.detect(gpu), "cuML")
        self.assertEqual(self.detect(cpu), "sklearn")
        wrapper = MultiOutputRegressor(cpu)
        pipeline = Pipeline([("scale", StandardScaler()), ("model", wrapper)])
        self.assertEqual(self.detect(pipeline), "sklearn")
        wrapper.estimators_ = [gpu]
        self.assertEqual(self.detect(pipeline), "cuML")

    def test_xgboost_device_and_actual_fallback(self):
        class FakeXGB:
            __module__ = "xgboost.sklearn"
            def get_params(self):
                return {"device": self.device}
            def get_booster(self):
                if self.actual is None:
                    raise ValueError("not fitted")
                config = {"learner": {"generic_param": {"device": self.actual}}}
                return SimpleNamespace(save_config=lambda: json.dumps(config))
        model = FakeXGB()
        for configured, actual, expected in (("cpu", None, "sklearn"), ("cuda:0", None, "xgboost"),
                                              ("cuda", "cpu", "sklearn"), ("cuda", "cuda:0", "xgboost")):
            model.device, model.actual = configured, actual
            self.assertEqual(self.detect(MultiOutputRegressor(model)), expected)

    @unittest.skipUnless(forecasting.CUDA_AVAILABLE and forecasting._cuml_RF_reg is not None,
                         "cuML RandomForest and CUDA required")
    def test_real_gpu_fit_and_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(64, 3)).astype(np.float32)
        y = np.column_stack((X[:, 0] + X[:, 1], X[:, 0] - X[:, 2])).astype(np.float32)
        model = MultiOutputRegressor(forecasting.EnhancedBitcoinForecaster._gpu_rf_reg(
            n_estimators=4, max_depth=3, random_state=42, n_jobs=-1))
        model.fit(X, y)
        prediction = model.predict(X[:5])
        self.assertEqual(prediction.shape, (5, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(self.detect(model), "cuML")


if __name__ == "__main__":
    unittest.main()
