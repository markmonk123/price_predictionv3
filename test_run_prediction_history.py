"""The scheduled runner (python/run_prediction.py) must record to and score from the ledger."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'python'))

import run_prediction as runner  # noqa: E402
from config.prediction_spec import HORIZON_MINUTES  # noqa: E402
from services.prediction_history import PredictionHistory  # noqa: E402

BASE = 1_790_000_100 - 1_790_000_100 % 300  # a five-minute boundary


def candles(start, count, close=100.0):
    return pd.DataFrame({
        'date': [datetime.fromtimestamp(start + i * 300, timezone.utc).replace(tzinfo=None)
                 for i in range(count)],
        'low': [close - 1] * count,
        'high': [close + 1] * count,
        'price': [close] * count,
    })


def no_fetch(start, end):
    raise AssertionError('test should not need network backfill')


class RunPredictionHistoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp.name, 'predictions.sqlite3')
        env = patch.dict(os.environ, {'PREDICTION_DB_PATH': self.db})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.temp.cleanup)

    def record(self, generated, frame, **overrides):
        from services.history_runtime import HistoryRuntime
        args = dict(candles_df=frame, generated_at=generated, reference_price=100.0,
                    direction=1, confidence=0.6, training_cutoff=frame['date'].iloc[-1],
                    runtime_factory=lambda *a, **k: HistoryRuntime(*a, fetcher=no_fetch, **k))
        args.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.record_prediction(**args)

    def rows(self):
        ledger = PredictionHistory(self.db)
        try:
            return ledger.rows()
        finally:
            ledger.close()

    def test_mid_candle_prediction_is_recorded_on_aligned_window(self):
        generated = BASE + 130  # 2m10s into a forming candle
        status = self.record(generated, candles(BASE - 300 * 50, 51))
        self.assertTrue(status['recorded'], status)
        [row] = self.rows()
        self.assertEqual(row['kind'], 'direction')
        self.assertEqual(row['signal'], runner.SIGNAL)
        self.assertEqual(row['model_version'], runner.MODEL_VERSION)
        self.assertEqual(row['target_end'], BASE + 300 + HORIZON_MINUTES * 60)
        self.assertEqual(row['target_start'], row['target_end'] - 300)
        self.assertEqual(row['status'], 'pending')
        self.assertEqual(row['historical'], 0)
        self.assertEqual(row['data_cutoff'], generated)

    def test_later_run_scores_earlier_prediction(self):
        self.record(BASE + 130, candles(BASE - 300 * 50, 51))
        target_end = BASE + 300 + HORIZON_MINUTES * 60
        later = target_end + 400
        # Price rose 1% by the target close: an UP call (+1) is correct.
        frame = candles(BASE - 300 * 50, (later - (BASE - 300 * 50)) // 300 + 1, close=101.0)
        status = self.record(later, frame)
        self.assertTrue(status['recorded'], status)
        self.assertEqual(status['newly_scored'], 1)
        first = min(self.rows(), key=lambda r: r['target_end'])
        self.assertEqual(first['status'], 'scored')
        self.assertEqual(first['actual_direction'], 1)
        self.assertEqual(first['direction_correct'], 1)

    def test_forming_candle_close_is_never_used_for_scoring(self):
        self.record(BASE + 130, candles(BASE - 300 * 50, 51))
        target_end = BASE + 300 + HORIZON_MINUTES * 60
        # Target bar is still forming at `now`, so the prediction must stay pending.
        frame = candles(BASE - 300 * 50, (target_end - 300 - (BASE - 300 * 50)) // 300 + 1, close=101.0)
        self.record(target_end - 60, frame)
        first = min(self.rows(), key=lambda r: r['target_end'])
        self.assertEqual(first['status'], 'pending')

    def test_disabled_and_failures_are_reported_not_raised(self):
        with patch.dict(os.environ, {'PREDICTION_HISTORY': 'off'}):
            self.assertFalse(self.record(BASE + 130, candles(BASE - 3000, 11))['recorded'])
        status = self.record(BASE + 130, candles(BASE - 3000, 11), direction=7)
        self.assertFalse(status['recorded'])
        self.assertIn('Direction', status['reason'])
        self.assertEqual(self.rows(), [])


if __name__ == '__main__':
    unittest.main()
