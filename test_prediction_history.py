"""Deterministic ledger tests: no network, ML dependencies, or real runtime DB."""
import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from services.prediction_history import PredictionHistory, epoch
from services.history_runtime import HistoryRuntime, model_identity


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'history.sqlite3'
        self.h = PredictionHistory(self.path)
        self.base = epoch('2026-10-01T00:00:00Z')
        self.run = self.h.start_run('test', self.base, 82000, self.base)

    def tearDown(self):
        self.h.close()
        self.temp.cleanup()

    def add(self, **overrides):
        values = dict(kind='price', signal='ensemble', target_start=self.base + 6*3600,
                      target_end=self.base + 6*3600 + 900, horizon_minutes=360,
                      reference_price=82000, predicted_price=82300, model_version='v1',
                      generated_at=self.base)
        values.update(overrides)
        self.h.record(self.run, **values)

    def candles(self, start=None, price=83000):
        start = start or self.base + 6*3600
        return [dict(date=start + i*300, price=price, high=price+10+i, low=price-10-i)
                for i in range(3)]

    def test_six_hour_prediction_scored_twelve_hours_later_once(self):
        self.add()
        self.assertEqual(self.h.reconcile(self.candles(), self.base+12*3600), 1)
        row = self.h.rows()[0]
        self.assertEqual(row['actual_price'], 83000)
        self.assertEqual(row['dollar_error'], -700)
        self.assertAlmostEqual(row['pct_error'], -700/83000*100)
        self.assertEqual(row['within_one_pct'], 1)
        self.assertEqual(row['direction_correct'], 1)
        self.assertEqual(self.h.reconcile(self.candles(), self.base+12*3600), 0)

    def test_missing_and_incomplete_candles_stay_pending(self):
        self.add()
        candles = self.candles()
        self.assertEqual(self.h.reconcile(candles, self.base+6*3600+899), 0)
        self.assertEqual(self.h.reconcile(candles[1:], self.base+12*3600), 0)
        # No nearest matching or forward filling from adjacent windows.
        shifted = [dict(r, date=r['date']+900) for r in candles]
        self.assertEqual(self.h.reconcile(shifted, self.base+12*3600), 0)
        self.assertEqual(self.h.reconcile(candles, self.base+12*3600), 1)

    def test_invalid_candles_not_scored(self):
        self.add()
        for bad in (float('nan'), -1, float('inf')):
            candles = self.candles()
            candles[1]['price'] = bad
            self.assertEqual(self.h.reconcile(candles, self.base+12*3600), 0)

    def test_one_percent_boundary_and_positive_error(self):
        self.add(predicted_price=83830)
        self.h.reconcile(self.candles(), self.base+12*3600)
        row = self.h.rows()[0]
        self.assertEqual(row['pct_error'], 1)
        self.assertEqual(row['within_one_pct'], 1)

    def test_direction_threshold_and_ranges(self):
        self.add(kind='direction', predicted_price=None, predicted_direction=0, threshold=.02)
        self.add(kind='high_low', predicted_price=None, predicted_high=83012, predicted_low=82988)
        self.h.reconcile(self.candles(), self.base+12*3600)
        direction, bounds = self.h.rows()
        self.assertEqual(direction['direction_correct'], 1)
        self.assertEqual(direction['actual_direction'], 0)
        self.assertIsNone(direction['pct_error'])
        self.assertEqual(bounds['high_error'], 0)
        self.assertEqual(bounds['low_error'], 0)
        self.assertEqual(bounds['range_contained'], 1)

    def test_forty_eight_windows_idempotency_overlapping_runs_and_restart(self):
        for step in range(1, 49):
            self.add(target_start=self.base+step*900, target_end=self.base+(step+1)*900,
                     horizon_minutes=step*15)
        self.add(target_start=self.base+900, target_end=self.base+1800, horizon_minutes=15)
        self.assertEqual(len(self.h.rows()), 48)
        self.run = self.h.start_run('test', self.base, 82000, self.base, run_id='second')
        self.add(target_start=self.base+900, target_end=self.base+1800, horizon_minutes=15)
        self.assertEqual(len(self.h.rows()), 49)
        self.h.close()
        self.h = PredictionHistory(self.path)
        self.assertEqual(len(self.h.rows()), 49)

    def test_live_historical_filters_and_csv(self):
        self.add()
        self.add(signal='old', generated_at=self.base+12*3600)
        self.h.reconcile(self.candles(), self.base+12*3600)
        self.assertEqual(len(self.h.rows(historical=0)), 1)
        self.assertEqual(len(self.h.rows(historical=1)), 1)
        self.assertEqual(len(self.h.rows(kind='direction')), 0)
        self.assertEqual(len(self.h.rows(horizon=360, model_version='v1',
                                         start='2026-10-01', end='2026-10-02')), 2)
        output = Path(self.temp.name) / 'out.csv'
        self.assertEqual(self.h.export(output, historical=0), 1)
        with output.open(newline='') as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row['target_start'], '2026-10-01T06:00:00+00:00')
        self.assertEqual(row['actual_price'], '83000.0')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.h.report(historical=0)
        self.assertIn('LIVE price +360m n=1', out.getvalue())
        self.assertNotIn('HISTORICAL', out.getvalue())

    def test_utc_dst(self):
        self.assertEqual(epoch('2026-11-01T01:30:00-04:00')+3600,
                         epoch('2026-11-01T01:30:00-05:00'))

    def test_direction_threshold_boundary(self):
        self.add(kind='direction', predicted_price=None, predicted_direction=1,
                 reference_price=100, threshold=.001)
        self.h.reconcile(self.candles(price=100.1), self.base+12*3600)
        self.assertEqual(self.h.rows()[0]['direction_correct'], 1)

    def test_rule_outputs_and_nonfatal_write_failure(self):
        runtime = HistoryRuntime.__new__(HistoryRuntime)
        runtime.history, runtime.run_id = self.h, self.run
        runtime.rule('long', dict(date=self.base, price=82000), 1,
                     dict(next_high=82500, next_low=81500), 15)
        rows = self.h.rows()
        self.assertEqual({r['kind'] for r in rows}, {'rule_signal', 'high_low'})
        bounds = next(r for r in rows if r['kind'] == 'high_low')
        self.assertEqual(bounds['target_start'], self.base+300)
        self.assertEqual(bounds['target_end'], self.base+1200)
        with patch.object(self.h, 'record', side_effect=OSError('disk full')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                runtime.prediction(kind='price', signal='failed', target_start=self.base,
                                   target_end=self.base+900, horizon_minutes=15, reference_price=82000)
        self.assertIn('disk full', output.getvalue())
        self.assertEqual(len(self.h.rows()), 2)

    def test_runtime_backfills_missing_old_ranges(self):
        self.add()
        class Frame:
            def to_dict(self, orient):
                return []
        runtime = HistoryRuntime.__new__(HistoryRuntime)
        runtime.history = self.h
        runtime.now = self.base+12*3600
        runtime.fetcher = lambda start, end: self.candles(start)
        with contextlib.redirect_stdout(io.StringIO()):
            runtime.reconcile(Frame())
        self.assertEqual(self.h.rows()[0]['status'], 'scored')

    def test_runtime_database_failure_is_nonfatal(self):
        with patch.dict(os.environ, {'PREDICTION_DB_PATH': self.temp.name}):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                runtime = HistoryRuntime('test', None)
                runtime.prediction(kind='price')
        self.assertIsNone(runtime.history)
        self.assertIn('initialization failed', out.getvalue())

    def test_model_identity_survives_cached_load(self):
        fresh = model_identity('classifier', self.base, self.base-900)
        cached = dict(fresh, cached=True)
        self.assertEqual(fresh['model_version'], cached['model_version'])
        self.assertEqual(cached['training_cutoff'], self.base-900)

    def test_report_cli_exports_filtered_csv(self):
        self.add()
        self.add(signal='historical', historical=True)
        self.h.reconcile(self.candles(), self.base+12*3600)
        output = Path(self.temp.name) / 'cli.csv'
        command = [sys.executable, '-B', '-m', 'services.prediction_history', '--db', str(self.path),
                   '--kind', 'price', '--horizon', '360', '--model-version', 'v1',
                   '--start', '2026-10-01', '--end', '2026-10-02', '--csv', str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=True,
                                cwd=Path(__file__).resolve().parent)
        self.assertIn('LIVE price +360m n=1', result.stdout)
        self.assertNotIn('HISTORICAL', result.stdout)
        with output.open(newline='') as f:
            self.assertEqual(len(list(csv.DictReader(f))), 1)


if __name__ == '__main__':
    unittest.main()
