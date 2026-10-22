"""Durable prediction ledger and exact, completed-candle evaluation (stdlib only)."""
from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def epoch(value):
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class PredictionHistory:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("PREDICTION_DB_PATH") or
                         Path(__file__).resolve().parents[1] / "data" / "predictions.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, generated_at INTEGER NOT NULL,
                data_cutoff INTEGER NOT NULL, source TEXT NOT NULL,
                reference_price REAL NOT NULL, market TEXT NOT NULL DEFAULT 'BTC-USD'
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                kind TEXT NOT NULL, signal TEXT NOT NULL, target_start INTEGER NOT NULL,
                target_end INTEGER NOT NULL, horizon_minutes INTEGER NOT NULL,
                reference_price REAL NOT NULL, predicted_price REAL, predicted_direction INTEGER,
                predicted_high REAL, predicted_low REAL, confidence REAL, threshold REAL NOT NULL,
                model_version TEXT NOT NULL, trained_at INTEGER, training_cutoff INTEGER,
                cached INTEGER NOT NULL, historical INTEGER NOT NULL, metadata TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', actual_price REAL, actual_high REAL,
                actual_low REAL, dollar_error REAL, pct_error REAL, abs_pct_error REAL,
                within_one_pct INTEGER, actual_direction INTEGER, direction_correct INTEGER,
                underlying_return_pct REAL, high_error REAL, low_error REAL,
                range_contained INTEGER, scored_at INTEGER,
                UNIQUE(run_id, kind, signal, target_start, target_end)
            );
            CREATE INDEX IF NOT EXISTS pending_targets ON predictions(status, target_start);
            PRAGMA user_version=1;
        """)

    def close(self):
        self.db.close()

    def start_run(self, source, data_cutoff, reference_price, generated_at=None, run_id=None):
        run_id = run_id or str(uuid.uuid4())
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?)",
                            (run_id, epoch(generated_at or datetime.now(timezone.utc)),
                             epoch(data_cutoff), str(source), float(reference_price), "BTC-USD"))
        return run_id

    def record(self, run_id, *, kind, signal, target_start, target_end, horizon_minutes,
               reference_price, model_version, predicted_price=None, predicted_direction=None,
               predicted_high=None, predicted_low=None, confidence=None, threshold=0,
               trained_at=None, training_cutoff=None, cached=False, historical=False,
               metadata=None, generated_at=None):
        start, end = epoch(target_start), epoch(target_end)
        if end <= start or start % 300 or end % 300:
            raise ValueError("Prediction windows must be positive and aligned to five-minute candles")
        if horizon_minutes <= 0 or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Horizon must be positive and threshold finite and nonnegative")
        if predicted_direction not in (None, -1, 0, 1):
            raise ValueError("Direction must be -1, 0, or 1")
        prices = [reference_price, predicted_price, predicted_high, predicted_low]
        if any(v is not None and (not math.isfinite(float(v)) or v <= 0) for v in prices):
            raise ValueError("Prediction prices must be positive and finite")
        # Capture actual generation time, including long fresh-training runs.
        generated = epoch(generated_at or datetime.now(timezone.utc))
        values = dict(run_id=run_id, kind=kind, signal=signal, target_start=start,
                      target_end=end, horizon_minutes=int(horizon_minutes),
                      reference_price=float(reference_price), predicted_price=predicted_price,
                      predicted_direction=predicted_direction, predicted_high=predicted_high,
                      predicted_low=predicted_low, confidence=confidence, threshold=float(threshold),
                      model_version=str(model_version), trained_at=epoch(trained_at) if trained_at is not None else None,
                      training_cutoff=epoch(training_cutoff) if training_cutoff is not None else None,
                      cached=int(cached), historical=int(historical or end <= generated),
                      metadata=json.dumps(dict(metadata or {}, generated_at=iso(generated)), default=str))
        with self.db:
            self.db.execute(f"INSERT OR IGNORE INTO predictions ({','.join(values)}) "
                            f"VALUES ({','.join('?' for _ in values)})", tuple(values.values()))

    def pending_windows(self, now=None):
        return self.db.execute("SELECT DISTINCT target_start,target_end FROM predictions "
                               "WHERE status='pending' AND target_end<=? ORDER BY target_start",
                               (epoch(now or datetime.now(timezone.utc)),)).fetchall()

    def reconcile(self, candles, now=None):
        """Candles: timestamp/open-time, price/close, high, low; exact five-minute bars only."""
        now = epoch(now or datetime.now(timezone.utc))
        bars = {}
        for row in candles:
            t = epoch(row['date'])
            close, high, low = (float(row[k]) for k in ('price', 'high', 'low'))
            if (t % 300 == 0 and t + 300 <= now and
                    all(math.isfinite(v) and v > 0 for v in (close, high, low))
                    and low <= close <= high):
                bars[t] = (close, high, low)
        scored = 0
        with self.db:
            rows = self.db.execute("SELECT * FROM predictions WHERE status='pending' AND target_end<=?", (now,)).fetchall()
            for p in rows:
                times = range(p['target_start'], p['target_end'], 300)
                if any(t not in bars for t in times):
                    continue
                actual = bars[p['target_end'] - 300][0]
                high = max(bars[t][1] for t in times)
                low = min(bars[t][2] for t in times)
                change = (actual - p['reference_price']) / p['reference_price']
                ret = change * 100
                meets_threshold = abs(change) >= p['threshold'] or math.isclose(
                    abs(change), p['threshold'], rel_tol=1e-12, abs_tol=1e-15)
                direction = ((change > 0) - (change < 0)) if meets_threshold else 0
                predicted_direction = p['predicted_direction']
                if predicted_direction is None and p['predicted_price'] is not None:
                    delta = p['predicted_price'] - p['reference_price']
                    predicted_direction = (delta > 0) - (delta < 0)
                error = None if p['predicted_price'] is None else p['predicted_price'] - actual
                pct = None if error is None else error / actual * 100
                high_error = None if p['predicted_high'] is None else p['predicted_high'] - high
                low_error = None if p['predicted_low'] is None else p['predicted_low'] - low
                contained = None if high_error is None or low_error is None else int(high_error >= 0 and low_error <= 0)
                self.db.execute("""UPDATE predictions SET status='scored', actual_price=?,actual_high=?,
                    actual_low=?,dollar_error=?,pct_error=?,abs_pct_error=?,within_one_pct=?,
                    actual_direction=?,direction_correct=?,underlying_return_pct=?,high_error=?,
                    low_error=?,range_contained=?,scored_at=? WHERE id=? AND status='pending'""",
                    (actual, high, low, error, pct, None if pct is None else abs(pct),
                     None if pct is None else int(abs(pct) <= 1 + 1e-12), direction,
                     None if predicted_direction is None else int(direction == predicted_direction),
                     ret, high_error, low_error, contained, now, p['id']))
                scored += 1
        return scored

    def rows(self, *, start=None, end=None, kind=None, horizon=None, model_version=None, historical=None):
        where, args = [], []
        for expression, value in (("r.generated_at>=?", epoch(start) if start else None),
                                  ("r.generated_at<?", epoch(end) if end else None),
                                  ("p.kind=?", kind), ("p.horizon_minutes=?", horizon),
                                  ("p.model_version=?", model_version), ("p.historical=?", historical)):
            if value is not None:
                where.append(expression)
                args.append(value)
        return [dict(row) for row in self.db.execute(
            "SELECT p.*,r.generated_at,r.data_cutoff,r.source,r.market FROM predictions p "
            "JOIN runs r ON r.id=p.run_id" + (" WHERE " + " AND ".join(where) if where else "") +
            " ORDER BY r.generated_at,p.id", args)]

    def report(self, **filters):
        rows = self.rows(**filters)
        print(f"Prediction history: {len(rows)} predictions, "
              f"{sum(r['status'] == 'pending' for r in rows)} pending")
        groups = {}
        for row in rows:
            if row['status'] == 'scored':
                groups.setdefault((row['historical'], row['kind'], row['horizon_minutes']), []).append(row)
        for (historical, kind, horizon), group in sorted(groups.items()):
            errors = [r['dollar_error'] for r in group if r['dollar_error'] is not None]
            directions = [r['direction_correct'] for r in group if r['direction_correct'] is not None]
            text = f"  {'HISTORICAL' if historical else 'LIVE'} {kind} +{horizon}m n={len(group)}"
            if errors:
                priced = [r for r in group if r['dollar_error'] is not None]
                text += (f" MAE=${sum(abs(e) for e in errors)/len(errors):.2f}"
                         f" RMSE=${math.sqrt(sum(e*e for e in errors)/len(errors)):.2f}"
                         f" MAPE={sum(r['abs_pct_error'] for r in priced)/len(priced):.3f}%"
                         f" within 1%={100*sum(r['within_one_pct'] for r in priced)/len(priced):.1f}%")
            if directions:
                text += f" direction={100*sum(directions)/len(directions):.1f}% (n={len(directions)})"
            ranges = [r for r in group if r['range_contained'] is not None]
            if ranges:
                text += (f" range contained={100*sum(r['range_contained'] for r in ranges)/len(ranges):.1f}%"
                         f" high MAE=${sum(abs(r['high_error']) for r in ranges)/len(ranges):.2f}"
                         f" low MAE=${sum(abs(r['low_error']) for r in ranges)/len(ranges):.2f}")
            if kind in ('trade', 'rule_signal'):
                text += f" mean underlying return={sum(r['underlying_return_pct'] for r in group)/len(group):+.3f}%"
            print(text)

    def export(self, path, **filters):
        rows = self.rows(**filters)
        fields = [d[0] for d in self.db.execute("SELECT p.*,r.generated_at,r.data_cutoff,r.source,r.market "
                  "FROM predictions p JOIN runs r ON r.id=p.run_id LIMIT 0").description]
        with open(path, 'w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                for key in ('target_start', 'target_end', 'trained_at', 'training_cutoff', 'scored_at', 'generated_at', 'data_cutoff'):
                    if row[key] is not None:
                        row[key] = iso(row[key])
                writer.writerow(row)
        return len(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Report/export stored BTC-USD predictions (dates in UTC).')
    parser.add_argument('--db')
    parser.add_argument('--start', help='Inclusive run date/time, ISO 8601')
    parser.add_argument('--end', help='Exclusive run date/time, ISO 8601')
    parser.add_argument('--kind')
    parser.add_argument('--horizon', type=int, help='Forecast horizon in minutes')
    parser.add_argument('--model-version')
    parser.add_argument('--history', choices=('live', 'historical', 'all'), default='live')
    parser.add_argument('--csv', help='CSV output path')
    args = parser.parse_args()
    history = PredictionHistory(args.db)
    try:
        filters = dict(start=args.start, end=args.end, kind=args.kind, horizon=args.horizon,
                       model_version=args.model_version,
                       historical={'live': 0, 'historical': 1, 'all': None}[args.history])
        history.report(**filters)
        if args.csv:
            print(f"Exported {history.export(args.csv, **filters)} predictions to {args.csv}")
    finally:
        history.close()


if __name__ == '__main__':
    main()
