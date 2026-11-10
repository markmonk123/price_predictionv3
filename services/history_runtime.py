"""Best-effort adapters between model DataFrames and the durable history ledger."""
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from services.prediction_history import PredictionHistory, epoch, iso

# Windows still missing candles this long after they close are marked unscorable.
PENDING_GRACE_SECONDS = 24 * 3600

# Coinbase rejects urllib's default "Python-urllib/x.y" User-Agent with HTTP 403.
USER_AGENT = 'priceprediction-history/1.0'


def best_effort(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.history is None:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            print(f"Prediction history {method.__name__} failed: {exc}")
            return None
    return wrapped


def fetch_actual_candles(start, end):
    """Fetch exact historical five-minute Coinbase candles; caller handles gaps."""
    query = urlencode(dict(granularity=300, start=iso(start), end=iso(end)))
    request = Request('https://api.exchange.coinbase.com/products/BTC-USD/candles?' + query,
                      headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    with urlopen(request, timeout=15) as response:
        rows = json.load(response)
    return [dict(date=int(r[0]), low=float(r[1]), high=float(r[2]), price=float(r[4])) for r in rows]


class HistoryRuntime:
    def __init__(self, source, raw_df, now=None, fetcher=fetch_actual_candles,
                 reference_price=None, data_cutoff=None):
        """raw_df: completed-or-forming candles (date=open time, price=close, high, low).

        By default the run's reference is the last row's close at its candle close.
        Live runners that predict from a ticker price mid-candle pass
        reference_price/data_cutoff explicitly instead.
        """
        self.history = None
        self.run_id = None
        self.now = epoch(now or datetime.now(timezone.utc))
        self.source = source
        self.raw_df = raw_df
        self.fetcher = fetcher
        self.last_scored = 0
        self.last_expired = 0
        try:
            self.history = PredictionHistory()
            self.reconcile(raw_df)
            latest = raw_df.iloc[-1]
            self.run_id = self.history.start_run(
                source,
                epoch(data_cutoff) if data_cutoff is not None else epoch(latest['date']) + 300,
                float(reference_price if reference_price is not None else latest['price']),
                self.now)
        except Exception as exc:
            print(f"Prediction history initialization failed: {exc}")
            if self.history is not None:
                self.history.close()
            self.history = None

    @best_effort
    def reconcile(self, raw_df):
        records = raw_df.to_dict('records')
        scored = self.history.reconcile(records, self.now)
        available = {epoch(r['date']) for r in records}
        # Merge only missing pending ranges, in API-sized pages (max 300 bars).
        missing = set()
        for window in self.history.pending_windows(self.now):
            missing.update(t for t in range(window['target_start'], window['target_end'], 300)
                           if t not in available)
        ordered = sorted(missing)
        ranges = []
        for t in ordered:
            if ranges and ranges[-1][1] == t and t - ranges[-1][0] < 300 * 299:
                ranges[-1][1] = t + 300
            else:
                ranges.append([t, t + 300])
        fetch_failed = False
        for index, (start, end) in enumerate(ranges):
            try:
                if index and self.fetcher is fetch_actual_candles:
                    time.sleep(0.35)
                records.extend(self.fetcher(start, end))
            except Exception as exc:
                fetch_failed = True
                print(f"Prediction history actual-data fetch failed ({iso(start)}): {exc}")
        if ranges:
            scored += self.history.reconcile(records, self.now)
        # Only give up on old windows once their candles were actually requested.
        expired = 0 if fetch_failed else self.history.expire(self.now - PENDING_GRACE_SECONDS, self.now)
        self.last_scored, self.last_expired = scored, expired
        print(f"Prediction history: {scored} newly scored"
              + (f", {expired} expired as unscorable" if expired else ""))
        self.history.report()

    @best_effort
    def prediction(self, *, kind, signal, target_start, target_end, horizon_minutes,
                   reference_price, model=None, **values):
        model = model or dict(model_version='rules-v1', cached=False)
        self.history.record(self.run_id, kind=kind, signal=signal, target_start=target_start,
                            target_end=target_end, horizon_minutes=horizon_minutes,
                            reference_price=float(reference_price), **model, **values)

    @best_effort
    def predictions(self, items):
        """Batch form of prediction(): every item is written in one transaction."""
        rows = []
        for item in items:
            item = dict(item)
            model = item.pop('model', None) or dict(model_version='rules-v1', cached=False)
            rows.append(dict(item, reference_price=float(item['reference_price']), **model))
        self.history.record_many(self.run_id, rows)

    def classifier(self, model, features, row, horizon_minutes, threshold, identity,
                   signal='classifier'):
        """Use actual estimator classes, including classifiers trained without all 3 classes."""
        proba = model.predict_proba(features)[0]
        index = int(proba.argmax())
        direction = int(model.classes_[index])
        confidence = float(proba[index])
        # Input date is a candle OPEN; reference price is known at its CLOSE.
        target_end = epoch(row['date']) + 300 + int(horizon_minutes) * 60
        self.prediction(kind='direction', signal=signal, target_start=target_end - 300,
                        target_end=target_end, horizon_minutes=horizon_minutes,
                        reference_price=row['price'], predicted_direction=direction,
                        confidence=confidence, threshold=threshold, model=identity)
        return direction, confidence

    def rule(self, name, row, direction, window, horizon_minutes):
        base = epoch(row['date']) + 300
        self.prediction(kind='rule_signal', signal=name, target_start=base + horizon_minutes * 60 - 300,
                        target_end=base + horizon_minutes * 60, horizon_minutes=horizon_minutes,
                        reference_price=row['price'], predicted_direction=direction)
        self.prediction(kind='high_low', signal=name, target_start=base,
                        target_end=base + 900, horizon_minutes=15,
                        reference_price=row['price'], predicted_high=float(window['next_high']),
                        predicted_low=float(window['next_low']))

    @best_effort
    def forecast(self, forecast_df, forecaster, reference_price):
        model = forecaster.history_identity
        for row in forecast_df.to_dict('records'):
            start = epoch(row['timestamp'])
            self.prediction(kind='price', signal='forecast_ensemble', target_start=start,
                            target_end=start + 900, horizon_minutes=int(row['interval_minutes']),
                            reference_price=reference_price, predicted_price=float(row['predicted_price']),
                            model=model, metadata={k: row[k] for k in ('prediction_std', 'baseline_adjusted',
                                                                      'baseline_gap_pct', 'baseline_scale')})

    @best_effort
    def finish(self):
        # Use the run's clock, not the wall clock: raw_df was fetched at self.now,
        # so a bar still forming then holds partial values even if it has closed since.
        self.history.reconcile(self.raw_df.to_dict('records'), self.now)
        self.history.report()
        self.history.close()
        self.history = None


def model_identity(family, trained_at, training_cutoff=None, cached=False, version=None):
    trained = epoch(trained_at)
    cutoff = epoch(training_cutoff) if training_cutoff is not None else None
    key = f'{family}:{trained}:{cutoff}'
    return dict(model_version=version or hashlib.sha256(key.encode()).hexdigest()[:20],
                trained_at=trained, training_cutoff=cutoff, cached=cached)
