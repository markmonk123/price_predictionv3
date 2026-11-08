const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const { tempDir, removeDir } = require('./helpers');
const { createLedger } = require('../predictions/ledger');

// Same schema as services/prediction_history.py (PRAGMA user_version=1).
const SCHEMA = `
  CREATE TABLE runs (
    id TEXT PRIMARY KEY, generated_at INTEGER NOT NULL, data_cutoff INTEGER NOT NULL,
    source TEXT NOT NULL, reference_price REAL NOT NULL, market TEXT NOT NULL DEFAULT 'BTC-USD');
  CREATE TABLE predictions (
    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), kind TEXT NOT NULL, signal TEXT NOT NULL,
    target_start INTEGER NOT NULL, target_end INTEGER NOT NULL, horizon_minutes INTEGER NOT NULL,
    reference_price REAL NOT NULL, predicted_price REAL, predicted_direction INTEGER, predicted_high REAL,
    predicted_low REAL, confidence REAL, threshold REAL NOT NULL, model_version TEXT NOT NULL, trained_at INTEGER,
    training_cutoff INTEGER, cached INTEGER NOT NULL, historical INTEGER NOT NULL, metadata TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', actual_price REAL, actual_high REAL, actual_low REAL, dollar_error REAL,
    pct_error REAL, abs_pct_error REAL, within_one_pct INTEGER, actual_direction INTEGER, direction_correct INTEGER,
    underlying_return_pct REAL, high_error REAL, low_error REAL, range_contained INTEGER, scored_at INTEGER,
    UNIQUE(run_id, kind, signal, target_start, target_end));`;

const BASE = Date.parse('2026-10-01T00:00:00Z') / 1000;

const fixture = (t) => {
  const dir = tempDir();
  const file = path.join(dir, 'predictions.sqlite3');
  const db = new DatabaseSync(file);
  db.exec(SCHEMA);
  db.prepare('INSERT INTO runs VALUES (?,?,?,?,?,?)').run('run-1', BASE, BASE, 'test', 100, 'BTC-USD');
  db.prepare('INSERT INTO runs VALUES (?,?,?,?,?,?)').run('run-2', BASE + 86400, BASE, 'test', 100, 'BTC-USD');
  const insert = db.prepare(`INSERT INTO predictions (run_id, kind, signal, target_start, target_end, horizon_minutes,
    reference_price, predicted_price, predicted_direction, threshold, model_version, cached, historical, metadata,
    status, dollar_error, abs_pct_error, within_one_pct, direction_correct, underlying_return_pct)
    VALUES ($run, $kind, $signal, $start, $start + 300, $horizon, 100, $pp, $dir, 0.001, 'v1', 0, $hist, '{"a":1}',
    $status, $err, $ape, $w1, $dc, $ret)`);
  const row = (overrides) => insert.run({
    $run: 'run-1', $kind: 'price', $signal: 'forecast', $start: BASE, $horizon: 15, $pp: null, $dir: null,
    $hist: 0, $status: 'scored', $err: null, $ape: null, $w1: null, $dc: null, $ret: null, ...overrides,
  });
  row({ $pp: 103, $err: 3, $ape: 3, $w1: 0, $dc: 1, $ret: 1 });
  row({ $pp: 99, $err: -4, $ape: 4, $w1: 0, $dc: 0, $ret: -1, $start: BASE + 300 });
  row({ $status: 'pending', $start: BASE + 600 });
  row({ $kind: 'direction', $signal: 'classifier', $dir: 1, $dc: 1, $ret: 0.2, $horizon: 15 });
  row({ $kind: 'direction', $signal: 'heldout_classifier', $dir: -1, $dc: 0, $hist: 1, $run: 'run-2' });
  row({ $kind: 'trade', $signal: 'best', $dir: 1, $dc: 1, $ret: 0.5, $status: 'unscorable', $start: BASE + 900 });
  db.close();
  const ledger = createLedger(file);
  t.after(() => { ledger.close(); removeDir(dir); });
  return ledger;
};

test('a missing ledger file yields empty results instead of creating a DB', (t) => {
  const dir = tempDir();
  t.after(() => removeDir(dir));
  const ledger = createLedger(path.join(dir, 'none.sqlite3'));
  assert.deepEqual(ledger.history(), { total: 0, rows: [] });
  assert.deepEqual(ledger.accuracy(), { groups: [] });
  assert.equal(ledger.exists(), false);
});

test('history filters, paginates and converts times to ISO', (t) => {
  const ledger = fixture(t);
  const all = ledger.history();
  assert.equal(all.total, 6);
  assert.equal(all.rows[0].run_id, 'run-2'); // newest run first
  assert.equal(all.rows[0].target_start, '2026-10-01T00:00:00.000Z');
  assert.equal(all.rows[0].historical, true);
  assert.deepEqual(all.rows[0].metadata, { a: 1 });

  assert.equal(ledger.history({ kind: 'price' }).total, 3);
  assert.equal(ledger.history({ status: 'pending' }).total, 1);
  assert.equal(ledger.history({ historical: false }).total, 5);
  assert.equal(ledger.history({ from: '2026-10-01T12:00:00Z' }).total, 1);
  const page = ledger.history({ limit: 2, offset: 2 });
  assert.equal(page.rows.length, 2);
  assert.equal(page.total, 6);
});

test('accuracy matches PredictionHistory.report() math', (t) => {
  const ledger = fixture(t);
  const groups = ledger.accuracy({ historical: false }).groups;
  const price = groups.find((g) => g.kind === 'price');
  assert.equal(price.total, 3);
  assert.equal(price.scored, 2);
  assert.equal(price.pending, 1);
  assert.equal(price.mae, 3.5);
  assert.equal(price.rmse, Math.sqrt((9 + 16) / 2));
  assert.equal(price.mapePct, 3.5);
  assert.equal(price.withinOnePct, 0);
  assert.equal(price.directionAccuracyPct, 50);
  assert.equal(price.meanReturnPct, null); // only trade/rule_signal report returns

  const trade = groups.find((g) => g.kind === 'trade');
  assert.equal(trade.unscorable, 1);
  assert.equal(trade.scored, 0);
  assert.equal(trade.meanReturnPct, null); // unscorable rows are excluded

  const bySignal = ledger.accuracy({ kind: 'direction', groupBy: 'signal' }).groups;
  assert.deepEqual(bySignal.map((g) => [g.historical, g.signal, g.directionAccuracyPct]),
    [[false, 'classifier', 100], [true, 'heldout_classifier', 0]]);
});
