/**
 * Read-only access to the Python prediction ledger (data/predictions.sqlite3),
 * written by services/prediction_history.py. Node never writes to it.
 *
 * Accuracy aggregates mirror PredictionHistory.report() so the UI and the CLI agree.
 */

const fs = require('fs');
const { DatabaseSync } = require('node:sqlite');

const TIME_COLUMNS = ['target_start', 'target_end', 'trained_at', 'training_cutoff', 'scored_at', 'generated_at', 'data_cutoff'];

const iso = (seconds) => (seconds === null || seconds === undefined ? null : new Date(seconds * 1000).toISOString());
const toEpoch = (value) => (value === undefined ? undefined : Math.floor(new Date(value).getTime() / 1000));

const createLedger = (file) => {
  let db = null;

  /** Open lazily: the ledger only exists after the Python model has run once. */
  const connection = () => {
    if (db) return db;
    if (!fs.existsSync(file)) return null;
    db = new DatabaseSync(file, { readOnly: true });
    db.exec('PRAGMA busy_timeout = 10000');
    return db;
  };

  const buildWhere = (filters) => {
    const where = [];
    const params = {};
    const add = (clause, key, value) => {
      if (value === undefined || value === null || value === '') return;
      where.push(clause);
      params[key] = value;
    };
    add('p.kind = $kind', '$kind', filters.kind);
    add('p.signal = $signal', '$signal', filters.signal);
    add('p.horizon_minutes = $horizon', '$horizon', filters.horizon);
    add('p.status = $status', '$status', filters.status);
    add('p.model_version = $model', '$model', filters.modelVersion);
    if (filters.historical !== undefined) add('p.historical = $historical', '$historical', filters.historical ? 1 : 0);
    add('r.generated_at >= $from', '$from', toEpoch(filters.from));
    add('r.generated_at < $to', '$to', toEpoch(filters.to));
    return { sql: where.length ? `WHERE ${where.join(' AND ')}` : '', params };
  };

  const history = (filters = {}) => {
    const conn = connection();
    if (!conn) return { total: 0, rows: [] };
    const { sql, params } = buildWhere(filters);
    const from = 'FROM predictions p JOIN runs r ON r.id = p.run_id';
    const total = conn.prepare(`SELECT COUNT(*) AS n ${from} ${sql}`).get(params).n;
    const rows = conn.prepare(`SELECT p.*, r.generated_at, r.data_cutoff, r.source, r.market ${from} ${sql}
      ORDER BY r.generated_at DESC, p.id DESC LIMIT $limit OFFSET $offset`)
      .all({ ...params, $limit: Math.min(Number(filters.limit) || 100, 1000), $offset: Number(filters.offset) || 0 })
      .map((row) => {
        const out = { ...row, cached: Boolean(row.cached), historical: Boolean(row.historical) };
        TIME_COLUMNS.forEach((column) => { out[column] = iso(row[column]); });
        try { out.metadata = JSON.parse(row.metadata); } catch { /* keep raw */ }
        return out;
      });
    return { total, rows };
  };

  const accuracy = (filters = {}) => {
    const conn = connection();
    if (!conn) return { groups: [] };
    const { sql, params } = buildWhere(filters);
    const bySignal = filters.groupBy === 'signal';
    const keys = ['p.historical', 'p.kind', 'p.horizon_minutes', ...(bySignal ? ['p.signal'] : [])];
    const rows = conn.prepare(`SELECT ${keys.join(', ')},
        COUNT(*) AS total,
        SUM(p.status = 'pending') AS pending,
        SUM(p.status = 'unscorable') AS unscorable,
        SUM(p.status = 'scored') AS scored,
        AVG(CASE WHEN p.status = 'scored' THEN ABS(p.dollar_error) END) AS mae,
        AVG(CASE WHEN p.status = 'scored' THEN p.dollar_error * p.dollar_error END) AS mse,
        AVG(CASE WHEN p.status = 'scored' THEN p.abs_pct_error END) AS mape,
        AVG(CASE WHEN p.status = 'scored' THEN p.within_one_pct END) AS within_one_pct,
        AVG(CASE WHEN p.status = 'scored' THEN p.direction_correct END) AS direction_accuracy,
        COUNT(CASE WHEN p.status = 'scored' THEN p.direction_correct END) AS direction_n,
        AVG(CASE WHEN p.status = 'scored' THEN p.range_contained END) AS range_contained,
        AVG(CASE WHEN p.status = 'scored' THEN p.underlying_return_pct END) AS mean_return_pct
      FROM predictions p JOIN runs r ON r.id = p.run_id ${sql}
      GROUP BY ${keys.join(', ')} ORDER BY ${keys.join(', ')}`).all(params);
    const pct = (value) => (value === null ? null : value * 100);
    return {
      groups: rows.map((row) => ({
        historical: Boolean(row.historical),
        kind: row.kind,
        horizonMinutes: row.horizon_minutes,
        ...(bySignal ? { signal: row.signal } : {}),
        total: row.total,
        scored: row.scored,
        pending: row.pending,
        unscorable: row.unscorable,
        mae: row.mae,
        rmse: row.mse === null ? null : Math.sqrt(row.mse),
        mapePct: row.mape,
        withinOnePct: pct(row.within_one_pct),
        directionAccuracyPct: pct(row.direction_accuracy),
        directionN: row.direction_n,
        rangeContainedPct: pct(row.range_contained),
        meanReturnPct: ['trade', 'rule_signal'].includes(row.kind) ? row.mean_return_pct : null,
      })),
    };
  };

  const close = () => { if (db) { db.close(); db = null; } };

  return { history, accuracy, close, exists: () => fs.existsSync(file) };
};

module.exports = { createLedger };
