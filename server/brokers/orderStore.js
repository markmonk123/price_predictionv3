/**
 * SQLite order ledger shared by every broker (data/trading.sqlite3).
 * Uses Node's built-in node:sqlite, so there is no native dependency to build.
 */

const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');

const OPEN_STATUSES = ['PENDING', 'OPEN', 'QUEUED'];

const toOrder = (row) => row && ({
  id: row.id,
  broker: row.broker,
  clientOrderId: row.client_order_id,
  externalId: row.external_id,
  productId: row.product_id,
  side: row.side,
  type: row.order_type,
  baseSize: row.base_size,
  quoteSize: row.quote_size,
  limitPrice: row.limit_price,
  postOnly: Boolean(row.post_only),
  status: row.status,
  filledSize: row.filled_size,
  averageFilledPrice: row.average_filled_price,
  totalFees: row.total_fees,
  reason: row.reason,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});

const openDatabase = (file) => {
  if (file !== ':memory:') fs.mkdirSync(path.dirname(file), { recursive: true });
  const db = new DatabaseSync(file);
  db.exec('PRAGMA busy_timeout = 10000; PRAGMA journal_mode = WAL;');
  return db;
};

/** Run fn inside a transaction; roll back on any error. */
const transaction = (db, fn) => {
  db.exec('BEGIN IMMEDIATE');
  try {
    const result = fn();
    db.exec('COMMIT');
    return result;
  } catch (error) {
    db.exec('ROLLBACK');
    throw error;
  }
};

const createOrderStore = (file) => {
  const db = openDatabase(file);
  db.exec(`
    CREATE TABLE IF NOT EXISTS orders (
      id TEXT PRIMARY KEY,
      broker TEXT NOT NULL,
      client_order_id TEXT NOT NULL UNIQUE,
      external_id TEXT,
      product_id TEXT NOT NULL,
      side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
      order_type TEXT NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT')),
      base_size REAL,
      quote_size REAL,
      limit_price REAL,
      post_only INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL,
      filled_size REAL NOT NULL DEFAULT 0,
      average_filled_price REAL,
      total_fees REAL NOT NULL DEFAULT 0,
      reason TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS orders_by_broker_status ON orders(broker, status);
    CREATE INDEX IF NOT EXISTS orders_by_created ON orders(created_at);
  `);

  const insertStmt = db.prepare(`
    INSERT INTO orders (id, broker, client_order_id, external_id, product_id, side, order_type,
      base_size, quote_size, limit_price, post_only, status, filled_size, average_filled_price,
      total_fees, reason, created_at, updated_at)
    VALUES ($id, $broker, $clientOrderId, $externalId, $productId, $side, $type, $baseSize,
      $quoteSize, $limitPrice, $postOnly, $status, $filledSize, $averageFilledPrice, $totalFees,
      $reason, $createdAt, $updatedAt)`);
  const getStmt = db.prepare('SELECT * FROM orders WHERE id = ?');

  const insert = (order) => {
    const now = new Date().toISOString();
    const row = {
      externalId: null, baseSize: null, quoteSize: null, limitPrice: null, filledSize: 0,
      averageFilledPrice: null, totalFees: 0, reason: null, createdAt: now, updatedAt: now,
      ...order,
      postOnly: order.postOnly ? 1 : 0,
    };
    insertStmt.run(Object.fromEntries(Object.entries(row).map(([k, v]) => [`$${k}`, v ?? null])));
    return get(order.id);
  };

  const get = (id) => toOrder(getStmt.get(id));

  const update = (id, fields) => {
    const columns = {
      status: 'status', externalId: 'external_id', filledSize: 'filled_size',
      averageFilledPrice: 'average_filled_price', totalFees: 'total_fees', reason: 'reason',
    };
    const entries = Object.entries(fields).filter(([key]) => columns[key]);
    if (!entries.length) return get(id);
    const sets = entries.map(([key]) => `${columns[key]} = $${key}`).join(', ');
    db.prepare(`UPDATE orders SET ${sets}, updated_at = $updatedAt WHERE id = $id`).run({
      ...Object.fromEntries(entries.map(([k, v]) => [`$${k}`, v ?? null])),
      $updatedAt: new Date().toISOString(),
      $id: id,
    });
    return get(id);
  };

  const list = ({ broker, status, productId, limit = 100 } = {}) => {
    const where = [];
    const params = {};
    if (broker) { where.push('broker = $broker'); params.$broker = broker; }
    if (productId) { where.push('product_id = $productId'); params.$productId = productId; }
    if (status === 'open') {
      where.push(`status IN (${OPEN_STATUSES.map((s) => `'${s}'`).join(',')})`);
    } else if (status) {
      where.push('status = $status'); params.$status = status;
    }
    params.$limit = Math.min(Math.max(Number(limit) || 100, 1), 1000);
    const sql = `SELECT * FROM orders ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
      ORDER BY created_at DESC, rowid DESC LIMIT $limit`;
    return db.prepare(sql).all(params).map(toOrder);
  };

  return { db, insert, get, update, list, close: () => db.close() };
};

module.exports = { createOrderStore, openDatabase, transaction, OPEN_STATUSES };
