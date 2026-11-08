/**
 * Test helpers: temp databases, a deterministic fake market, and a running app.
 */

const fs = require('fs');
const os = require('os');
const path = require('path');
const { EventEmitter } = require('events');
const { loadConfig } = require('../config');
const { createApp } = require('../app');
const { createOrderStore } = require('../brokers/orderStore');
const { createPaperBroker } = require('../brokers/paperBroker');
const { createLedger } = require('../predictions/ledger');

const tempDir = () => fs.mkdtempSync(path.join(os.tmpdir(), 'ppv3-test-'));

const removeDir = (dir) => {
  try { fs.rmSync(dir, { recursive: true, force: true }); } catch { /* Windows may hold WAL files briefly */ }
};

const fakeMarket = (initial = {}) => {
  let ticker = {
    productId: 'BTC-USD', price: 100, bid: 99, ask: 101, mid: 100, spread: 2,
    change24hPct: 0, volume24h: 10, time: '2026-11-08T00:00:00Z', source: 'test', ...initial,
  };
  return {
    offline: false,
    setTicker: (next) => { ticker = { ...ticker, ...next }; return ticker; },
    getTicker: async () => ticker,
    getCandles: async () => [{ time: 1794157200, open: 99, high: 102, low: 98, close: 100, volume: 5 }],
    getBook: async () => ({ productId: 'BTC-USD', bids: [{ price: 99, size: 1 }], asks: [{ price: 101, size: 1 }] }),
  };
};

const fakePredictions = (latest = null) => ({
  run: async () => latest,
  getLatest: () => latest,
  status: () => ({ running: false, lastRunAt: latest?.generatedAt ?? null, lastError: null }),
});

/** Start the full API on an ephemeral port. Call `close()` when done. */
const startApp = async ({ env = {}, market = fakeMarket(), predictions = fakePredictions(), brokers: extraBrokers = {} } = {}) => {
  const dir = tempDir();
  const config = loadConfig({
    TRADING_DB_PATH: path.join(dir, 'trading.sqlite3'),
    PREDICTION_DB_PATH: path.join(dir, 'predictions.sqlite3'),
    PAPER_STARTING_USD: '1000',
    PAPER_FEE_BPS: '100',
    ...env,
  });
  const store = createOrderStore(config.tradingDbPath);
  const brokers = { paper: createPaperBroker({ store, market, ...config.paper }), ...extraBrokers };
  const ledger = createLedger(config.predictionDbPath);
  const events = new EventEmitter();
  const app = createApp({ config, market, brokers, store, ledger, predictions, events });
  const server = await new Promise((resolve) => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  const base = `http://127.0.0.1:${server.address().port}/api/v1`;

  const request = async (method, url, { body, token } = {}) => {
    const response = await fetch(base + url, {
      method,
      headers: {
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    return { status: response.status, body: await response.json() };
  };

  const close = async () => {
    await new Promise((resolve) => server.close(resolve));
    store.close();
    ledger.close();
    removeDir(dir);
  };

  return { base, config, store, brokers, ledger, events, market, request, close, dir };
};

module.exports = { tempDir, removeDir, fakeMarket, fakePredictions, startApp };
