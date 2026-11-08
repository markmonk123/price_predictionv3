/**
 * Runtime configuration. Every environment variable the server reads is here.
 *
 * Trading safety model:
 *   TRADING_MODE=paper (default)  only the simulated paper broker is available.
 *   TRADING_MODE=live             the Coinbase Advanced Trade broker is enabled too,
 *                                 and startup fails unless API keys AND API_TOKEN are set.
 */

const path = require('path');

require('dotenv').config();

const ROOT = path.resolve(__dirname, '..');

const parseBoolean = (value) => ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase());

const parseNumber = (value, fallback, name) => {
  if (value === undefined || value === '') return fallback;
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`${name} must be a positive number (got "${value}")`);
  }
  return number;
};

const loadConfig = (env = process.env) => {
  const tradingMode = String(env.TRADING_MODE || 'paper').trim().toLowerCase();
  if (!['paper', 'live'].includes(tradingMode)) {
    throw new Error(`TRADING_MODE must be "paper" or "live" (got "${env.TRADING_MODE}")`);
  }

  const dataDir = env.DATA_DIR || path.join(ROOT, 'data');
  const config = {
    port: parseNumber(env.PORT, 5000, 'PORT'),
    // Bind to loopback by default: the API can place orders.
    host: env.HOST || '127.0.0.1',
    clientOrigins: (env.CLIENT_ORIGIN || 'http://localhost:5173,http://127.0.0.1:5173')
      .split(',').map((origin) => origin.trim()).filter(Boolean),
    tradingMode,
    apiToken: env.API_TOKEN || '',
    product: env.DEFAULT_PRODUCT || 'BTC-USD',
    offline: parseBoolean(env.DEMO_OFFLINE),
    maxOrderUsd: parseNumber(env.MAX_ORDER_USD, 100, 'MAX_ORDER_USD'),
    paper: {
      startingUsd: parseNumber(env.PAPER_STARTING_USD, 10000, 'PAPER_STARTING_USD'),
      feeBps: parseNumber(env.PAPER_FEE_BPS, 60, 'PAPER_FEE_BPS'),
    },
    coinbase: {
      keyId: env.COINBASE_API_KEY_ID || '',
      // .env files usually hold PEM keys on one line with literal "\n".
      keySecret: (env.COINBASE_API_KEY_SECRET || '').replace(/\\n/g, '\n'),
    },
    tickerIntervalMs: parseNumber(env.TICKER_INTERVAL_MS, 3000, 'TICKER_INTERVAL_MS'),
    predictionIntervalMs: parseNumber(env.PREDICTION_INTERVAL_MS, 5 * 60 * 1000, 'PREDICTION_INTERVAL_MS'),
    pythonPath: env.PYTHON_PATH || undefined,
    // Same default as services/prediction_history.py so Node reads the ledger Python writes.
    predictionDbPath: env.PREDICTION_DB_PATH || path.join(ROOT, 'data', 'predictions.sqlite3'),
    tradingDbPath: env.TRADING_DB_PATH || path.join(dataDir, 'trading.sqlite3'),
    clientBuildDir: path.join(ROOT, 'client', 'dist'),
    production: env.NODE_ENV === 'production',
  };

  if (tradingMode === 'live') {
    const missing = [
      ['COINBASE_API_KEY_ID', config.coinbase.keyId],
      ['COINBASE_API_KEY_SECRET', config.coinbase.keySecret],
      ['API_TOKEN', config.apiToken],
    ].filter(([, value]) => !value).map(([name]) => name);
    if (missing.length) {
      throw new Error(`TRADING_MODE=live requires ${missing.join(', ')} to be set`);
    }
  }

  return config;
};

module.exports = { loadConfig, ROOT };
