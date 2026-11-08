#!/usr/bin/env node
/**
 * Project MCP server (stdio) for the BTC prediction platform.
 *
 * Exposes market data, model predictions, the prediction ledger and PAPER trading
 * to MCP clients (Claude Code, VS Code). It is a thin client of the REST API at
 * API_BASE_URL, so the server stays the single source of truth.
 *
 * Safety: every order tool is pinned to the paper broker. No tool here can place
 * or cancel a real Coinbase order, regardless of the server's TRADING_MODE.
 *
 * Env: API_BASE_URL (default http://127.0.0.1:5000/api/v1), API_TOKEN (if the server sets one).
 * stdout carries the MCP protocol; diagnostics go to stderr only.
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { z } = require('zod');
const { version } = require('../package.json');

const GRANULARITIES = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'FOUR_HOUR', 'SIX_HOUR', 'ONE_DAY'];

const createApiClient = ({ baseUrl, token, fetchImpl = fetch }) => async (method, path, { query, body } = {}) => {
  const params = new URLSearchParams(Object.entries(query || {}).filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => [k, String(v)]));
  const url = `${baseUrl}${path}${params.size ? `?${params}` : ''}`;
  let response;
  try {
    response = await fetchImpl(url, {
      method,
      headers: {
        Accept: 'application/json',
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (error) {
    throw new Error(`Cannot reach the trading server at ${baseUrl} (${error.message}). Is \`npm start\` running?`);
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const err = payload?.error;
    const details = err?.details ? ` ${JSON.stringify(err.details)}` : '';
    throw new Error(`${response.status} ${err?.code || 'error'}: ${err?.message || response.statusText}${details}`);
  }
  return payload;
};

const ok = (data) => ({ content: [{ type: 'text', text: JSON.stringify(data, null, 2) }] });
const fail = (error) => ({ isError: true, content: [{ type: 'text', text: error.message }] });
const tool = (handler) => async (args) => {
  try { return ok(await handler(args || {})); } catch (error) { return fail(error); }
};

const productId = z.string().regex(/^[A-Z0-9]+-[A-Z0-9]+$/).optional().describe('Trading pair, e.g. BTC-USD (default: server product)');
const historyFilters = {
  kind: z.string().optional().describe('price | direction | trade | rule_signal | high_low'),
  signal: z.string().optional().describe('Signal name, e.g. classifier, heldout_classifier, forecast_ensemble, long'),
  horizon: z.number().int().positive().optional().describe('Horizon in minutes'),
  scope: z.enum(['live', 'historical', 'all']).optional().describe('live = forward predictions, historical = backtest rows'),
  from: z.string().optional().describe('Inclusive run time, ISO 8601'),
  to: z.string().optional().describe('Exclusive run time, ISO 8601'),
};
const orderFields = {
  side: z.enum(['BUY', 'SELL']),
  type: z.enum(['MARKET', 'LIMIT']),
  baseSize: z.number().positive().optional().describe('Size in base currency (BTC). Required for LIMIT.'),
  quoteSize: z.number().positive().optional().describe('Size in quote currency (USD). MARKET orders only.'),
  limitPrice: z.number().positive().optional().describe('Required for LIMIT orders'),
  postOnly: z.boolean().optional(),
  productId,
};

const createServer = ({ baseUrl, token, fetchImpl } = {}) => {
  const api = createApiClient({
    baseUrl: (baseUrl || process.env.API_BASE_URL || 'http://127.0.0.1:5000/api/v1').replace(/\/$/, ''),
    token: token ?? process.env.API_TOKEN,
    fetchImpl,
  });
  const server = new McpServer({ name: 'priceprediction', version });

  server.registerTool('get_server_status', {
    title: 'Server status',
    description: 'Trading mode (paper/live), enabled brokers, market-data health, last model run and ledger availability.',
    inputSchema: {},
  }, tool(() => api('GET', '/health')));

  server.registerTool('get_ticker', {
    title: 'Live ticker',
    description: 'Latest price, best bid/ask, spread, 24h change % and 24h volume from Coinbase.',
    inputSchema: { productId },
  }, tool((a) => api('GET', '/market/ticker', { query: a })));

  server.registerTool('get_candles', {
    title: 'OHLCV candles',
    description: 'Historical candles (oldest first). Times are UNIX seconds. Max 350 per call.',
    inputSchema: {
      productId,
      granularity: z.enum(GRANULARITIES).optional().describe('Default FIVE_MINUTE'),
      limit: z.number().int().min(1).max(350).optional().describe('Default 48'),
      start: z.number().int().positive().optional().describe('UNIX seconds'),
      end: z.number().int().positive().optional().describe('UNIX seconds'),
    },
  }, tool((a) => api('GET', '/market/candles', { query: { limit: 48, ...a } })));

  server.registerTool('get_order_book', {
    title: 'Order book',
    description: 'Top bids and asks from the live Coinbase order book.',
    inputSchema: { productId, limit: z.number().int().min(1).max(100).optional().describe('Levels per side, default 10') },
  }, tool((a) => api('GET', '/market/book', { query: { limit: 10, ...a } })));

  server.registerTool('get_latest_prediction', {
    title: 'Latest model prediction',
    description: 'Most recent output of the direction model: predicted_direction (-1/0/1), class probabilities, confidence, horizon and threshold.',
    inputSchema: {},
  }, tool(() => api('GET', '/predictions/latest')));

  server.registerTool('run_prediction', {
    title: 'Run the model now',
    description: 'Run the Python direction model on live data and return the new prediction. Takes up to a minute.',
    inputSchema: {},
  }, tool(() => api('POST', '/predictions/run')));

  server.registerTool('query_prediction_history', {
    title: 'Query prediction ledger',
    description: 'Rows from the durable prediction ledger (newest first) with their scored outcomes: actual price, '
      + 'dollar/percent error, direction_correct, range containment. Status is pending, scored or unscorable.',
    inputSchema: {
      ...historyFilters,
      status: z.enum(['pending', 'scored', 'unscorable']).optional(),
      limit: z.number().int().min(1).max(500).optional().describe('Default 50'),
      offset: z.number().int().min(0).optional(),
    },
  }, tool((a) => api('GET', '/predictions/history', { query: { limit: 50, ...a } })));

  server.registerTool('prediction_accuracy', {
    title: 'Prediction accuracy',
    description: 'Aggregate accuracy per (live/historical, kind, horizon[, signal]): MAE, RMSE, MAPE, within-1% rate, '
      + 'direction accuracy, range containment and mean underlying return. Mirrors the Python ledger report.',
    inputSchema: { ...historyFilters, groupBy: z.enum(['horizon', 'signal']).optional() },
  }, tool((a) => api('GET', '/predictions/accuracy', { query: a })));

  server.registerTool('get_paper_account', {
    title: 'Paper account balances',
    description: 'Simulated USD/BTC balances (available and on hold) of the paper broker.',
    inputSchema: {},
  }, tool(() => api('GET', '/account', { query: { broker: 'paper' } })));

  server.registerTool('preview_paper_order', {
    title: 'Preview a paper order',
    description: 'Price a simulated order against the live book without placing it: fill price, size, fees, total, and any errors.',
    inputSchema: orderFields,
  }, tool((a) => api('POST', '/orders/preview', { body: { ...a, broker: 'paper' } })));

  server.registerTool('place_paper_order', {
    title: 'Place a paper order',
    description: 'Place a SIMULATED order with the paper broker (never real money). Market orders fill at the live bid/ask; '
      + 'limit orders rest until the live price crosses them.',
    inputSchema: orderFields,
  }, tool((a) => api('POST', '/orders', { body: { ...a, broker: 'paper' } })));

  server.registerTool('list_orders', {
    title: 'List orders',
    description: 'Recent orders, newest first. status "open" returns resting orders only.',
    inputSchema: {
      status: z.string().optional().describe('open, FILLED, CANCELLED, ...'),
      limit: z.number().int().min(1).max(500).optional().describe('Default 50'),
    },
  }, tool((a) => api('GET', '/orders', { query: { limit: 50, ...a, broker: 'paper' } })));

  server.registerTool('cancel_order', {
    title: 'Cancel a paper order',
    description: 'Cancel a resting PAPER order and release its held funds. Refuses non-paper orders.',
    inputSchema: { orderId: z.string().min(1) },
  }, tool(async ({ orderId }) => {
    const order = await api('GET', `/orders/${encodeURIComponent(orderId)}`);
    if (order.broker !== 'paper') {
      throw new Error(`Order ${orderId} belongs to the ${order.broker} broker; this MCP server only manages paper orders.`);
    }
    return api('DELETE', `/orders/${encodeURIComponent(orderId)}`);
  }));

  return server;
};

if (require.main === module) {
  createServer().connect(new StdioServerTransport()).catch((error) => {
    process.stderr.write(`priceprediction MCP server failed: ${error.stack || error.message}\n`);
    process.exit(1);
  });
}

module.exports = { createServer };
