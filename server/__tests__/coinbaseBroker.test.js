const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const { tempDir, removeDir, fakeMarket } = require('./helpers');
const { createOrderStore } = require('../brokers/orderStore');
const { createCoinbaseBroker } = require('../brokers/coinbaseBroker');

const json = (status, body) => ({ ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(body) });

/** Stubbed Coinbase: records every call; responses are keyed by "METHOD path". */
const setup = (t, responses = {}) => {
  const dir = tempDir();
  const store = createOrderStore(path.join(dir, 'trading.sqlite3'));
  const calls = [];
  const jwtCalls = [];
  const fetchImpl = async (url, init) => {
    const { pathname, search } = new URL(url);
    const key = `${init.method} ${pathname}`;
    calls.push({ key, search, body: init.body ? JSON.parse(init.body) : undefined, auth: init.headers.Authorization });
    const handler = responses[key];
    if (!handler) throw new Error(`Unexpected call ${key}`);
    return typeof handler === 'function' ? handler(calls.at(-1)) : handler;
  };
  const jwt = async (options) => { jwtCalls.push(options); return 'signed.jwt.token'; };
  const broker = createCoinbaseBroker({
    keyId: 'key-id', keySecret: 'secret', maxOrderUsd: 100, store, market: fakeMarket(), fetchImpl, jwt,
  });
  t.after(() => { store.close(); removeDir(dir); });
  return { broker, store, calls, jwtCalls };
};

const previewOk = json(200, {
  order_total: '50.3', commission_total: '0.3', errs: [], warning: [], quote_size: '50', base_size: '0.5',
  best_bid: '99', best_ask: '101', is_max: false, preview_id: 'prev-1',
});

test('orders are previewed, then created with the preview id, and recorded locally', async (t) => {
  const { broker, calls, jwtCalls, store } = setup(t, {
    'POST /api/v3/brokerage/orders/preview': previewOk,
    'POST /api/v3/brokerage/orders': json(200, { success: true, success_response: { order_id: 'cb-1' } }),
  });
  const order = await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', quoteSize: 50 });

  assert.deepEqual(calls.map((c) => c.key), ['POST /api/v3/brokerage/orders/preview', 'POST /api/v3/brokerage/orders']);
  assert.deepEqual(calls[1].body.order_configuration, { market_market_ioc: { quote_size: '50' } });
  assert.equal(calls[1].body.preview_id, 'prev-1');
  assert.equal(calls[1].body.client_order_id, order.clientOrderId);
  assert.equal(calls[1].auth, 'Bearer signed.jwt.token');
  // One JWT per request, scoped to that exact method + host + path.
  assert.deepEqual(jwtCalls.map((j) => `${j.requestMethod} ${j.requestHost}${j.requestPath}`), [
    'POST api.coinbase.com/api/v3/brokerage/orders/preview',
    'POST api.coinbase.com/api/v3/brokerage/orders',
  ]);
  assert.equal(order.externalId, 'cb-1');
  assert.equal(order.status, 'PENDING');
  assert.equal(store.get(order.id).broker, 'coinbase');
});

test('MAX_ORDER_USD is enforced before Coinbase is called', async (t) => {
  const { broker, calls } = setup(t);
  await assert.rejects(broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', quoteSize: 100.01 }),
    { code: 'max_order_exceeded' });
  // Base-sized market orders are valued at the live ask (101): 1 BTC is ~$101.
  await assert.rejects(broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', baseSize: 1 }),
    { code: 'max_order_exceeded' });
  await assert.rejects(broker.preview({ productId: 'BTC-USD', side: 'SELL', type: 'LIMIT', baseSize: 2, limitPrice: 60 }),
    { code: 'max_order_exceeded' });
  assert.equal(calls.length, 0);
});

test('preview errors reject the order without creating it', async (t) => {
  const { broker, calls } = setup(t, {
    'POST /api/v3/brokerage/orders/preview': json(200, { errs: ['PREVIEW_INSUFFICIENT_FUND'], warning: [] }),
  });
  await assert.rejects(broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', quoteSize: 10 }),
    { code: 'order_rejected' });
  assert.equal(calls.length, 1);
});

test('a failed create is recorded as FAILED and surfaced', async (t) => {
  const { broker, store } = setup(t, {
    'POST /api/v3/brokerage/orders/preview': previewOk,
    'POST /api/v3/brokerage/orders': json(200, { success: false, error_response: { message: 'Insufficient balance' } }),
  });
  await assert.rejects(broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', quoteSize: 10 }),
    { code: 'order_rejected', message: 'Insufficient balance' });
  const [failed] = store.list({ broker: 'coinbase' });
  assert.equal(failed.status, 'FAILED');
});

test('list refreshes open orders, the query is excluded from the JWT path, and cancel uses batch_cancel', async (t) => {
  const { broker, calls, jwtCalls } = setup(t, {
    'POST /api/v3/brokerage/orders/preview': previewOk,
    'POST /api/v3/brokerage/orders': json(200, { success: true, success_response: { order_id: 'cb-9' } }),
    'GET /api/v3/brokerage/orders/historical/batch': json(200, {
      orders: [{ order_id: 'cb-9', status: 'OPEN', filled_size: '0.1', average_filled_price: '99', total_fees: '0.01' }],
      has_next: false,
    }),
    'POST /api/v3/brokerage/orders/batch_cancel': json(200, { results: [{ success: true, order_id: 'cb-9', failure_reason: 'UNKNOWN_CANCEL_FAILURE_REASON' }] }),
  });
  const placed = await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'LIMIT', baseSize: 0.5, limitPrice: 99 });
  assert.deepEqual(calls[1].body.order_configuration, { limit_limit_gtc: { base_size: '0.5', limit_price: '99', post_only: false } });

  const [listed] = await broker.list();
  assert.equal(listed.status, 'OPEN');
  assert.equal(listed.filledSize, 0.1);
  assert.equal(calls.at(-1).search, '?order_ids=cb-9');
  assert.equal(jwtCalls.at(-1).requestPath, '/api/v3/brokerage/orders/historical/batch');

  const cancelled = await broker.cancel(placed.id);
  assert.equal(cancelled.status, 'CANCEL_QUEUED');
  assert.deepEqual(calls.at(-1).body, { order_ids: ['cb-9'] });
});

test('HTTP errors from Coinbase become broker_error', async (t) => {
  const { broker } = setup(t, {
    'GET /api/v3/brokerage/accounts': json(401, { error: 'UNAUTHENTICATED', message: 'invalid signature' }),
  });
  await assert.rejects(broker.account(), { code: 'broker_error', message: 'invalid signature' });
});
