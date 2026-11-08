const test = require('node:test');
const assert = require('node:assert/strict');
const { startApp, fakePredictions } = require('./helpers');

test('health reports paper mode and no live trading', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const { status, body } = await app.request('GET', '/health');
  assert.equal(status, 200);
  assert.equal(body.tradingMode, 'paper');
  assert.equal(body.liveTradingEnabled, false);
  assert.deepEqual(body.brokers, ['paper']);
  assert.equal(body.market.ok, true);
  assert.equal(body.ledger.available, false);
});

test('market routes return normalized data and validate params', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const ticker = await app.request('GET', '/market/ticker');
  assert.equal(ticker.status, 200);
  assert.equal(ticker.body.bid, 99);
  const candles = await app.request('GET', '/market/candles?granularity=FIVE_MINUTE&limit=10');
  assert.equal(candles.status, 200);
  assert.equal(candles.body.candles.length, 1);
  const bad = await app.request('GET', '/market/candles?granularity=TEN_SECONDS');
  assert.equal(bad.status, 400);
  assert.equal(bad.body.error.code, 'validation_error');
  assert.equal(bad.body.error.details[0].path, 'granularity');
});

test('order validation errors use the error envelope', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const cases = [
    { side: 'BUY', type: 'MARKET' },
    { side: 'BUY', type: 'MARKET', baseSize: 1, quoteSize: 10 },
    { side: 'BUY', type: 'LIMIT', baseSize: 1 },
    { side: 'BUY', type: 'MARKET', baseSize: -1 },
    { side: 'HOLD', type: 'MARKET', baseSize: 1 },
    { side: 'BUY', type: 'MARKET', baseSize: 1, extra: true },
  ];
  for (const body of cases) {
    const response = await app.request('POST', '/orders', { body });
    assert.equal(response.status, 400, JSON.stringify(body));
    assert.equal(response.body.error.code, 'validation_error');
  }
});

test('coinbase broker is refused in paper mode', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const response = await app.request('POST', '/orders', {
    body: { broker: 'coinbase', side: 'BUY', type: 'MARKET', quoteSize: 5 },
  });
  assert.equal(response.status, 403);
  assert.equal(response.body.error.code, 'live_trading_disabled');
  const account = await app.request('GET', '/account?broker=coinbase');
  assert.equal(account.status, 403);
});

test('API token guards mutating routes only', async (t) => {
  const app = await startApp({ env: { API_TOKEN: 'secret-token' } });
  t.after(app.close);
  const order = { side: 'BUY', type: 'MARKET', quoteSize: 10 };
  assert.equal((await app.request('POST', '/orders', { body: order })).status, 401);
  assert.equal((await app.request('POST', '/orders', { body: order, token: 'wrong-token!' })).status, 401);
  assert.equal((await app.request('POST', '/orders', { body: order, token: 'secret-token' })).status, 201);
  assert.equal((await app.request('POST', '/predictions/run')).status, 401);
  assert.equal((await app.request('POST', '/account/reset')).status, 401);
  // Reads stay open.
  assert.equal((await app.request('GET', '/orders')).status, 200);
});

test('paper order lifecycle over HTTP emits order events', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const emitted = [];
  app.events.on('order', (order) => emitted.push(order.status));

  const preview = await app.request('POST', '/orders/preview', { body: { side: 'BUY', type: 'MARKET', baseSize: 2 } });
  assert.equal(preview.status, 200);
  assert.equal(preview.body.price, 101);
  assert.ok(Math.abs(preview.body.orderTotal - 204.02) < 1e-9);

  const placed = await app.request('POST', '/orders', { body: { side: 'BUY', type: 'LIMIT', baseSize: 1, limitPrice: 90 } });
  assert.equal(placed.status, 201);
  assert.equal(placed.body.status, 'OPEN');

  const listed = await app.request('GET', '/orders?status=open');
  assert.deepEqual(listed.body.orders.map((o) => o.id), [placed.body.id]);

  const cancelled = await app.request('DELETE', `/orders/${placed.body.id}`);
  assert.equal(cancelled.body.status, 'CANCELLED');
  const again = await app.request('DELETE', `/orders/${placed.body.id}`);
  assert.equal(again.status, 409);
  assert.equal((await app.request('GET', '/orders/does-not-exist')).status, 404);
  assert.deepEqual(emitted, ['OPEN', 'CANCELLED']);
});

test('insufficient paper funds are rejected with 422', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const response = await app.request('POST', '/orders', { body: { side: 'BUY', type: 'MARKET', baseSize: 50 } });
  assert.equal(response.status, 422);
  assert.deepEqual(response.body.error.details.errs, ['INSUFFICIENT_FUND']);
});

test('latest prediction is 404 until a model run succeeds', async (t) => {
  const empty = await startApp();
  t.after(empty.close);
  const missing = await empty.request('GET', '/predictions/latest');
  assert.equal(missing.status, 404);
  assert.equal(missing.body.error.code, 'no_prediction');

  const ready = await startApp({ predictions: fakePredictions({ predicted_direction: 1, generatedAt: 'x' }) });
  t.after(ready.close);
  assert.equal((await ready.request('GET', '/predictions/latest')).body.predicted_direction, 1);
});

test('unknown API routes return a JSON 404', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const { status, body } = await app.request('GET', '/nope');
  assert.equal(status, 404);
  assert.equal(body.error.code, 'not_found');
});
