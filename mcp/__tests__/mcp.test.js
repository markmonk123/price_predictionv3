const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { startApp } = require('../../server/__tests__/helpers');

const SERVER = path.join(__dirname, '..', 'server.js');

const connect = async (t, env) => {
  const client = new Client({ name: 'test', version: '1.0.0' });
  await client.connect(new StdioClientTransport({
    command: process.execPath,
    args: [SERVER],
    env: { ...process.env, ...env },
    stderr: 'pipe',
  }));
  t.after(() => client.close());
  const call = async (name, args = {}) => {
    const result = await client.callTool({ name, arguments: args });
    const text = result.content[0].text;
    return { isError: Boolean(result.isError), text, data: result.isError ? null : JSON.parse(text) };
  };
  return { client, call };
};

test('lists the project tools', async (t) => {
  const app = await startApp();
  t.after(app.close);
  const { client } = await connect(t, { API_BASE_URL: app.base });
  const { tools } = await client.listTools();
  assert.deepEqual(tools.map((tool) => tool.name).sort(), [
    'cancel_order', 'get_candles', 'get_latest_prediction', 'get_order_book', 'get_paper_account',
    'get_server_status', 'get_ticker', 'list_orders', 'place_paper_order', 'prediction_accuracy',
    'preview_paper_order', 'query_prediction_history', 'run_prediction',
  ]);
});

test('paper order round trip through MCP, with the token forwarded', async (t) => {
  const app = await startApp({ env: { API_TOKEN: 'mcp-token' } });
  t.after(app.close);
  const { call } = await connect(t, { API_BASE_URL: app.base, API_TOKEN: 'mcp-token' });

  const ticker = await call('get_ticker');
  assert.equal(ticker.data.ask, 101);

  const placed = await call('place_paper_order', { side: 'BUY', type: 'LIMIT', baseSize: 1, limitPrice: 90 });
  assert.equal(placed.isError, false, placed.text);
  assert.equal(placed.data.broker, 'paper');
  assert.equal(placed.data.status, 'OPEN');

  const open = await call('list_orders', { status: 'open' });
  assert.deepEqual(open.data.orders.map((o) => o.id), [placed.data.id]);

  const account = await call('get_paper_account');
  assert.equal(account.data.balances.find((b) => b.currency === 'USD').hold, 90 * 1.01);

  const cancelled = await call('cancel_order', { orderId: placed.data.id });
  assert.equal(cancelled.data.status, 'CANCELLED');
});

test('MCP order tools cannot reach a live broker and surface API errors', async (t) => {
  const app = await startApp();
  t.after(app.close);
  // Even if a caller sneaks in broker=coinbase, the tool pins broker=paper.
  const { call } = await connect(t, { API_BASE_URL: app.base });
  const sneaky = await call('preview_paper_order', { side: 'BUY', type: 'MARKET', quoteSize: 10, broker: 'coinbase' });
  assert.equal(sneaky.isError, false, sneaky.text);
  assert.equal(sneaky.data.broker, 'paper');

  // A live order recorded in the ledger can't be cancelled through MCP.
  app.store.insert({
    id: 'live-1', broker: 'coinbase', clientOrderId: 'live-1', productId: 'BTC-USD',
    side: 'BUY', type: 'LIMIT', baseSize: 0.001, limitPrice: 50000, status: 'OPEN',
  });
  const refused = await call('cancel_order', { orderId: 'live-1' });
  assert.equal(refused.isError, true);
  assert.match(refused.text, /only manages paper orders/);

  const missing = await call('get_latest_prediction');
  assert.equal(missing.isError, true);
  assert.match(missing.text, /404 no_prediction/);

  const history = await call('query_prediction_history', { kind: 'price' });
  assert.deepEqual(history.data, { total: 0, rows: [] });
});

test('reports a clear error when the trading server is down', async (t) => {
  const { call } = await connect(t, { API_BASE_URL: 'http://127.0.0.1:9/api/v1' });
  const result = await call('get_ticker');
  assert.equal(result.isError, true);
  assert.match(result.text, /Cannot reach the trading server/);
});
