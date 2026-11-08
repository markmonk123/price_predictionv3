const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const { tempDir, removeDir, fakeMarket } = require('./helpers');
const { createOrderStore } = require('../brokers/orderStore');
const { createPaperBroker } = require('../brokers/paperBroker');

const setup = (t) => {
  const dir = tempDir();
  const store = createOrderStore(path.join(dir, 'trading.sqlite3'));
  const market = fakeMarket();
  const broker = createPaperBroker({ store, market, startingUsd: 1000, feeBps: 100 });
  t.after(() => { store.close(); removeDir(dir); });
  const balances = async () => Object.fromEntries((await broker.account()).balances.map((b) => [b.currency, b]));
  return { store, market, broker, balances };
};

const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test('market buy fills at the ask and charges fees', async (t) => {
  const { broker, balances } = setup(t);
  const order = await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', baseSize: 2 });
  assert.equal(order.status, 'FILLED');
  assert.equal(order.averageFilledPrice, 101);
  close(order.totalFees, 2.02);
  const b = await balances();
  close(b.USD.available, 1000 - 202 - 2.02);
  close(b.BTC.available, 2);
});

test('quote-sized market sell converts at the bid', async (t) => {
  const { broker, balances } = setup(t);
  await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', baseSize: 2 });
  const sell = await broker.place({ productId: 'BTC-USD', side: 'SELL', type: 'MARKET', quoteSize: 99 });
  close(sell.filledSize, 1);
  const b = await balances();
  close(b.BTC.available, 1);
  close(b.USD.available, 1000 - 202 - 2.02 + 99 - 0.99);
});

test('resting limit buy holds funds, fills when crossed, and releases the hold', async (t) => {
  const { broker, market, balances } = setup(t);
  const order = await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'LIMIT', baseSize: 1, limitPrice: 95 });
  assert.equal(order.status, 'OPEN');
  let b = await balances();
  close(b.USD.hold, 95 * 1.01);
  close(b.USD.available, 1000 - 95 * 1.01);

  assert.deepEqual(broker.onTicker(market.setTicker({ bid: 96, ask: 97 })), []);
  const [filled] = broker.onTicker(market.setTicker({ bid: 94, ask: 95 }));
  assert.equal(filled.status, 'FILLED');
  assert.equal(filled.averageFilledPrice, 95);
  b = await balances();
  close(b.USD.hold, 0);
  close(b.USD.available, 1000 - 95 * 1.01);
  close(b.BTC.available, 1);
});

test('resting limit sell holds base and fills on a higher bid', async (t) => {
  const { broker, market, balances } = setup(t);
  await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'MARKET', baseSize: 1 });
  await broker.place({ productId: 'BTC-USD', side: 'SELL', type: 'LIMIT', baseSize: 1, limitPrice: 110 });
  let b = await balances();
  close(b.BTC.available, 0);
  close(b.BTC.hold, 1);
  const [filled] = broker.onTicker(market.setTicker({ bid: 111, ask: 112 }));
  assert.equal(filled.averageFilledPrice, 110);
  b = await balances();
  close(b.BTC.hold, 0);
  close(b.USD.available, 1000 - 101 * 1.01 + 110 * 0.99);
});

test('cancel releases the hold and marketable limits fill immediately at the better price', async (t) => {
  const { broker, balances } = setup(t);
  const resting = await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'LIMIT', baseSize: 1, limitPrice: 90 });
  await broker.cancel(resting.id);
  const b = await balances();
  close(b.USD.available, 1000);
  close(b.USD.hold, 0);
  await assert.rejects(broker.cancel(resting.id), { code: 'not_cancellable' });

  const marketable = await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'LIMIT', baseSize: 1, limitPrice: 150 });
  assert.equal(marketable.status, 'FILLED');
  assert.equal(marketable.averageFilledPrice, 101);
});

test('post-only orders that would take liquidity are cancelled, not filled', async (t) => {
  const { broker, balances } = setup(t);
  const order = await broker.place({
    productId: 'BTC-USD', side: 'BUY', type: 'LIMIT', baseSize: 1, limitPrice: 150, postOnly: true,
  });
  assert.equal(order.status, 'CANCELLED');
  close((await balances()).USD.available, 1000);
});

test('insufficient funds and reset', async (t) => {
  const { broker, balances } = setup(t);
  await assert.rejects(broker.place({ productId: 'BTC-USD', side: 'SELL', type: 'MARKET', baseSize: 1 }),
    { code: 'order_rejected' });
  await broker.place({ productId: 'BTC-USD', side: 'BUY', type: 'LIMIT', baseSize: 1, limitPrice: 50 });
  await broker.reset();
  const b = await balances();
  close(b.USD.available, 1000);
  assert.equal(b.BTC, undefined);
  assert.equal((await broker.list({ status: 'open' })).length, 0);
});
