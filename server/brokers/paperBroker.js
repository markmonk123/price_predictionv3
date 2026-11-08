/**
 * Paper broker: simulated fills against live Coinbase prices. No real money.
 *
 * - Market orders fill immediately at the best ask (BUY) or best bid (SELL).
 * - Marketable limit orders fill immediately at the best price.
 * - Resting limit orders hold funds and fill at their limit price once the
 *   polled ticker crosses them (see onTicker).
 * - Fees are simulated at feeBps of notional.
 */

const crypto = require('crypto');
const { ApiError } = require('../http');
const { transaction, OPEN_STATUSES } = require('./orderStore');

const currenciesOf = (productId) => {
  const [base, quote] = productId.split('-');
  return { base, quote };
};

const createPaperBroker = ({ store, market, startingUsd = 10000, feeBps = 60 }) => {
  const { db } = store;
  db.exec(`CREATE TABLE IF NOT EXISTS paper_balances (
    currency TEXT PRIMARY KEY,
    available REAL NOT NULL,
    hold REAL NOT NULL DEFAULT 0
  )`);

  const seedBalances = () => {
    db.prepare("INSERT OR IGNORE INTO paper_balances (currency, available) VALUES ('USD', ?)").run(startingUsd);
  };
  seedBalances();

  const balance = (currency) => db.prepare('SELECT * FROM paper_balances WHERE currency = ?').get(currency)
    || { currency, available: 0, hold: 0 };

  const adjust = (currency, availableDelta, holdDelta = 0) => {
    db.prepare(`INSERT INTO paper_balances (currency, available, hold) VALUES ($c, $a, $h)
      ON CONFLICT(currency) DO UPDATE SET available = available + $a, hold = hold + $h`)
      .run({ $c: currency, $a: availableDelta, $h: holdDelta });
  };

  const fee = (notional) => notional * (feeBps / 10000);

  /** Price an order against the current ticker without touching balances. */
  const quote = (order, ticker) => {
    const marketPrice = order.side === 'BUY' ? ticker.ask : ticker.bid;
    if (!marketPrice) throw new ApiError(503, 'market_unavailable', 'No live bid/ask to price the order');
    const marketable = order.type === 'MARKET'
      || (order.side === 'BUY' ? order.limitPrice >= ticker.ask : order.limitPrice <= ticker.bid);
    const price = marketable ? marketPrice : order.limitPrice;
    const baseSize = order.baseSize ?? order.quoteSize / price;
    const notional = baseSize * price;
    const commission = fee(notional);
    return { price, marketable, baseSize, notional, commission };
  };

  const checkFunds = (order, q) => {
    const { base, quote: quoteCurrency } = currenciesOf(order.productId);
    const errs = [];
    if (order.side === 'BUY' && balance(quoteCurrency).available + 1e-9 < q.notional + q.commission) {
      errs.push('INSUFFICIENT_FUND');
    }
    if (order.side === 'SELL' && balance(base).available + 1e-12 < q.baseSize) {
      errs.push('INSUFFICIENT_FUND');
    }
    if (!(q.baseSize > 0)) errs.push('INVALID_SIZE');
    return errs;
  };

  const preview = async (order) => {
    const ticker = await market.getTicker(order.productId);
    const q = quote(order, ticker);
    const total = order.side === 'BUY' ? q.notional + q.commission : q.notional - q.commission;
    return {
      broker: 'paper',
      productId: order.productId,
      side: order.side,
      type: order.type,
      price: q.price,
      baseSize: q.baseSize,
      quoteSize: q.notional,
      commissionTotal: q.commission,
      orderTotal: total,
      bestBid: ticker.bid,
      bestAsk: ticker.ask,
      fillsImmediately: q.marketable,
      errs: checkFunds(order, q),
      warnings: [],
    };
  };

  /** Move funds for a fill. Caller is inside a transaction. */
  const settleFill = (order, baseSize, price, { releaseHold = 0 } = {}) => {
    const { base, quote: quoteCurrency } = currenciesOf(order.productId);
    const notional = baseSize * price;
    const commission = fee(notional);
    if (order.side === 'BUY') {
      // Funds for resting orders sit in hold; return any over-reservation to available.
      adjust(quoteCurrency, releaseHold - notional - commission, -releaseHold);
      adjust(base, baseSize);
    } else {
      adjust(base, releaseHold ? 0 : -baseSize, -releaseHold);
      adjust(quoteCurrency, notional - commission);
    }
    return { filledSize: baseSize, averageFilledPrice: price, totalFees: commission, status: 'FILLED' };
  };

  const buyHold = (order) => order.baseSize * order.limitPrice * (1 + feeBps / 10000);

  const place = async (order) => {
    const ticker = await market.getTicker(order.productId);
    const q = quote(order, ticker);
    const errs = checkFunds(order, q);
    if (errs.length) {
      throw new ApiError(422, 'order_rejected', 'Paper order rejected', { errs });
    }
    const id = crypto.randomUUID();
    return transaction(db, () => {
      const stored = store.insert({
        id,
        broker: 'paper',
        clientOrderId: id,
        productId: order.productId,
        side: order.side,
        type: order.type,
        baseSize: order.baseSize ?? null,
        quoteSize: order.quoteSize ?? null,
        limitPrice: order.limitPrice ?? null,
        postOnly: order.postOnly,
        status: 'OPEN',
      });
      if (q.marketable) {
        if (order.postOnly) {
          return store.update(id, { status: 'CANCELLED', reason: 'Post-only order would have taken liquidity' });
        }
        return store.update(id, settleFill(order, q.baseSize, q.price));
      }
      const { base, quote: quoteCurrency } = currenciesOf(order.productId);
      if (order.side === 'BUY') {
        const hold = buyHold(order);
        adjust(quoteCurrency, -hold, hold);
      } else {
        adjust(base, -order.baseSize, order.baseSize);
      }
      return stored;
    });
  };

  const getOwn = (id) => {
    const order = store.get(id);
    if (!order || order.broker !== 'paper') throw new ApiError(404, 'not_found', `Paper order ${id} not found`);
    return order;
  };

  const cancel = async (id) => {
    const order = getOwn(id);
    if (!OPEN_STATUSES.includes(order.status)) {
      throw new ApiError(409, 'not_cancellable', `Order is ${order.status}`);
    }
    return transaction(db, () => {
      const { base, quote: quoteCurrency } = currenciesOf(order.productId);
      if (order.side === 'BUY') {
        const hold = buyHold(order);
        adjust(quoteCurrency, hold, -hold);
      } else {
        adjust(base, order.baseSize, -order.baseSize);
      }
      return store.update(id, { status: 'CANCELLED' });
    });
  };

  /** Fill resting limit orders the new ticker crosses. Returns the orders that changed. */
  const onTicker = (ticker) => {
    if (!ticker?.bid || !ticker?.ask) return [];
    const resting = store.list({ broker: 'paper', status: 'open', productId: ticker.productId, limit: 1000 });
    const filled = [];
    for (const order of resting) {
      const crosses = order.side === 'BUY' ? ticker.ask <= order.limitPrice : ticker.bid >= order.limitPrice;
      if (!crosses) continue;
      filled.push(transaction(db, () => {
        const releaseHold = order.side === 'BUY' ? buyHold(order) : order.baseSize;
        return store.update(order.id, settleFill(order, order.baseSize, order.limitPrice, { releaseHold }));
      }));
    }
    return filled;
  };

  const account = async () => ({
    broker: 'paper',
    balances: db.prepare('SELECT currency, available, hold FROM paper_balances ORDER BY currency').all()
      .map((row) => ({ currency: row.currency, available: row.available, hold: row.hold })),
  });

  /** Cancel every paper order and restore the starting balance. */
  const reset = async () => {
    transaction(db, () => {
      db.prepare(`UPDATE orders SET status = 'CANCELLED', updated_at = ? WHERE broker = 'paper'
        AND status IN (${OPEN_STATUSES.map((s) => `'${s}'`).join(',')})`).run(new Date().toISOString());
      db.exec('DELETE FROM paper_balances');
      seedBalances();
    });
    return account();
  };

  return {
    name: 'paper',
    preview,
    place,
    cancel,
    get: async (id) => getOwn(id),
    list: async (filters = {}) => store.list({ ...filters, broker: 'paper' }),
    account,
    reset,
    onTicker,
  };
};

module.exports = { createPaperBroker, currenciesOf };
