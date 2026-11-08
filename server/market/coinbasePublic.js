/**
 * Coinbase Advanced Trade public market data (no API key needed).
 * Docs: https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/public/
 *
 * Numbers arrive as strings from Coinbase and are converted here, so the rest
 * of the server and the client only ever see numbers.
 */

const NodeCache = require('node-cache');
const { ApiError } = require('../http');

const BASE_URL = 'https://api.coinbase.com/api/v3/brokerage/market';
const MAX_CANDLES = 350;

const GRANULARITY_SECONDS = {
  ONE_MINUTE: 60,
  FIVE_MINUTE: 300,
  FIFTEEN_MINUTE: 900,
  THIRTY_MINUTE: 1800,
  ONE_HOUR: 3600,
  TWO_HOUR: 7200,
  FOUR_HOUR: 14400,
  SIX_HOUR: 21600,
  ONE_DAY: 86400,
};

const num = (value) => (value === undefined || value === null || value === '' ? null : Number(value));

const createMarketData = ({ fetchImpl = fetch, offline = false, now = () => Date.now() } = {}) => {
  const cache = new NodeCache({ useClones: false });

  const cached = async (key, ttlSeconds, load) => {
    const hit = cache.get(key);
    if (hit !== undefined) return hit;
    const value = await load();
    cache.set(key, value, ttlSeconds);
    return value;
  };

  const getJson = async (path, query) => {
    const url = `${BASE_URL}${path}?${new URLSearchParams(query)}`;
    let response;
    try {
      response = await fetchImpl(url, { headers: { Accept: 'application/json' }, signal: AbortSignal.timeout(10000) });
    } catch (error) {
      throw new ApiError(503, 'market_unavailable', `Coinbase market data unreachable: ${error.message}`);
    }
    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new ApiError(response.status === 404 ? 404 : 503, 'market_unavailable',
        `Coinbase returned ${response.status} for ${path}`, body.slice(0, 500) || undefined);
    }
    return response.json();
  };

  // --- Offline synthetic data (DEMO_OFFLINE=1 only) -------------------------
  let syntheticPrice = 75000;
  const syntheticTicker = (productId) => {
    syntheticPrice *= 1 + (Math.random() - 0.5) * 0.001;
    const spread = syntheticPrice * 0.00002;
    return {
      productId, price: syntheticPrice, bid: syntheticPrice - spread / 2, ask: syntheticPrice + spread / 2,
      mid: syntheticPrice, spread, change24hPct: 0, volume24h: 0, time: new Date(now()).toISOString(), source: 'synthetic',
    };
  };

  const getTicker = (productId) => {
    if (offline) return Promise.resolve(syntheticTicker(productId));
    return cached(`ticker:${productId}`, 2, async () => {
      const [book, product] = await Promise.all([
        getJson('/product_book', { product_id: productId, limit: 1 }),
        cached(`product:${productId}`, 30, () => getJson(`/products/${encodeURIComponent(productId)}`, {})),
      ]);
      const bid = num(book.pricebook?.bids?.[0]?.price);
      const ask = num(book.pricebook?.asks?.[0]?.price);
      return {
        productId,
        price: num(book.last) ?? num(product.price),
        bid,
        ask,
        mid: num(book.mid_market),
        spread: num(book.spread_absolute),
        change24hPct: num(product.price_percentage_change_24h),
        volume24h: num(product.volume_24h),
        time: book.pricebook?.time || new Date(now()).toISOString(),
        source: 'coinbase',
      };
    });
  };

  const getCandles = (productId, { granularity = 'FIVE_MINUTE', start, end, limit = 300 } = {}) => {
    const seconds = GRANULARITY_SECONDS[granularity];
    if (!seconds) throw new ApiError(400, 'validation_error', `Unsupported granularity ${granularity}`);
    const count = Math.min(Math.max(Number(limit) || 300, 1), MAX_CANDLES);
    const endSec = end ? Math.floor(Number(end)) : Math.floor(now() / 1000);
    const startSec = start ? Math.floor(Number(start)) : endSec - seconds * count;

    if (offline) {
      const candles = [];
      let price = syntheticPrice;
      for (let t = startSec - (startSec % seconds); t < endSec; t += seconds) {
        const open = price;
        price *= 1 + (Math.random() - 0.5) * 0.004;
        candles.push({ time: t, open, high: Math.max(open, price) * 1.001, low: Math.min(open, price) * 0.999, close: price, volume: 1 });
      }
      return Promise.resolve(candles.slice(-count));
    }

    return cached(`candles:${productId}:${granularity}:${startSec}:${endSec}:${count}`, 15, async () => {
      const body = await getJson(`/products/${encodeURIComponent(productId)}/candles`, {
        start: String(startSec), end: String(endSec), granularity, limit: String(count),
      });
      return (body.candles || [])
        .map((c) => ({ time: Number(c.start), open: num(c.open), high: num(c.high), low: num(c.low), close: num(c.close), volume: num(c.volume) }))
        .sort((a, b) => a.time - b.time);
    });
  };

  const getBook = (productId, { limit = 20 } = {}) => {
    if (offline) {
      const { bid, ask, time } = syntheticTicker(productId);
      const level = (price, i, dir) => ({ price: price + dir * i * 0.5, size: Number((Math.random() * 0.5).toFixed(6)) });
      const levels = Array.from({ length: limit }, (_, i) => i);
      return Promise.resolve({ productId, bids: levels.map((i) => level(bid, i, -1)), asks: levels.map((i) => level(ask, i, 1)), time, source: 'synthetic' });
    }
    return cached(`book:${productId}:${limit}`, 2, async () => {
      const body = await getJson('/product_book', { product_id: productId, limit: String(limit) });
      const side = (rows) => (rows || []).map((row) => ({ price: num(row.price), size: num(row.size) }));
      return {
        productId,
        bids: side(body.pricebook?.bids),
        asks: side(body.pricebook?.asks),
        mid: num(body.mid_market),
        spreadBps: num(body.spread_bps),
        time: body.pricebook?.time,
        source: 'coinbase',
      };
    });
  };

  return { getTicker, getCandles, getBook, offline };
};

module.exports = { createMarketData, GRANULARITY_SECONDS, MAX_CANDLES };
