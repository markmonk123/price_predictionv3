/**
 * Coinbase Advanced Trade broker (REAL MONEY). Only constructed when TRADING_MODE=live.
 * Docs: https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/orders/
 *
 * Safety rails:
 *   - Every order is previewed first; any preview error rejects the order.
 *   - Notional above MAX_ORDER_USD is refused server-side, before Coinbase is called.
 *   - client_order_id is a fresh UUID; Coinbase returns the existing order if it is reused,
 *     so a retried request can never double-place.
 *   - Every order is also recorded in the local order ledger.
 */

const crypto = require('crypto');
const { ApiError } = require('../http');
const { OPEN_STATUSES } = require('./orderStore');

const HOST = 'api.coinbase.com';
const PREFIX = '/api/v3/brokerage';

const num = (value) => (value === undefined || value === null || value === '' ? null : Number(value));
const str = (value) => (value === undefined || value === null ? undefined : String(value));

const defaultJwt = (options) => {
  // Loaded lazily so paper mode never needs the SDK.
  // eslint-disable-next-line global-require
  const { generateJwt } = require('@coinbase/cdp-sdk/auth');
  return generateJwt(options);
};

const orderConfiguration = (order) => {
  if (order.type === 'MARKET') {
    return {
      market_market_ioc: order.baseSize !== undefined
        ? { base_size: str(order.baseSize) }
        : { quote_size: str(order.quoteSize) },
    };
  }
  return {
    limit_limit_gtc: {
      base_size: str(order.baseSize),
      limit_price: str(order.limitPrice),
      post_only: Boolean(order.postOnly),
    },
  };
};

const createCoinbaseBroker = ({
  keyId, keySecret, maxOrderUsd, store, market, fetchImpl = fetch, jwt = defaultJwt,
}) => {
  const request = async (method, path, { query, body } = {}) => {
    // The JWT "uri" claim covers method + host + path, never the query string.
    const token = await jwt({
      apiKeyId: keyId, apiKeySecret: keySecret, requestMethod: method, requestHost: HOST, requestPath: PREFIX + path,
    });
    const qs = query ? `?${query}` : '';
    let response;
    try {
      response = await fetchImpl(`https://${HOST}${PREFIX}${path}${qs}`, {
        method,
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', Accept: 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(15000),
      });
    } catch (error) {
      throw new ApiError(502, 'broker_unreachable', `Coinbase unreachable: ${error.message}`);
    }
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch { payload = { raw: text.slice(0, 500) }; }
    if (!response.ok) {
      throw new ApiError(502, 'broker_error', payload?.message || payload?.error || `Coinbase returned ${response.status}`, payload);
    }
    return payload;
  };

  /** Estimated USD notional, used for the MAX_ORDER_USD guard. */
  const estimateNotional = async (order) => {
    if (order.quoteSize !== undefined) return order.quoteSize;
    if (order.type === 'LIMIT') return order.baseSize * order.limitPrice;
    const ticker = await market.getTicker(order.productId);
    const price = order.side === 'BUY' ? ticker.ask : ticker.bid;
    if (!price) throw new ApiError(503, 'market_unavailable', 'No live price to size the order');
    return order.baseSize * price;
  };

  const enforceLimit = async (order) => {
    const notional = await estimateNotional(order);
    if (notional > maxOrderUsd) {
      throw new ApiError(422, 'max_order_exceeded',
        `Order notional ~$${notional.toFixed(2)} exceeds MAX_ORDER_USD ($${maxOrderUsd})`, { notional, maxOrderUsd });
    }
    return notional;
  };

  const preview = async (order) => {
    await enforceLimit(order);
    const body = await request('POST', '/orders/preview', {
      body: { product_id: order.productId, side: order.side, order_configuration: orderConfiguration(order) },
    });
    return {
      broker: 'coinbase',
      productId: order.productId,
      side: order.side,
      type: order.type,
      price: num(body.est_average_filled_price) ?? order.limitPrice ?? null,
      baseSize: num(body.base_size),
      quoteSize: num(body.quote_size),
      commissionTotal: num(body.commission_total),
      orderTotal: num(body.order_total),
      bestBid: num(body.best_bid),
      bestAsk: num(body.best_ask),
      slippage: num(body.slippage),
      previewId: body.preview_id,
      errs: body.errs || [],
      warnings: body.warning || [],
    };
  };

  const place = async (order) => {
    const checked = await preview(order);
    if (checked.errs.length) {
      throw new ApiError(422, 'order_rejected', 'Coinbase preview rejected the order', { errs: checked.errs });
    }
    const id = crypto.randomUUID();
    const result = await request('POST', '/orders', {
      body: {
        client_order_id: id,
        product_id: order.productId,
        side: order.side,
        order_configuration: orderConfiguration(order),
        preview_id: checked.previewId,
      },
    });
    const common = {
      id,
      broker: 'coinbase',
      clientOrderId: id,
      productId: order.productId,
      side: order.side,
      type: order.type,
      baseSize: order.baseSize ?? null,
      quoteSize: order.quoteSize ?? null,
      limitPrice: order.limitPrice ?? null,
      postOnly: order.postOnly,
    };
    if (!result?.success) {
      const reason = result?.error_response?.message || result?.error_response?.new_order_failure_reason || 'Order not created';
      store.insert({ ...common, status: 'FAILED', reason });
      throw new ApiError(422, 'order_rejected', reason, result?.error_response);
    }
    return store.insert({ ...common, externalId: result.success_response.order_id, status: 'PENDING' });
  };

  const applyRemote = (local, remote) => store.update(local.id, {
    status: remote.status,
    filledSize: num(remote.filled_size) ?? 0,
    averageFilledPrice: num(remote.average_filled_price),
    totalFees: num(remote.total_fees) ?? 0,
  });

  const getOwn = (id) => {
    const order = store.get(id);
    if (!order || order.broker !== 'coinbase') throw new ApiError(404, 'not_found', `Coinbase order ${id} not found`);
    return order;
  };

  const get = async (id) => {
    const order = getOwn(id);
    if (!order.externalId) return order;
    const body = await request('GET', `/orders/historical/${encodeURIComponent(order.externalId)}`);
    return body?.order ? applyRemote(order, body.order) : order;
  };

  const list = async (filters = {}) => {
    const orders = store.list({ ...filters, broker: 'coinbase' });
    const open = orders.filter((o) => o.externalId && OPEN_STATUSES.includes(o.status));
    if (!open.length) return orders;
    const query = new URLSearchParams(open.map((o) => ['order_ids', o.externalId])).toString();
    const body = await request('GET', '/orders/historical/batch', { query });
    const byExternal = new Map((body?.orders || []).map((remote) => [remote.order_id, remote]));
    return orders.map((o) => (byExternal.has(o.externalId) ? applyRemote(o, byExternal.get(o.externalId)) : o));
  };

  const cancel = async (id) => {
    const order = getOwn(id);
    if (!order.externalId || !OPEN_STATUSES.includes(order.status)) {
      throw new ApiError(409, 'not_cancellable', `Order is ${order.status}`);
    }
    const body = await request('POST', '/orders/batch_cancel', { body: { order_ids: [order.externalId] } });
    const result = body?.results?.[0];
    if (!result?.success) {
      throw new ApiError(409, 'not_cancellable', result?.failure_reason || 'Cancel was not accepted');
    }
    return store.update(id, { status: 'CANCEL_QUEUED' });
  };

  const account = async () => {
    const body = await request('GET', '/accounts', { query: 'limit=250' });
    return {
      broker: 'coinbase',
      balances: (body?.accounts || [])
        .map((a) => ({ currency: a.currency, available: num(a.available_balance?.value) ?? 0, hold: num(a.hold?.value) ?? 0 }))
        .filter((b) => b.available > 0 || b.hold > 0 || ['USD', 'BTC'].includes(b.currency)),
    };
  };

  return { name: 'coinbase', preview, place, cancel, get, list, account };
};

module.exports = { createCoinbaseBroker, orderConfiguration };
