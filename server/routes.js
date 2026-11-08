/**
 * REST API v1 routes. Mounted at /api/v1 by server/app.js.
 */

const express = require('express');
const { z } = require('zod');
const { ApiError, asyncRoute, validate } = require('./http');
const { GRANULARITY_SECONDS, MAX_CANDLES } = require('./market/coinbasePublic');

const productId = z.string().regex(/^[A-Z0-9]+-[A-Z0-9]+$/, 'Expected a product like BTC-USD');
const dateString = z.string().refine((value) => !Number.isNaN(Date.parse(value)), 'Expected an ISO date/time');
const brokerName = z.enum(['paper', 'coinbase']);

const orderSchema = (defaultProduct) => z.object({
  broker: brokerName.default('paper'),
  productId: productId.default(defaultProduct),
  side: z.enum(['BUY', 'SELL']),
  type: z.enum(['MARKET', 'LIMIT']),
  baseSize: z.number().positive().optional(),
  quoteSize: z.number().positive().optional(),
  limitPrice: z.number().positive().optional(),
  postOnly: z.boolean().default(false),
}).strict().superRefine((order, ctx) => {
  const issue = (path, message) => ctx.addIssue({ code: 'custom', path: [path], message });
  if (order.type === 'MARKET') {
    if ((order.baseSize === undefined) === (order.quoteSize === undefined)) {
      issue('baseSize', 'Market orders need exactly one of baseSize or quoteSize');
    }
    if (order.limitPrice !== undefined) issue('limitPrice', 'Market orders do not take a limit price');
  } else {
    if (order.baseSize === undefined) issue('baseSize', 'Limit orders need baseSize');
    if (order.limitPrice === undefined) issue('limitPrice', 'Limit orders need limitPrice');
    if (order.quoteSize !== undefined) issue('quoteSize', 'Limit orders are sized with baseSize');
  }
});

const marketRoutes = ({ config, market }) => {
  const router = express.Router();
  const product = z.object({ productId: productId.default(config.product) });

  router.get('/ticker', validate(product, 'query'), asyncRoute(async (req, res) => {
    res.json(await market.getTicker(req.valid.query.productId));
  }));

  router.get('/candles', validate(product.extend({
    granularity: z.enum(Object.keys(GRANULARITY_SECONDS)).default('FIVE_MINUTE'),
    start: z.coerce.number().int().positive().optional(),
    end: z.coerce.number().int().positive().optional(),
    limit: z.coerce.number().int().min(1).max(MAX_CANDLES).default(300),
  }), 'query'), asyncRoute(async (req, res) => {
    const { productId: id, ...options } = req.valid.query;
    res.json({ productId: id, granularity: options.granularity, candles: await market.getCandles(id, options) });
  }));

  router.get('/book', validate(product.extend({
    limit: z.coerce.number().int().min(1).max(100).default(20),
  }), 'query'), asyncRoute(async (req, res) => {
    res.json(await market.getBook(req.valid.query.productId, { limit: req.valid.query.limit }));
  }));

  return router;
};

const predictionRoutes = ({ predictions, ledger, auth }) => {
  const router = express.Router();

  router.get('/latest', (req, res, next) => {
    const latest = predictions.getLatest();
    if (!latest) {
      const { lastError } = predictions.status();
      return next(new ApiError(404, 'no_prediction', lastError
        ? `No prediction yet; last run failed: ${lastError.message}`
        : 'No prediction has been generated yet'));
    }
    return res.json(latest);
  });

  router.post('/run', auth, asyncRoute(async (req, res) => {
    res.json(await predictions.run());
  }));

  const filters = z.object({
    kind: z.string().max(40).optional(),
    signal: z.string().max(80).optional(),
    horizon: z.coerce.number().int().positive().optional(),
    status: z.enum(['pending', 'scored', 'unscorable']).optional(),
    scope: z.enum(['live', 'historical', 'all']).default('all'),
    modelVersion: z.string().max(80).optional(),
    from: dateString.optional(),
    to: dateString.optional(),
  });
  const toLedgerFilters = ({ scope, ...rest }) => ({
    ...rest,
    historical: scope === 'all' ? undefined : scope === 'historical',
  });

  router.get('/history', validate(filters.extend({
    limit: z.coerce.number().int().min(1).max(1000).default(100),
    offset: z.coerce.number().int().min(0).default(0),
  }), 'query'), (req, res) => {
    res.json(ledger.history(toLedgerFilters(req.valid.query)));
  });

  router.get('/accuracy', validate(filters.extend({
    groupBy: z.enum(['horizon', 'signal']).default('horizon'),
  }), 'query'), (req, res) => {
    res.json(ledger.accuracy(toLedgerFilters(req.valid.query)));
  });

  return router;
};

const orderRoutes = ({ config, brokers, store, events, auth }) => {
  const router = express.Router();

  const brokerFor = (name) => {
    if (brokers[name]) return brokers[name];
    if (name === 'coinbase') {
      throw new ApiError(403, 'live_trading_disabled', 'Live Coinbase trading is disabled (TRADING_MODE=paper)');
    }
    throw new ApiError(400, 'unknown_broker', `Unknown broker ${name}`);
  };

  const findOrder = (id) => {
    const order = store.get(id);
    if (!order) throw new ApiError(404, 'not_found', `Order ${id} not found`);
    return order;
  };

  const schema = orderSchema(config.product);

  router.post('/preview', auth, validate(schema), asyncRoute(async (req, res) => {
    const order = req.valid.body;
    res.json(await brokerFor(order.broker).preview(order));
  }));

  router.post('/', auth, validate(schema), asyncRoute(async (req, res) => {
    const order = req.valid.body;
    const placed = await brokerFor(order.broker).place(order);
    events.emit('order', placed);
    res.status(201).json(placed);
  }));

  router.get('/', validate(z.object({
    broker: brokerName.optional(),
    status: z.string().max(20).optional(),
    limit: z.coerce.number().int().min(1).max(1000).default(100),
  }), 'query'), asyncRoute(async (req, res) => {
    const { broker, ...filters } = req.valid.query;
    const names = broker ? [broker] : Object.keys(brokers);
    const lists = await Promise.all(names.map((name) => brokerFor(name).list(filters)));
    const orders = lists.flat()
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, filters.limit);
    res.json({ orders });
  }));

  router.get('/:id', asyncRoute(async (req, res) => {
    const order = findOrder(req.params.id);
    // Orders from a broker that is no longer enabled are still readable locally.
    res.json(brokers[order.broker] ? await brokers[order.broker].get(order.id) : order);
  }));

  router.delete('/:id', auth, asyncRoute(async (req, res) => {
    const order = findOrder(req.params.id);
    const cancelled = await brokerFor(order.broker).cancel(order.id);
    events.emit('order', cancelled);
    res.json(cancelled);
  }));

  return router;
};

const accountRoutes = ({ brokers, auth }) => {
  const router = express.Router();

  router.get('/', validate(z.object({ broker: brokerName.default('paper') }), 'query'), asyncRoute(async (req, res) => {
    const broker = brokers[req.valid.query.broker];
    if (!broker) throw new ApiError(403, 'live_trading_disabled', 'Live Coinbase trading is disabled (TRADING_MODE=paper)');
    res.json(await broker.account());
  }));

  router.post('/reset', auth, asyncRoute(async (req, res) => {
    res.json(await brokers.paper.reset());
  }));

  return router;
};

module.exports = { marketRoutes, predictionRoutes, orderRoutes, accountRoutes, orderSchema };
