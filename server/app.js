/**
 * Builds the Express app. Kept separate from server.js (which only does startup)
 * so tests can run the full API on an ephemeral port with injected dependencies.
 */

const express = require('express');
const cors = require('cors');
const { requireToken, notFound, errorHandler } = require('./http');
const { marketRoutes, predictionRoutes, orderRoutes, accountRoutes } = require('./routes');

const createApp = ({ config, market, brokers, store, ledger, predictions, events }) => {
  const app = express();
  app.disable('x-powered-by');
  app.use(cors({ origin: config.clientOrigins }));
  app.use(express.json({ limit: '100kb' }));

  const auth = requireToken(config);
  const api = express.Router();

  api.get('/health', async (req, res) => {
    let marketStatus;
    try {
      const ticker = await market.getTicker(config.product);
      marketStatus = { ok: true, source: ticker.source, time: ticker.time };
    } catch (error) {
      marketStatus = { ok: false, source: market.offline ? 'synthetic' : 'coinbase', error: error.message };
    }
    res.json({
      status: 'ok',
      tradingMode: config.tradingMode,
      brokers: Object.keys(brokers),
      liveTradingEnabled: Boolean(brokers.coinbase),
      product: config.product,
      maxOrderUsd: config.maxOrderUsd,
      paperFeeBps: config.paper.feeBps,
      authRequired: Boolean(config.apiToken),
      market: marketStatus,
      model: predictions.status(),
      ledger: { available: ledger.exists() },
    });
  });

  api.use('/market', marketRoutes({ config, market }));
  api.use('/predictions', predictionRoutes({ predictions, ledger, auth }));
  api.use('/orders', orderRoutes({ config, brokers, store, events, auth }));
  api.use('/account', accountRoutes({ brokers, auth }));

  app.use('/api/v1', api);
  app.use('/api', notFound);
  app.use(errorHandler);
  return app;
};

module.exports = { createApp };
