/**
 * Server startup: wires dependencies, Socket.IO and the background loops.
 * The HTTP API itself lives in server/app.js.
 */

const http = require('http');
const path = require('path');
const fs = require('fs');
const { EventEmitter } = require('events');
const express = require('express');
const { Server } = require('socket.io');

const { loadConfig } = require('./server/config');
const { createApp } = require('./server/app');
const { createMarketData } = require('./server/market/coinbasePublic');
const { createOrderStore } = require('./server/brokers/orderStore');
const { createPaperBroker } = require('./server/brokers/paperBroker');
const { createCoinbaseBroker } = require('./server/brokers/coinbaseBroker');
const { createLedger } = require('./server/predictions/ledger');
const { createPredictionService } = require('./services/predictionService');
const { logMessage, logError } = require('./utils/logger');

const config = loadConfig();
const events = new EventEmitter();
const market = createMarketData({ offline: config.offline });
const store = createOrderStore(config.tradingDbPath);
const brokers = { paper: createPaperBroker({ store, market, ...config.paper }) };
if (config.tradingMode === 'live') {
  brokers.coinbase = createCoinbaseBroker({
    keyId: config.coinbase.keyId,
    keySecret: config.coinbase.keySecret,
    maxOrderUsd: config.maxOrderUsd,
    store,
    market,
  });
}
const ledger = createLedger(config.predictionDbPath);
const predictions = createPredictionService({ market, config, events });

const app = createApp({ config, market, brokers, store, ledger, predictions, events });

// Serve the built client (client/dist) in production.
if (config.production && fs.existsSync(config.clientBuildDir)) {
  app.use(express.static(config.clientBuildDir));
  app.get('*', (req, res) => res.sendFile(path.join(config.clientBuildDir, 'index.html')));
}

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: config.clientOrigins } });

let lastTicker = null;
io.on('connection', (socket) => {
  if (lastTicker) socket.emit('ticker', lastTicker);
  const latest = predictions.getLatest();
  if (latest) socket.emit('prediction', latest);
});

events.on('order', (order) => io.emit('order', order));
events.on('prediction', (prediction) => io.emit('prediction', prediction));

// Poll the ticker, broadcast it, and let the paper broker fill resting limit orders.
setInterval(async () => {
  try {
    lastTicker = await market.getTicker(config.product);
    io.emit('ticker', lastTicker);
    brokers.paper.onTicker(lastTicker).forEach((order) => events.emit('order', order));
  } catch (error) {
    logError('Ticker update failed:', error);
  }
}, config.tickerIntervalMs);

predictions.schedule(config.predictionIntervalMs);

server.listen(config.port, config.host, () => {
  logMessage(`Server running on http://${config.host}:${config.port} (API at /api/v1)`);
  logMessage(`Trading mode: ${config.tradingMode.toUpperCase()}; brokers: ${Object.keys(brokers).join(', ')}; `
    + `market data: ${config.offline ? 'synthetic (DEMO_OFFLINE)' : 'Coinbase'}`);
  if (!config.apiToken) logMessage('API_TOKEN not set: order endpoints are open to anything that can reach this host.');
});

process.on('unhandledRejection', (error) => logError('Unhandled rejection:', error));
