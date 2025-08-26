// Timeout middleware: returns 504 if request takes more than 860 seconds
function timeoutHandler(req, res, next) {
  const timeout = setTimeout(() => {
    if (!res.headersSent) {
      res.status(504).json({ error: 'Request timed out after 860 seconds' });
    }
  }, 860000);
  res.on('finish', () => clearTimeout(timeout));
  next();
}

/**
 * Market Data API Routes for Bitcoin Trading Platform
 */

const express = require('express');
const fixService = require('../services/fixService');
const { logError } = require('../utils/logger');

const router = express.Router();
router.use(timeoutHandler);

/**
 * @route   GET /api/market/data
 * @desc    Get latest market data
 * @access  Public
 */
router.get('/data', async (req, res) => {
  try {
    const { symbol = 'BTC/USD' } = req.query;
    const marketData = await fixService.getMarketData(symbol);
    res.json(marketData);
  } catch (error) {
    logError('Error getting market data:', error);
    res.status(500).json({ error: 'Failed to fetch market data' });
  }
});

/**
 * @route   POST /api/market/subscribe
 * @desc    Subscribe to market data via FIX
 * @access  Private (would require auth middleware in production)
 */
router.post('/subscribe', async (req, res) => {
  try {
    const { symbol = 'BTC/USD' } = req.body;
    const success = await fixService.subscribeToMarketData(symbol);

    if (success) {
      res.json({ success: true, message: `Subscribed to ${symbol} market data` });
    } else {
      res.status(400).json({ success: false, message: 'Failed to subscribe to market data' });
    }
  } catch (error) {
    logError('Error subscribing to market data:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
