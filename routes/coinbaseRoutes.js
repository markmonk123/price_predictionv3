/**
 * Coinbase API Routes for Bitcoin Trading Platform
 */

const express = require('express');
const coinbaseService = require('../services/coinbaseService');
const { logError } = require('../utils/logger');

const router = express.Router();

/**
 * @route   GET /api/coinbase/price
 * @desc    Get current Bitcoin price from Coinbase
 * @access  Public
 */
router.get('/price', async (req, res) => {
  try {
    const { symbol = 'BTC-USD' } = req.query;
    const priceData = await coinbaseService.getBitcoinPrice(symbol);
    res.json(priceData);
  } catch (error) {
    logError('Error getting Coinbase price:', error);
    res.status(500).json({ error: 'Failed to fetch price from Coinbase' });
  }
});

/**
 * @route   GET /api/coinbase/historical
 * @desc    Get historical price data from Coinbase
 * @access  Public
 */
router.get('/historical', async (req, res) => {
  try {
    const { symbol = 'BTC-USD', granularity = 3600, limit = 100 } = req.query;
    const historicalData = await coinbaseService.getHistoricalPrices(
      symbol,
      parseInt(granularity),
      parseInt(limit)
    );
    res.json(historicalData);
  } catch (error) {
    logError('Error getting historical data from Coinbase:', error);
    res.status(500).json({ error: 'Failed to fetch historical data' });
  }
});

/**
 * @route   GET /api/coinbase/balance
 * @desc    Get account balance
 * @access  Private (would require auth middleware in production)
 */
router.get('/balance', async (req, res) => {
  try {
    const balance = await coinbaseService.getAccountBalance();
    res.json(balance);
  } catch (error) {
    logError('Error getting account balance:', error);
    res.status(500).json({ error: 'Failed to fetch account balance' });
  }
});

/**
 * @route   GET /api/coinbase/pair/:symbol
 * @desc    Get trading pair information
 * @access  Public
 */
router.get('/pair/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const pairInfo = await coinbaseService.getTradingPairInfo(symbol);
    res.json(pairInfo);
  } catch (error) {
    logError('Error getting trading pair info:', error);
    res.status(500).json({ error: 'Failed to fetch trading pair info' });
  }
});

/**
 * @route   GET /api/coinbase/depth
 * @desc    Get market depth (order book)
 * @access  Public
 */
router.get('/depth', async (req, res) => {
  try {
    const { symbol = 'BTC-USD', level = 2 } = req.query;
    const marketDepth = await coinbaseService.getMarketDepth(symbol, parseInt(level));
    res.json(marketDepth);
  } catch (error) {
    logError('Error getting market depth:', error);
    res.status(500).json({ error: 'Failed to fetch market depth' });
  }
});

module.exports = router;
