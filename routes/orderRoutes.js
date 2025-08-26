
/**
 * Order API Routes for Bitcoin Trading Platform
 *
 * Simulated endpoints now use actual BTC/USD market price from Coinbase API.
 */

const express = require('express');
const fixService = require('../services/fixService');
const { logError, logMessage } = require('../utils/logger');

const router = express.Router();

/**
 * @route   POST /api/orders
 * @desc    Submit a new order via FIX
 * @access  Private (would require auth middleware in production)
 */
router.post('/', async (req, res) => {
  try {
    const { symbol, side, orderType, quantity, price, timeInForce } = req.body;

    // Validate required fields
    if (!symbol || !side || !orderType || !quantity) {
      return res.status(400).json({ error: 'Missing required order parameters' });
    }

    // Validate order type and price
    if (orderType === 'LIMIT' && !price) {
      return res.status(400).json({ error: 'Price is required for limit orders' });
    }

    // Send order via FIX
    const result = await fixService.sendOrder({
      symbol,
      side,
      orderType,
      quantity: parseFloat(quantity),
      price: price ? parseFloat(price) : undefined,
      timeInForce: timeInForce || 'GTC'
    });

    logMessage(`Order submitted: ${JSON.stringify(result)}`);
    res.json(result);
  } catch (error) {
    logError('Error submitting order:', error);
    res.status(500).json({ error: 'Failed to submit order' });
  }
});

const fetchMarketPrice = async () => {
  try {
    const resp = await fetch('https://api.exchange.coinbase.com/products/BTC-USD/ticker');
    if (!resp.ok) throw new Error('Failed to fetch market price');
    const data = await resp.json();
    return parseFloat(data.price);
  } catch (e) {
    // fallback to 60000 if API fails
    return 60000;
  }
};

router.get('/:orderId', async (req, res) => {
  try {
    const { orderId } = req.params;
    const marketPrice = await fetchMarketPrice();
    const simulatedStatus = {
      orderId,
      status: ['NEW', 'PARTIALLY_FILLED', 'FILLED'][Math.floor(Math.random() * 3)],
      filledQuantity: Math.random() * 10,
      averagePrice: marketPrice,
      symbol: 'BTC/USD',
      timestamp: new Date()
    };
    res.json(simulatedStatus);
  } catch (error) {
    logError('Error getting order status:', error);
    res.status(500).json({ error: 'Failed to get order status' });
  }
});

router.get('/', async (req, res) => {
  try {
    const marketPrice = await fetchMarketPrice();
    const simulatedOrders = [];
    for (let i = 0; i < 10; i++) {
      simulatedOrders.push({
        orderId: `ORD${Date.now() - i * 1000000}`,
        status: ['NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED'][Math.floor(Math.random() * 4)],
        symbol: 'BTC/USD',
        side: Math.random() > 0.5 ? 'BUY' : 'SELL',
        orderType: Math.random() > 0.3 ? 'LIMIT' : 'MARKET',
        quantity: Math.random() * 10,
        filledQuantity: Math.random() * 5,
        price: marketPrice,
        averagePrice: marketPrice,
        timestamp: new Date(Date.now() - i * 3600000)
      });
    }
    res.json(simulatedOrders);
  } catch (error) {
    logError('Error getting orders:', error);
    res.status(500).json({ error: 'Failed to get orders' });
  }
});

/**
 * @route   DELETE /api/orders/:orderId
 * @desc    Cancel an order
 * @access  Private (would require auth middleware in production)
 */
router.delete('/:orderId', async (req, res) => {
  try {
    const { orderId } = req.params;

    // In a real implementation, this would send a cancel request via FIX
    // For demo purposes, return success
    res.json({
      success: true,
      message: `Order ${orderId} canceled successfully`,
      timestamp: new Date()
    });
  } catch (error) {
    logError('Error canceling order:', error);
    res.status(500).json({ error: 'Failed to cancel order' });
  }
});

module.exports = router;
