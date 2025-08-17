/**
 * FIX Protocol API Routes for Bitcoin Trading Platform
 */

const express = require('express');
const fixService = require('../services/fixService');
const { logError } = require('../utils/logger');

const router = express.Router();

/**
 * @route   GET /api/fix/status
 * @desc    Get FIX connection status
 * @access  Private (would require auth middleware in production)
 */
router.get('/status', async (req, res) => {
  try {
    // In a real implementation, this would check actual connection status
    // For demo purposes, simulate status
    res.json({
      connected: true,
      sessionId: 'BITCOIN_PREDICTION_CLIENT-EXCHANGE',
      uptime: Math.floor(Math.random() * 86400), // Random uptime in seconds
      lastHeartbeat: new Date(),
      messagesReceived: Math.floor(Math.random() * 1000),
      messagesSent: Math.floor(Math.random() * 800)
    });
  } catch (error) {
    logError('Error getting FIX status:', error);
    res.status(500).json({ error: 'Failed to get FIX connection status' });
  }
});

/**
 * @route   POST /api/fix/reconnect
 * @desc    Force FIX session reconnection
 * @access  Private (would require auth middleware in production)
 */
router.post('/reconnect', async (req, res) => {
  try {
    const result = await fixService.initializeFixSession();

    if (result) {
      res.json({ success: true, message: 'FIX session reconnected successfully' });
    } else {
      res.status(500).json({ success: false, message: 'Failed to reconnect FIX session' });
    }
  } catch (error) {
    logError('Error reconnecting FIX session:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
