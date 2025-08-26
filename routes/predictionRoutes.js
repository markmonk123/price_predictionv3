/**
 * Prediction API Routes for Bitcoin Trading Platform
 */

const express = require('express');
const predictionService = require('../services/predictionService');
const { logError } = require('../utils/logger');

const router = express.Router();

/**
 * @route   GET /api/predictions/latest
 * @desc    Get latest prediction
 * @access  Public
 */
router.get('/latest', async (req, res) => {
  try {
    const prediction = await predictionService.getLatestPrediction();
    res.json(prediction);
  } catch (error) {
    logError('Error getting latest prediction:', error);
    res.status(500).json({ error: 'Failed to fetch prediction data' });
  }
});

/**
 * @route   GET /api/predictions/history
 * @desc    Get historical predictions
 * @access  Public
 */
router.get('/history', async (req, res) => {
  try {
    const { timeframe = '1h' } = req.query;
    const allowedTimeframes = ['1h', '1d', '1w'];
    if (!allowedTimeframes.includes(timeframe)) {
      return res.status(400).json({ error: 'Invalid timeframe. Allowed: 1h, 1d, 1w' });
    }
    const predictions = await predictionService.getHistoricalPredictions(timeframe);
    res.json(predictions);
  } catch (error) {
    logError('Error getting historical predictions:', error);
    res.status(500).json({ error: 'Failed to fetch historical prediction data' });
  }
});

/**
 * @route   POST /api/predictions/run
 * @desc    Run prediction model manually
 * @access  Private (would require auth middleware in production)
 */
router.post('/run', async (req, res) => {
  try {
    // Example: If your model expects certain input, validate here
    // For demonstration, let's say it expects an optional 'features' array
    const { features } = req.body || {};
    if (features && (!Array.isArray(features) || features.length === 0)) {
      return res.status(400).json({ error: 'Invalid features: must be a non-empty array if provided.' });
    }
    const prediction = await predictionService.runPredictionModel(features);
    res.json(prediction);
  } catch (error) {
    logError('Error running prediction model:', error);
    res.status(500).json({ error: 'Failed to run prediction model' });
  }
});

module.exports = router;
