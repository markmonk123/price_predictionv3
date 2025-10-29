/**
 * Prediction Service for Bitcoin Trading Platform
 * Interfaces with Python prediction model
 */

const { PythonShell } = require('python-shell');
const path = require('path');
const NodeCache = require('node-cache');
const fixService = require('./fixService');
const { logMessage, logError } = require('../utils/logger');

// Cache for storing latest predictions
const predictionCache = new NodeCache({ stdTTL: 300 }); // 5 minutes TTL

/**
 * Run Bitcoin prediction model
 */
const runPredictionModel = async () => {
  try {
    logMessage('Running Bitcoin prediction model...');

    // Get latest market data from FIX service (real data)
    const marketData = await fixService.getMarketData('BTC/USD');
    
    if (marketData.simulated) {
      logMessage('Warning: Using simulated market data for prediction');
    }

    // Options for Python shell
    const options = {
      mode: 'json',
      scriptPath: path.join(__dirname, '../python'),
      args: [
        '--price', marketData.last.toString(),
        '--volume', marketData.volume.toString(),
        '--time', new Date().toISOString()
      ]
    };

    // Run Python prediction script
    return new Promise((resolve, reject) => {
      PythonShell.run('run_prediction.py', options, (err, results) => {
        if (err) {
          logError('Error running prediction model:', err);
          return reject(err);
        }

        if (!results || !results.length) {
          return reject(new Error('No prediction results returned'));
        }

        const prediction = results[0];
        logMessage(`Prediction model result: ${JSON.stringify(prediction)}`);

        // Cache prediction result with market data context
        predictionCache.set('latest', {
          ...prediction,
          timestamp: new Date(),
          price: marketData.last,
          simulated: false  // Using real market data for prediction
        });

        resolve(prediction);
      });
    });
  } catch (error) {
    logError('Error in prediction model execution:', error);
    throw error;
  }
};

/**
 * Get latest prediction
 */
const getLatestPrediction = async () => {
  const cachedPrediction = predictionCache.get('latest');

  if (cachedPrediction) {
    return cachedPrediction;
  }

  // If no cached prediction, run the model to generate a new one
  try {
    return await runPredictionModel();
  } catch (error) {
    logError('Unable to generate prediction:', error);
    throw new Error('Prediction service unavailable - please ensure FIX service and Python environment are running');
  }
};

/**
 * Schedule regular predictions
 */
const schedulePredictions = (io) => {
  // Run prediction every minute
  const predictionInterval = 60000; // 1 minute

  setInterval(async () => {
    try {
      const prediction = await runPredictionModel();

      // Broadcast to all connected clients
      if (io) {
        io.emit('predictionData', prediction);
      }
    } catch (error) {
      logError('Error in scheduled prediction:', error);
    }
  }, predictionInterval);

  logMessage(`Scheduled predictions every ${predictionInterval/1000} seconds`);
};

/**
 * Get historical predictions
 */
const getHistoricalPredictions = async (timeframe = '1h') => {
  try {
    // In a real implementation, this would query a database
    // For now, return empty array since we don't have historical predictions stored
    logMessage('Historical predictions not yet implemented - requires database storage');
    return [];
  } catch (error) {
    logError('Error getting historical predictions:', error);
    throw error;
  }
};

module.exports = {
  runPredictionModel,
  getLatestPrediction,
  schedulePredictions,
  getHistoricalPredictions
};
