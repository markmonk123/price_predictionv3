/**
 * Prediction Service for Bitcoin Trading Platform
 * Interfaces with Python prediction model
 */

const { PythonShell } = require('python-shell');
const path = require('path');
const NodeCache = require('node-cache');
const fixService = require('./fixService');
const { logMessage, logError } = require('../utils/logger');
const { isDemoMode } = require('../utils/runtimeMode');

// Cache for storing latest predictions
const predictionCache = new NodeCache({ stdTTL: 300 }); // 5 minutes TTL

/**
 * Run Bitcoin prediction model
 */
const runPredictionModel = async () => {
  try {
    logMessage('Running Bitcoin prediction model...');

    // Get latest market data
    const marketData = await fixService.getMarketData('BTC/USD');

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
    if (isDemoMode) {
      options.args.push('--allow-synthetic');
    }

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

        if (prediction && prediction.error) {
          return reject(new Error(`Python model error: ${prediction.error}`));
        }

        if (!isDemoMode && prediction && prediction.simulated) {
          return reject(new Error('Prediction payload flagged as simulated while DEMO_MODE is disabled'));
        }

        logMessage(`Prediction model result: ${JSON.stringify(prediction)}`);

        // Cache prediction result
        predictionCache.set('latest', {
          ...prediction,
          timestamp: new Date(),
          price: marketData.last
        });

        resolve(prediction);
      });
    });
  } catch (error) {
    logError('Error in prediction model execution:', error);
    if (isDemoMode) {
      return simulatePrediction();
    }
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

  if (isDemoMode) {
    // If no cached prediction, generate a demo prediction.
    return simulatePrediction();
  }

  // In non-demo mode, attempt a real model run instead of synthetic fallback.
  return runPredictionModel();
};

/**
 * Simulate prediction for demo purposes
 */
const simulatePrediction = async () => {
  try {
    const marketData = await fixService.getMarketData('BTC/USD');

    // Calculate random prediction probabilities
    const increaseProbability = Math.random() * 0.5 + 0.25; // 25-75%
    const decreaseProbability = Math.random() * 0.5 + 0.25; // 25-75%
    const noChangeProbability = 1 - increaseProbability - decreaseProbability;

    // Determine predicted direction based on highest probability
    let predictedDirection;
    if (increaseProbability > decreaseProbability && increaseProbability > noChangeProbability) {
      predictedDirection = 1; // Increase
    } else if (decreaseProbability > increaseProbability && decreaseProbability > noChangeProbability) {
      predictedDirection = -1; // Decrease
    } else {
      predictedDirection = 0; // No change
    }

    // Create prediction object
    const prediction = {
      price: marketData.last,
      timestamp: new Date(),
      predicted_direction: predictedDirection,
      increase_probability: increaseProbability,
      decrease_probability: decreaseProbability,
      no_change_probability: noChangeProbability,
      confidence: Math.max(increaseProbability, decreaseProbability, noChangeProbability),
      timeframe: '1 minute',
      threshold: 0.0005, // 0.05%
      simulated: true // Flag to indicate this is simulated
    };

    // Cache the prediction
    predictionCache.set('latest', prediction);

    return prediction;
  } catch (error) {
    logError('Error simulating prediction:', error);
    throw error;
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
    if (!isDemoMode) {
      throw new Error('Historical predictions are demo-only until a real persistence source is implemented');
    }

    // In a real implementation, this would query a database
    // For demo purposes, we'll generate simulated historical data

    const numDataPoints = timeframe === '1h' ? 60 : 
                          timeframe === '1d' ? 24 * 60 : 
                          timeframe === '1w' ? 7 * 24 * 60 : 60;

    const currentPrice = (await fixService.getMarketData('BTC/USD')).last;
    const predictions = [];

    // Generate simulated historical predictions
    for (let i = 0; i < numDataPoints; i++) {
      const timestamp = new Date(Date.now() - (i * 60 * 1000)); // Go back i minutes
      const priceDelta = (Math.random() - 0.5) * 100; // Random price change
      const price = currentPrice - priceDelta * i / 10;

      const increaseProbability = Math.random() * 0.5 + 0.25;
      const decreaseProbability = Math.random() * 0.5 + 0.25;
      const noChangeProbability = 1 - increaseProbability - decreaseProbability;

      let predictedDirection;
      if (increaseProbability > decreaseProbability && increaseProbability > noChangeProbability) {
        predictedDirection = 1;
      } else if (decreaseProbability > increaseProbability && decreaseProbability > noChangeProbability) {
        predictedDirection = -1;
      } else {
        predictedDirection = 0;
      }

      predictions.push({
        price,
        timestamp,
        predicted_direction: predictedDirection,
        increase_probability: increaseProbability,
        decrease_probability: decreaseProbability,
        no_change_probability: noChangeProbability,
        confidence: Math.max(increaseProbability, decreaseProbability, noChangeProbability),
        timeframe: '1 minute',
        threshold: 0.0005,
        simulated: true
      });
    }

    return predictions.reverse(); // Most recent first
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
