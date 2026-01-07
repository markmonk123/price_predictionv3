/**
 * Node.js Backend Connector for Price Prediction Service
 * 
 * Features:
 * - Coinbase market data integration via @coinbase/coinbase-sdk
 * - FIX protocol client skeleton (optional)
 * - Feature normalization and forwarding to Python ML service
 * - Secure handling of API keys via environment variables
 */

const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
require('dotenv').config();

// Initialize Express app
const app = express();
const PORT = process.env.NODE_PORT || 3001;

// Middleware
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true }));

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', process.env.CORS_ORIGIN || '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Configuration from environment
const COINBASE_API_KEY = process.env.COINBASE_API_KEY;
const COINBASE_API_SECRET = process.env.COINBASE_API_SECRET;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const USE_TLS = process.env.USE_TLS === 'true';

// Coinbase SDK initialization (placeholder - actual implementation depends on SDK version)
let coinbaseClient = null;

/**
 * Initialize Coinbase SDK client
 * 
 * TODO: Configure actual Coinbase SDK authentication
 * See: https://docs.cdp.coinbase.com/coinbase-sdk/docs/
 */
function initializeCoinbaseClient() {
  if (!COINBASE_API_KEY || !COINBASE_API_SECRET) {
    console.warn('WARNING: COINBASE_API_KEY or COINBASE_API_SECRET not set in environment');
    console.warn('Coinbase features will be disabled. Add keys to .env file.');
    return null;
  }

  try {
    // Example initialization - adjust based on actual SDK API
    // const { Coinbase } = require('@coinbase/coinbase-sdk');
    // coinbaseClient = new Coinbase({
    //   apiKey: COINBASE_API_KEY,
    //   apiSecret: COINBASE_API_SECRET,
    // });
    
    console.log('Coinbase client initialized (stub - implement actual SDK integration)');
    return {}; // Stub client object
  } catch (error) {
    console.error('Failed to initialize Coinbase client:', error.message);
    return null;
  }
}

/**
 * FIX Protocol Client Skeleton
 * 
 * Optional: Use quickfix library for FIX connectivity to exchanges
 * Uncomment and configure if FIX connectivity is required
 */
function initializeFIXClient() {
  // TODO: Implement FIX client if needed
  // Example using quickfix (optional dependency):
  /*
  try {
    const quickfix = require('quickfix');
    
    // Configure FIX initiator
    const settings = new quickfix.SessionSettings('path/to/fix_config.cfg');
    const storeFactory = new quickfix.FileStoreFactory(settings);
    const logFactory = new quickfix.FileLogFactory(settings);
    
    const application = {
      onCreate: (sessionID) => {
        console.log('FIX session created:', sessionID.toString());
      },
      onLogon: (sessionID) => {
        console.log('FIX session logged on:', sessionID.toString());
      },
      onLogout: (sessionID) => {
        console.log('FIX session logged out:', sessionID.toString());
      },
      toAdmin: (message, sessionID) => {},
      toApp: (message, sessionID) => {},
      fromAdmin: (message, sessionID) => {},
      fromApp: (message, sessionID) => {
        // Handle incoming FIX messages
        console.log('Received FIX message:', message.toString());
      }
    };
    
    const initiator = new quickfix.SocketInitiator(
      application,
      storeFactory,
      settings,
      logFactory
    );
    
    initiator.start();
    console.log('FIX client started');
    
    return initiator;
  } catch (error) {
    console.error('FIX client initialization failed:', error.message);
    return null;
  }
  */
  
  console.log('FIX client not configured (optional feature)');
  return null;
}

/**
 * Normalize features for ML model input
 * 
 * @param {Object} marketData - Raw market data from exchange
 * @returns {Array<number>} - Normalized feature vector
 */
function normalizeFeatures(marketData) {
  // Example feature extraction and normalization
  // Customize based on your model's expected inputs
  
  const features = [
    parseFloat(marketData.price || 0),
    parseFloat(marketData.volume || 0),
    parseFloat(marketData.bid || 0),
    parseFloat(marketData.ask || 0),
    parseFloat(marketData.high || 0),
    parseFloat(marketData.low || 0),
  ];
  
  // Basic normalization (replace with actual preprocessing logic)
  const normalized = features.map(f => {
    if (isNaN(f)) return 0;
    return f;
  });
  
  return normalized;
}

/**
 * Forward features to Python ML service for prediction
 * 
 * @param {Array<number>} features - Feature vector
 * @param {string} modelName - Model name to use
 * @returns {Promise<Object>} - Prediction response
 */
async function getPrediction(features, modelName = 'ensemble_hybrid') {
  const url = `${ML_SERVICE_URL}/predict`;
  
  try {
    const response = await axios.post(url, {
      features: features,
      model_name: modelName
    }, {
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 10000 // 10 second timeout
    });
    
    return response.data;
  } catch (error) {
    console.error('ML service prediction failed:', error.message);
    throw new Error(`Prediction service error: ${error.message}`);
  }
}

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  const health = {
    status: 'ok',
    timestamp: new Date().toISOString(),
    coinbase_connected: coinbaseClient !== null,
    ml_service_url: ML_SERVICE_URL,
  };
  
  res.json(health);
});

/**
 * Get market data endpoint (stub - implement actual Coinbase API calls)
 */
app.get('/market/:symbol', async (req, res) => {
  const symbol = req.params.symbol;
  
  try {
    // TODO: Implement actual Coinbase API call
    // Example stub response
    const marketData = {
      symbol: symbol,
      price: 50000 + Math.random() * 1000,
      volume: 1000000 + Math.random() * 100000,
      bid: 49900,
      ask: 50100,
      high: 51000,
      low: 49000,
      timestamp: new Date().toISOString()
    };
    
    res.json(marketData);
  } catch (error) {
    console.error('Failed to fetch market data:', error.message);
    res.status(500).json({ error: 'Failed to fetch market data' });
  }
});

/**
 * Prediction endpoint - normalizes data and forwards to ML service
 */
app.post('/predict', async (req, res) => {
  try {
    const { marketData, modelName } = req.body;
    
    if (!marketData) {
      return res.status(400).json({ error: 'marketData is required' });
    }
    
    // Normalize features
    const features = normalizeFeatures(marketData);
    
    // Get prediction from ML service
    const prediction = await getPrediction(features, modelName);
    
    res.json({
      success: true,
      input: marketData,
      features: features,
      prediction: prediction
    });
    
  } catch (error) {
    console.error('Prediction request failed:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Batch prediction endpoint
 */
app.post('/predict/batch', async (req, res) => {
  try {
    const { marketDataBatch, modelName } = req.body;
    
    if (!Array.isArray(marketDataBatch)) {
      return res.status(400).json({ error: 'marketDataBatch must be an array' });
    }
    
    // Normalize all samples
    const featuresBatch = marketDataBatch.map(normalizeFeatures);
    
    // Get batch prediction from ML service
    const url = `${ML_SERVICE_URL}/predict`;
    const response = await axios.post(url, {
      features: featuresBatch,
      model_name: modelName || 'ensemble_hybrid'
    }, {
      timeout: 30000 // 30 second timeout for batch
    });
    
    res.json({
      success: true,
      count: marketDataBatch.length,
      predictions: response.data
    });
    
  } catch (error) {
    console.error('Batch prediction failed:', error.message);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * ML service health check proxy
 */
app.get('/ml-service/health', async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`, {
      timeout: 5000
    });
    res.json(response.data);
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});

// Initialize clients
coinbaseClient = initializeCoinbaseClient();
const fixClient = initializeFIXClient();

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('Price Prediction Node Backend');
  console.log('='.repeat(60));
  console.log(`Server running on port ${PORT}`);
  console.log(`ML Service URL: ${ML_SERVICE_URL}`);
  console.log(`TLS Enabled: ${USE_TLS}`);
  console.log(`Coinbase Client: ${coinbaseClient ? 'Initialized' : 'Not configured'}`);
  console.log(`FIX Client: ${fixClient ? 'Initialized' : 'Not configured'}`);
  console.log('='.repeat(60));
  console.log('\nEndpoints:');
  console.log(`  GET  /health - Service health check`);
  console.log(`  GET  /market/:symbol - Get market data`);
  console.log(`  POST /predict - Single prediction`);
  console.log(`  POST /predict/batch - Batch predictions`);
  console.log(`  GET  /ml-service/health - ML service health`);
  console.log('='.repeat(60));
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  process.exit(0);
});
