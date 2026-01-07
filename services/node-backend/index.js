/**
 * Node.js Backend Service
 * 
 * This service demonstrates:
 * - Connecting to Coinbase via @coinbase/coinbase-sdk for market data
 * - Placeholder for FIX protocol integration (optional dependency)
 * - Feature normalization and forwarding to Python ML model service
 * - Secure environment-based configuration
 * 
 * Security notes:
 * - All secrets loaded from environment variables via dotenv
 * - HTTPS support for production model service calls
 * - Input validation and error handling
 * - No hardcoded credentials
 * 
 * Usage:
 *   1. Create .env file with required configuration
 *   2. npm install
 *   3. npm start
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { Coinbase } = require('@coinbase/coinbase-sdk');

// Configuration from environment
const PORT = process.env.PORT || 3000;
const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || 'http://localhost:8000';
const COINBASE_API_KEY = process.env.COINBASE_API_KEY;
const COINBASE_API_SECRET = process.env.COINBASE_API_SECRET;
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || 'http://localhost:3001').split(',');

// Initialize Express
const app = express();
app.use(express.json());
app.use(cors({
  origin: ALLOWED_ORIGINS,
  credentials: true
}));

// Coinbase client (initialize on startup if credentials provided)
let coinbaseClient = null;

/**
 * Initialize Coinbase SDK connection
 */
function initializeCoinbase() {
  if (!COINBASE_API_KEY || !COINBASE_API_SECRET) {
    console.warn('⚠️  COINBASE_API_KEY or COINBASE_API_SECRET not set');
    console.warn('⚠️  Coinbase integration will not be available');
    console.warn('⚠️  Add credentials to .env file to enable');
    return;
  }

  try {
    // TODO: Initialize Coinbase SDK client
    // This is a placeholder - actual SDK usage depends on SDK version
    // Refer to @coinbase/coinbase-sdk documentation for correct initialization
    coinbaseClient = Coinbase.configureFromJson({
      apiKey: COINBASE_API_KEY,
      apiSecret: COINBASE_API_SECRET
    });
    
    console.log('✓ Coinbase SDK initialized');
  } catch (error) {
    console.error('✗ Failed to initialize Coinbase SDK:', error.message);
  }
}

/**
 * Initialize FIX protocol connection (optional)
 * 
 * This is a placeholder for FIX integration.
 * Requires quickfix package (optional dependency).
 */
function initializeFIX() {
  try {
    // Check if quickfix is available
    const quickfix = require('quickfix');
    
    // TODO: Configure FIX connection
    // This is a skeleton - actual implementation depends on FIX server config
    console.log('✓ FIX library available (not configured)');
    console.log('  To use FIX: implement FIX initiator with your broker\'s settings');
    
    // Example FIX configuration structure:
    // const settings = new quickfix.SessionSettings();
    // const initiator = new quickfix.SocketInitiator(application, settings);
    // initiator.start();
    
  } catch (error) {
    console.log('ℹ FIX library not installed (optional)');
    console.log('  Install with: npm install quickfix');
  }
}

/**
 * Normalize market data features for ML model
 * 
 * @param {Object} marketData - Raw market data from Coinbase
 * @returns {Array<Array<number>>} - Normalized feature array
 */
function normalizeFeatures(marketData) {
  // TODO: Implement actual feature engineering based on your model
  // This is a placeholder that creates dummy features
  
  const features = [
    parseFloat(marketData.price || 0),
    parseFloat(marketData.volume || 0),
    parseFloat(marketData.bid || 0),
    parseFloat(marketData.ask || 0),
    parseFloat(marketData.high || 0),
    parseFloat(marketData.low || 0),
  ];
  
  // Return as 2D array (batch of 1 sample)
  return [features];
}

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    services: {
      coinbase: coinbaseClient !== null,
      model_service: MODEL_SERVICE_URL
    },
    timestamp: new Date().toISOString()
  });
});

/**
 * Get market data from Coinbase
 * 
 * Example endpoint to fetch current market data
 */
app.get('/market/:symbol', async (req, res) => {
  const { symbol } = req.params;
  
  if (!coinbaseClient) {
    return res.status(503).json({
      error: 'Coinbase not configured',
      message: 'Set COINBASE_API_KEY and COINBASE_API_SECRET in .env'
    });
  }
  
  try {
    // TODO: Implement actual Coinbase API call
    // This is a placeholder
    const marketData = {
      symbol: symbol.toUpperCase(),
      price: Math.random() * 50000 + 10000,
      volume: Math.random() * 1000000,
      bid: Math.random() * 50000 + 10000,
      ask: Math.random() * 50000 + 10000,
      high: Math.random() * 50000 + 10000,
      low: Math.random() * 50000 + 10000,
      timestamp: new Date().toISOString()
    };
    
    res.json(marketData);
  } catch (error) {
    console.error('Error fetching market data:', error);
    res.status(500).json({
      error: 'Failed to fetch market data',
      message: error.message
    });
  }
});

/**
 * Prediction endpoint
 * 
 * Fetches market data, normalizes features, and forwards to ML model
 */
app.post('/predict', async (req, res) => {
  try {
    const { symbol, marketData, model_name } = req.body;
    
    // Use provided market data or fetch from Coinbase
    let data = marketData;
    
    if (!data && symbol) {
      if (!coinbaseClient) {
        return res.status(503).json({
          error: 'Coinbase not configured',
          message: 'Provide marketData in request body or configure Coinbase'
        });
      }
      
      // TODO: Fetch from Coinbase
      data = {
        symbol: symbol.toUpperCase(),
        price: Math.random() * 50000 + 10000,
        volume: Math.random() * 1000000,
        bid: Math.random() * 50000 + 10000,
        ask: Math.random() * 50000 + 10000,
        high: Math.random() * 50000 + 10000,
        low: Math.random() * 50000 + 10000,
      };
    }
    
    if (!data) {
      return res.status(400).json({
        error: 'Missing data',
        message: 'Provide either symbol or marketData in request body'
      });
    }
    
    // Normalize features
    const features = normalizeFeatures(data);
    
    // Forward to ML model service
    const modelResponse = await axios.post(
      `${MODEL_SERVICE_URL}/predict`,
      {
        features,
        model_name: model_name || 'ensemble_balanced_stacking'
      },
      {
        timeout: 10000,
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
    
    // Return combined response
    res.json({
      marketData: data,
      prediction: modelResponse.data,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Prediction error:', error.message);
    
    if (error.response) {
      // Model service returned an error
      res.status(error.response.status).json({
        error: 'Model service error',
        message: error.response.data.detail || error.message
      });
    } else if (error.request) {
      // Model service not reachable
      res.status(503).json({
        error: 'Model service unavailable',
        message: 'Cannot connect to ML model service',
        service_url: MODEL_SERVICE_URL
      });
    } else {
      // Other error
      res.status(500).json({
        error: 'Internal error',
        message: error.message
      });
    }
  }
});

/**
 * Example: Subscribe to Coinbase WebSocket ticker
 * 
 * This demonstrates how to set up real-time market data streaming
 */
function subscribeToCoinbaseTicker(symbol = 'BTC-USD') {
  if (!coinbaseClient) {
    console.log('⚠️  Cannot subscribe to ticker: Coinbase not configured');
    return;
  }
  
  console.log(`📡 Setting up Coinbase ticker subscription for ${symbol}`);
  console.log('   (This is a placeholder - implement based on SDK version)');
  
  // TODO: Implement WebSocket subscription based on Coinbase SDK
  // Example structure:
  // coinbaseClient.websocket.subscribe(['ticker'], [symbol], (message) => {
  //   console.log('Ticker update:', message);
  //   // Process and potentially forward to model
  // });
}

// Initialize services on startup
console.log('🚀 Starting Node.js backend service...');
initializeCoinbase();
initializeFIX();

// Optional: Subscribe to ticker on startup
// subscribeToCoinbaseTicker('BTC-USD');

// Start server
app.listen(PORT, () => {
  console.log(`✓ Server running on port ${PORT}`);
  console.log(`✓ Model service URL: ${MODEL_SERVICE_URL}`);
  console.log('');
  console.log('Environment configuration:');
  console.log(`  COINBASE_API_KEY: ${COINBASE_API_KEY ? '***set***' : 'NOT SET'}`);
  console.log(`  COINBASE_API_SECRET: ${COINBASE_API_SECRET ? '***set***' : 'NOT SET'}`);
  console.log(`  MODEL_SERVICE_URL: ${MODEL_SERVICE_URL}`);
  console.log('');
  console.log('Available endpoints:');
  console.log(`  GET  /health`);
  console.log(`  GET  /market/:symbol`);
  console.log(`  POST /predict`);
  console.log('');
  console.log('Ready to accept requests! 🎯');
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
