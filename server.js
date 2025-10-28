const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const path = require('path');
const fixService = require('./services/fixService');
const predictionService = require('./services/predictionService');
const coinbaseService = require('./services/coinbaseService');
const dataFlowService = require('./services/dataFlowService');
const { logMessage, logError } = require('./utils/logger');

// Environment variables
require('dotenv').config();
const PORT = process.env.PORT || 5000;

// Initialize Express app
const app = express();
app.use(cors());
app.use(express.json());

// Create HTTP server
const server = http.createServer(app);

// Initialize Socket.IO
const io = socketIo(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  logMessage(`New client connected: ${socket.id}`);

  // Send initial data to new clients using secure data pipeline
  const sendInitialData = async () => {
    try {
      // Get data from multiple sources
      const [prediction, marketData, coinbasePrice] = await Promise.all([
        predictionService.getLatestPrediction(),
        fixService.getMarketData(),
        coinbaseService.getBitcoinPrice()
      ]);

      socket.emit('predictionData', prediction);
      socket.emit('marketData', marketData);
      socket.emit('coinbaseData', coinbasePrice);
    } catch (err) {
      logError('Error sending initial data:', err);
    }
  };

  sendInitialData();

  // Start streaming Coinbase price updates to this client
  const stopPriceStream = coinbaseService.streamPriceUpdates('BTC-USD', (priceData) => {
    socket.emit('coinbasePriceUpdate', priceData);
  }).catch(err => {
    logError('Error starting price stream:', err);
  });

  socket.on('disconnect', () => {
    logMessage(`Client disconnected: ${socket.id}`);
    // Clean up streams
    if (stopPriceStream && typeof stopPriceStream.then === 'function') {
      stopPriceStream.then(cleanup => cleanup && cleanup());
    }
  });
});

// API Routes
app.use('/api/market', require('./routes/marketRoutes'));
app.use('/api/predictions', require('./routes/predictionRoutes'));
app.use('/api/orders', require('./routes/orderRoutes'));
app.use('/api/fix', require('./routes/fixRoutes'));
app.use('/api/coinbase', require('./routes/coinbaseRoutes'));

// Health check endpoint
app.get('/api/health', (req, res) => {
  const health = {
    status: 'ok',
    timestamp: new Date(),
    services: {
      fix: fixService.isConnected || false,
      dataFlow: dataFlowService.globalBufferPool.getStats()
    }
  };
  res.json(health);
});

// Schedule regular market data updates via FIX
fixService.initializeFixSession();
fixService.scheduleMarketDataUpdates(io);

// Initialize Coinbase SDK
coinbaseService.initializeCoinbase().catch(err => {
  logError('Failed to initialize Coinbase SDK:', err);
});

// Schedule regular prediction model runs
predictionService.schedulePredictions(io);

// Serve static assets in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, 'client/build')));

  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'client/build', 'index.html'));
  });
}

// Start the server
server.listen(PORT, () => {
  logMessage(`Server running on port ${PORT}`);
});

// Handle unexpected errors
process.on('unhandledRejection', (err) => {
  logError('Unhandled Rejection:', err);
});
