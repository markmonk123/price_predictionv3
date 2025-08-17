const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const path = require('path');
const { PythonShell } = require('python-shell');
const fixService = require('./services/fixService');
const predictionService = require('./services/predictionService');
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

  // Send initial data to new clients
  predictionService.getLatestPrediction()
    .then(prediction => {
      socket.emit('predictionData', prediction);
    })
    .catch(err => logError('Error sending initial prediction data:', err));

  // Send market data updates
  fixService.getMarketData()
    .then(marketData => {
      socket.emit('marketData', marketData);
    })
    .catch(err => logError('Error sending market data:', err));

  socket.on('disconnect', () => {
    logMessage(`Client disconnected: ${socket.id}`);
  });
});

// API Routes
app.use('/api/market', require('./routes/marketRoutes'));
app.use('/api/predictions', require('./routes/predictionRoutes'));
app.use('/api/orders', require('./routes/orderRoutes'));
app.use('/api/fix', require('./routes/fixRoutes'));

// Schedule regular market data updates via FIX
fixService.initializeFixSession();
fixService.scheduleMarketDataUpdates(io);

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
