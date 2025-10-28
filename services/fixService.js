/**
 * FIX Protocol Service for Bitcoin Trading Platform
 * Implements FIX protocol for market data and order execution
 * Enhanced with secure data flows and streaming capabilities
 */

const NodeCache = require('node-cache');
const crypto = require('crypto');
const { logMessage, logError } = require('../utils/logger');
const { createSecureDataPipeline, createDataStream, createDataCollector } = require('./dataFlowService');

// Cache for storing latest market data
const marketDataCache = new NodeCache({ stdTTL: 300 }); // 5 minutes TTL

// Message queue for secure FIX message processing
const messageQueue = [];
const messageQueueMaxSize = 1000;

// Security: Message authentication
const generateMessageMAC = (message) => {
  const hmac = crypto.createHmac('sha256', process.env.FIX_SECRET || 'default-secret');
  hmac.update(JSON.stringify(message));
  return hmac.digest('hex');
};

const verifyMessageMAC = (message, mac) => {
  const expectedMAC = generateMessageMAC(message);
  return crypto.timingSafeEqual(Buffer.from(mac), Buffer.from(expectedMAC));
};

// FIX Message Types
const FIX_MSG_TYPES = {
  MARKET_DATA_REQUEST: 'V',
  MARKET_DATA_SNAPSHOT: 'W',
  NEW_ORDER_SINGLE: 'D',
  EXECUTION_REPORT: '8',
  HEARTBEAT: '0',
  LOGON: 'A',
  LOGOUT: '5'
};

// FIX Session configuration
let fixSession = null;
let isConnected = false;
let messageStream = null;

/**
 * Create a FIX message with secure formatting
 */
const createSecureFIXMessage = (msgType, fields) => {
  const message = {
    msgType: msgType,
    timestamp: new Date().toISOString(),
    fields: fields,
    sequenceNum: Date.now(),
    checksum: null
  };

  // Calculate checksum for message integrity
  message.checksum = generateMessageMAC(message);
  
  return message;
};

/**
 * Validate incoming FIX message
 */
const validateFIXMessage = (message) => {
  try {
    // Check required fields
    if (!message.msgType || !message.timestamp || !message.checksum) {
      return { valid: false, error: 'Missing required FIX fields' };
    }

    // Verify message integrity
    const messageWithoutChecksum = { ...message };
    delete messageWithoutChecksum.checksum;
    
    if (!verifyMessageMAC(messageWithoutChecksum, message.checksum)) {
      return { valid: false, error: 'Message integrity check failed' };
    }

    // Check message age (prevent replay attacks)
    const messageAge = Date.now() - new Date(message.timestamp).getTime();
    if (messageAge > 60000) { // 1 minute timeout
      return { valid: false, error: 'Message too old' };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: error.message };
  }
};

/**
 * Initialize FIX session with counterparty
 * Enhanced with secure connection handling
 */
const initializeFixSession = async () => {
  try {
    logMessage('Initializing enhanced FIX session with secure data flows...');

    // For this implementation, we'll simulate a secure FIX session
    // In production, this would connect to an actual FIX gateway
    
    fixSession = {
      id: crypto.randomUUID(),
      senderCompID: 'BITCOIN_PREDICTION_CLIENT',
      targetCompID: 'EXCHANGE',
      heartbeatInterval: 30,
      createdAt: new Date()
    };

    // Initialize secure message stream
    messageStream = createDataStream([]);
    
    isConnected = true;
    logMessage(`FIX session initialized successfully: ${fixSession.id}`);
    
    // Start heartbeat mechanism
    startHeartbeat();
    
    return true;
  } catch (error) {
    logError('Error initializing FIX session:', error);
    return false;
  }
};

/**
 * Start heartbeat mechanism
 */
const startHeartbeat = () => {
  if (!fixSession) return;
  
  setInterval(() => {
    if (isConnected) {
      const heartbeat = createSecureFIXMessage(FIX_MSG_TYPES.HEARTBEAT, {
        testReqID: Date.now().toString()
      });
      logMessage('Sending FIX heartbeat');
      processOutgoingMessage(heartbeat);
    }
  }, fixSession.heartbeatInterval * 1000);
};

/**
 * Process incoming FIX messages with secure validation
 */
const processIncomingMessage = (message) => {
  try {
    // Validate message security
    const validation = validateFIXMessage(message);
    if (!validation.valid) {
      logError('FIX message validation failed:', validation.error);
      return;
    }

    // Add to message queue with size limit
    if (messageQueue.length >= messageQueueMaxSize) {
      messageQueue.shift(); // Remove oldest message
    }
    messageQueue.push(message);

    const msgType = message.msgType;

    switch (msgType) {
      case FIX_MSG_TYPES.MARKET_DATA_SNAPSHOT:
        processMarketDataSnapshot(message);
        break;
      case FIX_MSG_TYPES.EXECUTION_REPORT:
        processExecutionReport(message);
        break;
      case FIX_MSG_TYPES.HEARTBEAT:
        logMessage('Received FIX heartbeat');
        break;
      default:
        logMessage(`Unhandled message type: ${msgType}`);
    }
  } catch (error) {
    logError('Error processing incoming FIX message:', error);
  }
};

/**
 * Process outgoing FIX messages
 */
const processOutgoingMessage = (message) => {
  try {
    // In a real implementation, this would send to FIX gateway
    logMessage(`Processing outgoing FIX message: ${message.msgType}`);
  } catch (error) {
    logError('Error processing outgoing FIX message:', error);
  }
};

/**
 * Process market data snapshot FIX message with secure data pipeline
 */
const processMarketDataSnapshot = async (message) => {
  try {
    // Extract relevant fields from secure message
    const { fields } = message;
    const symbol = fields.symbol || 'BTC/USD';
    const timestamp = message.timestamp;

    // Extract price data
    const bidPrice = fields.bidPrice || 0;
    const askPrice = fields.askPrice || 0;
    const lastPrice = fields.lastPrice || 0;
    const volume = fields.volume || 0;

    // Calculate mid price
    const midPrice = (parseFloat(bidPrice) + parseFloat(askPrice)) / 2;

    // Create market data object
    const marketData = {
      symbol,
      timestamp: new Date(timestamp),
      bid: parseFloat(bidPrice),
      ask: parseFloat(askPrice),
      last: parseFloat(lastPrice),
      mid: midPrice,
      volume: parseFloat(volume),
      receivedAt: new Date(),
      secure: true
    };

    // Process through secure data pipeline
    const pipeline = createSecureDataPipeline({
      validationRules: {
        required: ['symbol', 'timestamp', 'bid', 'ask', 'last'],
        types: {
          bid: 'number',
          ask: 'number',
          last: 'number',
          volume: 'number'
        },
        ranges: {
          bid: { min: 0, max: 1000000 },
          ask: { min: 0, max: 1000000 },
          last: { min: 0, max: 1000000 }
        }
      },
      compress: true,
      strictValidation: true
    });

    const dataSource = createDataStream([marketData]);
    const dataSink = createDataCollector();

    await pipeline.processData(dataSource, dataSink);
    const [validatedData] = dataSink.getData();

    // Cache the validated market data
    if (validatedData) {
      marketDataCache.set(symbol, validatedData);
      logMessage(`Updated secure market data for ${symbol}: ${JSON.stringify(validatedData)}`);
    }
  } catch (error) {
    logError('Error processing market data snapshot:', error);
  }
};

/**
 * Process execution report FIX message with secure validation
 */
const processExecutionReport = (message) => {
  try {
    // Extract relevant fields from secure message
    const { fields } = message;
    
    // Create execution report object with validation
    const execReport = {
      orderID: fields.orderID,
      execType: fields.execType,
      ordStatus: fields.ordStatus,
      symbol: fields.symbol,
      side: fields.side === '1' ? 'BUY' : 'SELL',
      orderQty: parseFloat(fields.orderQty || 0),
      price: parseFloat(fields.price || 0),
      transactTime: new Date(fields.transactTime),
      receivedAt: new Date(),
      secure: true
    };

    logMessage(`Received secure execution report: ${JSON.stringify(execReport)}`);

    // Update order status in database (would be implemented elsewhere)
    // orderService.updateOrderStatus(orderID, execReport);
  } catch (error) {
    logError('Error processing execution report:', error);
  }
};

/**
 * Subscribe to market data via FIX with secure messaging
 */
const subscribeToMarketData = (symbol) => {
  try {
    if (!isConnected || !fixSession) {
      logError('Cannot subscribe to market data: FIX session not established');
      return false;
    }

    logMessage(`Subscribing to market data for ${symbol}...`);

    // Create secure market data request message
    const message = createSecureFIXMessage(FIX_MSG_TYPES.MARKET_DATA_REQUEST, {
      mdReqID: Date.now().toString(),
      subscriptionRequestType: '1', // Subscribe
      marketDepth: '1', // Top of book
      mdUpdateType: '1', // Full refresh
      symbol: symbol,
      entryTypes: [0, 1, 2, 4, 7] // Bid, Offer, Trade, Opening Price, etc.
    });

    // Send the message
    processOutgoingMessage(message);

    logMessage(`Market data subscription request sent for ${symbol}`);
    return true;
  } catch (error) {
    logError('Error subscribing to market data:', error);
    return false;
  }
};

/**
 * Send a new order via FIX with secure validation
 */
const sendOrder = async (orderData) => {
  try {
    if (!isConnected || !fixSession) {
      throw new Error('FIX session not established');
    }

    const { symbol, side, orderType, quantity, price, timeInForce } = orderData;

    // Validate order data
    if (!symbol || !side || !orderType || !quantity) {
      throw new Error('Missing required order fields');
    }

    // Create secure new order single message
    const message = createSecureFIXMessage(FIX_MSG_TYPES.NEW_ORDER_SINGLE, {
      clOrdID: crypto.randomUUID(),
      symbol: symbol,
      side: side === 'BUY' ? '1' : '2',
      transactTime: new Date().toISOString(),
      orderQty: quantity.toString(),
      ordType: orderType === 'MARKET' ? '1' : '2',
      price: price ? price.toString() : undefined,
      timeInForce: {
        'DAY': '0',
        'GTC': '1',
        'IOC': '3',
        'FOK': '4'
      }[timeInForce] || '0'
    });

    // Send the message
    processOutgoingMessage(message);

    logMessage(`Secure order sent: ${JSON.stringify(orderData)}`);
    return { success: true, orderId: message.fields.clOrdID };
  } catch (error) {
    logError('Error sending order:', error);
    throw error;
  }
};

/**
 * Get latest market data
 */
const getMarketData = async (symbol = 'BTC/USD') => {
  const cachedData = marketDataCache.get(symbol);

  if (cachedData) {
    return cachedData;
  }

  // If no cached data, simulate market data for demo purposes
  return simulateMarketData(symbol);
};

/**
 * Simulate market data for demo purposes
 */
const simulateMarketData = (symbol) => {
  const basePrice = 60000 + (Math.random() * 10000 - 5000);
  const spread = basePrice * 0.0005; // 0.05% spread

  const marketData = {
    symbol,
    timestamp: new Date(),
    bid: basePrice - spread/2,
    ask: basePrice + spread/2,
    last: basePrice,
    mid: basePrice,
    volume: 10 + Math.random() * 100,
    receivedAt: new Date(),
    simulated: true // Flag to indicate this is simulated data
  };

  // Cache the simulated data
  marketDataCache.set(symbol, marketData);

  return marketData;
};

/**
 * Schedule regular market data updates
 */
const scheduleMarketDataUpdates = (io) => {
  // Update every 5 seconds for demo purposes
  const updateInterval = 5000;

  setInterval(async () => {
    try {
      const marketData = await getMarketData('BTC/USD');

      // Broadcast to all connected clients
      if (io) {
        io.emit('marketData', marketData);
      }
    } catch (error) {
      logError('Error in scheduled market data update:', error);
    }
  }, updateInterval);

  logMessage(`Scheduled market data updates every ${updateInterval/1000} seconds`);
};

module.exports = {
  initializeFixSession,
  subscribeToMarketData,
  sendOrder,
  getMarketData,
  scheduleMarketDataUpdates
};
