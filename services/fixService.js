/**
 * FIX Protocol Service for Bitcoin Trading Platform
 * Implements FIX protocol for market data and order execution
 */

const QuickFIX = require('quick-fix');
const NodeCache = require('node-cache');
const { logMessage, logError } = require('../utils/logger');

// Cache for storing latest market data
const marketDataCache = new NodeCache({ stdTTL: 300 }); // 5 minutes TTL

// FIX Message Types
const FIX_MSG_TYPES = {
  MARKET_DATA_REQUEST: 'V',
  MARKET_DATA_SNAPSHOT: 'W',
  NEW_ORDER_SINGLE: 'D',
  EXECUTION_REPORT: '8'
};

// FIX Session configuration
let fixSession = null;
let isConnected = false;

/**
 * Initialize FIX session with counterparty
 */
const initializeFixSession = async () => {
  try {
    logMessage('Initializing FIX session...');

    const fixConfig = {
      SenderCompID: 'BITCOIN_PREDICTION_CLIENT',
      TargetCompID: 'EXCHANGE',
      HeartBtInt: 30,  // Heartbeat interval in seconds
      ReconnectInterval: 5  // Reconnect interval in seconds
    };

    // Create FIX initiator
    const fixInitiator = new QuickFIX.Initiator(
      new QuickFIX.FileStoreFactory('./fixlogs'),
      new QuickFIX.FileLogFactory('./fixlogs'),
      new QuickFIX.Dictionary(fixConfig)
    );

    // Define application callbacks
    const fixApplication = {
      onCreate: (sessionID) => {
        logMessage(`FIX session created: ${sessionID}`);
        fixSession = sessionID;
      },
      onLogon: (sessionID) => {
        logMessage(`FIX session logged on: ${sessionID}`);
        isConnected = true;
        // Subscribe to Bitcoin market data after logon
        subscribeToMarketData('BTC/USD');
      },
      onLogout: (sessionID) => {
        logMessage(`FIX session logged out: ${sessionID}`);
        isConnected = false;
      },
      toAdmin: (message, sessionID) => {
        // Called before admin message is sent
        logMessage(`Sending admin message: ${message.toString()}`);
      },
      fromAdmin: (message, sessionID) => {
        // Called when admin message is received
        logMessage(`Received admin message: ${message.toString()}`);
        return true;
      },
      toApp: (message, sessionID) => {
        // Called before app message is sent
        logMessage(`Sending app message: ${message.toString()}`);
      },
      fromApp: (message, sessionID) => {
        // Called when app message is received
        logMessage(`Received app message: ${message.toString()}`);
        processIncomingMessage(message);
        return true;
      }
    };

    // Start the FIX initiator
    fixInitiator.start();

    logMessage('FIX session initialized successfully');
    return true;
  } catch (error) {
    logError('Error initializing FIX session:', error);
    return false;
  }
};

/**
 * Process incoming FIX messages
 */
const processIncomingMessage = (message) => {
  try {
    const msgType = message.getHeader().getField(35); // MsgType field

    switch (msgType) {
      case FIX_MSG_TYPES.MARKET_DATA_SNAPSHOT:
        processMarketDataSnapshot(message);
        break;
      case FIX_MSG_TYPES.EXECUTION_REPORT:
        processExecutionReport(message);
        break;
      default:
        logMessage(`Unhandled message type: ${msgType}`);
    }
  } catch (error) {
    logError('Error processing incoming FIX message:', error);
  }
};

/**
 * Process market data snapshot FIX message
 */
const processMarketDataSnapshot = (message) => {
  try {
    // Extract relevant fields
    const symbol = message.getField(55); // Symbol
    const timestamp = message.getField(52); // Sending time

    // Extract price data from different FIX fields
    const bidPrice = message.getField(132); // BidPx
    const askPrice = message.getField(133); // OfferPx
    const lastPrice = message.getField(31); // LastPx
    const volume = message.getField(32); // LastQty

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
      receivedAt: new Date()
    };

    // Cache the market data
    marketDataCache.set(symbol, marketData);

    logMessage(`Updated market data for ${symbol}: ${JSON.stringify(marketData)}`);

    // If this is simulated data, we'll need to broadcast it to connected clients
    // This would be handled in scheduleMarketDataUpdates
  } catch (error) {
    logError('Error processing market data snapshot:', error);
  }
};

/**
 * Process execution report FIX message
 */
const processExecutionReport = (message) => {
  try {
    // Extract relevant fields
    const orderID = message.getField(37); // OrderID
    const execType = message.getField(150); // ExecType
    const ordStatus = message.getField(39); // OrdStatus
    const symbol = message.getField(55); // Symbol
    const side = message.getField(54); // Side (1=Buy, 2=Sell)
    const orderQty = message.getField(38); // OrderQty
    const price = message.getField(44); // Price
    const transactTime = message.getField(60); // TransactTime

    // Create execution report object
    const execReport = {
      orderID,
      execType,
      ordStatus,
      symbol,
      side: side === '1' ? 'BUY' : 'SELL',
      orderQty: parseFloat(orderQty),
      price: parseFloat(price),
      transactTime: new Date(transactTime),
      receivedAt: new Date()
    };

    logMessage(`Received execution report: ${JSON.stringify(execReport)}`);

    // Update order status in database (would be implemented elsewhere)
    // orderService.updateOrderStatus(orderID, execReport);
  } catch (error) {
    logError('Error processing execution report:', error);
  }
};

/**
 * Subscribe to market data via FIX
 */
const subscribeToMarketData = (symbol) => {
  try {
    if (!isConnected || !fixSession) {
      logError('Cannot subscribe to market data: FIX session not established');
      return false;
    }

    logMessage(`Subscribing to market data for ${symbol}...`);

    // Create market data request message
    const message = new QuickFIX.Message();
    message.getHeader().setField(35, FIX_MSG_TYPES.MARKET_DATA_REQUEST); // MsgType

    // Set message fields
    message.setField(262, Date.now().toString()); // MDReqID (unique request ID)
    message.setField(263, '1'); // SubscriptionRequestType (1=Subscribe)
    message.setField(264, '1'); // MarketDepth (1=Top of book)
    message.setField(265, '1'); // MDUpdateType (1=Full refresh)

    // Add symbol
    message.setField(55, symbol); // Symbol

    // Request specific market data entry types
    // 0=Bid, 1=Offer, 2=Trade, 3=Index Value, 4=Opening Price, etc.
    const entryTypes = [0, 1, 2, 4, 7];
    message.setField(267, entryTypes.length); // NoMDEntryTypes

    entryTypes.forEach((type, index) => {
      message.setField(269 + index, type.toString()); // MDEntryType
    });

    // Send the message
    QuickFIX.Session.sendToTarget(message, fixSession);

    logMessage(`Market data subscription request sent for ${symbol}`);
    return true;
  } catch (error) {
    logError('Error subscribing to market data:', error);
    return false;
  }
};

/**
 * Send a new order via FIX
 */
const sendOrder = async (orderData) => {
  try {
    if (!isConnected || !fixSession) {
      throw new Error('FIX session not established');
    }

    const { symbol, side, orderType, quantity, price, timeInForce } = orderData;

    // Create new order single message
    const message = new QuickFIX.Message();
    message.getHeader().setField(35, FIX_MSG_TYPES.NEW_ORDER_SINGLE); // MsgType

    // Set message fields
    message.setField(11, Date.now().toString()); // ClOrdID (unique client order ID)
    message.setField(55, symbol); // Symbol
    message.setField(54, side === 'BUY' ? '1' : '2'); // Side (1=Buy, 2=Sell)
    message.setField(60, new Date().toISOString()); // TransactTime
    message.setField(38, quantity.toString()); // OrderQty
    message.setField(40, orderType === 'MARKET' ? '1' : '2'); // OrdType (1=Market, 2=Limit)

    // Add price for limit orders
    if (orderType === 'LIMIT' && price) {
      message.setField(44, price.toString()); // Price
    }

    // Set time in force
    const tifMap = {
      'DAY': '0',
      'GTC': '1', // Good Till Cancel
      'IOC': '3', // Immediate or Cancel
      'FOK': '4'  // Fill or Kill
    };
    message.setField(59, tifMap[timeInForce] || '0'); // TimeInForce

    // Send the message
    QuickFIX.Session.sendToTarget(message, fixSession);

    logMessage(`Order sent: ${JSON.stringify(orderData)}`);
    return { success: true, orderId: message.getField(11) };
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

  // Fetch real market data from exchange
  try {
    return await fetchRealMarketData(symbol);
  } catch (error) {
    logError('Failed to fetch real market data, service unavailable', error);
    throw new Error('Market data service unavailable');
  }
};

/**
 * Fetch real Bitcoin market data from Coinbase API (FIX alternative)
 * Uses REST API as fallback when FIX connection is not available
 */
const fetchRealMarketData = async (symbol) => {
  try {
    // Use Coinbase Pro REST API to get real-time market data
    const https = require('https');
    const symbolMap = {
      'BTC/USD': 'BTC-USD',
      'BTCUSD': 'BTC-USD',
      'BTC-USD': 'BTC-USD'
    };
    
    const product = symbolMap[symbol] || 'BTC-USD';
    
    // Fetch ticker data
    const tickerPromise = new Promise((resolve, reject) => {
      https.get(`https://api.exchange.coinbase.com/products/${product}/ticker`, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(e);
          }
        });
      }).on('error', reject);
    });
    
    const tickerData = await tickerPromise;
    
    // Parse real market data
    const marketData = {
      symbol,
      timestamp: new Date(tickerData.time),
      bid: parseFloat(tickerData.bid) || 0,
      ask: parseFloat(tickerData.ask) || 0,
      last: parseFloat(tickerData.price),
      mid: (parseFloat(tickerData.bid) + parseFloat(tickerData.ask)) / 2,
      volume: parseFloat(tickerData.volume) || 0,
      receivedAt: new Date(),
      simulated: false // Real data from exchange
    };
    
    // Cache the market data
    marketDataCache.set(symbol, marketData);
    
    logMessage(`Fetched real market data for ${symbol}: $${marketData.last.toFixed(2)}`);
    
    return marketData;
  } catch (error) {
    logError('Error fetching real market data:', error);
    throw error;
  }
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

/**
 * Fetch historical Bitcoin market data for Python models
 * @param {string} symbol - Trading pair symbol
 * @param {number} granularity - Candle granularity in seconds (60, 300, 900, 3600, 21600, 86400)
 * @param {number} limit - Number of data points to fetch (max 300 per request)
 * @returns {Promise<Array>} Array of historical candle data
 */
const getHistoricalMarketData = async (symbol = 'BTC/USD', granularity = 60, limit = 300) => {
  try {
    const https = require('https');
    const symbolMap = {
      'BTC/USD': 'BTC-USD',
      'BTCUSD': 'BTC-USD',
      'BTC-USD': 'BTC-USD'
    };
    
    const product = symbolMap[symbol] || 'BTC-USD';
    
    // Calculate time range for historical data
    const endTime = new Date();
    const startTime = new Date(endTime.getTime() - (limit * granularity * 1000));
    
    logMessage(`Fetching ${limit} candles of ${granularity}s granularity for ${symbol}`);
    
    // Fetch candles data from Coinbase
    const candlesPromise = new Promise((resolve, reject) => {
      const params = new URLSearchParams({
        start: startTime.toISOString(),
        end: endTime.toISOString(),
        granularity: granularity.toString()
      });
      
      const url = `https://api.exchange.coinbase.com/products/${product}/candles?${params}`;
      
      https.get(url, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(e);
          }
        });
      }).on('error', reject);
    });
    
    const candlesData = await candlesPromise;
    
    // Transform candles data to standard format
    // Coinbase format: [timestamp, low, high, open, close, volume]
    const historicalData = candlesData.map(candle => ({
      timestamp: new Date(candle[0] * 1000),
      open: candle[3],
      high: candle[2],
      low: candle[1],
      close: candle[4],
      volume: candle[5],
      price: candle[4] // Use close price as the price
    }));
    
    // Sort by timestamp (oldest first)
    historicalData.sort((a, b) => a.timestamp - b.timestamp);
    
    logMessage(`Successfully fetched ${historicalData.length} historical data points`);
    
    return historicalData;
  } catch (error) {
    logError('Error fetching historical market data:', error);
    throw error;
  }
};

module.exports = {
  initializeFixSession,
  subscribeToMarketData,
  sendOrder,
  getMarketData,
  scheduleMarketDataUpdates,
  getHistoricalMarketData
};
