/**
 * FIX Protocol Service for Bitcoin Trading Platform
 * Implements QuickFIX (node-quickfix) for market data and order execution.
 */

const path = require('path');
const NodeCache = require('node-cache');
const { logMessage, logError } = require('../utils/logger');

// Attempt to load node-quickfix. If the native addon is missing we fall back to simulation mode.
let quickfix = null;
try {
  // eslint-disable-next-line global-require
  quickfix = require('node-quickfix');
} catch (error) {
  logMessage('node-quickfix module not available; FIX connectivity will run in simulation mode.');
}

const marketDataCache = new NodeCache({ stdTTL: 300 });

const FIX_MSG_TYPES = {
  MARKET_DATA_REQUEST: 'V',
  MARKET_DATA_SNAPSHOT: 'W',
  NEW_ORDER_SINGLE: 'D',
  EXECUTION_REPORT: '8'
};

const FIX_IDENTIFIERS = {
  beginString: process.env.FIX_BEGIN_STRING || 'FIX.4.4',
  senderCompId: process.env.FIX_SENDER_COMP_ID || 'BITCOIN_PREDICTION_CLIENT',
  targetCompId: process.env.FIX_TARGET_COMP_ID || 'EXCHANGE'
};

const FIX_CONFIG_PATH = path.join(__dirname, '..', 'config', 'quickfix-initiator.cfg');
const DEFAULT_SYMBOL = 'BTC/USD';

let fixClient = null;
let isConnected = false;

const formatFixTimestamp = (date) => {
  const pad = (value, size = 2) => String(value).padStart(size, '0');
  return [
    `${date.getUTCFullYear()}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}`,
    `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}.${pad(date.getUTCMilliseconds(), 3)}`
  ].join('-');
};

const buildHeader = (msgType) => ({
  8: FIX_IDENTIFIERS.beginString,
  35: msgType,
  49: FIX_IDENTIFIERS.senderCompId,
  56: FIX_IDENTIFIERS.targetCompId
});

const startInitiator = (client) => new Promise((resolve, reject) => {
  try {
    client.start(() => resolve());
  } catch (error) {
    reject(error);
  }
});

const getTag = (message, tag) => {
  if (!message) {
    return undefined;
  }
  if (message.tags && message.tags[tag] !== undefined) {
    return message.tags[tag];
  }
  if (message.header && message.header[tag] !== undefined) {
    return message.header[tag];
  }
  return undefined;
};

const findGroup = (message, index) => {
  if (!message || !message.groups) {
    return null;
  }
  return message.groups.find((group) => String(group.index) === String(index)) || null;
};

const getEntryTag = (entry, tagCandidates) => {
  if (!entry) {
    return undefined;
  }
  for (const tag of tagCandidates) {
    if (entry[tag] !== undefined) {
      return entry[tag];
    }
    if (entry.tags && entry.tags[tag] !== undefined) {
      return entry.tags[tag];
    }
  }
  return undefined;
};

const initializeFixSession = async () => {
  if (!quickfix) {
    logMessage('QuickFIX initiator not started because node-quickfix is unavailable.');
    return false;
  }

  if (fixClient) {
    return isConnected;
  }

  try {
    logMessage('Initializing QuickFIX initiator...');

    const initiatorOptions = {
      propertiesFile: FIX_CONFIG_PATH,
      storeFactory: process.env.FIX_STORE_FACTORY || 'file'
    };

    if (process.env.FIX_USE_SSL) {
      initiatorOptions.ssl = process.env.FIX_USE_SSL === 'true';
    }

    if (process.env.FIX_USERNAME && process.env.FIX_PASSWORD) {
      initiatorOptions.credentials = {
        username: process.env.FIX_USERNAME,
        password: process.env.FIX_PASSWORD
      };
    }

    fixClient = new quickfix.initiator({
      onCreate: (sessionID) => {
        logMessage(`FIX session created: ${sessionID}`);
      },
      onLogon: (sessionID) => {
        logMessage(`FIX session logged on: ${sessionID}`);
        isConnected = true;
        subscribeToMarketData(DEFAULT_SYMBOL);
      },
      onLogout: (sessionID) => {
        logMessage(`FIX session logged out: ${sessionID}`);
        isConnected = false;
      },
      onLogonAttempt: (message, sessionID) => {
        logMessage(`FIX logon attempt for ${sessionID}: ${JSON.stringify(message)}`);
      },
      toAdmin: (message, sessionID) => {
        logMessage(`Sending admin message [${sessionID}]: ${JSON.stringify(message)}`);
      },
      fromAdmin: (message, sessionID) => {
        logMessage(`Received admin message [${sessionID}]: ${JSON.stringify(message)}`);
      },
      fromApp: (message, sessionID) => {
        logMessage(`Received application message [${sessionID}]: ${JSON.stringify(message)}`);
        processIncomingMessage(message);
      }
    }, initiatorOptions);

    await startInitiator(fixClient);
    logMessage('QuickFIX initiator started successfully');
    return true;
  } catch (error) {
    logError('Error initializing FIX session:', error);
    fixClient = null;
    isConnected = false;
    return false;
  }
};

const processIncomingMessage = (message) => {
  try {
    const msgType = getTag(message, 35);

    switch (msgType) {
      case FIX_MSG_TYPES.MARKET_DATA_SNAPSHOT:
        processMarketDataSnapshot(message);
        break;
      case FIX_MSG_TYPES.EXECUTION_REPORT:
        processExecutionReport(message);
        break;
      default:
        logMessage(`Unhandled FIX message type: ${msgType}`);
    }
  } catch (error) {
    logError('Error processing incoming FIX message:', error);
  }
};

const processMarketDataSnapshot = (message) => {
  try {
    const symbol = getTag(message, 55) || DEFAULT_SYMBOL;
    const timestamp = getTag(message, 52);

    const mdGroup = findGroup(message, 268); // NoMDEntries
    const mdEntries = mdGroup ? mdGroup.entries || [] : [];

    const entryByType = {};
    mdEntries.forEach((entry) => {
      const entryType = getEntryTag(entry, [269]);
      if (entryType !== undefined) {
        entryByType[entryType] = entry;
      }
    });

    const bidPrice = parseFloat(getEntryTag(entryByType['0'], [270, 132]));
    const askPrice = parseFloat(getEntryTag(entryByType['1'], [270, 133]));
    const lastPrice = parseFloat(getEntryTag(entryByType['2'], [270, 31]));
    const volume = parseFloat(getEntryTag(entryByType['2'], [271, 32]));

    if (Number.isNaN(bidPrice) || Number.isNaN(askPrice)) {
      logMessage('Market data snapshot missing bid/ask; ignoring.');
      return;
    }

    const midPrice = (bidPrice + askPrice) / 2;

    const marketData = {
      symbol,
      timestamp: timestamp ? new Date(timestamp) : new Date(),
      bid: bidPrice,
      ask: askPrice,
      last: Number.isNaN(lastPrice) ? midPrice : lastPrice,
      mid: midPrice,
      volume: Number.isNaN(volume) ? 0 : volume,
      receivedAt: new Date(),
      simulated: false
    };

    marketDataCache.set(symbol, marketData);
    logMessage(`Updated market data for ${symbol}: ${JSON.stringify(marketData)}`);
  } catch (error) {
    logError('Error processing market data snapshot:', error);
  }
};

const processExecutionReport = (message) => {
  try {
    const symbol = getTag(message, 55);
    const side = getTag(message, 54);

    const execReport = {
      orderID: getTag(message, 37),
      execType: getTag(message, 150),
      ordStatus: getTag(message, 39),
      symbol,
      side: side === '1' ? 'BUY' : 'SELL',
      orderQty: parseFloat(getTag(message, 38)),
      price: parseFloat(getTag(message, 44)),
      transactTime: (() => {
        const value = getTag(message, 60);
        return value ? new Date(value) : new Date();
      })(),
      receivedAt: new Date()
    };

    logMessage(`Received execution report: ${JSON.stringify(execReport)}`);
  } catch (error) {
    logError('Error processing execution report:', error);
  }
};

const sendFixMessage = (message, description) => {
  if (!quickfix || !fixClient || !isConnected) {
    logError(`Cannot send ${description}: FIX session not established`);
    return false;
  }

  try {
    fixClient.send(message, () => {
      logMessage(`Sent ${description}: ${JSON.stringify(message)}`);
    });
    return true;
  } catch (error) {
    logError(`Error sending ${description}:`, error);
    return false;
  }
};

const subscribeToMarketData = (symbol) => {
  const message = {
    header: buildHeader(FIX_MSG_TYPES.MARKET_DATA_REQUEST),
    tags: {
      262: Date.now().toString(),
      263: '1',
      264: '1',
      265: '1',
      146: 1,
      55: symbol,
      267: 3
    },
    groups: [
      {
        index: 146,
        delim: 55,
        entries: [{ 55: symbol }]
      },
      {
        index: 267,
        delim: 269,
        entries: [{ 269: '0' }, { 269: '1' }, { 269: '2' }]
      }
    ]
  };

  return sendFixMessage(message, `market data request for ${symbol}`);
};

const sendOrder = async (orderData) => {
  if (!quickfix || !fixClient || !isConnected) {
    throw new Error('FIX session not established');
  }

  const { symbol, side, orderType, quantity, price, timeInForce } = orderData;

  const tifMap = {
    DAY: '0',
    GTC: '1',
    IOC: '3',
    FOK: '4'
  };

  const message = {
    header: buildHeader(FIX_MSG_TYPES.NEW_ORDER_SINGLE),
    tags: {
      11: Date.now().toString(36),
      55: symbol,
      54: side === 'BUY' ? '1' : '2',
      38: quantity.toString(),
      40: orderType === 'MARKET' ? '1' : '2',
      59: tifMap[timeInForce] || '0',
      60: formatFixTimestamp(new Date())
    }
  };

  if (orderType === 'LIMIT' && price) {
    message.tags[44] = price.toString();
  }

  if (!sendFixMessage(message, `order for ${symbol}`)) {
    throw new Error('Failed to dispatch FIX order');
  }

  return { success: true, orderId: message.tags[11] };
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
