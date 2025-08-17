import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || '';

/**
 * Get FIX connection status
 */
export const getFixStatus = () => {
  return axios.get(`${API_URL}/api/fix/status`);
};

/**
 * Force reconnection of FIX session
 */
export const reconnectFixSession = () => {
  return axios.post(`${API_URL}/api/fix/reconnect`);
};

/**
 * Parse FIX message for display
 */
export const parseFixMessage = (message) => {
  if (!message) return [];

  // FIX message tag definitions
  const fixTags = {
    '8': 'BeginString',
    '9': 'BodyLength',
    '35': 'MsgType',
    '49': 'SenderCompID',
    '56': 'TargetCompID',
    '34': 'MsgSeqNum',
    '52': 'SendingTime',
    '10': 'CheckSum',
    '11': 'ClOrdID',
    '37': 'OrderID',
    '38': 'OrderQty',
    '39': 'OrdStatus',
    '40': 'OrdType',
    '44': 'Price',
    '54': 'Side',
    '55': 'Symbol',
    '59': 'TimeInForce',
    '60': 'TransactTime',
    '150': 'ExecType',
    '151': 'LeavesQty',
    '14': 'CumQty'
  };

  // Message type descriptions
  const msgTypes = {
    '0': 'Heartbeat',
    '1': 'Test Request',
    '2': 'Resend Request',
    '3': 'Reject',
    '4': 'Sequence Reset',
    '5': 'Logout',
    '8': 'Execution Report',
    'D': 'New Order - Single',
    'F': 'Order Cancel Request',
    'G': 'Order Cancel/Replace Request',
    'V': 'Market Data Request',
    'W': 'Market Data Snapshot',
    'X': 'Market Data Incremental Refresh'
  };

  // Side values
  const sideValues = {
    '1': 'Buy',
    '2': 'Sell'
  };

  // Order type values
  const orderTypeValues = {
    '1': 'Market',
    '2': 'Limit',
    '3': 'Stop',
    '4': 'Stop Limit'
  };

  // Order status values
  const orderStatusValues = {
    '0': 'New',
    '1': 'Partially Filled',
    '2': 'Filled',
    '4': 'Canceled',
    '8': 'Rejected'
  };

  // Parse the message into tag-value pairs
  const pairs = message.split('\u0001');
  const parsedMessage = [];

  pairs.forEach(pair => {
    if (!pair) return;

    const [tag, value] = pair.split('=');
    let tagName = fixTags[tag] || tag;
    let displayValue = value;

    // Format display value based on tag
    if (tag === '35') { // MsgType
      displayValue = `${value} (${msgTypes[value] || 'Unknown'})`;
    } else if (tag === '54') { // Side
      displayValue = `${value} (${sideValues[value] || 'Unknown'})`;
    } else if (tag === '40') { // OrdType
      displayValue = `${value} (${orderTypeValues[value] || 'Unknown'})`;
    } else if (tag === '39') { // OrdStatus
      displayValue = `${value} (${orderStatusValues[value] || 'Unknown'})`;
    }

    parsedMessage.push({
      tag,
      tagName,
      value: displayValue
    });
  });

  return parsedMessage;
};

/**
 * Format FIX message for display
 */
export const formatFixMessageForDisplay = (message) => {
  if (!message) return '';

  return message
    .split('\u0001')
    .filter(segment => segment.trim() !== '')
    .map(segment => {
      const [tag, value] = segment.split('=');
      return `<span class="fix-tag">${tag}</span>=<span class="fix-value">${value}</span><span class="fix-delimiter">|</span>`;
    })
    .join(' ');
};
