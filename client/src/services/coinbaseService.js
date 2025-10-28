/**
 * Coinbase Service for Frontend
 * Handles communication with Coinbase API endpoints
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

/**
 * Get current Bitcoin price from Coinbase
 */
export const getBitcoinPrice = async (symbol = 'BTC-USD') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/coinbase/price`, {
      params: { symbol }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching Bitcoin price from Coinbase:', error);
    throw error;
  }
};

/**
 * Get historical price data from Coinbase
 */
export const getHistoricalPrices = async (symbol = 'BTC-USD', granularity = 3600, limit = 100) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/coinbase/historical`, {
      params: { symbol, granularity, limit }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching historical data from Coinbase:', error);
    throw error;
  }
};

/**
 * Get account balance
 */
export const getAccountBalance = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/coinbase/balance`);
    return response.data;
  } catch (error) {
    console.error('Error fetching account balance:', error);
    throw error;
  }
};

/**
 * Get trading pair information
 */
export const getTradingPairInfo = async (symbol = 'BTC-USD') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/coinbase/pair/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching trading pair info:', error);
    throw error;
  }
};

/**
 * Get market depth (order book)
 */
export const getMarketDepth = async (symbol = 'BTC-USD', level = 2) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/coinbase/depth`, {
      params: { symbol, level }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching market depth:', error);
    throw error;
  }
};

export default {
  getBitcoinPrice,
  getHistoricalPrices,
  getAccountBalance,
  getTradingPairInfo,
  getMarketDepth
};
