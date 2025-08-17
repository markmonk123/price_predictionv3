import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || '';

/**
 * Fetch latest market data
 */
export const fetchMarketData = (symbol = 'BTC/USD') => {
  return axios.get(`${API_URL}/api/market/data`, {
    params: { symbol }
  });
};

/**
 * Subscribe to market data updates
 */
export const subscribeToMarketData = (symbol = 'BTC/USD') => {
  return axios.post(`${API_URL}/api/market/subscribe`, { symbol });
};

/**
 * Format price for display
 */
export const formatPrice = (price, decimals = 2) => {
  if (price === undefined || price === null) return '--';
  return price.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
};

/**
 * Calculate price change percentage
 */
export const calculatePriceChange = (currentPrice, previousPrice) => {
  if (!currentPrice || !previousPrice) return { value: 0, percentage: 0 };

  const change = currentPrice - previousPrice;
  const percentage = (change / previousPrice) * 100;

  return {
    value: change,
    percentage: percentage
  };
};
