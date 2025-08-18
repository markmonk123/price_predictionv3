import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || '';

/**
 * Place a new order
 */
export const placeOrder = (orderData) => {
  return axios.post(`${API_URL}/api/orders`, orderData);
};

/**
 * Get order by ID
 */
export const getOrder = (orderId) => {
  return axios.get(`${API_URL}/api/orders/${orderId}`);
};

/**
 * Get all orders
 */
export const getOrders = () => {
  return axios.get(`${API_URL}/api/orders`);
};

/**
 * Cancel an order
 */
export const cancelOrder = (orderId) => {
  return axios.delete(`${API_URL}/api/orders/${orderId}`);
};

/**
 * Get order status text
 */
export const getOrderStatusText = (status) => {
  const statusMap = {
    'NEW': 'New',
    'PARTIALLY_FILLED': 'Partially Filled',
    'FILLED': 'Filled',
    'CANCELED': 'Canceled',
    'REJECTED': 'Rejected',
    'EXPIRED': 'Expired'
  };

  return statusMap[status] || status;
};

/**
 * Get order status color
 */
export const getOrderStatusColor = (status) => {
  const colorMap = {
    'NEW': 'info',
    'PARTIALLY_FILLED': 'warning',
    'FILLED': 'success',
    'CANCELED': 'default',
    'REJECTED': 'error',
    'EXPIRED': 'default'
  };

  return colorMap[status] || 'default';
};

/**
 * Get side color (buy/sell)
 */
export const getSideColor = (side) => {
  return side === 'BUY' ? 'success.main' : 'error.main';
};
