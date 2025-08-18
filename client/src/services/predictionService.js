import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || '';

/**
 * Fetch latest prediction
 */
export const fetchLatestPrediction = () => {
  return axios.get(`${API_URL}/api/predictions/latest`);
};

/**
 * Fetch historical predictions
 */
export const fetchHistoricalPredictions = (timeframe = '1h') => {
  return axios.get(`${API_URL}/api/predictions/history`, {
    params: { timeframe }
  });
};

/**
 * Manually run prediction model
 */
export const runPredictionModel = () => {
  return axios.post(`${API_URL}/api/predictions/run`);
};

/**
 * Get color for prediction direction
 */
export const getPredictionColor = (direction) => {
  switch (direction) {
    case 1:
      return '#4caf50'; // green
    case -1:
      return '#f44336'; // red
    default:
      return '#9e9e9e'; // grey
  }
};

/**
 * Get text label for prediction direction
 */
export const getPredictionText = (direction) => {
  switch (direction) {
    case 1:
      return 'Increase';
    case -1:
      return 'Decrease';
    default:
      return 'No Change';
  }
};

/**
 * Format confidence percentage
 */
export const formatConfidence = (confidence) => {
  if (confidence === undefined || confidence === null) return '--';
  return `${(confidence * 100).toFixed(1)}%`;
};
