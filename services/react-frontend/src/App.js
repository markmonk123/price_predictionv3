/**
 * React Frontend Application
 * 
 * Simple React app that demonstrates:
 * - Sending sample tick data to Node backend
 * - Displaying prediction results
 * - Basic UI for testing the prediction pipeline
 */

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// Configuration
const NODE_BACKEND_URL = process.env.REACT_APP_NODE_BACKEND_URL || 'http://localhost:3001';

function App() {
  const [tickData, setTickData] = useState({
    price: 45123.45,
    volume: 123.456,
    bid: 45120.00,
    ask: 45125.00,
    high: 45500.00,
    low: 44800.00,
    open: 45000.00,
    model_name: 'ensemble_balanced_stacking'
  });
  
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Handle input changes
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setTickData(prev => ({
      ...prev,
      [name]: name === 'model_name' ? value : parseFloat(value) || 0
    }));
  };

  /**
   * Send tick data to Node backend
   */
  const sendPredictionRequest = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      console.log('Sending tick data:', tickData);
      
      const result = await axios.post(
        `${NODE_BACKEND_URL}/forward-tick`,
        tickData,
        {
          timeout: 10000,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      console.log('Received response:', result.data);
      setResponse(result.data);
      
    } catch (err) {
      console.error('Request failed:', err);
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Generate random tick data for testing
   */
  const generateRandomTick = () => {
    const basePrice = 45000 + Math.random() * 5000;
    setTickData({
      price: basePrice,
      volume: Math.random() * 1000,
      bid: basePrice - Math.random() * 50,
      ask: basePrice + Math.random() * 50,
      high: basePrice + Math.random() * 500,
      low: basePrice - Math.random() * 500,
      open: basePrice + Math.random() * 200 - 100,
      model_name: tickData.model_name
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🚀 Crypto Price Prediction</h1>
        <p>Send tick data for ML-based prediction</p>
      </header>

      <main className="App-main">
        <div className="form-container">
          <h2>Tick Data Input</h2>
          
          <div className="form-grid">
            <div className="form-group">
              <label>Price:</label>
              <input
                type="number"
                name="price"
                value={tickData.price}
                onChange={handleInputChange}
                step="0.01"
              />
            </div>

            <div className="form-group">
              <label>Volume:</label>
              <input
                type="number"
                name="volume"
                value={tickData.volume}
                onChange={handleInputChange}
                step="0.001"
              />
            </div>

            <div className="form-group">
              <label>Bid:</label>
              <input
                type="number"
                name="bid"
                value={tickData.bid}
                onChange={handleInputChange}
                step="0.01"
              />
            </div>

            <div className="form-group">
              <label>Ask:</label>
              <input
                type="number"
                name="ask"
                value={tickData.ask}
                onChange={handleInputChange}
                step="0.01"
              />
            </div>

            <div className="form-group">
              <label>High:</label>
              <input
                type="number"
                name="high"
                value={tickData.high}
                onChange={handleInputChange}
                step="0.01"
              />
            </div>

            <div className="form-group">
              <label>Low:</label>
              <input
                type="number"
                name="low"
                value={tickData.low}
                onChange={handleInputChange}
                step="0.01"
              />
            </div>

            <div className="form-group">
              <label>Open:</label>
              <input
                type="number"
                name="open"
                value={tickData.open}
                onChange={handleInputChange}
                step="0.01"
              />
            </div>

            <div className="form-group full-width">
              <label>Model:</label>
              <select
                name="model_name"
                value={tickData.model_name}
                onChange={handleInputChange}
              >
                <option value="ensemble_balanced_stacking">Balanced Stacking</option>
                <option value="ensemble_voting">Balanced Voting</option>
                <option value="ensemble_hybrid">Hybrid Zoo</option>
              </select>
            </div>
          </div>

          <div className="button-group">
            <button
              onClick={sendPredictionRequest}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? 'Processing...' : 'Get Prediction'}
            </button>
            
            <button
              onClick={generateRandomTick}
              className="btn btn-secondary"
            >
              Generate Random Data
            </button>
          </div>
        </div>

        {error && (
          <div className="error-container">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {response && (
          <div className="response-container">
            <h2>✅ Prediction Result</h2>
            
            <div className="response-section">
              <h3>Prediction</h3>
              <div className="prediction-value">
                {response.prediction?.predictions?.[0]?.toFixed(4) || 'N/A'}
              </div>
              <p className="prediction-label">Probability</p>
            </div>

            <div className="response-section">
              <h3>Model Info</h3>
              <p><strong>Model:</strong> {response.prediction?.model_name}</p>
              <p><strong>Features:</strong> {response.features?.length || 0} features</p>
              <p><strong>Shape:</strong> {response.prediction?.shape?.join(' × ')}</p>
            </div>

            <div className="response-section">
              <h3>Input Features</h3>
              <pre>{JSON.stringify(response.features, null, 2)}</pre>
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Backend: {NODE_BACKEND_URL}</p>
      </footer>
    </div>
  );
}

export default App;
