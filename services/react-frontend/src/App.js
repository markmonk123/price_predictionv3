/**
 * ML Prediction Dashboard
 * 
 * A minimal React frontend that:
 * - Connects to Node.js backend for predictions
 * - Displays model status and health
 * - Shows prediction results
 * 
 * Configuration:
 * - Backend URL configurable via REACT_APP_BACKEND_URL env var
 * - Defaults to http://localhost:3000
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:3000';

function App() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  
  // Sample market data for demo
  const [marketData, setMarketData] = useState({
    price: 45000,
    volume: 1000000,
    bid: 44950,
    ask: 45050,
    high: 46000,
    low: 44000
  });

  // Fetch health status on mount
  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/health`);
      setHealth(response.data);
      setError(null);
    } catch (err) {
      console.error('Health check failed:', err);
      setError('Backend service unavailable');
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post(`${BACKEND_URL}/predict`, {
        marketData: marketData,
        model_name: 'ensemble_balanced_stacking'
      });

      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction failed:', err);
      setError(err.response?.data?.message || 'Prediction request failed');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setMarketData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 ML Prediction Dashboard</h1>
        <p>Real-time machine learning predictions for market data</p>
      </header>

      <main className="App-main">
        {/* Health Status */}
        <section className="status-section">
          <h2>Service Status</h2>
          {health ? (
            <div className="status-card">
              <div className="status-item">
                <span className="status-label">Backend:</span>
                <span className={`status-badge ${health.status === 'healthy' ? 'healthy' : 'error'}`}>
                  {health.status}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Coinbase:</span>
                <span className={`status-badge ${health.services.coinbase ? 'healthy' : 'warning'}`}>
                  {health.services.coinbase ? 'Connected' : 'Not Configured'}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Model Service:</span>
                <span className="status-value">{health.services.model_service}</span>
              </div>
              <button onClick={fetchHealth} className="btn-secondary">
                Refresh Status
              </button>
            </div>
          ) : (
            <div className="status-card error">
              <p>{error || 'Loading...'}</p>
              <button onClick={fetchHealth} className="btn-secondary">
                Retry
              </button>
            </div>
          )}
        </section>

        {/* Market Data Input */}
        <section className="input-section">
          <h2>Market Data</h2>
          <div className="input-grid">
            {Object.entries(marketData).map(([key, value]) => (
              <div key={key} className="input-group">
                <label>{key.charAt(0).toUpperCase() + key.slice(1)}:</label>
                <input
                  type="number"
                  value={value}
                  onChange={(e) => handleInputChange(key, e.target.value)}
                  step="0.01"
                />
              </div>
            ))}
          </div>
        </section>

        {/* Prediction Button */}
        <section className="action-section">
          <button 
            onClick={handlePredict} 
            disabled={loading || !health}
            className="btn-primary"
          >
            {loading ? '⏳ Predicting...' : '🎯 Get Prediction'}
          </button>
        </section>

        {/* Error Display */}
        {error && (
          <section className="error-section">
            <div className="error-card">
              <h3>❌ Error</h3>
              <p>{error}</p>
            </div>
          </section>
        )}

        {/* Prediction Results */}
        {prediction && (
          <section className="results-section">
            <h2>Prediction Results</h2>
            <div className="results-card">
              <div className="result-item">
                <span className="result-label">Model Used:</span>
                <span className="result-value">{prediction.prediction.model_used}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Prediction:</span>
                <span className="result-value prediction-value">
                  {prediction.prediction.predictions.join(', ')}
                </span>
              </div>
              {prediction.prediction.probabilities && (
                <div className="result-item">
                  <span className="result-label">Confidence:</span>
                  <span className="result-value">
                    {(Math.max(...prediction.prediction.probabilities[0]) * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              <div className="result-item">
                <span className="result-label">Timestamp:</span>
                <span className="result-value">
                  {new Date(prediction.timestamp).toLocaleString()}
                </span>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="App-footer">
        <p>Powered by imbalanced-learn ensemble models</p>
      </footer>
    </div>
  );
}

export default App;
