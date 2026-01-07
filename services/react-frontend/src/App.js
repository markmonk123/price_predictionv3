/**
 * Minimal React App for Price Prediction Service
 * 
 * Displays model status and allows making predictions via Node.js backend
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:3001';

function App() {
  const [health, setHealth] = useState(null);
  const [mlHealth, setMlHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  
  // Sample market data for testing
  const [marketData, setMarketData] = useState({
    price: 50000,
    volume: 1000000,
    bid: 49900,
    ask: 50100,
    high: 51000,
    low: 49000
  });

  // Fetch health status on mount
  useEffect(() => {
    fetchHealth();
    fetchMLHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/health`);
      setHealth(response.data);
    } catch (err) {
      console.error('Failed to fetch health:', err);
      setHealth({ status: 'error', error: err.message });
    }
  };

  const fetchMLHealth = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/ml-service/health`);
      setMlHealth(response.data);
    } catch (err) {
      console.error('Failed to fetch ML health:', err);
      setMlHealth({ status: 'error', error: err.message });
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post(`${BACKEND_URL}/predict`, {
        marketData: marketData,
        modelName: 'ensemble_hybrid'
      });
      
      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction failed:', err);
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setMarketData({
      ...marketData,
      [field]: parseFloat(value) || 0
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔮 Price Prediction Service</h1>
        <p>ML-powered trading predictions with imbalanced-learn ensembles</p>
      </header>

      <main className="App-main">
        {/* Health Status Section */}
        <section className="status-section">
          <h2>📊 Service Status</h2>
          
          <div className="status-cards">
            <div className={`status-card ${health?.status === 'ok' ? 'healthy' : 'unhealthy'}`}>
              <h3>Node Backend</h3>
              <p>Status: {health?.status || 'Loading...'}</p>
              <p>Coinbase: {health?.coinbase_connected ? '✅ Connected' : '❌ Not configured'}</p>
            </div>

            <div className={`status-card ${mlHealth?.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
              <h3>ML Service</h3>
              <p>Status: {mlHealth?.status || 'Loading...'}</p>
              <p>Models: {mlHealth?.models_loaded?.length || 0} loaded</p>
              {mlHealth?.models_loaded && (
                <div className="models-list">
                  {mlHealth.models_loaded.map(model => (
                    <span key={model} className="model-badge">{model}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Prediction Input Section */}
        <section className="prediction-section">
          <h2>🎯 Make Prediction</h2>
          
          <div className="input-grid">
            <div className="input-field">
              <label>Price:</label>
              <input
                type="number"
                value={marketData.price}
                onChange={(e) => handleInputChange('price', e.target.value)}
              />
            </div>
            
            <div className="input-field">
              <label>Volume:</label>
              <input
                type="number"
                value={marketData.volume}
                onChange={(e) => handleInputChange('volume', e.target.value)}
              />
            </div>
            
            <div className="input-field">
              <label>Bid:</label>
              <input
                type="number"
                value={marketData.bid}
                onChange={(e) => handleInputChange('bid', e.target.value)}
              />
            </div>
            
            <div className="input-field">
              <label>Ask:</label>
              <input
                type="number"
                value={marketData.ask}
                onChange={(e) => handleInputChange('ask', e.target.value)}
              />
            </div>
            
            <div className="input-field">
              <label>High:</label>
              <input
                type="number"
                value={marketData.high}
                onChange={(e) => handleInputChange('high', e.target.value)}
              />
            </div>
            
            <div className="input-field">
              <label>Low:</label>
              <input
                type="number"
                value={marketData.low}
                onChange={(e) => handleInputChange('low', e.target.value)}
              />
            </div>
          </div>

          <button 
            className="predict-button"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? 'Predicting...' : '🚀 Get Prediction'}
          </button>
        </section>

        {/* Results Section */}
        {error && (
          <section className="error-section">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </section>
        )}

        {prediction && (
          <section className="results-section">
            <h2>✨ Prediction Results</h2>
            
            <div className="result-card">
              <h3>Prediction</h3>
              <p className="prediction-value">
                Class: {prediction.prediction?.predictions?.[0] ?? 'N/A'}
              </p>
              <p>Model: {prediction.prediction?.model_name}</p>
              
              {prediction.prediction?.probabilities && (
                <div className="probabilities">
                  <h4>Probabilities:</h4>
                  {prediction.prediction.probabilities[0].map((prob, idx) => (
                    <div key={idx} className="probability-bar">
                      <span>Class {idx}:</span>
                      <div className="bar-container">
                        <div 
                          className="bar-fill" 
                          style={{ width: `${(prob * 100).toFixed(1)}%` }}
                        />
                        <span className="bar-label">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="result-card">
              <h3>Input Features</h3>
              <pre>{JSON.stringify(prediction.features, null, 2)}</pre>
            </div>
          </section>
        )}
      </main>

      <footer className="App-footer">
        <p>Powered by imbalanced-learn, FastAPI, and React</p>
        <p>⚠️ For demonstration purposes only - not financial advice</p>
      </footer>
    </div>
  );
}

export default App;
