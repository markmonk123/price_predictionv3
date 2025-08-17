import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { io } from 'socket.io-client';
import { Box } from '@mui/material';

// Layout components
import Navigation from './components/layout/Navigation';
import Header from './components/layout/Header';

// Page components
import Dashboard from './pages/Dashboard';
import TradingView from './pages/TradingView';
import PredictionAnalysis from './pages/PredictionAnalysis';
import OrderBook from './pages/OrderBook';
import Settings from './pages/Settings';

// API Services
import { fetchMarketData } from './services/marketService';
import { fetchLatestPrediction } from './services/predictionService';

function App() {
  // Application state
  const [socket, setSocket] = useState(null);
  const [marketData, setMarketData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initialize socket connection and fetch initial data
  useEffect(() => {
    // Initialize Socket.IO connection
    const socketConnection = io(process.env.REACT_APP_API_URL || 'http://localhost:5000');
    setSocket(socketConnection);

    // Fetch initial data
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        // Fetch market data
        const marketResponse = await fetchMarketData();
        setMarketData(marketResponse.data);

        // Fetch prediction data
        const predictionResponse = await fetchLatestPrediction();
        setPredictionData(predictionResponse.data);

        setLoading(false);
      } catch (err) {
        console.error('Error fetching initial data:', err);
        setError('Failed to load initial data. Please refresh.');
        setLoading(false);
      }
    };

    fetchInitialData();

    // Socket event listeners
    socketConnection.on('connect', () => {
      console.log('Socket connected:', socketConnection.id);
    });

    socketConnection.on('marketData', (data) => {
      console.log('Received market data update:', data);
      setMarketData(data);
    });

    socketConnection.on('predictionData', (data) => {
      console.log('Received prediction update:', data);
      setPredictionData(data);
    });

    socketConnection.on('disconnect', () => {
      console.log('Socket disconnected');
    });

    socketConnection.on('error', (error) => {
      console.error('Socket error:', error);
      setError('Connection error. Please refresh.');
    });

    // Cleanup function
    return () => {
      console.log('Cleaning up socket connection');
      socketConnection.disconnect();
    };
  }, []);

  return (
    <Box sx={{ display: 'flex' }}>
      <Navigation />
      <Box component="main" sx={{ flexGrow: 1 }}>
        <Header marketData={marketData} predictionData={predictionData} loading={loading} />
        <Box sx={{ p: 3, mt: 8 }}>
          <Routes>
            <Route path="/" element={
              <Dashboard 
                marketData={marketData} 
                predictionData={predictionData} 
                loading={loading} 
                error={error} 
              />
            } />
            <Route path="/trading" element={
              <TradingView 
                socket={socket} 
                marketData={marketData} 
                predictionData={predictionData} 
              />
            } />
            <Route path="/predictions" element={
              <PredictionAnalysis 
                predictionData={predictionData} 
                loading={loading} 
              />
            } />
            <Route path="/orders" element={
              <OrderBook 
                socket={socket} 
                marketData={marketData} 
              />
            } />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Box>
      </Box>
    </Box>
  );
}

export default App;
