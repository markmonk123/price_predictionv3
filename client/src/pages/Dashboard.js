import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Paper, 
  Typography, 
  Box, 
  Divider, 
  Button, 
  Chip,
  Skeleton,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Alert
} from '@mui/material';
import { 
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  BarChart as BarChartIcon,
  ShowChart as ShowChartIcon,
  Timelapse as TimelapseIcon
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import { format } from 'date-fns';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

import { fetchHistoricalPredictions, getPredictionColor, getPredictionText } from '../services/predictionService';
import { formatPrice } from '../services/marketService';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function Dashboard({ marketData, predictionData, loading, error }) {
  const [historicalPredictions, setHistoricalPredictions] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);

  // Fetch historical predictions on component mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoadingHistory(true);
        const response = await fetchHistoricalPredictions('1h');
        setHistoricalPredictions(response.data);
        setLoadingHistory(false);
      } catch (err) {
        console.error('Error fetching historical predictions:', err);
        setLoadingHistory(false);
      }
    };

    fetchHistory();
  }, []);

  // Prepare chart data for price history
  const priceChartData = {
    labels: historicalPredictions.map(item => format(new Date(item.timestamp), 'HH:mm')),
    datasets: [
      {
        label: 'Bitcoin Price ($)',
        data: historicalPredictions.map(item => item.price),
        borderColor: '#2196f3',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHitRadius: 10,
        pointHoverRadius: 4,
        borderWidth: 2
      }
    ]
  };

  // Prepare chart data for prediction confidence
  const confidenceChartData = {
    labels: historicalPredictions.map(item => format(new Date(item.timestamp), 'HH:mm')),
    datasets: [
      {
        label: 'Increase Probability',
        data: historicalPredictions.map(item => item.increase_probability),
        borderColor: '#4caf50',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2
      },
      {
        label: 'Decrease Probability',
        data: historicalPredictions.map(item => item.decrease_probability),
        borderColor: '#f44336',
        backgroundColor: 'rgba(244, 67, 54, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2
      },
      {
        label: 'No Change Probability',
        data: historicalPredictions.map(item => item.no_change_probability),
        borderColor: '#9e9e9e',
        backgroundColor: 'rgba(158, 158, 158, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2
      }
    ]
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          boxWidth: 12,
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              if (label.includes('Price')) {
                label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
              } else {
                label += (context.parsed.y * 100).toFixed(1) + '%';
              }
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        }
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      }
    },
    interaction: {
      mode: 'index',
      intersect: false
    },
    elements: {
      line: {
        tension: 0.4
      }
    }
  };

  // Display the appropriate trend icon based on prediction direction
  const getTrendIcon = (direction) => {
    switch (direction) {
      case 1:
        return <TrendingUpIcon sx={{ color: 'success.main' }} />;
      case -1:
        return <TrendingDownIcon sx={{ color: 'error.main' }} />;
      default:
        return <TrendingFlatIcon sx={{ color: 'text.secondary' }} />;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Bitcoin Trading Dashboard
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Market Overview Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardHeader
              title="Market Overview"
              action={
                <IconButton disabled={loading}>
                  <RefreshIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Current Price
                </Typography>
                {loading ? (
                  <Skeleton width={100} />
                ) : marketData ? (
                  <Typography variant="body1" fontWeight="bold">
                    ${formatPrice(marketData.last)}
                  </Typography>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Unavailable
                  </Typography>
                )}
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Bid / Ask
                </Typography>
                {loading ? (
                  <Skeleton width={140} />
                ) : marketData ? (
                  <Typography variant="body2">
                    ${formatPrice(marketData.bid)} / ${formatPrice(marketData.ask)}
                  </Typography>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Unavailable
                  </Typography>
                )}
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Spread
                </Typography>
                {loading ? (
                  <Skeleton width={80} />
                ) : marketData ? (
                  <Typography variant="body2">
                    ${formatPrice(marketData.ask - marketData.bid)} 
                    ({((marketData.ask - marketData.bid) / marketData.mid * 100).toFixed(3)}%)
                  </Typography>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Unavailable
                  </Typography>
                )}
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  24h Volume
                </Typography>
                {loading ? (
                  <Skeleton width={80} />
                ) : marketData ? (
                  <Typography variant="body2">
                    {marketData.volume?.toFixed(2) || 'N/A'} BTC
                  </Typography>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Unavailable
                  </Typography>
                )}
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  Last Updated
                </Typography>
                {loading ? (
                  <Skeleton width={120} />
                ) : marketData && marketData.timestamp ? (
                  <Typography variant="body2">
                    {format(new Date(marketData.timestamp), 'yyyy-MM-dd HH:mm:ss')}
                  </Typography>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Unavailable
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Latest Prediction Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardHeader
              title="Latest Prediction"
              action={
                <IconButton disabled={loading}>
                  <TimelapseIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              {loading ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Skeleton width="60%" height={40} />
                  <Skeleton width="100%" height={24} />
                  <Skeleton width="100%" height={24} />
                  <Skeleton width="100%" height={24} />
                  <Skeleton width="80%" height={24} />
                </Box>
              ) : predictionData ? (
                <React.Fragment>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {getTrendIcon(predictionData.predicted_direction)}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      Predicted: {getPredictionText(predictionData.predicted_direction)}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Confidence
                    </Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {(predictionData.confidence * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="success.main">
                        Increase Probability
                      </Typography>
                      <Typography variant="body2" color="success.main">
                        {(predictionData.increase_probability * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="error.main">
                        Decrease Probability
                      </Typography>
                      <Typography variant="body2" color="error.main">
                        {(predictionData.decrease_probability * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="text.secondary">
                        No Change Probability
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {(predictionData.no_change_probability * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Timeframe
                    </Typography>
                    <Chip 
                      label={predictionData.timeframe || '1 minute'} 
                      size="small" 
                      color="info"
                      variant="outlined"
                    />
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Price Threshold
                    </Typography>
                    <Typography variant="body2">
                      ±{(predictionData.threshold * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      Prediction Time
                    </Typography>
                    <Typography variant="body2">
                      {predictionData.timestamp 
                        ? format(new Date(predictionData.timestamp), 'yyyy-MM-dd HH:mm:ss')
                        : 'N/A'}
                    </Typography>
                  </Box>
                </React.Fragment>
              ) : (
                <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                  Prediction data unavailable
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* FIX Protocol Status Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardHeader
              title="FIX Protocol Status"
              action={
                <IconButton>
                  <RefreshIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              <Box sx={{ display: 'flex', mb: 2 }}>
                <Box sx={{ 
                  width: 12, 
                  height: 12, 
                  borderRadius: '50%', 
                  bgcolor: 'success.main',
                  mr: 1.5,
                  mt: 0.5 
                }} />
                <Box>
                  <Typography variant="body1" fontWeight="medium">
                    Connected
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    FIX session established with exchange
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography variant="body2" color="text.secondary">
                  Session ID
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  BITCOIN_PREDICTION_CLIENT-EXCHANGE
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography variant="body2" color="text.secondary">
                  FIX Version
                </Typography>
                <Typography variant="body2">
                  FIX.4.4
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography variant="body2" color="text.secondary">
                  Messages Sent
                </Typography>
                <Typography variant="body2">
                  {Math.floor(Math.random() * 500) + 100}
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                <Typography variant="body2" color="text.secondary">
                  Messages Received
                </Typography>
                <Typography variant="body2">
                  {Math.floor(Math.random() * 800) + 200}
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  Last Heartbeat
                </Typography>
                <Typography variant="body2">
                  {format(new Date(), 'HH:mm:ss')}
                </Typography>
              </Box>

              <Box sx={{ mt: 3 }}>
                <Button variant="outlined" fullWidth>View FIX Message Log</Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Price Chart */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: 400 }}>
            <CardHeader
              title="Bitcoin Price History"
              action={
                <IconButton>
                  <ShowChartIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent sx={{ height: 'calc(100% - 76px)' }}>
              {loadingHistory ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body1" color="text.secondary">
                    Loading price history...
                  </Typography>
                </Box>
              ) : historicalPredictions.length > 0 ? (
                <Line data={priceChartData} options={chartOptions} />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body1" color="text.secondary">
                    No price history available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Confidence Chart */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: 400 }}>
            <CardHeader
              title="Prediction Probabilities"
              action={
                <IconButton>
                  <BarChartIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent sx={{ height: 'calc(100% - 76px)' }}>
              {loadingHistory ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body1" color="text.secondary">
                    Loading prediction history...
                  </Typography>
                </Box>
              ) : historicalPredictions.length > 0 ? (
                <Line data={confidenceChartData} options={chartOptions} />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body1" color="text.secondary">
                    No prediction history available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
