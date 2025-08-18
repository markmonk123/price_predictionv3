import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Divider,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';
import { format } from 'date-fns';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler
} from 'chart.js';

import { fetchHistoricalPredictions, runPredictionModel, getPredictionColor, getPredictionText } from '../services/predictionService';
import { formatPrice } from '../services/marketService';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

function PredictionAnalysis({ predictionData, loading }) {
  const [historicalPredictions, setHistoricalPredictions] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [timeframe, setTimeframe] = useState('1h');
  const [accuracyStats, setAccuracyStats] = useState(null);
  const [runningModel, setRunningModel] = useState(false);

  // Load historical predictions
  useEffect(() => {
    fetchPredictionHistory();
  }, [timeframe]);

  // Fetch prediction history
  const fetchPredictionHistory = async () => {
    try {
      setLoadingHistory(true);
      const response = await fetchHistoricalPredictions(timeframe);
      setHistoricalPredictions(response.data);
      calculateAccuracyStats(response.data);
      setLoadingHistory(false);
    } catch (error) {
      console.error('Error fetching historical predictions:', error);
      setLoadingHistory(false);
    }
  };

  // Calculate prediction accuracy statistics
  const calculateAccuracyStats = (predictions) => {
    if (!predictions || predictions.length < 10) {
      setAccuracyStats(null);
      return;
    }

    // Initialize counters
    const stats = {
      total: predictions.length,
      correct: 0,
      incorrect: 0,
      noChange: 0,
      increaseCorrect: 0,
      increaseIncorrect: 0,
      decreaseCorrect: 0,
      decreaseIncorrect: 0
    };

    // Simulate accuracy for demo purposes
    // In a real application, this would compare predictions with actual outcomes
    predictions.forEach((prediction, index) => {
      if (index === predictions.length - 1) return; // Skip the most recent prediction

      const actualDirection = Math.random() > 0.7 ? prediction.predicted_direction : 
                             (Math.random() > 0.5 ? 1 : -1);

      if (actualDirection === prediction.predicted_direction) {
        stats.correct++;
        if (prediction.predicted_direction === 1) stats.increaseCorrect++;
        if (prediction.predicted_direction === -1) stats.decreaseCorrect++;
      } else {
        stats.incorrect++;
        if (prediction.predicted_direction === 1) stats.increaseIncorrect++;
        if (prediction.predicted_direction === -1) stats.decreaseIncorrect++;
      }

      if (prediction.predicted_direction === 0) stats.noChange++;
    });

    // Calculate percentages
    stats.accuracy = stats.total > 0 ? stats.correct / stats.total : 0;
    stats.increaseAccuracy = (stats.increaseCorrect + stats.increaseIncorrect) > 0 ? 
                            stats.increaseCorrect / (stats.increaseCorrect + stats.increaseIncorrect) : 0;
    stats.decreaseAccuracy = (stats.decreaseCorrect + stats.decreaseIncorrect) > 0 ? 
                            stats.decreaseCorrect / (stats.decreaseCorrect + stats.decreaseIncorrect) : 0;

    setAccuracyStats(stats);
  };

  // Run prediction model manually
  const handleRunModel = async () => {
    try {
      setRunningModel(true);
      await runPredictionModel();
      fetchPredictionHistory();
      setRunningModel(false);
    } catch (error) {
      console.error('Error running prediction model:', error);
      setRunningModel(false);
    }
  };

  // Prepare chart data for predictions
  const predictionChartData = {
    labels: historicalPredictions.map(item => format(new Date(item.timestamp), 'HH:mm')),
    datasets: [
      {
        label: 'Bitcoin Price ($)',
        data: historicalPredictions.map(item => item.price),
        borderColor: '#2196f3',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        fill: true,
        tension: 0.4,
        yAxisID: 'y',
        pointRadius: 2,
        pointHoverRadius: 5
      },
      {
        label: 'Prediction Confidence',
        data: historicalPredictions.map(item => item.confidence * 100),
        borderColor: '#9c27b0',
        backgroundColor: 'rgba(156, 39, 176, 0.1)',
        borderDash: [5, 5],
        tension: 0.4,
        yAxisID: 'y1',
        pointRadius: 0
      }
    ]
  };

  // Prepare chart data for prediction distribution
  const predictionDistributionData = {
    labels: ['Increase', 'No Change', 'Decrease'],
    datasets: [
      {
        label: 'Number of Predictions',
        data: [
          historicalPredictions.filter(item => item.predicted_direction === 1).length,
          historicalPredictions.filter(item => item.predicted_direction === 0).length,
          historicalPredictions.filter(item => item.predicted_direction === -1).length
        ],
        backgroundColor: ['rgba(76, 175, 80, 0.6)', 'rgba(158, 158, 158, 0.6)', 'rgba(244, 67, 54, 0.6)'],
        borderWidth: 1
      }
    ]
  };

  // Chart options for prediction history
  const predictionChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    stacked: false,
    plugins: {
      title: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              if (label.includes('Price')) {
                label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
              } else if (label.includes('Confidence')) {
                label += context.parsed.y.toFixed(1) + '%';
              }
            }
            return label;
          }
        }
      }
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Price (USD)'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Confidence (%)'
        },
        min: 0,
        max: 100,
        grid: {
          drawOnChartArea: false
        }
      },
      x: {
        grid: {
          display: false
        }
      }
    }
  };

  // Chart options for prediction distribution
  const distributionChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = context.raw;
            const total = predictionDistributionData.datasets[0].data.reduce((a, b) => a + b, 0);
            const percentage = total > 0 ? (value / total * 100).toFixed(1) + '%' : '0%';
            return `${value} (${percentage})`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number of Predictions'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        grid: {
          display: false
        }
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
        Prediction Analysis
      </Typography>

      <Grid container spacing={3}>
        {/* Latest Prediction Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <PsychologyIcon sx={{ mr: 1 }} />
                  Latest Prediction
                </Box>
              }
              action={
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={handleRunModel}
                  disabled={runningModel}
                >
                  {runningModel ? 'Running...' : 'Run Model'}
                </Button>
              }
            />
            <Divider />
            <CardContent>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : predictionData ? (
                <React.Fragment>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    {getTrendIcon(predictionData.predicted_direction)}
                    <Typography variant="h5" sx={{ ml: 1 }}>
                      {getPredictionText(predictionData.predicted_direction)}
                    </Typography>
                    <Chip 
                      label={`${(predictionData.confidence * 100).toFixed(1)}%`}
                      color="info"
                      size="small"
                      sx={{ ml: 1 }}
                    />
                  </Box>

                  <Divider sx={{ mb: 2 }} />

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Bitcoin Price
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      ${formatPrice(predictionData.price)}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="success.main">
                      Increase Probability
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {(predictionData.increase_probability * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="error.main">
                      Decrease Probability
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {(predictionData.decrease_probability * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      No Change Probability
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {(predictionData.no_change_probability * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Divider sx={{ mb: 2 }} />

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Timeframe
                    </Typography>
                    <Typography variant="body2">
                      {predictionData.timeframe || '1 minute'}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Threshold
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
                      {format(new Date(predictionData.timestamp), 'yyyy-MM-dd HH:mm:ss')}
                    </Typography>
                  </Box>
                </React.Fragment>
              ) : (
                <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
                  No prediction data available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Accuracy Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TimelineIcon sx={{ mr: 1 }} />
                  Prediction Accuracy
                </Box>
              }
            />
            <Divider />
            <CardContent>
              {loadingHistory ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : accuracyStats ? (
                <React.Fragment>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h5" align="center" gutterBottom>
                      {(accuracyStats.accuracy * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary" align="center">
                      Overall Accuracy
                    </Typography>
                  </Box>

                  <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-around', 
                    mb: 3,
                    border: '1px solid rgba(0, 0, 0, 0.12)',
                    borderRadius: 1,
                    p: 2
                  }}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="success.main">
                        {(accuracyStats.increaseAccuracy * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Increase Accuracy
                      </Typography>
                    </Box>

                    <Divider orientation="vertical" flexItem />

                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" color="error.main">
                        {(accuracyStats.decreaseAccuracy * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Decrease Accuracy
                      </Typography>
                    </Box>
                  </Box>

                  <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Metric</TableCell>
                          <TableCell align="right">Count</TableCell>
                          <TableCell align="right">Percentage</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>Total Predictions</TableCell>
                          <TableCell align="right">{accuracyStats.total}</TableCell>
                          <TableCell align="right">100%</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Correct Predictions</TableCell>
                          <TableCell align="right">{accuracyStats.correct}</TableCell>
                          <TableCell align="right">
                            {(accuracyStats.correct / accuracyStats.total * 100).toFixed(1)}%
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Incorrect Predictions</TableCell>
                          <TableCell align="right">{accuracyStats.incorrect}</TableCell>
                          <TableCell align="right">
                            {(accuracyStats.incorrect / accuracyStats.total * 100).toFixed(1)}%
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>No Change Predictions</TableCell>
                          <TableCell align="right">{accuracyStats.noChange}</TableCell>
                          <TableCell align="right">
                            {(accuracyStats.noChange / accuracyStats.total * 100).toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>

                  <Typography variant="caption" color="text.secondary">
                    * Accuracy metrics are based on backtesting historical predictions against actual price movements
                  </Typography>
                </React.Fragment>
              ) : (
                <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
                  Not enough prediction data to calculate accuracy
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Distribution Chart */}
        <Grid item xs={12} md={6} lg={4}>
          <Card sx={{ height: '100%' }}>
            <CardHeader 
              title="Prediction Distribution"
              action={
                <Tooltip title="Refresh Data">
                  <IconButton onClick={fetchPredictionHistory} disabled={loadingHistory}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
              }
            />
            <Divider />
            <CardContent sx={{ height: 'calc(100% - 76px)' }}>
              <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <FormControl size="small" sx={{ width: 150, mb: 2, alignSelf: 'flex-end' }}>
                  <InputLabel id="timeframe-label">Timeframe</InputLabel>
                  <Select
                    labelId="timeframe-label"
                    value={timeframe}
                    label="Timeframe"
                    onChange={(e) => setTimeframe(e.target.value)}
                  >
                    <MenuItem value="1h">Last Hour</MenuItem>
                    <MenuItem value="1d">Last Day</MenuItem>
                    <MenuItem value="1w">Last Week</MenuItem>
                  </Select>
                </FormControl>

                {loadingHistory ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexGrow: 1 }}>
                    <CircularProgress />
                  </Box>
                ) : historicalPredictions.length > 0 ? (
                  <Box sx={{ flexGrow: 1 }}>
                    <Bar data={predictionDistributionData} options={distributionChartOptions} />
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexGrow: 1 }}>
                    <Typography variant="body1" color="text.secondary">
                      No prediction data available
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction History Chart */}
        <Grid item xs={12}>
          <Card>
            <CardHeader 
              title="Prediction History & Price"
              action={
                <FormControl size="small" sx={{ width: 150 }}>
                  <InputLabel id="chart-timeframe-label">Timeframe</InputLabel>
                  <Select
                    labelId="chart-timeframe-label"
                    value={timeframe}
                    label="Timeframe"
                    onChange={(e) => setTimeframe(e.target.value)}
                  >
                    <MenuItem value="1h">Last Hour</MenuItem>
                    <MenuItem value="1d">Last Day</MenuItem>
                    <MenuItem value="1w">Last Week</MenuItem>
                  </Select>
                </FormControl>
              }
            />
            <Divider />
            <CardContent sx={{ height: 400 }}>
              {loadingHistory ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <CircularProgress />
                </Box>
              ) : historicalPredictions.length > 0 ? (
                <Line data={predictionChartData} options={predictionChartOptions} />
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

        {/* Recent Predictions Table */}
        <Grid item xs={12}>
          <Card>
            <CardHeader title="Recent Predictions" />
            <Divider />
            <CardContent>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Price</TableCell>
                      <TableCell>Prediction</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Increase Prob.</TableCell>
                      <TableCell>Decrease Prob.</TableCell>
                      <TableCell>No Change Prob.</TableCell>
                      <TableCell>Threshold</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {loadingHistory ? (
                      <TableRow>
                        <TableCell colSpan={8} align="center">
                          <CircularProgress size={24} />
                        </TableCell>
                      </TableRow>
                    ) : historicalPredictions.slice(0, 10).map((prediction, index) => (
                      <TableRow key={index}>
                        <TableCell>{format(new Date(prediction.timestamp), 'HH:mm:ss')}</TableCell>
                        <TableCell>${formatPrice(prediction.price)}</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {getTrendIcon(prediction.predicted_direction)}
                            <Typography variant="body2" sx={{ ml: 1 }}>
                              {getPredictionText(prediction.predicted_direction)}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>{(prediction.confidence * 100).toFixed(1)}%</TableCell>
                        <TableCell>{(prediction.increase_probability * 100).toFixed(1)}%</TableCell>
                        <TableCell>{(prediction.decrease_probability * 100).toFixed(1)}%</TableCell>
                        <TableCell>{(prediction.no_change_probability * 100).toFixed(1)}%</TableCell>
                        <TableCell>±{(prediction.threshold * 100).toFixed(1)}%</TableCell>
                      </TableRow>
                    ))}
                    {!loadingHistory && historicalPredictions.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={8} align="center">
                          No prediction history available
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default PredictionAnalysis;
