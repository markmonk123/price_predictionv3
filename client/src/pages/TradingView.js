import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormHelperText,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Alert,
  Tooltip
} from '@mui/material';
import {
  ShoppingCart as OrderIcon,
  Delete as DeleteIcon,
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { format } from 'date-fns';

import { placeOrder, getOrders, cancelOrder, getOrderStatusText, getOrderStatusColor, getSideColor } from '../services/orderService';
import { formatPrice } from '../services/marketService';
import { getPredictionColor, getPredictionText } from '../services/predictionService';

function TradingView({ socket, marketData, predictionData }) {
  // Order form state
  const [orderType, setOrderType] = useState('MARKET');
  const [side, setSide] = useState('BUY');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [timeInForce, setTimeInForce] = useState('GTC');

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [orders, setOrders] = useState([]);
  const [loadingOrders, setLoadingOrders] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [orderError, setOrderError] = useState('');
  const [orderSuccess, setOrderSuccess] = useState('');

  // Load orders on component mount
  useEffect(() => {
    fetchOrders();
  }, []);

  // Update price field when market data changes
  useEffect(() => {
    if (marketData && marketData.last) {
      setPrice(marketData.last.toFixed(2));
    }
  }, [marketData]);

  // Fetch orders from API
  const fetchOrders = async () => {
    try {
      setLoadingOrders(true);
      const response = await getOrders();
      setOrders(response.data);
      setLoadingOrders(false);
    } catch (error) {
      console.error('Error fetching orders:', error);
      setLoadingOrders(false);
    }
  };

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Reset status messages
    setOrderError('');
    setOrderSuccess('');

    // Validate inputs
    if (!quantity || parseFloat(quantity) <= 0) {
      setOrderError('Please enter a valid quantity');
      return;
    }

    if (orderType === 'LIMIT' && (!price || parseFloat(price) <= 0)) {
      setOrderError('Please enter a valid price for limit orders');
      return;
    }

    try {
      setSubmitting(true);

      // Create order object
      const orderData = {
        symbol: 'BTC/USD',
        side,
        orderType,
        quantity: parseFloat(quantity),
        timeInForce
      };

      // Add price for limit orders
      if (orderType === 'LIMIT') {
        orderData.price = parseFloat(price);
      }

      // Submit order via API
      const response = await placeOrder(orderData);

      // Display success message
      setOrderSuccess(`Order successfully submitted with ID: ${response.data.orderId}`);

      // Refresh orders list
      fetchOrders();

      // Reset form if desired
      // setQuantity('');

    } catch (error) {
      console.error('Error submitting order:', error);
      setOrderError(error.response?.data?.error || 'Failed to submit order. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  // Handle order cancellation
  const handleCancelOrder = async (orderId) => {
    try {
      await cancelOrder(orderId);
      fetchOrders(); // Refresh the orders list
    } catch (error) {
      console.error('Error canceling order:', error);
      setOrderError('Failed to cancel order. Please try again.');
    }
  };

  // Calculate total order value
  const calculateTotal = () => {
    if (!quantity || quantity <= 0) return '0.00';

    let calculatedPrice = price;
    if (!calculatedPrice && marketData) {
      calculatedPrice = side === 'BUY' ? marketData.ask : marketData.bid;
    }

    if (!calculatedPrice) return '0.00';

    return (parseFloat(quantity) * parseFloat(calculatedPrice)).toFixed(2);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Trading View
      </Typography>

      <Grid container spacing={3}>
        {/* Order Entry Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <OrderIcon sx={{ mr: 1 }} />
                  Order Entry
                </Box>
              }
            />
            <Divider />
            <CardContent>
              {orderError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {orderError}
                </Alert>
              )}

              {orderSuccess && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  {orderSuccess}
                </Alert>
              )}

              <form onSubmit={handleSubmit}>
                <FormControl fullWidth margin="normal">
                  <InputLabel id="side-label">Side</InputLabel>
                  <Select
                    labelId="side-label"
                    value={side}
                    label="Side"
                    onChange={(e) => setSide(e.target.value)}
                  >
                    <MenuItem value="BUY">
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <ArrowUpIcon sx={{ color: 'success.main', mr: 1 }} />
                        Buy
                      </Box>
                    </MenuItem>
                    <MenuItem value="SELL">
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <ArrowDownIcon sx={{ color: 'error.main', mr: 1 }} />
                        Sell
                      </Box>
                    </MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth margin="normal">
                  <InputLabel id="order-type-label">Order Type</InputLabel>
                  <Select
                    labelId="order-type-label"
                    value={orderType}
                    label="Order Type"
                    onChange={(e) => setOrderType(e.target.value)}
                  >
                    <MenuItem value="MARKET">Market</MenuItem>
                    <MenuItem value="LIMIT">Limit</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  fullWidth
                  margin="normal"
                  label="Quantity (BTC)"
                  type="number"
                  InputProps={{ inputProps: { min: 0, step: 0.001 } }}
                  value={quantity}
                  onChange={(e) => setQuantity(e.target.value)}
                  required
                />

                {orderType === 'LIMIT' && (
                  <TextField
                    fullWidth
                    margin="normal"
                    label="Price (USD)"
                    type="number"
                    InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                    value={price}
                    onChange={(e) => setPrice(e.target.value)}
                    required
                  />
                )}

                {orderType === 'LIMIT' && (
                  <FormControl fullWidth margin="normal">
                    <InputLabel id="tif-label">Time In Force</InputLabel>
                    <Select
                      labelId="tif-label"
                      value={timeInForce}
                      label="Time In Force"
                      onChange={(e) => setTimeInForce(e.target.value)}
                    >
                      <MenuItem value="DAY">Day</MenuItem>
                      <MenuItem value="GTC">Good Till Cancel</MenuItem>
                      <MenuItem value="IOC">Immediate or Cancel</MenuItem>
                      <MenuItem value="FOK">Fill or Kill</MenuItem>
                    </Select>
                  </FormControl>
                )}

                <Box sx={{ mt: 2, mb: 3 }}>
                  <Typography variant="body2" color="text.secondary">
                    Order Total
                  </Typography>
                  <Typography variant="h6">
                    ${calculateTotal()}
                  </Typography>
                </Box>

                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  color={side === 'BUY' ? 'success' : 'error'}
                  type="submit"
                  disabled={submitting}
                >
                  {submitting ? 'Submitting...' : side === 'BUY' ? 'Buy Bitcoin' : 'Sell Bitcoin'}
                </Button>
              </form>
            </CardContent>
          </Card>
        </Grid>

        {/* Market Information Panel */}
        <Grid item xs={12} md={8}>
          <Card sx={{ mb: 3 }}>
            <CardHeader 
              title="Current Market Information"
              action={
                <IconButton onClick={fetchOrders}>
                  <RefreshIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Last Price
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                      ${marketData ? formatPrice(marketData.last) : '--'}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="success.main">
                      Bid
                    </Typography>
                    <Typography variant="body1" color="success.main" fontWeight="medium">
                      ${marketData ? formatPrice(marketData.bid) : '--'}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="error.main">
                      Ask
                    </Typography>
                    <Typography variant="body1" color="error.main" fontWeight="medium">
                      ${marketData ? formatPrice(marketData.ask) : '--'}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      Spread
                    </Typography>
                    <Typography variant="body2">
                      {marketData ? `$${formatPrice(marketData.ask - marketData.bid)} (${((marketData.ask - marketData.bid) / marketData.mid * 100).toFixed(3)}%)` : '--'}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Prediction (1 Minute)
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Chip
                        label={predictionData ? getPredictionText(predictionData.predicted_direction) : 'N/A'}
                        color={predictionData ? getOrderStatusColor(getPredictionText(predictionData.predicted_direction).toLowerCase()) : 'default'}
                        sx={{ mr: 1 }}
                      />
                      <Typography variant="body1">
                        {predictionData ? `${(predictionData.confidence * 100).toFixed(1)}% confidence` : ''}
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="success.main">
                      Increase Probability
                    </Typography>
                    <Typography variant="body2">
                      {predictionData ? `${(predictionData.increase_probability * 100).toFixed(1)}%` : '--'}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="error.main">
                      Decrease Probability
                    </Typography>
                    <Typography variant="body2">
                      {predictionData ? `${(predictionData.decrease_probability * 100).toFixed(1)}%` : '--'}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      No Change Probability
                    </Typography>
                    <Typography variant="body2">
                      {predictionData ? `${(predictionData.no_change_probability * 100).toFixed(1)}%` : '--'}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Orders Panel */}
          <Card>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange}>
                <Tab label="Open Orders" />
                <Tab label="Order History" />
              </Tabs>
            </Box>
            <CardContent>
              {tabValue === 0 && (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Order ID</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Side</TableCell>
                        <TableCell>Quantity</TableCell>
                        <TableCell>Price</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Time</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {loadingOrders ? (
                        <TableRow>
                          <TableCell colSpan={8} align="center">Loading orders...</TableCell>
                        </TableRow>
                      ) : orders.filter(order => ['NEW', 'PARTIALLY_FILLED'].includes(order.status)).length > 0 ? (
                        orders.filter(order => ['NEW', 'PARTIALLY_FILLED'].includes(order.status)).map((order) => (
                          <TableRow key={order.orderId}>
                            <TableCell sx={{ fontFamily: 'monospace' }}>{order.orderId}</TableCell>
                            <TableCell>{order.orderType}</TableCell>
                            <TableCell>
                              <Typography sx={{ color: getSideColor(order.side) }}>
                                {order.side}
                              </Typography>
                            </TableCell>
                            <TableCell>{order.quantity}</TableCell>
                            <TableCell>
                              {order.orderType === 'LIMIT' ? 
                                `$${formatPrice(order.price)}` : 
                                'Market'}
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={getOrderStatusText(order.status)} 
                                color={getOrderStatusColor(order.status)} 
                                size="small" 
                              />
                            </TableCell>
                            <TableCell>{format(new Date(order.timestamp), 'HH:mm:ss')}</TableCell>
                            <TableCell>
                              <Tooltip title="Cancel Order">
                                <IconButton 
                                  size="small" 
                                  onClick={() => handleCancelOrder(order.orderId)}
                                  disabled={order.status === 'FILLED'}
                                >
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={8} align="center">No open orders</TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}

              {tabValue === 1 && (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Order ID</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Side</TableCell>
                        <TableCell>Quantity</TableCell>
                        <TableCell>Price</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {loadingOrders ? (
                        <TableRow>
                          <TableCell colSpan={7} align="center">Loading order history...</TableCell>
                        </TableRow>
                      ) : orders.filter(order => !['NEW', 'PARTIALLY_FILLED'].includes(order.status)).length > 0 ? (
                        orders.filter(order => !['NEW', 'PARTIALLY_FILLED'].includes(order.status)).map((order) => (
                          <TableRow key={order.orderId}>
                            <TableCell sx={{ fontFamily: 'monospace' }}>{order.orderId}</TableCell>
                            <TableCell>{order.orderType}</TableCell>
                            <TableCell>
                              <Typography sx={{ color: getSideColor(order.side) }}>
                                {order.side}
                              </Typography>
                            </TableCell>
                            <TableCell>{order.quantity}</TableCell>
                            <TableCell>
                              {order.orderType === 'LIMIT' ? 
                                `$${formatPrice(order.price)}` : 
                                'Market'}
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={getOrderStatusText(order.status)} 
                                color={getOrderStatusColor(order.status)} 
                                size="small" 
                              />
                            </TableCell>
                            <TableCell>{format(new Date(order.timestamp), 'HH:mm:ss')}</TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={7} align="center">No order history</TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default TradingView;
