import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Chip,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  Tabs,
  Tab,
  TextField
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  ReceiptLong as ReceiptIcon,
  Code as CodeIcon,
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon
} from '@mui/icons-material';

import { getOrders, getOrderStatusText, getOrderStatusColor, getSideColor } from '../services/orderService';
import { formatPrice } from '../services/marketService';
import { formatFixMessageForDisplay, parseFixMessage } from '../services/fixService';

function OrderBook({ socket, marketData }) {
  const [orders, setOrders] = useState([]);
  const [loadingOrders, setLoadingOrders] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [fixMessages, setFixMessages] = useState([]);

  // Load orders on component mount
  useEffect(() => {
    fetchOrders();
    initializeFixMessages();
  }, []);

  // Handle FIX messages from socket
  useEffect(() => {
    if (socket) {
      socket.on('fixMessage', (message) => {
        addFixMessage(message);
      });

      return () => {
        socket.off('fixMessage');
      };
    }
  }, [socket]);

  // Initialize simulated FIX messages
  const initializeFixMessages = () => {
    const simulatedMessages = [
      // Market Data FIX messages
      '8=FIX.4.4\u00019=219\u000135=W\u000134=12\u000149=EXCHANGE\u000156=BITCOIN_PREDICTION_CLIENT\u000152=20250817-12:34:56.789\u000155=BTC/USD\u0001268=2\u0001269=0\u0001270=59876.50\u0001271=10.5\u0001272=20250817-12:34:56.789\u0001269=1\u0001270=59885.25\u0001271=8.2\u0001272=20250817-12:34:56.789\u000110=231\u0001',
      // Order Acknowledgment
      '8=FIX.4.4\u00019=253\u000135=8\u000134=15\u000149=EXCHANGE\u000156=BITCOIN_PREDICTION_CLIENT\u000152=20250817-12:35:01.123\u000137=ORDER123\u000111=CLIENT456\u000117=EXEC789\u000120=0\u000139=0\u000155=BTC/USD\u000154=1\u000138=1.5\u000140=2\u000144=60000.00\u000159=1\u000160=20250817-12:35:01.123\u0001150=0\u0001151=1.5\u000110=021\u0001',
      // Order Fill
      '8=FIX.4.4\u00019=267\u000135=8\u000134=16\u000149=EXCHANGE\u000156=BITCOIN_PREDICTION_CLIENT\u000152=20250817-12:35:05.456\u000137=ORDER123\u000111=CLIENT456\u000117=EXEC790\u000120=0\u000139=2\u000155=BTC/USD\u000154=1\u000138=1.5\u000140=2\u000144=60000.00\u000159=1\u000160=20250817-12:35:01.123\u0001150=F\u0001151=0\u000132=1.5\u000131=60010.50\u000110=058\u0001'
    ];

    setFixMessages(simulatedMessages.map(msg => ({
      message: msg,
      timestamp: new Date(),
      parsed: parseFixMessage(msg)
    })));
  };

  // Add a new FIX message
  const addFixMessage = (message) => {
    setFixMessages(prevMessages => [
      {
        message,
        timestamp: new Date(),
        parsed: parseFixMessage(message)
      },
      ...prevMessages
    ]);
  };

  // Simulate adding a new FIX message
  const simulateNewFixMessage = () => {
    // Simulated market data update
    const basePrice = 60000 + (Math.random() * 1000 - 500);
    const bidPrice = (basePrice - (Math.random() * 10)).toFixed(2);
    const askPrice = (basePrice + (Math.random() * 10)).toFixed(2);
    const bidQty = (Math.random() * 15).toFixed(2);
    const askQty = (Math.random() * 15).toFixed(2);
    const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').substring(0, 21);

    const message = `8=FIX.4.4\u00019=219\u000135=W\u000134=${Math.floor(Math.random() * 1000)}\u000149=EXCHANGE\u000156=BITCOIN_PREDICTION_CLIENT\u000152=${timestamp}\u000155=BTC/USD\u0001268=2\u0001269=0\u0001270=${bidPrice}\u0001271=${bidQty}\u0001272=${timestamp}\u0001269=1\u0001270=${askPrice}\u0001271=${askQty}\u0001272=${timestamp}\u000110=231\u0001`;

    addFixMessage(message);
  };

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

  // Generate a simulated order book
  const generateOrderBook = () => {
    if (!marketData) return { bids: [], asks: [] };

    const basePrice = marketData.last;
    const bids = [];
    const asks = [];

    // Generate 10 bids below current price
    for (let i = 0; i < 10; i++) {
      const price = basePrice - (i * 10) - (Math.random() * 5);
      const quantity = 10 + (Math.random() * 20);
      bids.push({
        price,
        quantity,
        total: price * quantity,
        orders: Math.floor(Math.random() * 10) + 1
      });
    }

    // Generate 10 asks above current price
    for (let i = 0; i < 10; i++) {
      const price = basePrice + (i * 10) + (Math.random() * 5);
      const quantity = 10 + (Math.random() * 20);
      asks.push({
        price,
        quantity,
        total: price * quantity,
        orders: Math.floor(Math.random() * 10) + 1
      });
    }

    return { bids, asks };
  };

  const orderBook = generateOrderBook();

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Order Book & FIX Protocol
      </Typography>

      <Grid container spacing={3}>
        {/* Order Book */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <ReceiptIcon sx={{ mr: 1 }} />
                  Order Book
                </Box>
              }
              action={
                <IconButton onClick={fetchOrders}>
                  <RefreshIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                {/* Ask Orders */}
                <Grid item xs={6}>
                  <Typography variant="subtitle2" gutterBottom color="error.main" sx={{ display: 'flex', alignItems: 'center' }}>
                    <ArrowDownIcon fontSize="small" sx={{ mr: 0.5 }} />
                    Sell Orders
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Price</TableCell>
                          <TableCell align="right">Quantity</TableCell>
                          <TableCell align="right">Total</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {orderBook.asks.map((ask, index) => (
                          <TableRow key={index}>
                            <TableCell sx={{ color: 'error.main' }}>
                              ${formatPrice(ask.price)}
                            </TableCell>
                            <TableCell align="right">{ask.quantity.toFixed(3)}</TableCell>
                            <TableCell align="right">${formatPrice(ask.total)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>

                {/* Bid Orders */}
                <Grid item xs={6}>
                  <Typography variant="subtitle2" gutterBottom color="success.main" sx={{ display: 'flex', alignItems: 'center' }}>
                    <ArrowUpIcon fontSize="small" sx={{ mr: 0.5 }} />
                    Buy Orders
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Price</TableCell>
                          <TableCell align="right">Quantity</TableCell>
                          <TableCell align="right">Total</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {orderBook.bids.map((bid, index) => (
                          <TableRow key={index}>
                            <TableCell sx={{ color: 'success.main' }}>
                              ${formatPrice(bid.price)}
                            </TableCell>
                            <TableCell align="right">{bid.quantity.toFixed(3)}</TableCell>
                            <TableCell align="right">${formatPrice(bid.total)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>

              {/* Spread Information */}
              <Box sx={{ 
                mt: 3, 
                p: 2, 
                bgcolor: 'background.default', 
                borderRadius: 1,
                display: 'flex',
                justifyContent: 'space-around'
              }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">Last Price</Typography>
                  <Typography variant="h6">
                    ${marketData ? formatPrice(marketData.last) : '--'}
                  </Typography>
                </Box>

                <Divider orientation="vertical" flexItem />

                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">Spread</Typography>
                  <Typography variant="h6">
                    {marketData ? 
                      `$${formatPrice(marketData.ask - marketData.bid)}` : 
                      '--'}
                  </Typography>
                </Box>

                <Divider orientation="vertical" flexItem />

                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">24h Volume</Typography>
                  <Typography variant="h6">
                    {marketData && marketData.volume ? 
                      `${marketData.volume.toFixed(2)} BTC` : 
                      '--'}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* FIX Protocol Section */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <CodeIcon sx={{ mr: 1 }} />
                  FIX Protocol Messages
                </Box>
              }
              action={
                <Button 
                  variant="outlined" 
                  size="small" 
                  startIcon={<RefreshIcon />}
                  onClick={simulateNewFixMessage}
                >
                  Simulate Message
                </Button>
              }
            />
            <Divider />
            <CardContent>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  About FIX Protocol
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  The Financial Information eXchange (FIX) protocol is an electronic communications protocol initiated in 1992 for international real-time exchange of information related to securities transactions and markets.
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <TextField
                  fullWidth
                  label="Search FIX Messages"
                  variant="outlined"
                  size="small"
                  placeholder="Search by tag or value..."
                />
              </Box>

              {fixMessages.length > 0 ? (
                fixMessages.map((fixMessage, index) => (
                  <Accordion key={index} sx={{ mb: 1 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Chip 
                            label={fixMessage.parsed.find(p => p.tag === '35')?.value || 'Unknown'} 
                            size="small" 
                            color="primary" 
                            sx={{ mr: 1 }}
                          />
                          <Typography variant="body2">
                            {fixMessage.parsed.find(p => p.tag === '55')?.value || 'BTC/USD'}
                          </Typography>
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          {fixMessage.timestamp.toLocaleTimeString()}
                        </Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Raw FIX Message:
                        </Typography>
                        <Box 
                          className="fix-message"
                          dangerouslySetInnerHTML={{ 
                            __html: formatFixMessageForDisplay(fixMessage.message) 
                          }}
                        />

                        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                          Parsed Fields:
                        </Typography>
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Tag</TableCell>
                                <TableCell>Name</TableCell>
                                <TableCell>Value</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {fixMessage.parsed.map((field, fieldIndex) => (
                                <TableRow key={fieldIndex}>
                                  <TableCell>{field.tag}</TableCell>
                                  <TableCell>{field.tagName}</TableCell>
                                  <TableCell>{field.value}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                ))
              ) : (
                <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
                  No FIX messages available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Orders Section */}
        <Grid item xs={12}>
          <Card>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange}>
                <Tab label="Open Orders" />
                <Tab label="Order History" />
                <Tab label="Filled Orders" />
              </Tabs>
            </Box>
            <CardContent>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Order ID</TableCell>
                      <TableCell>Symbol</TableCell>
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
                        <TableCell colSpan={8} align="center">
                          <CircularProgress size={24} />
                        </TableCell>
                      </TableRow>
                    ) : orders.length > 0 ? (
                      orders
                        .filter(order => {
                          if (tabValue === 0) return ['NEW', 'PARTIALLY_FILLED'].includes(order.status);
                          if (tabValue === 1) return true; // All orders in history
                          if (tabValue === 2) return order.status === 'FILLED';
                          return false;
                        })
                        .map((order, index) => (
                          <TableRow key={index}>
                            <TableCell sx={{ fontFamily: 'monospace' }}>{order.orderId}</TableCell>
                            <TableCell>{order.symbol}</TableCell>
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
                            <TableCell>
                              {new Date(order.timestamp).toLocaleString()}
                            </TableCell>
                          </TableRow>
                        ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={8} align="center">
                          No orders found
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

export default OrderBook;
