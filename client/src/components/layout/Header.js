import React from 'react';
import { AppBar, Toolbar, Typography, Box, Chip, CircularProgress, Paper } from '@mui/material';
import { ArrowUpward, ArrowDownward, ArrowForward } from '@mui/icons-material';
import { format } from 'date-fns';

function Header({ marketData, predictionData, loading }) {
  // Format price with comma separators and fixed decimal places
  const formatPrice = (price) => {
    if (!price && price !== 0) return '--';
    return price.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  // Get direction icon based on prediction
  const getDirectionIcon = (direction) => {
    switch (direction) {
      case 1:
        return <ArrowUpward fontSize="small" sx={{ color: 'success.main' }} />;
      case -1:
        return <ArrowDownward fontSize="small" sx={{ color: 'error.main' }} />;
      default:
        return <ArrowForward fontSize="small" sx={{ color: 'text.secondary' }} />;
    }
  };

  // Get direction text based on prediction
  const getDirectionText = (direction) => {
    switch (direction) {
      case 1:
        return 'Increase';
      case -1:
        return 'Decrease';
      default:
        return 'No Change';
    }
  };

  // Get color based on prediction
  const getDirectionColor = (direction) => {
    switch (direction) {
      case 1:
        return 'success';
      case -1:
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <AppBar position="fixed" color="default" elevation={0} sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
          {/* Left side - Bitcoin price and market data */}
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="h6" component="div" sx={{ fontWeight: 'bold', mr: 2 }}>
              Bitcoin
            </Typography>

            {loading ? (
              <CircularProgress size={24} thickness={4} />
            ) : marketData ? (
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography variant="h5" component="div" sx={{ fontWeight: 'bold', mr: 2 }}>
                  ${formatPrice(marketData.last)}
                </Typography>
                <Box sx={{ display: { xs: 'none', sm: 'flex' }, alignItems: 'center', mr: 2 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
                    Bid: ${formatPrice(marketData.bid)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Ask: ${formatPrice(marketData.ask)}
                  </Typography>
                </Box>
              </Box>
            ) : (
              <Typography variant="body1" color="text.secondary">
                Market data unavailable
              </Typography>
            )}
          </Box>

          {/* Spacer */}
          <Box sx={{ flexGrow: 1 }} />

          {/* Right side - Prediction data */}
          {!loading && predictionData && (
            <Paper 
              elevation={0} 
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                py: 1, 
                px: 2, 
                borderRadius: 2,
                bgcolor: 'background.default'
              }}
            >
              <Box sx={{ mr: 2, display: { xs: 'none', md: 'block' } }}>
                <Typography variant="body2" color="text.secondary">
                  Prediction
                </Typography>
                <Typography variant="body2">
                  {predictionData.timestamp ? format(new Date(predictionData.timestamp), 'HH:mm:ss') : '--'}
                </Typography>
              </Box>

              <Chip
                icon={getDirectionIcon(predictionData.predicted_direction)}
                label={getDirectionText(predictionData.predicted_direction)}
                color={getDirectionColor(predictionData.predicted_direction)}
                size="small"
                variant="outlined"
                sx={{ mr: 1 }}
              />

              <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                <Chip
                  label={`${(predictionData.confidence * 100).toFixed(0)}% confidence`}
                  size="small"
                  color="info"
                  sx={{ height: 24 }}
                />
              </Box>
            </Paper>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;
