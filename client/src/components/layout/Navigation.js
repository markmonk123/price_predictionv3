import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
  Box,
  Typography,
  Tooltip,
  styled
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  ShowChart as ChartIcon,
  Insights as InsightsIcon,
  Receipt as ReceiptIcon,
  Settings as SettingsIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
  Info as InfoIcon,
  AccountBalance as AccountIcon
} from '@mui/icons-material';

// Custom styled components
const StyledDrawer = styled(Drawer)(({ theme, open }) => ({
  width: open ? 240 : 72,
  flexShrink: 0,
  whiteSpace: 'nowrap',
  boxSizing: 'border-box',
  '& .MuiDrawer-paper': {
    width: open ? 240 : 72,
    transition: theme.transitions.create('width', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen,
    }),
    backgroundColor: theme.palette.background.dark,
    color: theme.palette.common.white,
    overflowX: 'hidden',
  },
}));

const StyledListItem = styled(ListItem)(({ theme, selected }) => ({
  marginBottom: 4,
  marginTop: 4,
  marginLeft: 8,
  marginRight: 8,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: selected ? theme.palette.primary.main : 'transparent',
  '&:hover': {
    backgroundColor: selected ? theme.palette.primary.main : 'rgba(255, 255, 255, 0.08)',
  },
}));

function Navigation() {
  const [open, setOpen] = useState(true);
  const location = useLocation();

  const handleDrawerToggle = () => {
    setOpen(!open);
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Trading View', icon: <ChartIcon />, path: '/trading' },
    { text: 'Prediction Analysis', icon: <InsightsIcon />, path: '/predictions' },
    { text: 'Order Book', icon: <ReceiptIcon />, path: '/orders' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' }
  ];

  return (
    <StyledDrawer
      variant="permanent"
      open={open}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'space-between' : 'center',
          padding: '16px',
        }}
      >
        {open && (
          <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center' }}>
            <AccountIcon sx={{ mr: 1 }} />
            FIX Trading
          </Typography>
        )}
        {!open && <AccountIcon />}
        <IconButton onClick={handleDrawerToggle} sx={{ color: 'white' }}>
          {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </Box>
      <Divider sx={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
      <List sx={{ mt: 2 }}>
        {menuItems.map((item) => {
          const isSelected = location.pathname === item.path;
          return (
            <Tooltip 
              key={item.text} 
              title={open ? '' : item.text} 
              placement="right"
            >
              <StyledListItem
                button
                component={Link}
                to={item.path}
                selected={isSelected}
              >
                <ListItemIcon sx={{ color: isSelected ? 'white' : 'rgba(255, 255, 255, 0.7)', minWidth: 40 }}>
                  {item.icon}
                </ListItemIcon>
                {open && (
                  <ListItemText 
                    primary={item.text} 
                    sx={{ 
                      '& .MuiListItemText-primary': {
                        color: isSelected ? 'white' : 'rgba(255, 255, 255, 0.7)'
                      }
                    }} 
                  />
                )}
              </StyledListItem>
            </Tooltip>
          );
        })}
      </List>
      <Box sx={{ flexGrow: 1 }} />
      <Divider sx={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'flex-start' : 'center',
        }}
      >
        <InfoIcon sx={{ color: 'rgba(255, 255, 255, 0.7)', mr: open ? 1 : 0 }} />
        {open && (
          <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
            FIX Protocol v4.4
          </Typography>
        )}
      </Box>
    </StyledDrawer>
  );
}

export default Navigation;
