import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  AppBar, Box, Drawer, IconButton, List, ListItemButton, ListItemIcon, ListItemText, Toolbar, Tooltip, Typography,
  useMediaQuery,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import SpaceDashboardOutlinedIcon from '@mui/icons-material/SpaceDashboardOutlined';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import InsightsOutlinedIcon from '@mui/icons-material/InsightsOutlined';
import ReceiptLongOutlinedIcon from '@mui/icons-material/ReceiptLongOutlined';
import SettingsOutlinedIcon from '@mui/icons-material/SettingsOutlined';
import DarkModeOutlinedIcon from '@mui/icons-material/DarkModeOutlined';
import LightModeOutlinedIcon from '@mui/icons-material/LightModeOutlined';
import CircleIcon from '@mui/icons-material/Circle';
import { useHealth, useTicker } from '../api/hooks';
import { useRealtime } from '../realtime/RealtimeProvider';
import { useSettings } from '../settings/SettingsContext';
import { ModeBadge } from '../components/ModeBadge';
import { Change } from '../components/Direction';
import { formatUsd } from '../format';
import { status } from '../theme/tokens';

const DRAWER_WIDTH = 220;

const NAV = [
  { to: '/', label: 'Dashboard', icon: <SpaceDashboardOutlinedIcon /> },
  { to: '/trade', label: 'Trade', icon: <SwapHorizIcon /> },
  { to: '/predictions', label: 'Predictions', icon: <InsightsOutlinedIcon /> },
  { to: '/orders', label: 'Orders', icon: <ReceiptLongOutlinedIcon /> },
  { to: '/settings', label: 'Settings', icon: <SettingsOutlinedIcon /> },
];

const Connection = () => {
  const { connected } = useRealtime();
  return (
    <Tooltip title={connected ? 'Receiving live updates' : 'Live updates disconnected; falling back to polling'}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, typography: 'caption', color: 'text.secondary' }}>
        <CircleIcon sx={{ fontSize: 10, color: connected ? status.good : status.warning }} aria-hidden />
        {connected ? 'Live' : 'Reconnecting'}
      </Box>
    </Tooltip>
  );
};

const TickerSummary = () => {
  const { data: ticker } = useTicker();
  if (!ticker) return null;
  return (
    <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, minWidth: 0 }} aria-live="off">
      <Typography variant="body2" color="text.secondary" sx={{ display: { xs: 'none', sm: 'block' } }}>BTC-USD</Typography>
      <Typography sx={{ fontWeight: 700, fontSize: '1.05rem' }} className="num">{formatUsd(ticker.price)}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ display: { xs: 'none', sm: 'flex' } }}>
        <Change value={ticker.change24hPct} />&nbsp;24h
      </Typography>
    </Box>
  );
};

const Nav = ({ onNavigate }) => (
  <List sx={{ px: 1 }}>
    {NAV.map((item) => (
      <ListItemButton
        key={item.to}
        component={NavLink}
        to={item.to}
        end={item.to === '/'}
        onClick={onNavigate}
        sx={{ borderRadius: 1, mb: 0.5, '&.active': { bgcolor: 'action.selected', fontWeight: 700 } }}
      >
        <ListItemIcon sx={{ minWidth: 36 }}>{item.icon}</ListItemIcon>
        <ListItemText primary={item.label} />
      </ListItemButton>
    ))}
  </List>
);

export const AppShell = () => {
  const theme = useTheme();
  const desktop = useMediaQuery(theme.breakpoints.up('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const { data: health } = useHealth();
  const { themeMode, update } = useSettings();

  const brand = (
    <Toolbar sx={{ gap: 1 }}>
      <Typography variant="h2" component="div" sx={{ fontWeight: 700 }}>BTC Predictor</Typography>
    </Toolbar>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar position="fixed" sx={{ width: { md: `calc(100% - ${DRAWER_WIDTH}px)` }, ml: { md: `${DRAWER_WIDTH}px` } }}>
        <Toolbar sx={{ gap: 2 }}>
          {!desktop && (
            <IconButton edge="start" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
              <MenuIcon />
            </IconButton>
          )}
          <TickerSummary />
          <Box sx={{ flex: 1 }} />
          <Connection />
          <ModeBadge mode={health?.tradingMode} />
          <Tooltip title={themeMode === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}>
            <IconButton onClick={() => update({ themeMode: themeMode === 'dark' ? 'light' : 'dark' })} aria-label="Toggle theme">
              {themeMode === 'dark' ? <LightModeOutlinedIcon /> : <DarkModeOutlinedIcon />}
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Box component="nav" sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}>
        {desktop ? (
          <Drawer variant="permanent" open sx={{ '& .MuiDrawer-paper': { width: DRAWER_WIDTH } }}>
            {brand}
            <Nav />
          </Drawer>
        ) : (
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={() => setMobileOpen(false)}
            sx={{ '& .MuiDrawer-paper': { width: DRAWER_WIDTH } }}
          >
            {brand}
            <Nav onNavigate={() => setMobileOpen(false)} />
          </Drawer>
        )}
      </Box>

      <Box component="main" sx={{ flex: 1, minWidth: 0, p: { xs: 1.5, md: 3 } }}>
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  );
};
