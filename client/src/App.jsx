import { useMemo } from 'react';
import { Route, Routes } from 'react-router-dom';
import { CssBaseline } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { createAppTheme } from './theme/theme';
import { useSettings } from './settings/SettingsContext';
import { RealtimeProvider } from './realtime/RealtimeProvider';
import { AppShell } from './layout/AppShell';
import { Dashboard } from './pages/Dashboard';
import { Trade } from './pages/Trade';
import { Predictions } from './pages/Predictions';
import { Orders } from './pages/Orders';
import { Settings } from './pages/Settings';
import { Empty } from './components/QueryState';

export const App = () => {
  const { themeMode } = useSettings();
  const theme = useMemo(() => createAppTheme(themeMode), [themeMode]);
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <RealtimeProvider>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Dashboard />} />
            <Route path="trade" element={<Trade />} />
            <Route path="predictions" element={<Predictions />} />
            <Route path="orders" element={<Orders />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<Empty>Page not found.</Empty>} />
          </Route>
        </Routes>
      </RealtimeProvider>
    </ThemeProvider>
  );
};
