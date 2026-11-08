import { createTheme } from '@mui/material/styles';
import { tokens, status, fontFamily } from './tokens';

export const createAppTheme = (mode) => {
  const t = tokens[mode];
  return createTheme({
    palette: {
      mode,
      primary: { main: t.accent },
      error: { main: status.critical },
      warning: { main: status.warning },
      success: { main: status.good },
      background: { default: t.page, paper: t.surface },
      text: { primary: t.textPrimary, secondary: t.textSecondary },
      divider: t.border,
      up: { main: t.up, contrastText: '#ffffff' },
      down: { main: t.down, contrastText: '#ffffff' },
    },
    tokens: t,
    shape: { borderRadius: 8 },
    typography: {
      fontFamily,
      h1: { fontSize: '1.5rem', fontWeight: 600 },
      h2: { fontSize: '1.125rem', fontWeight: 600 },
      h3: { fontSize: '0.9375rem', fontWeight: 600 },
      body2: { fontSize: '0.8125rem' },
      caption: { color: t.textMuted },
      button: { textTransform: 'none', fontWeight: 600 },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: { backgroundColor: t.page },
          '.num': { fontVariantNumeric: 'tabular-nums' },
        },
      },
      MuiPaper: {
        defaultProps: { elevation: 0 },
        styleOverrides: { root: { backgroundImage: 'none', border: `1px solid ${t.border}` } },
      },
      MuiAppBar: {
        styleOverrides: { root: { border: 0, borderBottom: `1px solid ${t.border}`, backgroundColor: t.surface, color: t.textPrimary } },
      },
      MuiDrawer: {
        styleOverrides: { paper: { border: 0, borderRight: `1px solid ${t.border}`, backgroundColor: t.surface } },
      },
      MuiTableCell: {
        styleOverrides: {
          root: { borderBottom: `1px solid ${t.grid}`, padding: '6px 12px' },
          head: { color: t.textMuted, fontWeight: 600, fontSize: '0.75rem' },
        },
      },
      MuiButton: { defaultProps: { disableElevation: true } },
      MuiToggleButton: { styleOverrides: { root: { textTransform: 'none', fontWeight: 600, padding: '4px 12px' } } },
    },
  });
};
