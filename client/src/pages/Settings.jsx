import { useState } from 'react';
import {
  Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, IconButton,
  InputAdornment, TextField, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import { useHealth, useResetPaper } from '../api/hooks';
import { describeError } from '../api/client';
import { useSettings } from '../settings/SettingsContext';
import { Panel } from '../components/Panel';
import { ModeBadge } from '../components/ModeBadge';
import { QueryState } from '../components/QueryState';
import { formatTime, formatUsd } from '../format';

const Fact = ({ label, children }) => (
  <Box sx={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: 2, py: 0.75, borderBottom: 1, borderColor: 'divider' }}>
    <Typography variant="body2" color="text.secondary">{label}</Typography>
    <Box sx={{ typography: 'body2', minWidth: 0, overflowWrap: 'anywhere' }}>{children}</Box>
  </Box>
);

const ServerPanel = () => {
  const query = useHealth();
  return (
    <Panel title="Server" subtitle="Configured with environment variables on the server (see .env.example)">
      <QueryState query={query}>
        {(h) => (
          <Box>
            <Fact label="Trading mode"><ModeBadge mode={h.tradingMode} /></Fact>
            <Fact label="Brokers">{h.brokers.join(', ')}</Fact>
            <Fact label="Live order cap">{formatUsd(h.maxOrderUsd)} per order</Fact>
            <Fact label="Paper fee">{h.paperFeeBps / 100}%</Fact>
            <Fact label="API token">{h.authRequired ? 'Required for orders and model runs' : 'Not required (local paper mode)'}</Fact>
            <Fact label="Market data">{h.market.ok ? `${h.market.source} · OK` : `Unavailable: ${h.market.error}`}</Fact>
            <Fact label="Model">
              {h.model.running ? 'Running now' : h.model.lastRunAt ? `Last run ${formatTime(h.model.lastRunAt)}` : 'Not run yet'}
              {h.model.lastError && <Alert severity="error" sx={{ mt: 1 }}>Last run failed: {h.model.lastError.message}</Alert>}
            </Fact>
            <Fact label="Prediction ledger">{h.ledger.available ? 'Available' : 'Not created yet (run priceprediction.py)'}</Fact>
          </Box>
        )}
      </QueryState>
    </Panel>
  );
};

const TokenPanel = () => {
  const { apiToken, update } = useSettings();
  const [show, setShow] = useState(false);
  return (
    <Panel title="API token" subtitle="Sent as a Bearer token with orders and model runs. Stored only in this browser.">
      <TextField
        fullWidth
        size="small"
        label="API_TOKEN"
        type={show ? 'text' : 'password'}
        value={apiToken}
        onChange={(e) => update({ apiToken: e.target.value })}
        autoComplete="off"
        slotProps={{
          input: {
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={() => setShow((s) => !s)} aria-label={show ? 'Hide token' : 'Show token'} edge="end">
                  {show ? <VisibilityOffIcon /> : <VisibilityIcon />}
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
      />
    </Panel>
  );
};

const AppearancePanel = () => {
  const { themeMode, update } = useSettings();
  return (
    <Panel title="Appearance">
      <ToggleButtonGroup size="small" exclusive value={themeMode} onChange={(_, v) => v && update({ themeMode: v })} aria-label="Theme">
        <ToggleButton value="dark">Dark</ToggleButton>
        <ToggleButton value="light">Light</ToggleButton>
      </ToggleButtonGroup>
    </Panel>
  );
};

const PaperResetPanel = () => {
  const reset = useResetPaper();
  const [open, setOpen] = useState(false);
  return (
    <Panel title="Paper account" subtitle="Cancel every open paper order and restore the starting balance">
      {reset.isError && <Alert severity="error" sx={{ mb: 1 }}>{describeError(reset.error)}</Alert>}
      {reset.isSuccess && <Alert severity="success" sx={{ mb: 1 }}>Paper account reset.</Alert>}
      <Button variant="outlined" color="error" onClick={() => setOpen(true)} disabled={reset.isPending}>Reset paper account</Button>
      <Dialog open={open} onClose={() => setOpen(false)}>
        <DialogTitle>Reset the paper account?</DialogTitle>
        <DialogContent>
          <DialogContentText>Open paper orders are cancelled and balances return to the starting amount. Order history is kept.</DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Keep</Button>
          <Button color="error" variant="contained" onClick={() => { setOpen(false); reset.mutate(); }}>Reset</Button>
        </DialogActions>
      </Dialog>
    </Panel>
  );
};

const McpPanel = () => (
  <Panel title="MCP server" subtitle="Lets Claude Code / VS Code query predictions and paper-trade through this server">
    <Typography variant="body2" sx={{ mb: 1 }}>
      Configured in <code>.mcp.json</code> (Claude Code) and <code>.vscode/mcp.json</code> (VS Code). Keep this server running,
      then open <code>/mcp</code> in Claude Code. Order tools are pinned to the paper broker; the MCP server cannot place real orders.
    </Typography>
  </Panel>
);

export const Settings = () => (
  <Box sx={{ display: 'grid', gap: 2, maxWidth: 900 }}>
    <Typography variant="h1">Settings</Typography>
    <ServerPanel />
    <TokenPanel />
    <AppearancePanel />
    <PaperResetPanel />
    <McpPanel />
  </Box>
);
