/**
 * Order ticket: build → preview → place.
 *
 * - Every order is previewed first (price, size, fees, total, rejection reasons).
 *   Editing any field discards the preview, so what you place is what you saw.
 * - The limit price is prefilled once from the book when you switch to LIMIT and
 *   is never overwritten by live ticks afterwards.
 * - A real-money (Coinbase) order requires an explicit confirmation dialog.
 */
import { useState } from 'react';
import {
  Alert, Box, Button, Checkbox, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle,
  FormControlLabel, InputAdornment, MenuItem, TextField, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useHealth, usePlaceOrder, usePreviewOrder, useTicker } from '../api/hooks';
import { describeError } from '../api/client';
import { formatBtc, formatUsd } from '../format';
import { Panel } from './Panel';

const emptyForm = { side: 'BUY', type: 'MARKET', unit: 'USD', size: '', limitPrice: '', postOnly: false };

/** Validate the form and build the API order body. Returns { order } or { error }. */
export const buildOrder = (form, broker) => {
  const size = Number(form.size);
  if (!form.size || !Number.isFinite(size) || size <= 0) return { error: 'Enter a size greater than zero' };
  const order = { broker, side: form.side, type: form.type };
  if (form.type === 'LIMIT') {
    const limitPrice = Number(form.limitPrice);
    if (!form.limitPrice || !Number.isFinite(limitPrice) || limitPrice <= 0) return { error: 'Enter a limit price' };
    return { order: { ...order, baseSize: size, limitPrice, postOnly: form.postOnly } };
  }
  return { order: form.unit === 'USD' ? { ...order, quoteSize: size } : { ...order, baseSize: size } };
};

const Row = ({ label, children, strong }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 0.25 }}>
    <Typography variant="body2" color="text.secondary">{label}</Typography>
    <Typography variant="body2" className="num" sx={{ fontWeight: strong ? 700 : 500 }}>{children}</Typography>
  </Box>
);

export const OrderTicket = () => {
  const { tokens } = useTheme();
  const { data: health } = useHealth();
  const { data: ticker } = useTicker();
  const preview = usePreviewOrder();
  const place = usePlaceOrder();
  const [broker, setBroker] = useState('paper');
  const [form, setForm] = useState(emptyForm);
  const [formError, setFormError] = useState('');
  const [confirmOpen, setConfirmOpen] = useState(false);

  const liveEnabled = Boolean(health?.liveTradingEnabled);
  const activeBroker = liveEnabled ? broker : 'paper';
  const isLive = activeBroker === 'coinbase';

  const change = (patch) => {
    setForm((prev) => {
      const next = { ...prev, ...patch };
      if (patch.type === 'LIMIT') {
        next.unit = 'BTC';
        if (!prev.limitPrice && ticker) {
          next.limitPrice = String(next.side === 'BUY' ? ticker.bid : ticker.ask);
        }
      }
      return next;
    });
    setFormError('');
    preview.reset();
    place.reset();
  };

  const built = buildOrder(form, activeBroker);
  const previewData = preview.data;
  const canPlace = Boolean(previewData) && !previewData.errs?.length && !place.isPending;

  const runPreview = () => {
    if (built.error) {
      setFormError(built.error);
      return;
    }
    place.reset();
    preview.mutate(built.order);
  };

  const submit = () => {
    place.mutate(built.order, {
      onSuccess: () => {
        setForm((prev) => ({ ...prev, size: '' }));
        preview.reset();
      },
    });
    setConfirmOpen(false);
  };

  const sideColor = form.side === 'BUY' ? tokens.up : tokens.down;

  return (
    <Panel title="Place order" subtitle={isLive ? 'Coinbase Advanced Trade · REAL MONEY' : 'Paper broker · simulated fills at live prices'}>
      <Box component="form" noValidate onSubmit={(e) => { e.preventDefault(); runPreview(); }} sx={{ display: 'grid', gap: 1.5 }}>
        {liveEnabled && (
          <TextField select size="small" label="Broker" value={broker} onChange={(e) => { setBroker(e.target.value); preview.reset(); }}>
            <MenuItem value="paper">Paper (simulated)</MenuItem>
            <MenuItem value="coinbase">Coinbase (real money)</MenuItem>
          </TextField>
        )}

        <ToggleButtonGroup
          exclusive
          fullWidth
          value={form.side}
          onChange={(_, side) => side && change({ side })}
          aria-label="Side"
        >
          <ToggleButton value="BUY" sx={{ '&.Mui-selected, &.Mui-selected:hover': { bgcolor: tokens.up, color: '#fff' } }}>Buy</ToggleButton>
          <ToggleButton value="SELL" sx={{ '&.Mui-selected, &.Mui-selected:hover': { bgcolor: tokens.down, color: '#fff' } }}>Sell</ToggleButton>
        </ToggleButtonGroup>

        <ToggleButtonGroup size="small" exclusive fullWidth value={form.type} onChange={(_, type) => type && change({ type })} aria-label="Order type">
          <ToggleButton value="MARKET">Market</ToggleButton>
          <ToggleButton value="LIMIT">Limit</ToggleButton>
        </ToggleButtonGroup>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            size="small"
            label="Size"
            value={form.size}
            onChange={(e) => change({ size: e.target.value })}
            slotProps={{ htmlInput: { inputMode: 'decimal' } }}
            sx={{ flex: 1 }}
          />
          <ToggleButtonGroup
            size="small"
            exclusive
            value={form.unit}
            onChange={(_, unit) => unit && change({ unit })}
            aria-label="Size unit"
          >
            <ToggleButton value="USD" disabled={form.type === 'LIMIT'}>USD</ToggleButton>
            <ToggleButton value="BTC">BTC</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {form.type === 'LIMIT' && (
          <>
            <TextField
              size="small"
              label="Limit price"
              value={form.limitPrice}
              onChange={(e) => change({ limitPrice: e.target.value })}
              slotProps={{
                htmlInput: { inputMode: 'decimal' },
                input: { startAdornment: <InputAdornment position="start">$</InputAdornment> },
              }}
            />
            <FormControlLabel
              control={<Checkbox size="small" checked={form.postOnly} onChange={(e) => change({ postOnly: e.target.checked })} />}
              label={<Typography variant="body2">Post-only (never take liquidity)</Typography>}
            />
          </>
        )}

        {ticker && (
          <Typography variant="caption" className="num">
            Bid {formatUsd(ticker.bid)} · Ask {formatUsd(ticker.ask)}
          </Typography>
        )}

        {formError && <Alert severity="warning">{formError}</Alert>}
        {preview.isError && <Alert severity="error">{describeError(preview.error)}</Alert>}

        {previewData && (
          <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5 }} aria-label="Order preview">
            <Row label="Est. price">{formatUsd(previewData.price)}</Row>
            <Row label="Size">{formatBtc(previewData.baseSize)}</Row>
            <Row label="Notional">{formatUsd(previewData.quoteSize)}</Row>
            <Row label="Fees">{formatUsd(previewData.commissionTotal)}</Row>
            <Row label={form.side === 'BUY' ? 'Total cost' : 'Net proceeds'} strong>{formatUsd(previewData.orderTotal)}</Row>
            {previewData.fillsImmediately === false && (
              <Typography variant="caption" component="div" sx={{ mt: 0.5 }}>Rests on the book until the price reaches your limit.</Typography>
            )}
            {previewData.errs?.length > 0 && (
              <Alert severity="error" sx={{ mt: 1 }}>Would be rejected: {previewData.errs.join(', ')}</Alert>
            )}
            {previewData.warnings?.length > 0 && (
              <Alert severity="warning" sx={{ mt: 1 }}>{previewData.warnings.join(', ')}</Alert>
            )}
          </Box>
        )}

        {place.isError && <Alert severity="error">{describeError(place.error)}</Alert>}
        {place.isSuccess && (
          <Alert severity="success">
            Order {place.data.status.toLowerCase()}
            {place.data.status === 'FILLED' && ` at ${formatUsd(place.data.averageFilledPrice)}`}.
          </Alert>
        )}

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button type="submit" variant="outlined" sx={{ flex: 1 }} disabled={preview.isPending}>
            {preview.isPending ? 'Previewing…' : 'Preview'}
          </Button>
          <Button
            variant="contained"
            sx={{ flex: 1, bgcolor: sideColor, '&:hover': { bgcolor: sideColor, filter: 'brightness(0.92)' } }}
            disabled={!canPlace}
            onClick={() => (isLive ? setConfirmOpen(true) : submit())}
          >
            {place.isPending ? 'Placing…' : `Place ${form.side === 'BUY' ? 'buy' : 'sell'}${isLive ? ' (real money)' : ''}`}
          </Button>
        </Box>
      </Box>

      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)}>
        <DialogTitle>Place a real Coinbase order?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This {form.side} order spends real money on your Coinbase account:
            {' '}<strong>{formatUsd(previewData?.orderTotal)}</strong> including {formatUsd(previewData?.commissionTotal)} in fees.
            The server caps live orders at {formatUsd(health?.maxOrderUsd)}.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)}>Cancel</Button>
          <Button color="error" variant="contained" onClick={submit}>Place real order</Button>
        </DialogActions>
      </Dialog>
    </Panel>
  );
};
