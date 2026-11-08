import { useState } from 'react';
import { Box, MenuItem, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { useHealth, useOrders } from '../api/hooks';
import { OrdersTable } from '../components/OrdersTable';
import { Panel } from '../components/Panel';
import { QueryState } from '../components/QueryState';

export const Orders = () => {
  const { data: health } = useHealth();
  const live = Boolean(health?.liveTradingEnabled);
  const [status, setStatus] = useState('');
  const [broker, setBroker] = useState('');
  const query = useOrders({ status: status || undefined, broker: broker || undefined, limit: 200 });

  return (
    <Box sx={{ display: 'grid', gap: 2 }}>
      <Typography variant="h1">Orders</Typography>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
        <ToggleButtonGroup size="small" exclusive value={status} onChange={(_, v) => v !== null && setStatus(v)} aria-label="Status">
          <ToggleButton value="">All</ToggleButton>
          <ToggleButton value="open">Open</ToggleButton>
          <ToggleButton value="FILLED">Filled</ToggleButton>
          <ToggleButton value="CANCELLED">Cancelled</ToggleButton>
        </ToggleButtonGroup>
        {live && (
          <TextField select size="small" label="Broker" value={broker} onChange={(e) => setBroker(e.target.value)} sx={{ minWidth: 160 }}>
            <MenuItem value="">All brokers</MenuItem>
            <MenuItem value="paper">Paper</MenuItem>
            <MenuItem value="coinbase">Coinbase</MenuItem>
          </TextField>
        )}
      </Box>
      <Panel title="Order history" subtitle="Newest first · up to 200 orders">
        <QueryState query={query} isEmpty={(d) => !d?.orders?.length} empty="No orders yet. Place one from the Trade page.">
          {(data) => <OrdersTable orders={data.orders} showBroker={live} />}
        </QueryState>
      </Panel>
    </Box>
  );
};
