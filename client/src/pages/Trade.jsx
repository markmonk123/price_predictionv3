import { Box, Typography } from '@mui/material';
import { useHealth, useOrders } from '../api/hooks';
import { OrderTicket } from '../components/OrderTicket';
import { OrderBookView } from '../components/OrderBookView';
import { BalancesCard } from '../components/BalancesCard';
import { PriceChart } from '../components/PriceChart';
import { OrdersTable } from '../components/OrdersTable';
import { Panel } from '../components/Panel';
import { QueryState } from '../components/QueryState';

const OpenOrders = ({ showBroker }) => {
  const query = useOrders({ status: 'open', limit: 50 });
  return (
    <Panel title="Open orders" subtitle="Resting limit orders fill when the live price crosses them">
      <QueryState query={query} isEmpty={(d) => !d?.orders?.length} empty="No open orders.">
        {(data) => <OrdersTable orders={data.orders} showBroker={showBroker} dense />}
      </QueryState>
    </Panel>
  );
};

export const Trade = () => {
  const { data: health } = useHealth();
  const live = Boolean(health?.liveTradingEnabled);
  return (
    <Box sx={{ display: 'grid', gap: 2 }}>
      <Typography variant="h1">Trade</Typography>
      <Box
        sx={{
          display: 'grid',
          gap: 2,
          alignItems: 'start',
          gridTemplateColumns: { xs: '1fr', md: '320px minmax(0, 1fr)', xl: '320px minmax(0, 1fr) 360px' },
        }}
      >
        <Box sx={{ display: 'grid', gap: 2 }}>
          <OrderTicket />
          <BalancesCard broker="paper" />
          {live && <BalancesCard broker="coinbase" />}
        </Box>
        <Box sx={{ display: 'grid', gap: 2, minWidth: 0 }}>
          <PriceChart height={320} showMarkers={false} />
          <OpenOrders showBroker={live} />
        </Box>
        <Box sx={{ gridColumn: { md: '2', xl: 'auto' } }}>
          <OrderBookView />
        </Box>
      </Box>
    </Box>
  );
};
