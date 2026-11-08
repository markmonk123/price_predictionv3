import { Box, Typography } from '@mui/material';
import { useAccount, useTicker } from '../api/hooks';
import { formatNumber, formatUsd } from '../format';
import { Panel } from './Panel';
import { QueryState } from './QueryState';

/** Account value in USD: cash plus BTC (available + held) marked at the bid. */
export const accountValue = (balances, bid) => balances.reduce((sum, b) => {
  const units = b.available + b.hold;
  if (b.currency === 'USD') return sum + units;
  if (b.currency === 'BTC' && bid) return sum + units * bid;
  return sum;
}, 0);

export const BalancesCard = ({ broker = 'paper' }) => {
  const query = useAccount(broker);
  const { data: ticker } = useTicker();
  return (
    <Panel title={broker === 'paper' ? 'Paper account' : 'Coinbase account'} subtitle="Available · on hold for resting orders">
      <QueryState query={query}>
        {(account) => (
          <Box sx={{ display: 'grid', gap: 1 }}>
            {account.balances.map((b) => (
              <Box key={b.currency} sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>{b.currency}</Typography>
                <Typography variant="body2" className="num" sx={{ textAlign: 'right' }}>
                  {b.currency === 'USD' ? formatUsd(b.available) : formatNumber(b.available, 8)}
                  {b.hold > 0 && (
                    <Box component="span" sx={{ color: 'text.secondary' }}>
                      {' '}· {b.currency === 'USD' ? formatUsd(b.hold) : formatNumber(b.hold, 8)} held
                    </Box>
                  )}
                </Typography>
              </Box>
            ))}
            <Box sx={{ borderTop: 1, borderColor: 'divider', pt: 1, display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Total value</Typography>
              <Typography variant="body2" className="num" sx={{ fontWeight: 700 }}>
                {formatUsd(accountValue(account.balances, ticker?.bid))}
              </Typography>
            </Box>
          </Box>
        )}
      </QueryState>
    </Panel>
  );
};
