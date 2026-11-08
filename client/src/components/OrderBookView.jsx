/**
 * Live order book. Depth bars show cumulative size per side (one hue per side,
 * low-opacity fill); prices and sizes are always printed.
 */
import { Box, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';
import { useOrderBook } from '../api/hooks';
import { formatNumber } from '../format';
import { Panel } from './Panel';
import { QueryState } from './QueryState';

const withCumulative = (levels) => {
  let total = 0;
  return levels.map((level) => {
    total += level.size;
    return { ...level, total };
  });
};

const Side = ({ levels, color, max, label }) => (
  <>
    {levels.map((level) => (
      <TableRow key={`${label}-${level.price}`}>
        <TableCell
          className="num"
          sx={{
            color: 'text.primary',
            fontWeight: 600,
            background: `linear-gradient(to left, ${alpha(color, 0.18)} ${(level.total / max) * 100}%, transparent 0)`,
            borderLeft: `3px solid ${color}`,
          }}
        >
          {formatNumber(level.price)}
        </TableCell>
        <TableCell align="right" className="num">{formatNumber(level.size, 6)}</TableCell>
        <TableCell align="right" className="num">{formatNumber(level.total, 6)}</TableCell>
      </TableRow>
    ))}
  </>
);

export const OrderBookView = ({ depth = 12 }) => {
  const { tokens: t } = useTheme();
  const query = useOrderBook(depth);
  return (
    <Panel title="Order book" subtitle="Coinbase · top of book, cumulative depth">
      <QueryState query={query} isEmpty={(d) => !d?.bids?.length && !d?.asks?.length} empty="The book is empty.">
        {(book) => {
          const asks = withCumulative(book.asks);
          const bids = withCumulative(book.bids);
          const max = Math.max(asks.at(-1)?.total ?? 0, bids.at(-1)?.total ?? 0) || 1;
          const spread = book.asks[0] && book.bids[0] ? book.asks[0].price - book.bids[0].price : null;
          return (
            <Table size="small" aria-label="Order book">
              <TableHead>
                <TableRow>
                  <TableCell>Price (USD)</TableCell>
                  <TableCell align="right">Size (BTC)</TableCell>
                  <TableCell align="right">Total</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <Side levels={[...asks].reverse()} color={t.down} max={max} label="ask" />
                <TableRow>
                  <TableCell colSpan={3} sx={{ textAlign: 'center' }}>
                    <Typography variant="caption" className="num">
                      Spread {spread === null ? '–' : `$${formatNumber(spread)}`} · asks above (red) · bids below (blue)
                    </Typography>
                  </TableCell>
                </TableRow>
                <Side levels={bids} color={t.up} max={max} label="bid" />
              </TableBody>
            </Table>
          );
        }}
      </QueryState>
    </Panel>
  );
};
