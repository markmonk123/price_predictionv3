import { Box, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { useAccount, useAccuracy, useLatestPrediction, usePredictionHistory, useTicker } from '../api/hooks';
import { PriceChart } from '../components/PriceChart';
import { PredictionCard } from '../components/PredictionCard';
import { Panel, StatTile } from '../components/Panel';
import { Change, Direction } from '../components/Direction';
import { QueryState } from '../components/QueryState';
import { PredictionStatus, Outcome, describePrediction } from '../components/PredictionRows';
import { accountValue } from '../components/BalancesCard';
import { formatPct, formatTime, formatUsd } from '../format';

const RECENT = { scope: 'live', limit: 8 };
const LIVE_DIRECTION = { scope: 'live', kind: 'direction', groupBy: 'signal' };

const Tiles = () => {
  const { data: ticker } = useTicker();
  const { data: prediction } = useLatestPrediction();
  const { data: accuracy } = useAccuracy(LIVE_DIRECTION);
  const { data: account } = useAccount('paper');
  const classifier = accuracy?.groups?.find((g) => g.signal === 'classifier');

  return (
    <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr 1fr', lg: 'repeat(4, 1fr)' } }}>
      <StatTile
        label="BTC-USD"
        value={formatUsd(ticker?.price)}
        detail={ticker ? <span><Change value={ticker.change24hPct} /> in 24h</span> : 'Waiting for market data…'}
      />
      <StatTile
        label="Model says (next 15 min)"
        value={prediction ? <Direction direction={prediction.predicted_direction} /> : '–'}
        detail={prediction ? `${formatPct(prediction.confidence * 100)} confidence · ${formatTime(prediction.generatedAt)}` : 'No prediction yet'}
      />
      <StatTile
        label="Live direction accuracy"
        value={classifier?.directionN ? formatPct(classifier.directionAccuracyPct) : '–'}
        detail={classifier?.directionN
          ? `${classifier.directionN} scored forward predictions`
          : 'Appears once live predictions are scored'}
      />
      <StatTile
        label="Paper account value"
        value={account ? formatUsd(accountValue(account.balances, ticker?.bid)) : '–'}
        detail={<RouterLink to="/trade" style={{ color: 'inherit' }}>Trade with paper money →</RouterLink>}
      />
    </Box>
  );
};

const RecentPredictions = () => {
  const query = usePredictionHistory(RECENT);
  return (
    <Panel
      title="Recent forward predictions"
      subtitle="Scored automatically once the target candles close"
      actions={<Typography variant="body2" component={RouterLink} to="/predictions" sx={{ color: 'primary.main' }}>All predictions →</Typography>}
    >
      <QueryState
        query={query}
        isEmpty={(d) => !d?.rows?.length}
        empty="No predictions recorded yet. Run priceprediction.py (or wait for the next scheduled run) to populate the ledger."
      >
        {(data) => (
          <Box sx={{ overflowX: 'auto' }}>
            <Table size="small" aria-label="Recent predictions">
              <TableHead>
                <TableRow>
                  <TableCell>Made</TableCell>
                  <TableCell>Signal</TableCell>
                  <TableCell>Prediction</TableCell>
                  <TableCell>Target</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Outcome</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.rows.map((row) => (
                  <TableRow key={row.id} hover>
                    <TableCell>{formatTime(row.generated_at)}</TableCell>
                    <TableCell>{row.signal}</TableCell>
                    <TableCell>{describePrediction(row)}</TableCell>
                    <TableCell>{formatTime(row.target_end)}</TableCell>
                    <TableCell><PredictionStatus status={row.status} /></TableCell>
                    <TableCell><Outcome row={row} /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}
      </QueryState>
    </Panel>
  );
};

export const Dashboard = () => (
  <Box sx={{ display: 'grid', gap: 2 }}>
    <Typography variant="h1">Dashboard</Typography>
    <Tiles />
    <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 2fr) minmax(320px, 1fr)' }, alignItems: 'start' }}>
      <PriceChart />
      <PredictionCard />
    </Box>
    <RecentPredictions />
  </Box>
);
