import { useState } from 'react';
import {
  Box, MenuItem, Table, TableBody, TableCell, TableHead, TablePagination, TableRow, TextField,
  ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useAccuracy, usePredictionHistory } from '../api/hooks';
import { Panel } from '../components/Panel';
import { QueryState } from '../components/QueryState';
import { Outcome, PredictionStatus, describePrediction } from '../components/PredictionRows';
import { formatPct, formatTime, formatUsd } from '../format';

const KINDS = [
  { value: '', label: 'All kinds' },
  { value: 'direction', label: 'Direction' },
  { value: 'price', label: 'Price' },
  { value: 'high_low', label: 'High / low range' },
  { value: 'rule_signal', label: 'Rule signal' },
  { value: 'trade', label: 'Trade' },
];

/** Inline data bar: one hue (accent) for every row, value always printed beside it. */
const PctBar = ({ value }) => {
  const { tokens: t } = useTheme();
  if (value === null || value === undefined) return <span>–</span>;
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 140 }}>
      <Box sx={{ flex: 1, height: 8, bgcolor: t.grid, borderRadius: '4px', overflow: 'hidden' }} aria-hidden>
        <Box sx={{ width: `${Math.min(Math.max(value, 0), 100)}%`, height: '100%', bgcolor: t.accent, borderRadius: '4px' }} />
      </Box>
      <Box component="span" className="num" sx={{ width: 48, textAlign: 'right' }}>{formatPct(value)}</Box>
    </Box>
  );
};

const AccuracyTable = ({ filters, groupBy }) => {
  const query = useAccuracy({ ...filters, groupBy });
  return (
    <Panel
      title="Accuracy"
      subtitle="Scored predictions only. Live = forward predictions; backtest = held-out historical rows."
    >
      <QueryState query={query} isEmpty={(d) => !d?.groups?.length} empty="No predictions match these filters yet.">
        {(data) => (
          <Box sx={{ overflowX: 'auto' }}>
            <Table size="small" aria-label="Accuracy by group">
              <TableHead>
                <TableRow>
                  <TableCell>Scope</TableCell>
                  <TableCell>Kind</TableCell>
                  <TableCell align="right">Horizon</TableCell>
                  {groupBy === 'signal' && <TableCell>Signal</TableCell>}
                  <TableCell align="right">Scored / total</TableCell>
                  <TableCell>Direction accuracy</TableCell>
                  <TableCell align="right">MAE</TableCell>
                  <TableCell align="right">RMSE</TableCell>
                  <TableCell align="right">MAPE</TableCell>
                  <TableCell align="right">Within 1%</TableCell>
                  <TableCell align="right">Range hit</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.groups.map((g) => (
                  <TableRow key={`${g.historical}-${g.kind}-${g.horizonMinutes}-${g.signal ?? ''}`} hover>
                    <TableCell>{g.historical ? 'Backtest' : 'Live'}</TableCell>
                    <TableCell>{g.kind}</TableCell>
                    <TableCell align="right" className="num">{g.horizonMinutes}m</TableCell>
                    {groupBy === 'signal' && <TableCell>{g.signal}</TableCell>}
                    <TableCell align="right" className="num">
                      {g.scored} / {g.total}
                      {g.pending > 0 && <Box component="span" sx={{ color: 'text.secondary' }}> · {g.pending} pending</Box>}
                    </TableCell>
                    <TableCell><PctBar value={g.directionN ? g.directionAccuracyPct : null} /></TableCell>
                    <TableCell align="right" className="num">{g.mae == null ? '–' : formatUsd(g.mae)}</TableCell>
                    <TableCell align="right" className="num">{g.rmse == null ? '–' : formatUsd(g.rmse)}</TableCell>
                    <TableCell align="right" className="num">{formatPct(g.mapePct, 2)}</TableCell>
                    <TableCell align="right" className="num">{formatPct(g.withinOnePct)}</TableCell>
                    <TableCell align="right" className="num">{formatPct(g.rangeContainedPct)}</TableCell>
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

const HistoryTable = ({ filters }) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const query = usePredictionHistory({ ...filters, limit: rowsPerPage, offset: page * rowsPerPage });
  return (
    <Panel title="Predictions" subtitle="Newest run first">
      <QueryState query={query} isEmpty={(d) => !d?.rows?.length} empty="No predictions match these filters.">
        {(data) => (
          <>
            <Box sx={{ overflowX: 'auto' }}>
              <Table size="small" aria-label="Prediction history">
                <TableHead>
                  <TableRow>
                    <TableCell>Made</TableCell>
                    <TableCell>Kind</TableCell>
                    <TableCell>Signal</TableCell>
                    <TableCell align="right">Horizon</TableCell>
                    <TableCell align="right">Reference</TableCell>
                    <TableCell>Prediction</TableCell>
                    <TableCell>Target window</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Outcome</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.rows.map((row) => (
                    <TableRow key={row.id} hover>
                      <TableCell>{formatTime(row.generated_at)}</TableCell>
                      <TableCell>{row.kind}{row.historical && <Typography variant="caption" component="span"> · backtest</Typography>}</TableCell>
                      <TableCell>{row.signal}</TableCell>
                      <TableCell align="right" className="num">{row.horizon_minutes}m</TableCell>
                      <TableCell align="right" className="num">{formatUsd(row.reference_price)}</TableCell>
                      <TableCell>{describePrediction(row)}</TableCell>
                      <TableCell>{formatTime(row.target_start)} → {formatTime(row.target_end)}</TableCell>
                      <TableCell><PredictionStatus status={row.status} /></TableCell>
                      <TableCell><Outcome row={row} /></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
            <TablePagination
              component="div"
              count={data.total}
              page={page}
              rowsPerPage={rowsPerPage}
              rowsPerPageOptions={[25, 50, 100]}
              onPageChange={(_, next) => setPage(next)}
              onRowsPerPageChange={(e) => { setRowsPerPage(Number(e.target.value)); setPage(0); }}
            />
          </>
        )}
      </QueryState>
    </Panel>
  );
};

export const Predictions = () => {
  const [scope, setScope] = useState('live');
  const [kind, setKind] = useState('');
  const [status, setStatus] = useState('');
  const [groupBy, setGroupBy] = useState('signal');
  const filters = { scope, kind: kind || undefined, status: status || undefined };

  return (
    <Box sx={{ display: 'grid', gap: 2 }}>
      <Typography variant="h1">Predictions</Typography>
      {/* One filter row scopes everything below it. */}
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
        <ToggleButtonGroup size="small" exclusive value={scope} onChange={(_, v) => v && setScope(v)} aria-label="Scope">
          <ToggleButton value="live">Live</ToggleButton>
          <ToggleButton value="historical">Backtest</ToggleButton>
          <ToggleButton value="all">All</ToggleButton>
        </ToggleButtonGroup>
        <TextField select size="small" label="Kind" value={kind} onChange={(e) => setKind(e.target.value)} sx={{ minWidth: 170 }}>
          {KINDS.map((k) => <MenuItem key={k.value} value={k.value}>{k.label}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="Status" value={status} onChange={(e) => setStatus(e.target.value)} sx={{ minWidth: 150 }}>
          <MenuItem value="">Any status</MenuItem>
          <MenuItem value="pending">Pending</MenuItem>
          <MenuItem value="scored">Scored</MenuItem>
          <MenuItem value="unscorable">Unscorable</MenuItem>
        </TextField>
        <ToggleButtonGroup size="small" exclusive value={groupBy} onChange={(_, v) => v && setGroupBy(v)} aria-label="Group accuracy by">
          <ToggleButton value="horizon">By horizon</ToggleButton>
          <ToggleButton value="signal">By signal</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <AccuracyTable filters={filters} groupBy={groupBy} />
      <HistoryTable key={JSON.stringify(filters)} filters={filters} />
    </Box>
  );
};
