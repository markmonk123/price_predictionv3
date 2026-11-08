/** Shared rendering for prediction-ledger rows. */
import { Box, Chip } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import BlockIcon from '@mui/icons-material/Block';
import TaskAltIcon from '@mui/icons-material/TaskAlt';
import { Direction } from './Direction';
import { formatPct, formatUsd } from '../format';

export const PredictionStatus = ({ status }) => {
  if (status === 'scored') return <Chip size="small" variant="outlined" icon={<TaskAltIcon />} label="Scored" />;
  if (status === 'unscorable') return <Chip size="small" variant="outlined" icon={<BlockIcon />} label="Unscorable" />;
  return <Chip size="small" variant="outlined" color="primary" icon={<HourglassEmptyIcon />} label="Pending" />;
};

/** What the row predicted, in words + icon. */
export const describePrediction = (row) => {
  if (row.predicted_price != null) return <span className="num">{formatUsd(row.predicted_price)}</span>;
  if (row.predicted_high != null) {
    return <span className="num">{formatUsd(row.predicted_low)} – {formatUsd(row.predicted_high)}</span>;
  }
  if (row.predicted_direction != null) {
    return (
      <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 1 }}>
        <Direction direction={row.predicted_direction} />
        {row.confidence != null && <Box component="span" sx={{ color: 'text.secondary' }}>{formatPct(row.confidence * 100)}</Box>}
      </Box>
    );
  }
  return '–';
};

/** Scored outcome: correct/incorrect (icon + word) or the price error. */
export const Outcome = ({ row }) => {
  if (row.status !== 'scored') return <Box component="span" sx={{ color: 'text.secondary' }}>–</Box>;
  if (row.direction_correct != null && row.predicted_price == null) {
    const ok = row.direction_correct === 1;
    return (
      <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
        {ok ? <CheckIcon fontSize="small" color="success" /> : <CloseIcon fontSize="small" color="error" />}
        {ok ? 'Correct' : 'Wrong'}
      </Box>
    );
  }
  if (row.dollar_error != null) {
    return <span className="num">{formatUsd(row.dollar_error)} ({formatPct(row.pct_error, 2)})</span>;
  }
  if (row.range_contained != null) return row.range_contained ? 'Inside range' : 'Outside range';
  return <span className="num">{formatUsd(row.actual_price)}</span>;
};
