import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { useHealth, useLatestPrediction, useRunPrediction } from '../api/hooks';
import { describeError } from '../api/client';
import { formatPct, formatTime, formatUsd } from '../format';
import { Panel } from './Panel';
import { Direction } from './Direction';
import { ProbabilityBar } from './ProbabilityBar';
import { Empty, QueryState } from './QueryState';

export const PredictionCard = () => {
  const query = useLatestPrediction();
  const run = useRunPrediction();
  const { data: health } = useHealth();
  const modelError = health?.model?.lastError;

  return (
    <Panel
      title="Latest prediction"
      subtitle={query.data ? `Next ${query.data.timeframe} · threshold ±${formatPct(query.data.threshold * 100, 2)}` : 'Direction model'}
      actions={(
        <Button
          size="small"
          variant="outlined"
          startIcon={run.isPending ? <CircularProgress size={14} /> : <PlayArrowIcon />}
          disabled={run.isPending || health?.model?.running}
          onClick={() => run.mutate()}
        >
          {run.isPending ? 'Running…' : 'Run model'}
        </Button>
      )}
    >
      {run.isError && <Alert severity="error" sx={{ mb: 1.5 }}>{describeError(run.error)}</Alert>}
      <QueryState query={query} loadingLabel="Loading prediction…">
        {(prediction) => (prediction ? (
          <Box sx={{ display: 'grid', gap: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
              <Typography sx={{ fontSize: '1.75rem', fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                <Direction direction={prediction.predicted_direction} />
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Confidence <strong>{formatPct(prediction.confidence * 100)}</strong>
              </Typography>
            </Box>
            <ProbabilityBar
              down={prediction.decrease_probability}
              flat={prediction.no_change_probability}
              up={prediction.increase_probability}
            />
            <Typography variant="caption">
              Made {formatTime(prediction.generatedAt)} at {formatUsd(prediction.price)}
              {prediction.data_source === 'synthetic' && ' · SYNTHETIC data (offline mode)'}
            </Typography>
          </Box>
        ) : (
          <Empty>
            {modelError ? `The last model run failed: ${modelError.message}` : 'No prediction yet. The model runs every few minutes, or run it now.'}
          </Empty>
        ))}
      </QueryState>
    </Panel>
  );
};
