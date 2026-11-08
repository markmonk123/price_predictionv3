/** Loading / error / empty states shared by every data panel. */
import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material';
import { describeError } from '../api/client';

export const Loading = ({ label = 'Loading…', minHeight = 120 }) => (
  <Box sx={{ minHeight, display: 'grid', placeItems: 'center' }} role="status">
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, color: 'text.secondary' }}>
      <CircularProgress size={18} />
      <Typography variant="body2">{label}</Typography>
    </Box>
  </Box>
);

export const ErrorState = ({ error, onRetry, title = 'Could not load data' }) => (
  <Alert
    severity="error"
    action={onRetry ? <Button color="inherit" size="small" onClick={() => onRetry()}>Retry</Button> : undefined}
  >
    <strong>{title}.</strong> {describeError(error)}
  </Alert>
);

export const Empty = ({ children }) => (
  <Box sx={{ py: 4, textAlign: 'center' }}>
    <Typography variant="body2" color="text.secondary">{children}</Typography>
  </Box>
);

/**
 * Render a react-query result: spinner on first load, error with retry, empty message,
 * otherwise children(data). Refetches keep the previous render (no skeleton flash).
 */
export const QueryState = ({ query, isEmpty, empty, loadingLabel, minHeight, children }) => {
  if (query.isPending) return <Loading label={loadingLabel} minHeight={minHeight} />;
  if (query.isError && query.data === undefined) return <ErrorState error={query.error} onRetry={query.refetch} />;
  if (isEmpty?.(query.data)) return <Empty>{empty}</Empty>;
  return (
    <Box sx={{ opacity: query.isPlaceholderData ? 0.6 : 1, transition: 'opacity 150ms' }}>
      {children(query.data)}
    </Box>
  );
};
