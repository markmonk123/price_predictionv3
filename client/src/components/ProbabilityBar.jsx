/**
 * Direction probabilities as one part-to-whole bar: down (red pole) · no change
 * (neutral midpoint) · up (blue pole), 2px surface gaps between segments.
 * Every value is also printed in the legend below, so hover is never required.
 */
import { Box, Tooltip, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { formatPct } from '../format';

export const ProbabilityBar = ({ down = 0, flat = 0, up = 0 }) => {
  const { tokens: t } = useTheme();
  const segments = [
    { key: 'down', label: 'Down', value: down, color: t.down },
    { key: 'flat', label: 'No change', value: flat, color: t.neutralMid },
    { key: 'up', label: 'Up', value: up, color: t.up },
  ];
  return (
    <Box>
      <Box
        role="img"
        aria-label={segments.map((s) => `${s.label} ${formatPct(s.value * 100)}`).join(', ')}
        sx={{ display: 'flex', gap: '2px', height: 12, borderRadius: '4px', overflow: 'hidden', bgcolor: 'background.paper' }}
      >
        {segments.filter((s) => s.value > 0).map((s) => (
          <Tooltip key={s.key} title={`${s.label}: ${formatPct(s.value * 100)}`}>
            <Box sx={{ flexGrow: s.value, flexBasis: 0, bgcolor: s.color, minWidth: 2 }} />
          </Tooltip>
        ))}
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1, gap: 1 }}>
        {segments.map((s) => (
          <Box key={s.key} sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
            <Box aria-hidden sx={{ width: 8, height: 8, borderRadius: '2px', bgcolor: s.color }} />
            <Typography variant="body2" color="text.secondary">
              {s.label} <Box component="span" className="num" sx={{ color: 'text.primary', fontWeight: 600 }}>{formatPct(s.value * 100)}</Box>
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
};
