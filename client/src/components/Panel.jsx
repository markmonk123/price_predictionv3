import { Paper, Box, Typography } from '@mui/material';

/** A titled card. `actions` sit on the right of the title row. */
export const Panel = ({ title, subtitle, actions, children, sx, contentSx }) => (
  <Paper sx={{ display: 'flex', flexDirection: 'column', minWidth: 0, ...sx }}>
    {(title || actions) && (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 2, pt: 1.5, pb: 1, flexWrap: 'wrap' }}>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {title && <Typography variant="h3" component="h2">{title}</Typography>}
          {subtitle && <Typography variant="caption">{subtitle}</Typography>}
        </Box>
        {actions}
      </Box>
    )}
    <Box sx={{ px: 2, pb: 2, flex: 1, minWidth: 0, ...contentSx }}>{children}</Box>
  </Paper>
);

/** Stat tile: the number is the chart. Proportional figures for the hero value. */
export const StatTile = ({ label, value, detail, icon }) => (
  <Paper sx={{ px: 2, py: 1.5, minWidth: 0 }}>
    <Typography variant="caption" component="div" sx={{ mb: 0.5 }}>{label}</Typography>
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
      {icon}
      <Typography sx={{ fontSize: '1.5rem', fontWeight: 600, lineHeight: 1.2 }} noWrap>{value}</Typography>
    </Box>
    {detail && <Box sx={{ mt: 0.5, typography: 'body2', color: 'text.secondary' }}>{detail}</Box>}
  </Paper>
);
