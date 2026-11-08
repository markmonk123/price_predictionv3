/** Direction cues: the arrow + word always travel with the color, never color alone. */
import { Box } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import RemoveIcon from '@mui/icons-material/Remove';
import { directionLabel, formatSignedPct } from '../format';

export const useDirectionColor = () => {
  const { tokens } = useTheme();
  return (direction) => (direction > 0 ? tokens.up : direction < 0 ? tokens.down : tokens.textMuted);
};

export const DirectionIcon = ({ direction, fontSize = 'small' }) => {
  const color = useDirectionColor()(direction);
  const Icon = direction > 0 ? ArrowUpwardIcon : direction < 0 ? ArrowDownwardIcon : RemoveIcon;
  return <Icon fontSize={fontSize} sx={{ color }} aria-hidden />;
};

export const Direction = ({ direction, showLabel = true }) => (
  <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
    <DirectionIcon direction={direction} />
    {showLabel && <span>{directionLabel(direction)}</span>}
  </Box>
);

/** Signed % change; the text stays in ink color, the arrow carries the sign. */
export const Change = ({ value }) => {
  if (value === null || value === undefined) return <span>–</span>;
  const direction = Math.sign(value);
  return (
    <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
      <DirectionIcon direction={direction} fontSize="inherit" />
      <span className="num">{formatSignedPct(value)}</span>
    </Box>
  );
};
