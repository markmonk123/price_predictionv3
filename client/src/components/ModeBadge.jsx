/**
 * PAPER / LIVE trading-mode badge. LIVE means real money and must be impossible
 * to miss: status color + icon + label (never color alone).
 */
import { Chip, Tooltip } from '@mui/material';
import ScienceOutlinedIcon from '@mui/icons-material/ScienceOutlined';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import HelpOutlineIcon from '@mui/icons-material/HelpOutlineOutlined';

export const ModeBadge = ({ mode }) => {
  if (mode === 'live') {
    return (
      <Tooltip title="Live trading is enabled: Coinbase orders use real money">
        <Chip
          icon={<WarningAmberIcon />}
          label="LIVE · real money"
          size="small"
          color="error"
          sx={{ fontWeight: 700 }}
        />
      </Tooltip>
    );
  }
  if (mode === 'paper') {
    return (
      <Tooltip title="Paper trading: orders are simulated against live prices">
        <Chip icon={<ScienceOutlinedIcon />} label="PAPER" size="small" variant="outlined" sx={{ fontWeight: 700 }} />
      </Tooltip>
    );
  }
  return <Chip icon={<HelpOutlineIcon />} label="Mode unknown" size="small" variant="outlined" />;
};
