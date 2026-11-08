import {
  Alert, Button, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlined';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import HighlightOffIcon from '@mui/icons-material/HighlightOff';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutlineOutlined';
import { useCancelOrder } from '../api/hooks';
import { describeError } from '../api/client';
import { formatBtc, formatTime, formatUsd } from '../format';
import { Direction } from './Direction';

const OPEN = ['PENDING', 'OPEN', 'QUEUED'];

/** Status chip: icon + label, so state is never conveyed by color alone. */
export const OrderStatus = ({ status }) => {
  if (status === 'FILLED') return <Chip size="small" color="success" variant="outlined" icon={<CheckCircleOutlineIcon />} label="Filled" />;
  if (OPEN.includes(status)) return <Chip size="small" color="primary" variant="outlined" icon={<HourglassEmptyIcon />} label={status === 'OPEN' ? 'Open' : 'Pending'} />;
  if (status === 'FAILED') return <Chip size="small" color="error" variant="outlined" icon={<ErrorOutlineIcon />} label="Failed" />;
  return <Chip size="small" variant="outlined" icon={<HighlightOffIcon />} label={status === 'CANCEL_QUEUED' ? 'Cancelling' : status.charAt(0) + status.slice(1).toLowerCase()} />;
};

const sizeOf = (order) => {
  if (order.filledSize) return formatBtc(order.filledSize);
  if (order.baseSize) return formatBtc(order.baseSize);
  return `${formatUsd(order.quoteSize)} of BTC`;
};

export const OrdersTable = ({ orders, showBroker = false, dense = false }) => {
  const cancel = useCancelOrder();
  return (
    <>
      {cancel.isError && <Alert severity="error" sx={{ mb: 1 }}>{describeError(cancel.error)}</Alert>}
      <TableContainer>
        <Table size="small" aria-label="Orders">
          <TableHead>
            <TableRow>
              <TableCell>Time</TableCell>
              {showBroker && <TableCell>Broker</TableCell>}
              <TableCell>Side</TableCell>
              <TableCell>Type</TableCell>
              <TableCell align="right">Size</TableCell>
              <TableCell align="right">Price</TableCell>
              {!dense && <TableCell align="right">Fees</TableCell>}
              <TableCell>Status</TableCell>
              <TableCell align="right" />
            </TableRow>
          </TableHead>
          <TableBody>
            {orders.map((order) => (
              <TableRow key={order.id} hover>
                <TableCell>{formatTime(order.createdAt)}</TableCell>
                {showBroker && <TableCell>{order.broker === 'coinbase' ? 'Coinbase' : 'Paper'}</TableCell>}
                <TableCell><Direction direction={order.side === 'BUY' ? 1 : -1} showLabel={false} /> {order.side === 'BUY' ? 'Buy' : 'Sell'}</TableCell>
                <TableCell>{order.type === 'LIMIT' ? 'Limit' : 'Market'}</TableCell>
                <TableCell align="right" className="num">{sizeOf(order)}</TableCell>
                <TableCell align="right" className="num">{formatUsd(order.averageFilledPrice ?? order.limitPrice)}</TableCell>
                {!dense && <TableCell align="right" className="num">{formatUsd(order.totalFees)}</TableCell>}
                <TableCell><OrderStatus status={order.status} /></TableCell>
                <TableCell align="right">
                  {OPEN.includes(order.status) && (
                    <Button
                      size="small"
                      color="inherit"
                      disabled={cancel.isPending && cancel.variables === order.id}
                      onClick={() => cancel.mutate(order.id)}
                    >
                      Cancel
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
};
