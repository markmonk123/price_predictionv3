/**
 * Candlestick chart (lightweight-charts) with model prediction markers.
 *
 * - Up/down candles use the validated blue/red diverging pair (see theme/tokens.js).
 * - Hover: built-in crosshair plus an OHLC readout that tracks it.
 * - Live: socket ticks update the last candle (or open a new one) in place.
 * - Markers: one arrow per live classifier prediction at the candle it was made on;
 *   only the newest is labeled (no number on every point).
 * - Table view: the same candles as rows, for keyboard / screen-reader access.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Box, Table, TableBody, TableCell, TableHead, TableRow, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { CandlestickSeries, ColorType, CrosshairMode, createChart, createSeriesMarkers } from 'lightweight-charts';
import { useCandles, usePredictionHistory, useTicker } from '../api/hooks';
import { fontFamily } from '../theme/tokens';
import { formatNumber, formatPct } from '../format';
import { Panel } from './Panel';
import { ErrorState, Loading } from './QueryState';

export const TIMEFRAMES = [
  { value: 'FIVE_MINUTE', label: '5m', seconds: 300 },
  { value: 'FIFTEEN_MINUTE', label: '15m', seconds: 900 },
  { value: 'ONE_HOUR', label: '1h', seconds: 3600 },
  { value: 'SIX_HOUR', label: '6h', seconds: 21600 },
  { value: 'ONE_DAY', label: '1D', seconds: 86400 },
];

const MARKER_FILTERS = { kind: 'direction', signal: 'classifier', scope: 'live', limit: 200 };

/** Candle open time a live classifier prediction was made on (see services/history_runtime.py). */
const predictionCandleTime = (row) => Date.parse(row.target_end) / 1000 - 300 - row.horizon_minutes * 60;

/** Axis ticks in the viewer's local zone (lightweight-charts defaults to UTC). */
const localTick = (time, tickType) => {
  const d = new Date(time * 1000);
  if (tickType === 0) return String(d.getFullYear());
  if (tickType === 1) return d.toLocaleString(undefined, { month: 'short' });
  if (tickType === 2) return d.toLocaleString(undefined, { month: 'short', day: 'numeric' });
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
};

const Readout = ({ bar, up, down }) => {
  if (!bar) return null;
  const color = bar.close >= bar.open ? up : down;
  const items = [['O', bar.open], ['H', bar.high], ['L', bar.low], ['C', bar.close]];
  return (
    <Box className="num" sx={{ display: 'flex', gap: 1.5, typography: 'body2', color: 'text.secondary', flexWrap: 'wrap' }}>
      {items.map(([label, value]) => (
        <span key={label}>{label} <Box component="span" sx={{ color: 'text.primary' }}>{formatNumber(value)}</Box></span>
      ))}
      <span>
        Δ{' '}
        <Box component="span" sx={{ color: 'text.primary' }}>
          {formatPct(((bar.close - bar.open) / bar.open) * 100, 2)}
        </Box>
        <Box component="span" aria-hidden sx={{ display: 'inline-block', width: 8, height: 8, borderRadius: '2px', bgcolor: color, ml: 0.75 }} />
      </span>
    </Box>
  );
};

export const PriceChart = ({ height = 380, showMarkers = true }) => {
  const theme = useTheme();
  const t = theme.tokens;
  const [timeframe, setTimeframe] = useState('FIVE_MINUTE');
  const [view, setView] = useState('chart');
  const [hoverBar, setHoverBar] = useState(null);
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const markersRef = useRef(null);
  const lastBarRef = useRef(null);

  const candlesQuery = useCandles(timeframe);
  const { data: ticker } = useTicker();
  const { data: history } = usePredictionHistory(MARKER_FILTERS);
  const seconds = TIMEFRAMES.find((tf) => tf.value === timeframe).seconds;
  const candles = useMemo(() => candlesQuery.data?.candles ?? [], [candlesQuery.data]);

  // Create the chart once per theme; data is applied by the effects below.
  useEffect(() => {
    if (view !== 'chart' || !containerRef.current) return undefined;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: t.surface },
        textColor: t.textMuted,
        fontFamily,
        fontSize: 11,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: t.grid } },
      rightPriceScale: { borderColor: t.axis },
      timeScale: { borderColor: t.axis, timeVisible: true, secondsVisible: false, tickMarkFormatter: localTick },
      crosshair: { mode: CrosshairMode.Normal },
      localization: {
        priceFormatter: (price) => formatNumber(price),
        timeFormatter: (time) => new Date(time * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }),
      },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: t.up, downColor: t.down, wickUpColor: t.up, wickDownColor: t.down, borderVisible: false,
    });
    chartRef.current = chart;
    seriesRef.current = series;
    markersRef.current = createSeriesMarkers(series, []);
    chart.subscribeCrosshairMove((param) => {
      setHoverBar(param.time ? param.seriesData.get(series) ?? null : null);
    });
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
    };
  }, [t, view]);

  useEffect(() => {
    if (!seriesRef.current || !candles.length) return;
    seriesRef.current.setData(candles.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    lastBarRef.current = { ...candles.at(-1) };
    chartRef.current.timeScale().scrollToRealTime();
  }, [candles, view, t]);

  // Fold live ticks into the current candle.
  useEffect(() => {
    const last = lastBarRef.current;
    if (!seriesRef.current || !last || !ticker?.price) return;
    const bucket = Math.floor(Date.parse(ticker.time) / 1000 / seconds) * seconds;
    if (Number.isNaN(bucket) || bucket < last.time) return;
    const next = bucket === last.time
      ? { ...last, close: ticker.price, high: Math.max(last.high, ticker.price), low: Math.min(last.low, ticker.price) }
      : { time: bucket, open: last.close, high: ticker.price, low: ticker.price, close: ticker.price };
    lastBarRef.current = next;
    seriesRef.current.update({ time: next.time, open: next.open, high: next.high, low: next.low, close: next.close });
  }, [ticker, seconds]);

  useEffect(() => {
    if (!markersRef.current) return;
    if (!showMarkers || !candles.length || !history?.rows) {
      markersRef.current.setMarkers([]);
      return;
    }
    const first = candles[0].time;
    const rows = history.rows
      .map((row) => ({ row, time: Math.floor(predictionCandleTime(row) / seconds) * seconds }))
      .filter(({ time }) => time >= first)
      .sort((a, b) => a.time - b.time);
    markersRef.current.setMarkers(rows.map(({ row, time }, index) => {
      const dir = row.predicted_direction;
      const isLatest = index === rows.length - 1;
      return {
        time,
        position: dir > 0 ? 'belowBar' : dir < 0 ? 'aboveBar' : 'inBar',
        shape: dir > 0 ? 'arrowUp' : dir < 0 ? 'arrowDown' : 'circle',
        color: dir > 0 ? t.up : dir < 0 ? t.down : t.textMuted,
        size: 1,
        text: isLatest ? `Model ${dir > 0 ? 'up' : dir < 0 ? 'down' : 'flat'} ${Math.round((row.confidence ?? 0) * 100)}%` : undefined,
      };
    }));
  }, [history, candles, seconds, showMarkers, t, view]);

  const shownBar = hoverBar ?? (candles.length ? candles.at(-1) : null);

  return (
    <Panel
      title="BTC-USD"
      subtitle={`Coinbase candles · times in your local zone${showMarkers ? ' · arrows = model predictions' : ''}`}
      actions={(
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <ToggleButtonGroup size="small" exclusive value={timeframe} onChange={(_, v) => v && setTimeframe(v)} aria-label="Timeframe">
            {TIMEFRAMES.map((tf) => <ToggleButton key={tf.value} value={tf.value}>{tf.label}</ToggleButton>)}
          </ToggleButtonGroup>
          <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)} aria-label="View">
            <ToggleButton value="chart">Chart</ToggleButton>
            <ToggleButton value="table">Table</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      )}
    >
      {candlesQuery.isError && !candles.length && <ErrorState error={candlesQuery.error} onRetry={candlesQuery.refetch} />}
      {candlesQuery.isPending && <Loading minHeight={height} label="Loading candles…" />}
      {view === 'chart' ? (
        <>
          <Box sx={{ minHeight: 24, mb: 1 }}>
            <Readout bar={shownBar} up={t.up} down={t.down} />
          </Box>
          <Box
            ref={containerRef}
            sx={{ height, display: candlesQuery.isPending ? 'none' : 'block', opacity: candlesQuery.isPlaceholderData ? 0.6 : 1 }}
          />
        </>
      ) : (
        <Box sx={{ maxHeight: height + 32, overflow: 'auto' }}>
          <Table size="small" stickyHeader aria-label="Candles">
            <TableHead>
              <TableRow>
                {['Time', 'Open', 'High', 'Low', 'Close', 'Volume'].map((h) => (
                  <TableCell key={h} align={h === 'Time' ? 'left' : 'right'}>{h}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {[...candles].reverse().slice(0, 100).map((c) => (
                <TableRow key={c.time}>
                  <TableCell>{new Date(c.time * 1000).toLocaleString()}</TableCell>
                  {[c.open, c.high, c.low, c.close].map((v, i) => <TableCell key={i} align="right" className="num">{formatNumber(v)}</TableCell>)}
                  <TableCell align="right" className="num">{formatNumber(c.volume, 4)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <Typography variant="caption">Newest 100 candles.</Typography>
        </Box>
      )}
    </Panel>
  );
};
