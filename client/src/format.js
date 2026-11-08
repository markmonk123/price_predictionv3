const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });

export const formatUsd = (value) => (value === null || value === undefined || Number.isNaN(value) ? '–' : usd.format(value));

export const formatNumber = (value, digits = 2) => (value === null || value === undefined || Number.isNaN(value)
  ? '–'
  : Number(value).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }));

export const formatBtc = (value) => (value === null || value === undefined ? '–' : `${formatNumber(value, 8).replace(/\.?0+$/, '') || '0'} BTC`);

export const formatPct = (value, digits = 1) => (value === null || value === undefined || Number.isNaN(value)
  ? '–'
  : `${Number(value).toFixed(digits)}%`);

export const formatSignedPct = (value, digits = 2) => (value === null || value === undefined
  ? '–'
  : `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(digits)}%`);

export const formatTime = (value) => {
  if (!value) return '–';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '–' : date.toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
};

/** Model direction (-1/0/1) as a word; always shown alongside an arrow icon. */
export const directionLabel = (direction) => ({ 1: 'Up', [-1]: 'Down', 0: 'No change' }[direction] ?? '–');
