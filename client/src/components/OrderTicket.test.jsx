import { describe, expect, it } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { OrderTicket, buildOrder } from './OrderTicket';
import { ModeBadge } from './ModeBadge';
import { mockApi, renderWithProviders } from '../test/utils';

const health = (live = false) => ({
  tradingMode: live ? 'live' : 'paper', brokers: live ? ['paper', 'coinbase'] : ['paper'],
  liveTradingEnabled: live, maxOrderUsd: 100,
});
const ticker = { productId: 'BTC-USD', price: 100, bid: 99, ask: 101, time: '2026-11-08T00:00:00Z' };
const previewBody = {
  broker: 'paper', price: 101, baseSize: 0.5, quoteSize: 50.5, commissionTotal: 0.3, orderTotal: 50.8, errs: [], warnings: [],
};

describe('buildOrder', () => {
  it('validates size and limit price', () => {
    expect(buildOrder({ size: '', type: 'MARKET' }, 'paper').error).toMatch(/size/);
    expect(buildOrder({ size: '-1', type: 'MARKET' }, 'paper').error).toMatch(/size/);
    expect(buildOrder({ size: '1', type: 'LIMIT', limitPrice: '' }, 'paper').error).toMatch(/limit price/);
  });

  it('sizes market orders in USD or BTC and limit orders in BTC', () => {
    expect(buildOrder({ side: 'BUY', type: 'MARKET', unit: 'USD', size: '50' }, 'paper').order)
      .toEqual({ broker: 'paper', side: 'BUY', type: 'MARKET', quoteSize: 50 });
    expect(buildOrder({ side: 'SELL', type: 'MARKET', unit: 'BTC', size: '0.1' }, 'paper').order)
      .toEqual({ broker: 'paper', side: 'SELL', type: 'MARKET', baseSize: 0.1 });
    expect(buildOrder({ side: 'BUY', type: 'LIMIT', unit: 'BTC', size: '0.1', limitPrice: '90', postOnly: true }, 'coinbase').order)
      .toEqual({ broker: 'coinbase', side: 'BUY', type: 'LIMIT', baseSize: 0.1, limitPrice: 90, postOnly: true });
  });
});

describe('OrderTicket', () => {
  it('requires a preview before placing, then places exactly the previewed order', async () => {
    const user = userEvent.setup();
    const calls = mockApi({
      'GET /health': health(),
      'GET /market/ticker': ticker,
      'POST /orders/preview': previewBody,
      'POST /orders': [201, { id: 'o1', status: 'FILLED', averageFilledPrice: 101 }],
    });
    renderWithProviders(<OrderTicket />, { settings: { apiToken: 'tok' } });

    const buy = screen.getByRole('button', { name: 'Place buy' });
    expect(buy).toBeDisabled();
    await user.type(screen.getByLabelText('Size'), '50');
    await user.click(screen.getByRole('button', { name: 'Preview' }));

    const preview = await screen.findByLabelText('Order preview');
    expect(within(preview).getByText('$50.80')).toBeInTheDocument();
    await user.click(buy);
    await screen.findByText(/Order filled at \$101.00/);

    const placed = calls.find((c) => c.method === 'POST' && c.path === '/orders');
    expect(placed.body).toEqual({ broker: 'paper', side: 'BUY', type: 'MARKET', quoteSize: 50 });
    expect(placed.headers.Authorization).toBe('Bearer tok');
  });

  it('editing the order discards the stale preview', async () => {
    const user = userEvent.setup();
    mockApi({ 'GET /health': health(), 'GET /market/ticker': ticker, 'POST /orders/preview': previewBody });
    renderWithProviders(<OrderTicket />);
    await user.type(screen.getByLabelText('Size'), '50');
    await user.click(screen.getByRole('button', { name: 'Preview' }));
    await screen.findByLabelText('Order preview');
    await user.type(screen.getByLabelText('Size'), '0');
    expect(screen.queryByLabelText('Order preview')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Place buy' })).toBeDisabled();
  });

  it('shows validation and API rejections', async () => {
    const user = userEvent.setup();
    mockApi({
      'GET /health': health(),
      'GET /market/ticker': ticker,
      'POST /orders/preview': { ...previewBody, errs: ['INSUFFICIENT_FUND'] },
    });
    renderWithProviders(<OrderTicket />);
    await user.click(screen.getByRole('button', { name: 'Preview' }));
    expect(screen.getByText('Enter a size greater than zero')).toBeInTheDocument();
    await user.type(screen.getByLabelText('Size'), '5000');
    await user.click(screen.getByRole('button', { name: 'Preview' }));
    await screen.findByText(/Would be rejected: INSUFFICIENT_FUND/);
    expect(screen.getByRole('button', { name: 'Place buy' })).toBeDisabled();
  });

  it('prefills the limit price once and requires confirmation for real-money orders', async () => {
    const user = userEvent.setup();
    const calls = mockApi({
      'GET /health': health(true),
      'GET /market/ticker': ticker,
      'POST /orders/preview': { ...previewBody, broker: 'coinbase' },
      'POST /orders': [201, { id: 'o2', status: 'PENDING' }],
    });
    renderWithProviders(<OrderTicket />);
    await screen.findByLabelText('Broker');
    await user.click(screen.getByLabelText('Broker'));
    await user.click(await screen.findByRole('option', { name: 'Coinbase (real money)' }));
    await screen.findByText(/REAL MONEY/);
    await screen.findByText(/Bid \$99\.00/);

    await user.click(screen.getByRole('button', { name: 'Limit' }));
    expect(screen.getByLabelText('Limit price')).toHaveValue('99');
    await user.type(screen.getByLabelText('Size'), '0.5');
    await user.click(screen.getByRole('button', { name: 'Preview' }));
    await screen.findByLabelText('Order preview');

    await user.click(screen.getByRole('button', { name: 'Place buy (real money)' }));
    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByText(/spends real money/)).toBeInTheDocument();
    expect(calls.some((c) => c.method === 'POST' && c.path === '/orders')).toBe(false);

    await user.click(within(dialog).getByRole('button', { name: 'Place real order' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/orders')).toBe(true));
    expect(calls.find((c) => c.path === '/orders' && c.method === 'POST').body)
      .toEqual({ broker: 'coinbase', side: 'BUY', type: 'LIMIT', baseSize: 0.5, limitPrice: 99, postOnly: false });
  });
});

describe('ModeBadge', () => {
  it('labels paper and live mode with text, not just color', () => {
    const { rerender } = renderWithProviders(<ModeBadge mode="paper" />);
    expect(screen.getByText('PAPER')).toBeInTheDocument();
    rerender(<ModeBadge mode="live" />);
    expect(screen.getByText('LIVE · real money')).toBeInTheDocument();
  });
});
