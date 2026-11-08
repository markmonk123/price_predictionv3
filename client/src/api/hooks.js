/** React Query hooks over the REST API. Query keys are shared with the realtime layer. */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError, apiFetch } from './client';
import { useSettings } from '../settings/SettingsContext';

export const keys = {
  health: ['health'],
  ticker: ['ticker'],
  candles: (granularity, limit) => ['candles', granularity, limit],
  book: (limit) => ['book', limit],
  prediction: ['prediction', 'latest'],
  history: (filters) => ['history', filters],
  accuracy: (filters) => ['accuracy', filters],
  orders: (filters) => ['orders', filters],
  account: (broker) => ['account', broker],
};

export const useHealth = () => useQuery({
  queryKey: keys.health,
  queryFn: ({ signal }) => apiFetch('/health', { signal }),
  refetchInterval: 15000,
});

export const useTicker = () => useQuery({
  queryKey: keys.ticker,
  queryFn: ({ signal }) => apiFetch('/market/ticker', { signal }),
  // Socket.IO pushes ticks every few seconds; polling is the fallback.
  refetchInterval: 15000,
});

export const useCandles = (granularity, limit = 300) => useQuery({
  queryKey: keys.candles(granularity, limit),
  queryFn: ({ signal }) => apiFetch('/market/candles', { query: { granularity, limit }, signal }),
  refetchInterval: 60000,
  placeholderData: keepPreviousData,
});

export const useOrderBook = (limit = 15) => useQuery({
  queryKey: keys.book(limit),
  queryFn: ({ signal }) => apiFetch('/market/book', { query: { limit }, signal }),
  refetchInterval: 3000,
  placeholderData: keepPreviousData,
});

export const useLatestPrediction = () => useQuery({
  queryKey: keys.prediction,
  queryFn: async ({ signal }) => {
    try {
      return await apiFetch('/predictions/latest', { signal });
    } catch (error) {
      // 404 just means the model hasn't produced a prediction yet.
      if (error instanceof ApiError && error.status === 404) return null;
      throw error;
    }
  },
  refetchInterval: 60000,
});

export const usePredictionHistory = (filters) => useQuery({
  queryKey: keys.history(filters),
  queryFn: ({ signal }) => apiFetch('/predictions/history', { query: filters, signal }),
  placeholderData: keepPreviousData,
  refetchInterval: 60000,
});

export const useAccuracy = (filters) => useQuery({
  queryKey: keys.accuracy(filters),
  queryFn: ({ signal }) => apiFetch('/predictions/accuracy', { query: filters, signal }),
  placeholderData: keepPreviousData,
  refetchInterval: 60000,
});

export const useOrders = (filters = {}) => useQuery({
  queryKey: keys.orders(filters),
  queryFn: ({ signal }) => apiFetch('/orders', { query: filters, signal }),
  placeholderData: keepPreviousData,
});

export const useAccount = (broker = 'paper', { enabled = true } = {}) => useQuery({
  queryKey: keys.account(broker),
  queryFn: ({ signal }) => apiFetch('/account', { query: { broker }, signal }),
  enabled,
});

/** Mutations send the API token from settings and refresh what they change. */
const useAuthedMutation = (request, invalidate = []) => {
  const { apiToken } = useSettings();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (variables) => request(variables, apiToken),
    onSuccess: () => invalidate.forEach((key) => queryClient.invalidateQueries({ queryKey: key })),
  });
};

export const usePreviewOrder = () => useAuthedMutation(
  (order, token) => apiFetch('/orders/preview', { method: 'POST', body: order, token }),
);

export const usePlaceOrder = () => useAuthedMutation(
  (order, token) => apiFetch('/orders', { method: 'POST', body: order, token }),
  [['orders'], ['account']],
);

export const useCancelOrder = () => useAuthedMutation(
  (id, token) => apiFetch(`/orders/${encodeURIComponent(id)}`, { method: 'DELETE', token }),
  [['orders'], ['account']],
);

export const useRunPrediction = () => useAuthedMutation(
  (_, token) => apiFetch('/predictions/run', { method: 'POST', token }),
  [keys.prediction, ['history'], ['accuracy'], keys.health],
);

export const useResetPaper = () => useAuthedMutation(
  (_, token) => apiFetch('/account/reset', { method: 'POST', token }),
  [['orders'], ['account']],
);
