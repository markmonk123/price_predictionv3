import { render } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '@mui/material/styles';
import { MemoryRouter } from 'react-router-dom';
import { vi } from 'vitest';
import { createAppTheme } from '../theme/theme';
import { SettingsProvider } from '../settings/SettingsContext';

/**
 * Route fetch calls by "METHOD /path" (path without /api/v1 and query).
 * Handlers return [status, body] or a body (200). Every call is recorded.
 */
export const mockApi = (routes) => {
  const calls = [];
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init = {}) => {
    const { pathname, search } = new URL(url, 'http://localhost');
    const method = init.method || 'GET';
    const path = pathname.replace(/^\/api\/v1/, '');
    const body = init.body ? JSON.parse(init.body) : undefined;
    calls.push({ method, path, search, body, headers: init.headers });
    const handler = routes[`${method} ${path}`];
    if (!handler) return new Response(JSON.stringify({ error: { code: 'not_found', message: `No mock for ${method} ${path}` } }), { status: 404 });
    const result = typeof handler === 'function' ? handler(body) : handler;
    const [status, payload] = Array.isArray(result) ? result : [200, result];
    return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } });
  });
  return calls;
};

export const renderWithProviders = (ui, { settings, mode = 'dark' } = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <SettingsProvider initial={settings}>
        <ThemeProvider theme={createAppTheme(mode)}>
          <MemoryRouter>{ui}</MemoryRouter>
        </ThemeProvider>
      </SettingsProvider>
    </QueryClientProvider>,
  );
};
