/**
 * Fetch wrapper for the /api/v1 REST API.
 * Non-2xx responses carry { error: { code, message, details } } and become ApiError.
 */

export class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

const BASE = `${import.meta.env.VITE_API_BASE ?? ''}/api/v1`;

export const buildQuery = (query = {}) => {
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
  });
  const text = params.toString();
  return text ? `?${text}` : '';
};

export const apiFetch = async (path, { method = 'GET', body, query, token, signal } = {}) => {
  let response;
  try {
    response = await fetch(`${BASE}${path}${buildQuery(query)}`, {
      method,
      signal,
      headers: {
        Accept: 'application/json',
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    throw new ApiError(0, 'network_error', 'Cannot reach the trading server. Is it running?');
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const err = payload?.error;
    throw new ApiError(response.status, err?.code ?? 'http_error',
      err?.message ?? `Request failed (${response.status})`, err?.details);
  }
  return payload;
};

/** Human-readable message for any error thrown by apiFetch, including validation details. */
export const describeError = (error) => {
  if (!error) return '';
  if (error instanceof ApiError && Array.isArray(error.details) && error.details.length) {
    return `${error.message}: ${error.details.map((d) => (d.path ? `${d.path}: ${d.message}` : d.message)).join('; ')}`;
  }
  if (error instanceof ApiError && error.details?.errs?.length) {
    return `${error.message} (${error.details.errs.join(', ')})`;
  }
  return error.message || String(error);
};
