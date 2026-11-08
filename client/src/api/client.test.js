import { describe, expect, it, vi } from 'vitest';
import { ApiError, apiFetch, buildQuery, describeError } from './client';

const respond = (status, body) => vi.spyOn(globalThis, 'fetch')
  .mockResolvedValue(new Response(body === undefined ? '' : JSON.stringify(body), { status }));

describe('apiFetch', () => {
  it('returns JSON and sends the token and body', async () => {
    const spy = respond(200, { ok: true });
    await expect(apiFetch('/orders', { method: 'POST', body: { a: 1 }, token: 'tok', query: { x: 1, y: undefined } }))
      .resolves.toEqual({ ok: true });
    const [url, init] = spy.mock.calls[0];
    expect(url).toBe('/api/v1/orders?x=1');
    expect(init.headers.Authorization).toBe('Bearer tok');
    expect(init.body).toBe('{"a":1}');
  });

  it('turns the error envelope into an ApiError', async () => {
    respond(422, { error: { code: 'order_rejected', message: 'Paper order rejected', details: { errs: ['INSUFFICIENT_FUND'] } } });
    const error = await apiFetch('/orders').catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 422, code: 'order_rejected' });
    expect(describeError(error)).toBe('Paper order rejected (INSUFFICIENT_FUND)');
  });

  it('describes validation details and copes with non-JSON errors', async () => {
    respond(400, { error: { code: 'validation_error', message: 'Invalid request', details: [{ path: 'baseSize', message: 'Required' }] } });
    expect(describeError(await apiFetch('/x').catch((e) => e))).toBe('Invalid request: baseSize: Required');
    respond(502);
    const error = await apiFetch('/x').catch((e) => e);
    expect(error).toMatchObject({ status: 502, code: 'http_error', message: 'Request failed (502)' });
  });

  it('reports network failures clearly', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
    const error = await apiFetch('/health').catch((e) => e);
    expect(error).toMatchObject({ status: 0, code: 'network_error' });
  });
});

describe('buildQuery', () => {
  it('drops empty values', () => {
    expect(buildQuery({ a: 'x', b: '', c: null, d: 0 })).toBe('?a=x&d=0');
    expect(buildQuery({})).toBe('');
  });
});
