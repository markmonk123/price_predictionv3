/**
 * HTTP helpers shared by every route: error envelope, validation, auth.
 *
 * Every error response has the shape { error: { code, message, details? } }.
 */

const crypto = require('crypto');
const { logError } = require('../utils/logger');

class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/** Express 4 does not catch rejected promises; forward them to the error handler. */
const asyncRoute = (handler) => (req, res, next) => {
  Promise.resolve(handler(req, res, next)).catch(next);
};

/** Validate req[source] with a zod schema and store the parsed value on req.valid[source]. */
const validate = (schema, source = 'body') => (req, res, next) => {
  const result = schema.safeParse(req[source] ?? {});
  if (!result.success) {
    return next(new ApiError(400, 'validation_error', 'Invalid request',
      result.error.issues.map((issue) => ({ path: issue.path.join('.'), message: issue.message }))));
  }
  req.valid = { ...req.valid, [source]: result.data };
  return next();
};

const tokensMatch = (expected, provided) => {
  const a = Buffer.from(expected);
  const b = Buffer.from(provided);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
};

/**
 * Guard mutating routes. With no API_TOKEN configured (local paper trading) the
 * guard is open; live mode refuses to start without a token (see config.js).
 */
const requireToken = (config) => (req, res, next) => {
  if (!config.apiToken) return next();
  const header = req.get('authorization') || '';
  const provided = header.startsWith('Bearer ') ? header.slice(7) : '';
  if (!provided || !tokensMatch(config.apiToken, provided)) {
    return next(new ApiError(401, 'unauthorized', 'A valid API token is required for this action'));
  }
  return next();
};

const notFound = (req, res, next) => next(new ApiError(404, 'not_found', `No route for ${req.method} ${req.originalUrl}`));

// eslint-disable-next-line no-unused-vars
const errorHandler = (err, req, res, next) => {
  if (err instanceof ApiError) {
    return res.status(err.status).json({ error: { code: err.code, message: err.message, details: err.details } });
  }
  if (err?.type === 'entity.parse.failed') {
    return res.status(400).json({ error: { code: 'invalid_json', message: 'Request body is not valid JSON' } });
  }
  logError(`Unhandled error on ${req.method} ${req.originalUrl}:`, err);
  return res.status(500).json({ error: { code: 'internal_error', message: 'Internal server error' } });
};

module.exports = { ApiError, asyncRoute, validate, requireToken, notFound, errorHandler };
