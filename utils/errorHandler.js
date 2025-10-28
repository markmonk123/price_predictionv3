/**
 * Error Handler Utility
 * Provides comprehensive error handling and recovery mechanisms
 */

const { logError, logMessage } = require('./logger');

/**
 * Custom error classes for different error types
 */
class DataFlowError extends Error {
  constructor(message, originalError = null) {
    super(message);
    this.name = 'DataFlowError';
    this.originalError = originalError;
    this.timestamp = new Date();
  }
}

class FIXProtocolError extends Error {
  constructor(message, originalError = null) {
    super(message);
    this.name = 'FIXProtocolError';
    this.originalError = originalError;
    this.timestamp = new Date();
  }
}

class CoinbaseAPIError extends Error {
  constructor(message, statusCode = null, originalError = null) {
    super(message);
    this.name = 'CoinbaseAPIError';
    this.statusCode = statusCode;
    this.originalError = originalError;
    this.timestamp = new Date();
  }
}

class ValidationError extends Error {
  constructor(message, field = null) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
    this.timestamp = new Date();
  }
}

/**
 * Error handler with retry logic
 */
class ErrorHandler {
  constructor(options = {}) {
    this.maxRetries = options.maxRetries || 3;
    this.retryDelay = options.retryDelay || 1000; // ms
    this.backoffMultiplier = options.backoffMultiplier || 2;
    this.errorCallbacks = {};
  }

  /**
   * Register error callback for specific error type
   */
  onError(errorType, callback) {
    if (!this.errorCallbacks[errorType]) {
      this.errorCallbacks[errorType] = [];
    }
    this.errorCallbacks[errorType].push(callback);
  }

  /**
   * Trigger error callbacks
   */
  async triggerCallbacks(errorType, error) {
    const callbacks = this.errorCallbacks[errorType] || [];
    for (const callback of callbacks) {
      try {
        await callback(error);
      } catch (err) {
        logError('Error in error callback:', err);
      }
    }
  }

  /**
   * Execute function with retry logic
   */
  async executeWithRetry(fn, context = null) {
    let lastError = null;
    let delay = this.retryDelay;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        if (attempt > 0) {
          logMessage(`Retry attempt ${attempt}/${this.maxRetries}...`);
          await this.sleep(delay);
          delay *= this.backoffMultiplier;
        }

        const result = await fn.call(context);
        
        if (attempt > 0) {
          logMessage(`Operation succeeded after ${attempt} retries`);
        }
        
        return result;
      } catch (error) {
        lastError = error;
        logError(`Attempt ${attempt + 1} failed:`, error);

        // Don't retry on validation errors
        if (error instanceof ValidationError) {
          throw error;
        }

        // Trigger error callbacks
        await this.triggerCallbacks(error.name, error);

        if (attempt === this.maxRetries) {
          logError(`All ${this.maxRetries + 1} attempts failed`);
          throw error;
        }
      }
    }

    throw lastError;
  }

  /**
   * Sleep helper for retry delays
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Wrap async function with error handling
   */
  wrapAsync(fn) {
    return async (...args) => {
      try {
        return await fn(...args);
      } catch (error) {
        return this.handleError(error);
      }
    };
  }

  /**
   * Handle error and return safe response
   */
  handleError(error) {
    logError('Handling error:', error);

    // Sanitize error for client response
    const safeError = {
      message: error.message || 'An unexpected error occurred',
      type: error.name || 'Error',
      timestamp: error.timestamp || new Date()
    };

    // Don't expose internal details in production
    if (process.env.NODE_ENV !== 'production') {
      safeError.stack = error.stack;
      safeError.details = error.originalError;
    }

    return safeError;
  }
}

/**
 * Circuit breaker pattern for fault tolerance
 */
class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000; // 1 minute
    this.monitoringPeriod = options.monitoringPeriod || 10000; // 10 seconds
    
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failureCount = 0;
    this.lastFailureTime = null;
    this.successCount = 0;
    this.requests = [];
  }

  /**
   * Execute function with circuit breaker
   */
  async execute(fn) {
    if (this.state === 'OPEN') {
      // Check if we should try to reset
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        logMessage('Circuit breaker entering HALF_OPEN state');
        this.state = 'HALF_OPEN';
        this.failureCount = 0;
      } else {
        throw new Error('Circuit breaker is OPEN - service unavailable');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  /**
   * Handle successful execution
   */
  onSuccess() {
    this.successCount++;
    
    if (this.state === 'HALF_OPEN') {
      logMessage('Circuit breaker closing after successful test');
      this.state = 'CLOSED';
      this.failureCount = 0;
    }

    this.recordRequest(true);
  }

  /**
   * Handle failed execution
   */
  onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.failureThreshold) {
      logMessage(`Circuit breaker opening after ${this.failureCount} failures`);
      this.state = 'OPEN';
    }

    this.recordRequest(false);
  }

  /**
   * Record request for monitoring
   */
  recordRequest(success) {
    const now = Date.now();
    this.requests.push({ timestamp: now, success });
    
    // Clean old requests
    this.requests = this.requests.filter(
      r => now - r.timestamp < this.monitoringPeriod
    );
  }

  /**
   * Get circuit breaker statistics
   */
  getStats() {
    const recentRequests = this.requests.length;
    const successfulRequests = this.requests.filter(r => r.success).length;
    const failureRate = recentRequests > 0 
      ? ((recentRequests - successfulRequests) / recentRequests) * 100 
      : 0;

    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      recentRequests,
      successfulRequests,
      failureRate: failureRate.toFixed(2) + '%',
      lastFailureTime: this.lastFailureTime
    };
  }
}

/**
 * Global error handler instance
 */
const globalErrorHandler = new ErrorHandler({
  maxRetries: 3,
  retryDelay: 1000,
  backoffMultiplier: 2
});

/**
 * Circuit breakers for different services
 */
const circuitBreakers = {
  coinbase: new CircuitBreaker({ failureThreshold: 5, resetTimeout: 60000 }),
  fix: new CircuitBreaker({ failureThreshold: 3, resetTimeout: 30000 }),
  prediction: new CircuitBreaker({ failureThreshold: 5, resetTimeout: 60000 })
};

module.exports = {
  DataFlowError,
  FIXProtocolError,
  CoinbaseAPIError,
  ValidationError,
  ErrorHandler,
  CircuitBreaker,
  globalErrorHandler,
  circuitBreakers
};
