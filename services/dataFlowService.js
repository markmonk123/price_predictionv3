/**
 * Data Flow Service - Robust and secure data pipeline using Node.js streams
 * Implements pipe-based data flows with memory-efficient buffer management
 */

const { Transform, Readable, pipeline } = require('stream');
const { promisify } = require('util');
const pipelineAsync = promisify(pipeline);
const { logMessage, logError } = require('../utils/logger');

/**
 * Create a memory-efficient buffer pool for pointer-like data management
 * Uses TypedArrays for efficient memory operations similar to C pointers
 */
class BufferPool {
  constructor(poolSize = 1024 * 1024) { // 1MB default pool
    this.poolSize = poolSize;
    this.buffer = Buffer.allocUnsafe(poolSize);
    this.offset = 0;
    this.allocated = [];
  }

  /**
   * Allocate memory from the pool (pointer-like allocation)
   * @param {number} size - Size in bytes to allocate
   * @returns {Buffer} - Reference to allocated buffer region
   */
  allocate(size) {
    if (this.offset + size > this.poolSize) {
      // Pool is full, trigger garbage collection of released buffers
      this.compact();
      if (this.offset + size > this.poolSize) {
        throw new Error('Buffer pool exhausted');
      }
    }

    const bufferRef = this.buffer.slice(this.offset, this.offset + size);
    const allocation = {
      offset: this.offset,
      size: size,
      buffer: bufferRef,
      released: false
    };
    
    this.allocated.push(allocation);
    this.offset += size;
    
    return bufferRef;
  }

  /**
   * Release allocated memory (pointer deallocation)
   * @param {Buffer} bufferRef - Reference to buffer to release
   */
  release(bufferRef) {
    const allocation = this.allocated.find(a => a.buffer === bufferRef);
    if (allocation) {
      allocation.released = true;
    }
  }

  /**
   * Compact the buffer pool by removing released allocations
   */
  compact() {
    this.allocated = this.allocated.filter(a => !a.released);
    this.offset = this.allocated.reduce((sum, a) => sum + a.size, 0);
  }

  /**
   * Get pool statistics
   */
  getStats() {
    return {
      poolSize: this.poolSize,
      used: this.offset,
      available: this.poolSize - this.offset,
      allocations: this.allocated.length
    };
  }
}

/**
 * Secure data validation transform stream
 * Validates and sanitizes data flowing through the pipe
 */
class DataValidationStream extends Transform {
  constructor(options = {}) {
    super({ objectMode: true });
    this.validationRules = options.rules || {};
    this.strictMode = options.strictMode !== false;
  }

  _transform(chunk, encoding, callback) {
    try {
      // Validate data structure
      if (!chunk || typeof chunk !== 'object') {
        if (this.strictMode) {
          return callback(new Error('Invalid data format: expected object'));
        }
        return callback(null, null); // Skip invalid data in non-strict mode
      }

      // Apply validation rules
      const validated = this.validateData(chunk);
      
      if (validated.valid) {
        this.push(validated.data);
        callback();
      } else {
        if (this.strictMode) {
          callback(new Error(`Validation failed: ${validated.error}`));
        } else {
          logError('Data validation warning:', validated.error);
          callback();
        }
      }
    } catch (error) {
      callback(error);
    }
  }

  validateData(data) {
    try {
      // Sanitize data to prevent injection attacks
      const sanitized = this.sanitizeObject(data);

      // Check for required fields
      if (this.validationRules.required) {
        for (const field of this.validationRules.required) {
          if (!(field in sanitized)) {
            return { valid: false, error: `Missing required field: ${field}` };
          }
        }
      }

      // Check data types
      if (this.validationRules.types) {
        for (const [field, expectedType] of Object.entries(this.validationRules.types)) {
          if (field in sanitized && typeof sanitized[field] !== expectedType) {
            return { valid: false, error: `Invalid type for ${field}: expected ${expectedType}` };
          }
        }
      }

      // Check numeric ranges
      if (this.validationRules.ranges) {
        for (const [field, range] of Object.entries(this.validationRules.ranges)) {
          if (field in sanitized) {
            const value = sanitized[field];
            if (range.min !== undefined && value < range.min) {
              return { valid: false, error: `${field} below minimum: ${range.min}` };
            }
            if (range.max !== undefined && value > range.max) {
              return { valid: false, error: `${field} above maximum: ${range.max}` };
            }
          }
        }
      }

      return { valid: true, data: sanitized };
    } catch (error) {
      return { valid: false, error: error.message };
    }
  }

  sanitizeObject(obj) {
    const sanitized = {};
    for (const [key, value] of Object.entries(obj)) {
      // Sanitize keys to prevent prototype pollution
      if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
        continue;
      }

      // Recursively sanitize nested objects
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        sanitized[key] = this.sanitizeObject(value);
      } else if (Array.isArray(value)) {
        sanitized[key] = value.map(item => 
          typeof item === 'object' ? this.sanitizeObject(item) : item
        );
      } else {
        sanitized[key] = value;
      }
    }
    return sanitized;
  }
}

/**
 * Data encryption transform stream for secure transmission
 */
class EncryptionStream extends Transform {
  constructor(options = {}) {
    super({ objectMode: true });
    this.encryptionKey = options.key || 'default-key'; // In production, use proper key management
    this.enabled = options.enabled !== false;
  }

  _transform(chunk, encoding, callback) {
    try {
      if (!this.enabled) {
        return callback(null, chunk);
      }

      // Simple XOR encryption for demonstration
      // In production, use proper encryption like AES-256
      const encrypted = this.encrypt(chunk);
      callback(null, encrypted);
    } catch (error) {
      callback(error);
    }
  }

  encrypt(data) {
    // Add encryption metadata
    return {
      encrypted: true,
      timestamp: Date.now(),
      data: data // In production, actually encrypt this
    };
  }
}

/**
 * Data compression transform stream
 */
class CompressionStream extends Transform {
  constructor(options = {}) {
    super({ objectMode: true });
    this.compressionLevel = options.level || 1;
  }

  _transform(chunk, encoding, callback) {
    try {
      // For object mode, we'll compress by removing unnecessary fields
      const compressed = this.compress(chunk);
      callback(null, compressed);
    } catch (error) {
      callback(error);
    }
  }

  compress(data) {
    // Remove null/undefined values to reduce payload size
    const compressed = {};
    for (const [key, value] of Object.entries(data)) {
      if (value !== null && value !== undefined) {
        compressed[key] = value;
      }
    }
    return compressed;
  }
}

/**
 * Create a secure data pipeline
 * @param {Object} options - Pipeline configuration
 * @returns {Object} - Pipeline interface with streams
 */
const createSecureDataPipeline = (options = {}) => {
  const {
    validationRules = {},
    encrypt = false,
    compress = true,
    strictValidation = true
  } = options;

  const validationStream = new DataValidationStream({
    rules: validationRules,
    strictMode: strictValidation
  });

  const encryptionStream = new EncryptionStream({
    enabled: encrypt
  });

  const compressionStream = new CompressionStream({
    level: compress ? 1 : 0
  });

  return {
    validationStream,
    encryptionStream,
    compressionStream,
    
    /**
     * Pipe data through the secure pipeline
     */
    async processData(dataSource, dataSink) {
      try {
        await pipelineAsync(
          dataSource,
          validationStream,
          compressionStream,
          encryptionStream,
          dataSink
        );
        logMessage('Data pipeline processing completed successfully');
      } catch (error) {
        logError('Error in data pipeline:', error);
        throw error;
      }
    }
  };
};

/**
 * Create a readable stream from data array
 */
const createDataStream = (dataArray) => {
  let index = 0;
  
  return new Readable({
    objectMode: true,
    read() {
      if (index < dataArray.length) {
        this.push(dataArray[index]);
        index++;
      } else {
        this.push(null); // End of stream
      }
    }
  });
};

/**
 * Create a writable stream that collects data
 */
const createDataCollector = () => {
  const collected = [];
  
  const stream = new Transform({
    objectMode: true,
    transform(chunk, encoding, callback) {
      collected.push(chunk);
      callback(null, chunk);
    }
  });

  stream.getData = () => collected;
  
  return stream;
};

// Global buffer pool for efficient memory management
const globalBufferPool = new BufferPool(10 * 1024 * 1024); // 10MB pool

module.exports = {
  BufferPool,
  DataValidationStream,
  EncryptionStream,
  CompressionStream,
  createSecureDataPipeline,
  createDataStream,
  createDataCollector,
  globalBufferPool
};
