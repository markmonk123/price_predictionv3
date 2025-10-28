/**
 * Test suite for data flow service
 * Tests secure data pipeline, buffer pool, and stream processing
 */

const {
  BufferPool,
  DataValidationStream,
  createSecureDataPipeline,
  createDataStream,
  createDataCollector
} = require('../services/dataFlowService');

/**
 * Simple test runner
 */
class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  test(name, fn) {
    this.tests.push({ name, fn });
  }

  async run() {
    console.log('\n=== Running Data Flow Tests ===\n');
    
    for (const test of this.tests) {
      try {
        await test.fn();
        this.passed++;
        console.log(`✓ ${test.name}`);
      } catch (error) {
        this.failed++;
        console.log(`✗ ${test.name}`);
        console.log(`  Error: ${error.message}`);
      }
    }

    console.log(`\n=== Test Results ===`);
    console.log(`Passed: ${this.passed}`);
    console.log(`Failed: ${this.failed}`);
    console.log(`Total: ${this.tests.length}\n`);

    return this.failed === 0;
  }
}

// Create test runner
const runner = new TestRunner();

// Test BufferPool allocation and deallocation
runner.test('BufferPool: allocate and release memory', () => {
  const pool = new BufferPool(1024);
  
  // Allocate memory
  const buffer1 = pool.allocate(256);
  const buffer2 = pool.allocate(256);
  
  if (buffer1.length !== 256) {
    throw new Error('Buffer allocation size mismatch');
  }
  
  if (pool.offset !== 512) {
    throw new Error('Pool offset not updated correctly');
  }
  
  // Release memory
  pool.release(buffer1);
  pool.compact();
  
  const stats = pool.getStats();
  if (stats.allocations !== 1) {
    throw new Error('Buffer release failed');
  }
});

// Test BufferPool exhaustion
runner.test('BufferPool: handle exhaustion gracefully', () => {
  const pool = new BufferPool(100);
  
  try {
    // Try to allocate more than pool size
    pool.allocate(50);
    pool.allocate(60); // Should trigger compact and then fail
    throw new Error('Should have thrown exhaustion error');
  } catch (error) {
    if (!error.message.includes('exhausted')) {
      throw error;
    }
  }
});

// Test data validation stream
runner.test('DataValidationStream: validate correct data', async () => {
  const validationRules = {
    required: ['price', 'symbol'],
    types: { price: 'number', symbol: 'string' }
  };
  
  const stream = new DataValidationStream({ rules: validationRules });
  const data = [{ price: 60000, symbol: 'BTC-USD', volume: 1000 }];
  
  const source = createDataStream(data);
  const sink = createDataCollector();
  
  await new Promise((resolve, reject) => {
    source
      .pipe(stream)
      .pipe(sink)
      .on('finish', resolve)
      .on('error', reject);
  });
  
  const result = sink.getData();
  if (result.length !== 1) {
    throw new Error('Validation failed for valid data');
  }
  
  if (result[0].price !== 60000) {
    throw new Error('Data corrupted during validation');
  }
});

// Test data validation stream with invalid data
runner.test('DataValidationStream: reject invalid data in strict mode', async () => {
  const validationRules = {
    required: ['price'],
    types: { price: 'number' }
  };
  
  const stream = new DataValidationStream({ 
    rules: validationRules,
    strictMode: true
  });
  
  const data = [{ symbol: 'BTC-USD' }]; // Missing required 'price'
  
  const source = createDataStream(data);
  const sink = createDataCollector();
  
  let errorCaught = false;
  
  try {
    await new Promise((resolve, reject) => {
      const pipeline = source.pipe(stream).pipe(sink);
      
      pipeline.on('finish', resolve);
      pipeline.on('error', (err) => {
        errorCaught = true;
        reject(err);
      });
      
      // Also catch errors on individual streams
      source.on('error', reject);
      stream.on('error', (err) => {
        errorCaught = true;
        reject(err);
      });
      sink.on('error', reject);
    });
    
    if (!errorCaught) {
      throw new Error('Should have rejected invalid data');
    }
  } catch (error) {
    if (!error.message.includes('required')) {
      throw error;
    }
  }
});

// Test prototype pollution prevention
runner.test('DataValidationStream: prevent prototype pollution', async () => {
  const stream = new DataValidationStream({ rules: {} });
  
  const maliciousData = [{
    __proto__: { polluted: true },
    constructor: { polluted: true },
    prototype: { polluted: true },
    price: 60000
  }];
  
  const source = createDataStream(maliciousData);
  const sink = createDataCollector();
  
  await new Promise((resolve, reject) => {
    source
      .pipe(stream)
      .pipe(sink)
      .on('finish', resolve)
      .on('error', reject);
  });
  
  const result = sink.getData();
  
  // Check that dangerous keys were removed during sanitization
  const sanitizedKeys = Object.keys(result[0]);
  const hasDangerousKeys = sanitizedKeys.some(key => 
    key === '__proto__' || key === 'constructor' || key === 'prototype'
  );
  
  if (hasDangerousKeys) {
    throw new Error('Prototype pollution prevention failed');
  }
  
  if (result[0].price !== 60000) {
    throw new Error('Valid data was incorrectly filtered');
  }
});

// Test secure data pipeline end-to-end
runner.test('SecureDataPipeline: process data through full pipeline', async () => {
  const pipeline = createSecureDataPipeline({
    validationRules: {
      required: ['price'],
      types: { price: 'number' },
      ranges: { price: { min: 0, max: 100000 } }
    },
    compress: true,
    encrypt: false,
    strictValidation: true
  });
  
  const data = [
    { price: 60000, symbol: 'BTC-USD', volume: 1000, metadata: null }
  ];
  
  const source = createDataStream(data);
  const sink = createDataCollector();
  
  await pipeline.processData(source, sink);
  
  const result = sink.getData();
  
  if (result.length !== 1) {
    throw new Error('Pipeline did not process data');
  }
  
  // Compression should have removed null values
  if ('metadata' in result[0]) {
    throw new Error('Compression did not remove null values');
  }
  
  if (result[0].price !== 60000) {
    throw new Error('Data corrupted in pipeline');
  }
});

// Test range validation
runner.test('DataValidationStream: enforce range limits', async () => {
  const validationRules = {
    required: ['price'],
    types: { price: 'number' },
    ranges: { price: { min: 1000, max: 100000 } }
  };
  
  const stream = new DataValidationStream({ 
    rules: validationRules,
    strictMode: true
  });
  
  const data = [{ price: 200000 }]; // Above maximum
  
  const source = createDataStream(data);
  const sink = createDataCollector();
  
  let errorCaught = false;
  
  try {
    await new Promise((resolve, reject) => {
      const pipeline = source.pipe(stream).pipe(sink);
      
      pipeline.on('finish', resolve);
      pipeline.on('error', reject);
      
      stream.on('error', (err) => {
        errorCaught = true;
        reject(err);
      });
    });
    
    if (!errorCaught) {
      throw new Error('Should have rejected out-of-range data');
    }
  } catch (error) {
    if (!error.message.includes('maximum')) {
      throw error;
    }
  }
});

// Test data stream with empty array
runner.test('DataStream: handle empty data array', async () => {
  const source = createDataStream([]);
  const sink = createDataCollector();
  
  await new Promise((resolve, reject) => {
    source
      .pipe(sink)
      .on('finish', resolve)
      .on('error', reject);
  });
  
  const result = sink.getData();
  if (result.length !== 0) {
    throw new Error('Empty stream produced data');
  }
});

// Test data stream with large dataset
runner.test('DataStream: handle large dataset efficiently', async () => {
  const largeData = Array.from({ length: 1000 }, (_, i) => ({
    price: 60000 + i,
    timestamp: Date.now(),
    volume: 1000
  }));
  
  const source = createDataStream(largeData);
  const sink = createDataCollector();
  
  const startTime = Date.now();
  
  await new Promise((resolve, reject) => {
    source
      .pipe(sink)
      .on('finish', resolve)
      .on('error', reject);
  });
  
  const duration = Date.now() - startTime;
  
  const result = sink.getData();
  if (result.length !== 1000) {
    throw new Error('Large dataset not processed completely');
  }
  
  // Should complete in reasonable time (< 1 second)
  if (duration > 1000) {
    throw new Error('Large dataset processing too slow');
  }
});

// Run all tests
if (require.main === module) {
  runner.run().then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = runner;
