const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('events');
const { fakeMarket } = require('./helpers');
const { createPredictionService } = require('../../services/predictionService');

const config = { product: 'BTC-USD', pythonPath: undefined };

test('passes the live price and latest candle volume to the model and broadcasts the result', async () => {
  const events = new EventEmitter();
  const seen = [];
  events.on('prediction', (p) => seen.push(p));
  let args;
  const service = createPredictionService({
    market: fakeMarket({ price: 75000 }),
    config,
    events,
    runPython: async (a) => { args = a; return [{ predicted_direction: 1, confidence: 0.7 }]; },
  });
  const result = await service.run();
  assert.deepEqual(args.slice(0, 4), ['--price', '75000', '--volume', '5']);
  assert.ok(!args.includes('--allow-synthetic'));
  assert.equal(result.predicted_direction, 1);
  assert.equal(service.getLatest(), result);
  assert.deepEqual(seen, [result]);
});

test('python errors surface as model_error and are reported in status', async () => {
  const service = createPredictionService({
    market: fakeMarket(), config, runPython: async () => [{ error: 'boom', errorType: 'ValueError' }],
  });
  await assert.rejects(service.run(), { code: 'model_error', message: /boom/ });
  assert.match(service.status().lastError.message, /boom/);
  assert.equal(service.getLatest(), null);

  const crashing = createPredictionService({
    market: fakeMarket(), config, runPython: async () => { throw new Error('exit code 1'); },
  });
  await assert.rejects(crashing.run(), { code: 'model_error', message: /exit code 1/ });
});

test('concurrent runs share a single python process', async () => {
  let calls = 0;
  const service = createPredictionService({
    market: fakeMarket(),
    config,
    runPython: async () => { calls += 1; await new Promise((r) => setTimeout(r, 20)); return [{ predicted_direction: 0 }]; },
  });
  const [a, b] = await Promise.all([service.run(), service.run()]);
  assert.equal(calls, 1);
  assert.equal(a, b);
  await service.run();
  assert.equal(calls, 2);
});
