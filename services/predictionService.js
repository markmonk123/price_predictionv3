/**
 * Runs the Python model (python/run_prediction.py) with live Coinbase prices.
 */

const path = require('path');
const { PythonShell } = require('python-shell');
const { ApiError } = require('../server/http');
const { logMessage, logError } = require('../utils/logger');

const SCRIPT_DIR = path.join(__dirname, '..', 'python');

const defaultRunner = (args, pythonPath) => PythonShell.run('run_prediction.py', {
  mode: 'json',
  scriptPath: SCRIPT_DIR,
  pythonPath,
  args,
});

const createPredictionService = ({ market, config, events, runPython = defaultRunner }) => {
  let latest = null;
  let lastError = null;
  let inFlight = null;
  let timer = null;

  const execute = async () => {
    const ticker = await market.getTicker(config.product);
    // run_prediction.py replaces the newest bar's volume with --volume, so pass the
    // latest 5-minute candle's volume rather than a 24h figure.
    const candles = await market.getCandles(config.product, { granularity: 'FIVE_MINUTE', limit: 2 });
    const volume = candles.at(-1)?.volume ?? 0;
    const args = ['--price', String(ticker.price), '--volume', String(volume), '--time', new Date().toISOString()];
    if (market.offline) args.push('--allow-synthetic');

    let results;
    try {
      results = await runPython(args, config.pythonPath);
    } catch (error) {
      throw new ApiError(502, 'model_error', `Prediction script failed: ${error.message}`);
    }
    const prediction = (results || []).find((row) => row && typeof row === 'object');
    if (!prediction) throw new ApiError(502, 'model_error', 'Prediction script returned no result');
    if (prediction.error) throw new ApiError(502, 'model_error', `Python model error: ${prediction.error}`);

    latest = { ...prediction, productId: config.product, generatedAt: new Date().toISOString() };
    lastError = null;
    logMessage(`Prediction: direction=${latest.predicted_direction} confidence=${latest.confidence}`);
    events?.emit('prediction', latest);
    return latest;
  };

  /** Run the model; concurrent callers share one in-flight run. */
  const run = () => {
    if (!inFlight) {
      inFlight = execute()
        .catch((error) => {
          lastError = { message: error.message, at: new Date().toISOString() };
          throw error;
        })
        .finally(() => { inFlight = null; });
    }
    return inFlight;
  };

  const schedule = (intervalMs) => {
    const tick = () => run().catch((error) => logError('Scheduled prediction failed:', error));
    tick();
    timer = setInterval(tick, intervalMs);
    return () => clearInterval(timer);
  };

  return {
    run,
    schedule,
    getLatest: () => latest,
    status: () => ({ running: Boolean(inFlight), lastRunAt: latest?.generatedAt ?? null, lastError }),
  };
};

module.exports = { createPredictionService };
