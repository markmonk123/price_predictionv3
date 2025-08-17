/**
 * Logger utility for Bitcoin Trading Platform
 */

const fs = require('fs');
const path = require('path');

// Ensure logs directory exists
const logsDir = path.join(__dirname, '../logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Log file paths
const generalLogPath = path.join(logsDir, 'general.log');
const errorLogPath = path.join(logsDir, 'error.log');

/**
 * Write to log file
 */
const writeToLog = (logPath, message) => {
  const timestamp = new Date().toISOString();
  const logEntry = `${timestamp} - ${message}\n`;

  fs.appendFile(logPath, logEntry, (err) => {
    if (err) {
      console.error('Error writing to log file:', err);
    }
  });
};

/**
 * Log general message
 */
const logMessage = (message, ...args) => {
  const formattedMessage = args.length ? `${message} ${args.join(' ')}` : message;
  console.log(`[INFO] ${formattedMessage}`);
  writeToLog(generalLogPath, `[INFO] ${formattedMessage}`);
};

/**
 * Log error message
 */
const logError = (message, error) => {
  let errorMessage = message;

  if (error) {
    if (error instanceof Error) {
      errorMessage += ` ${error.message}\n${error.stack}`;
    } else {
      errorMessage += ` ${JSON.stringify(error)}`;
    }
  }

  console.error(`[ERROR] ${errorMessage}`);
  writeToLog(errorLogPath, `[ERROR] ${errorMessage}`);
};

module.exports = {
  logMessage,
  logError
};
