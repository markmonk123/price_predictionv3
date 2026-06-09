const parseBoolean = (value, defaultValue = false) => {
  if (value === undefined || value === null || value === '') {
    return defaultValue;
  }

  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
};

const isProduction = process.env.NODE_ENV === 'production';

// Default to demo mode outside production, fail closed in production.
const isDemoMode = parseBoolean(process.env.DEMO_MODE, !isProduction);

module.exports = {
  isDemoMode,
  isProduction
};
