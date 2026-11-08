/** Client-side preferences persisted in localStorage (theme, API token). */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const STORAGE_KEY = 'btc-predictor-settings';
const DEFAULTS = { themeMode: 'dark', apiToken: '' };

const load = () => {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') };
  } catch {
    return DEFAULTS;
  }
};

const SettingsContext = createContext(null);

export const SettingsProvider = ({ children, initial }) => {
  const [settings, setSettings] = useState(() => ({ ...load(), ...initial }));

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const update = useCallback((patch) => setSettings((prev) => ({ ...prev, ...patch })), []);
  const value = useMemo(() => ({ ...settings, update }), [settings, update]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
};

export const useSettings = () => {
  const value = useContext(SettingsContext);
  if (!value) throw new Error('useSettings must be used inside SettingsProvider');
  return value;
};
