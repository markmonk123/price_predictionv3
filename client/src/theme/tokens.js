/**
 * Design tokens. Values come from the dataviz reference palette, with dark mode
 * as its own selected steps (not an automatic inversion).
 *
 * Price direction uses the validated blue (up) / red (down) diverging pair
 * rather than green/red, which deuteranopes and protanopes cannot separate.
 * Direction is never color-only: arrows and labels always accompany it.
 */

export const tokens = {
  light: {
    page: '#f9f9f7',
    surface: '#fcfcfb',
    surfaceRaised: '#ffffff',
    textPrimary: '#0b0b0b',
    textSecondary: '#52514e',
    textMuted: '#898781',
    grid: '#e1e0d9',
    axis: '#c3c2b7',
    border: 'rgba(11,11,11,0.10)',
    neutralMid: '#c3c2b7',
    up: '#2a78d6',
    down: '#e34948',
    accent: '#2a78d6',
    deltaGoodText: '#006300',
  },
  dark: {
    page: '#0d0d0d',
    surface: '#1a1a19',
    surfaceRaised: '#21211f',
    textPrimary: '#f0efec',
    textSecondary: '#c3c2b7',
    textMuted: '#898781',
    grid: '#2c2c2a',
    axis: '#383835',
    border: 'rgba(255,255,255,0.10)',
    neutralMid: '#5c5b56',
    up: '#3987e5',
    down: '#e66767',
    accent: '#3987e5',
    deltaGoodText: '#0ca30c',
  },
};

/** Status colors are reserved for state (never series) and always ship with an icon + label. */
export const status = {
  good: '#0ca30c',
  warning: '#fab219',
  serious: '#ec835a',
  critical: '#d03b3b',
};

export const fontFamily = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
