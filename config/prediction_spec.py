"""
Single source of truth for the prediction product spec (F4 / SOP D5 fix).

Every model path in the repo MUST derive its horizon and threshold from this
module; do NOT hardcode new ones anywhere else. If a new entry point needs
a different spec, parameterize the call site — keep this file authoritative.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Classification (Increase/Decrease/No-change) product spec
# Used by: run_prediction.py, enhanced_prediction.py (create_enhanced_features,
#          evaluate_threshold_horizon_grid), priceprediction.py (only where
#          explicitly opting in via SHARED).
# ---------------------------------------------------------------------------
HORIZON_MINUTES: int = 15
PCT_THRESHOLD: float = 0.015           # ±1.50% (raised from ±0.10% to reduce label noise)
TIMEFRAME_LABEL: str = f"{HORIZON_MINUTES} minutes"

# ---------------------------------------------------------------------------
# Regression (continuous multi-step forecaster) product spec
# Used by: enhanced_forecasting.py.
# ---------------------------------------------------------------------------
HORIZON_HOURS: int = 12

# ---------------------------------------------------------------------------
# Candle / bar-size spec
# Used by: enhanced_prediction.py, priceprediction.py (fetch_bitcoin_data),
#          create_enhanced_features (converts minute-based windows into bars).
# ---------------------------------------------------------------------------
BAR_MINUTES: int = 5                   # 5-minute candles
NUM_DATA_POINTS: int = 11000           # 11000 x 5m bars ~= 38 days of history

# ---------------------------------------------------------------------------
# Walk-forward CV search grid (centered on the spec above)
# Used by: enhanced_prediction.evaluate_threshold_horizon_grid.
# ---------------------------------------------------------------------------
HORIZON_GRID_MINUTES: tuple = (3, 6, 12, 24)     # 15m/30m/1h/2h in 5m bars
THRESHOLD_GRID: tuple = (0.010, 0.015, 0.020)


def spec_summary() -> str:
    """Return a one-line human-readable spec for logs / error messages."""
    return (
        f"bar={BAR_MINUTES}m, points={NUM_DATA_POINTS}, "
        f"horizon={HORIZON_MINUTES}m ({TIMEFRAME_LABEL}), "
        f"threshold=±{PCT_THRESHOLD * 100:.2f}%, "
        f"regression_horizon={HORIZON_HOURS}h"
    )


__all__ = [
    "HORIZON_MINUTES",
    "BAR_MINUTES",
    "NUM_DATA_POINTS",
    "PCT_THRESHOLD",
    "TIMEFRAME_LABEL",
    "HORIZON_HOURS",
    "HORIZON_GRID_MINUTES",
    "THRESHOLD_GRID",
    "spec_summary",
]
