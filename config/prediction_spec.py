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
PCT_THRESHOLD: float = 0.005          # ±0.5%
TIMEFRAME_LABEL: str = f"{HORIZON_MINUTES} minutes"

# ---------------------------------------------------------------------------
# Regression (continuous multi-step forecaster) product spec
# Used by: enhanced_forecasting.py.
# ---------------------------------------------------------------------------
HORIZON_HOURS: int = 12

# ---------------------------------------------------------------------------
# Walk-forward CV search grid (centered on the spec above)
# Used by: enhanced_prediction.evaluate_threshold_horizon_grid.
# ---------------------------------------------------------------------------
HORIZON_GRID_MINUTES: tuple = (5, 15, 30, 60)
THRESHOLD_GRID: tuple = (0.003, 0.005, 0.008)


def spec_summary() -> str:
    """Return a one-line human-readable spec for logs / error messages."""
    return (
        f"horizon={HORIZON_MINUTES}m ({TIMEFRAME_LABEL}), "
        f"threshold=±{PCT_THRESHOLD * 100:.1f}%, "
        f"regression_horizon={HORIZON_HOURS}h"
    )


__all__ = [
    "HORIZON_MINUTES",
    "PCT_THRESHOLD",
    "TIMEFRAME_LABEL",
    "HORIZON_HOURS",
    "HORIZON_GRID_MINUTES",
    "THRESHOLD_GRID",
    "spec_summary",
]
