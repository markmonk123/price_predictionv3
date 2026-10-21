"""
Model state persistence for enhanced_prediction.py.

Saves and loads the fitted artefacts that enhanced_prediction.main() builds at the
end of every run, so the next run can either use the cached estimators directly
(load fast path) or fall back to a fresh training pass if the cache is missing,
corrupt, or schema-mismatched.

Scope (v1):
    * Save: VotingClassifier ensemble (post-calibration), EnsembleConfidenceBooster,
      top-3 ScalerEvaluator tuples, the feature column list, the last cv/accuracy.
    * Load: returns a frozen ModelState dataclass with all of the above. Any
      problem (missing file, decode error, schema mismatch, missing class)
      raises ModelStateError with a single-line message; callers are expected to
      catch it and run a fresh training pass instead.
    * I/O: atomic temp-file + os.replace; never leaves a half-written state file.
    * Stale-state management: optional purge of cache files older than N days.

Out of scope (intentional, for a future v2):
    * Warm-start retraining ("use saved state + new data"). v1 saves and reloads,
      but reload is followed by a fresh training pass; a caller can choose to
      short-circuit and predict with the loaded state until the next save lands.
    * Background retrain thread.
    * Distributed locks (file-lock or DB-backed).

Why joblib rather than pickle:
    joblib handles large numpy arrays more efficiently (mmap on reload, optional
    compression). It also plays nicer with sklearn's Bunch-like nested objects.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

LOG = logging.getLogger("model_state")

# Bump this whenever the on-disk schema changes incompatibly. The loader refuses
# to deserialize a state file whose schema_version does not equal this value.
SCHEMA_VERSION = 1

# Default cache directory: <repo-root>/services/_state/. Siblings of services/
# are addressed relative to this file. Callers can override via save_state(..., path=...).
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "_state"
_DEFAULT_STATE_NAME = "enhanced_prediction_state.joblib"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ModelStateError(Exception):
    """Raised when loading a state file fails for any reason.

    The .__cause__ attribute will carry the underlying exception. The .message is
    always a single line so callers can log it without further formatting.
    """


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelState:
    """A successfully loaded snapshot of the prediction pipeline's fitted state.

    Attributes:
        ensemble:        The fitted VotingClassifier (or CalibratedClassifierCV
                          wrapper applied on top), ready for .predict / .predict_proba.
        booster:         The fitted EnsembleConfidenceBooster, ready for
                          .predict_with_boost(...).
        scaler_results:  List of (name, avg_cv_score, fitted_scaler_instance) tuples,
                          length >= 1 (top-3 from ScalerEvaluator).
        feature_cols:    List/tuple of feature column names used at fit time. Required
                          to align new data on the predict path.
        cv_mean:         Last walk-forward CV mean accuracy.
        accuracy:        Last held-out accuracy.
        saved_at:        UTC datetime when this snapshot was written.
        schema_version:  Int -- always SCHEMA_VERSION at write time.
        metadata:        Optional caller-supplied dict (e.g. {"trigger": "scheduler",
                          "git_sha": "..."}).
    """
    ensemble: Any
    booster: Any
    scaler_results: Sequence[Tuple[str, float, Any]]
    feature_cols: Sequence[str]
    cv_mean: float
    accuracy: float
    saved_at: _dt.datetime
    schema_version: int
    metadata: dict = field(default_factory=dict)

    def is_fresh(self, max_age_seconds: float) -> bool:
        """True if this snapshot is no older than max_age_seconds ago (UTC)."""
        age = (_dt.datetime.now(_dt.timezone.utc) - self.saved_at).total_seconds()
        return 0 <= age <= max_age_seconds

    def describe(self) -> dict:
        """A small dict suitable for logging/print(). Stripping non-serialisables."""
        return {
            "schema_version": self.schema_version,
            "saved_at_utc": self.saved_at.isoformat(),
            "cv_mean": round(float(self.cv_mean), 4),
            "accuracy": round(float(self.accuracy), 4),
            "n_scalers": len(list(self.scaler_results)),
            "n_features": len(list(self.feature_cols)),
            "ensemble_type": f"{type(self.ensemble).__module__}.{type(self.ensemble).__name__}",
            "booster_type":  f"{type(self.booster).__module__}.{type(self.booster).__name__}",
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_state(
    *,
    ensemble: Any,
    booster: Any,
    scaler_results: Iterable[Tuple[str, float, Any]],
    feature_cols: Iterable[str],
    cv_mean: float,
    accuracy: float,
    cache_dir: Optional[Path] = None,
    state_name: str = _DEFAULT_STATE_NAME,
    metadata: Optional[dict] = None,
    overwrite: bool = True,
    compress: int = 3,
) -> Path:
    """Persist fitted estimators to <cache_dir>/<state_name>.

    Args:
        ensemble, booster, scaler_results, feature_cols, cv_mean, accuracy:
            The fitted artefacts (see ModelState).
        cache_dir:   Directory to write into. Defaults to services/_state/.
        state_name:  File name. Defaults to enhanced_prediction_state.joblib.
        metadata:    Optional JSON-serialisable dict to record alongside (e.g.
                     {"trigger": "main-end", "git_sha": "..."}).
        overwrite:   If False and the target file exists, raises FileExistsError.
                     Defaults to True for convenience.
        compress:    joblib compression level (0 disables). Default 3 keeps the
                     state file roughly 1/4 the size of an uncompressed dump.

    Returns:
        The Path of the written state file.

    Raises:
        RuntimeError if joblib is not importable.
        OSError on filesystem failure (the file is never left half-written).
    """
    if joblib is None:
        raise RuntimeError("joblib is required for save_state() but is not importable")

    target_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    target_path = (target_dir / state_name).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "saved_at": _dt.datetime.now(_dt.timezone.utc),
        "ensemble": ensemble,
        "booster": booster,
        "scaler_results": list(scaler_results),
        "feature_cols": list(feature_cols),
        "cv_mean": float(cv_mean),
        "accuracy": float(accuracy),
        "metadata": dict(metadata or {}),
    }

    _atomic_write_joblib(target_path, payload, compress=compress, overwrite=overwrite)
    LOG.info("saved model state to %s (%.1f KB)", target_path, target_path.stat().st_size / 1024)
    return target_path


def _atomic_write_joblib(target_path: Path, payload: Any, *, compress: int, overwrite: bool) -> None:
    """Write payload to target_path atomically via a sibling temp file."""
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing state: {target_path}")
    target_dir = target_path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_state_", suffix=".joblib", dir=str(target_dir))
    tmp_path = Path(tmp_name)
    try:
        import joblib as _joblib
        with os.fdopen(fd, "wb") as fh:
            _joblib.dump(payload, fh, compress=compress)
        # os.replace is atomic on POSIX and best-effort on Windows.
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_state(
    cache_dir: Optional[Path] = None,
    state_name: str = _DEFAULT_STATE_NAME,
) -> ModelState:
    """Load a previously-saved ModelState.

    Args:
        cache_dir: Directory containing the state file. Defaults to services/_state/.
        state_name: Filename. Defaults to enhanced_prediction_state.joblib.

    Returns:
        A ModelState frozen dataclass.

    Raises:
        FileNotFoundError  if the state file does not exist.
        ModelStateError    for any other failure (corrupt file, schema mismatch,
                           missing class, unpicklable object, etc.).
    """
    if joblib is None:
        raise ModelStateError("joblib is required for load_state() but is not importable")

    target_path = (Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR).resolve() / state_name
    if not target_path.exists():
        raise FileNotFoundError(f"no model state at {target_path}")

    try:
        with open(target_path, "rb") as fh:
            payload = joblib.load(fh)
    except Exception as exc:
        raise ModelStateError(
            f"failed to deserialize model state at {target_path}: {type(exc).__name__}: {exc}"
        ) from exc

    _require(payload, "schema_version", target_path)
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ModelStateError(
            f"schema mismatch in {target_path.name}: file has v{payload['schema_version']}, "
            f"this code expects v{SCHEMA_VERSION}. Delete the file or run a fresh training pass."
        )

    try:
        return ModelState(
            ensemble=payload["ensemble"],
            booster=payload["booster"],
            scaler_results=payload["scaler_results"],
            feature_cols=payload["feature_cols"],
            cv_mean=float(payload.get("cv_mean", 0.0)),
            accuracy=float(payload.get("accuracy", 0.0)),
            saved_at=payload.get("saved_at", _dt.datetime.now(_dt.timezone.utc)),
            schema_version=int(payload["schema_version"]),
            metadata=dict(payload.get("metadata", {})),
        )
    except Exception as exc:
        raise ModelStateError(
            f"model state at {target_path} is missing required fields "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _require(payload: dict, key: str, path: Path) -> None:
    if key not in payload:
        raise ModelStateError(f"model state at {path} is missing required field {key!r}")


# ---------------------------------------------------------------------------
# Stale-state management
# ---------------------------------------------------------------------------

def purge_stale_states(
    cache_dir: Optional[Path] = None,
    max_age_days: float = 7.0,
    pattern: str = "*.joblib",
) -> int:
    """Delete state files older than max_age_days. Returns count removed."""
    target_dir = (Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR).resolve()
    if not target_dir.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400.0
    removed = 0
    for path in target_dir.glob(pattern):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def ensure_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Create the cache dir if missing and return its absolute path."""
    target = (Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = [
    "SCHEMA_VERSION",
    "ModelState",
    "ModelStateError",
    "save_state",
    "load_state",
    "purge_stale_states",
    "ensure_cache_dir",
]