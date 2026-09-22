#!/usr/bin/env bash
# =============================================================================
# setup-cuml-wsl.sh -- one-shot cuML/CUDA install for the price_predictionv3
# repo inside WSL.
#
# What this script does:
#   1. Verifies WSL/Linux prerequisites (Python 3.10-3.12, NVIDIA driver >= 525,
#      CUDA toolkit visible, network access to pypi.nvidia.com).
#   2. Creates or recreates a Python 3.12 .venv at $PROJECT_DIR/.venv.
#   3. Installs pip >= 24 and the project's pinned cuML/cupy/nvidia-cuda wheels
#      (versions read from requirements.txt; can be overridden on the CLI).
#   4. Runs the same signature probe that enhanced_prediction.py uses
#      (sample_weight in cuML .fit() for both RF and LR) and prints a clear
#      pass/fail summary.
#
# Run this from inside the WSL distro (Ubuntu-22.04 or whatever you use):
#   bash scripts/setup-cuml-wsl.sh
#
# Flags:
#   --recreate-venv         Delete and recreate $VENV_DIR if it already exists.
#   --target-cuml VERSION   Override the cuML version (default: from requirements.txt).
#   --target-cupy VERSION   Override the cupy version (default: from requirements.txt).
#   --python VERSION        Override the Python interpreter to use (default: python3.12).
#   --project-dir PATH      Override the project root (default: /home/monki/projects/price_predictionv3).
#   --verify-only           Only run diagnostics, do not install anything.
#   --dry-run               Print what would be done without executing pip/venv.
#   -h, --help              Show this help.
#
# Exit codes:
#   0   success (install + verify both passed)
#   1   prerequisite check failed (Python version, NVIDIA driver, etc.)
#   2   venv creation failed
#   3   pip install failed
#   4   post-install verification failed (cuML signature probe missing sample_weight)
# =============================================================================

set -u
# We deliberately do NOT use 'set -e' -- we want to continue past non-fatal
# diagnostics and only exit on hard failures with structured codes.

PROJECT_DIR="${PROJECT_DIR:-/home/monki/projects/price_predictionv3}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN_DEFAULT="python3.12"
NVIDIA_INDEX="https://pypi.nvidia.com"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

# Defaults read from requirements.txt pin; overridable via flags.
TARGET_CUML=""
TARGET_CUPY=""
PYTHON_BIN=""
DO_RECREATE_VENV=0
DO_VERIFY_ONLY=0
DRY_RUN=0

# --- helpers -----------------------------------------------------------------

section() {
  printf '\n=== %s ===\n' "$1"
}

say() { printf '  %s\n' "$*"; }
warn() { printf '  WARN: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

run_or_echo() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  [dry-run] %s\n' "$*"
    return 0
  else
    printf '  %s\n' "$*"
    "$@"
  fi
}

# --- arg parsing -------------------------------------------------------------

usage() {
  sed -n '2,40p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate-venv)    DO_RECREATE_VENV=1; shift ;;
    --target-cuml)      TARGET_CUML="$2"; shift 2 ;;
    --target-cupy)      TARGET_CUPY="$2"; shift 2 ;;
    --python)           PYTHON_BIN="$2"; shift 2 ;;
    --project-dir)      PROJECT_DIR="$2"; VENV_DIR="$PROJECT_DIR/.venv"; REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"; shift 2 ;;
    --verify-only)      DO_VERIFY_ONLY=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage ;;
    *) die "Unknown argument: $1" ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

# --- 0. guard rails ----------------------------------------------------------

section "0. Prerequisites"

# Are we actually inside WSL/Linux?
if [[ "$(uname -s)" != "Linux" ]]; then
  die "This script must be run from inside WSL (got uname=$(uname -s))."
fi

# Project directory exists?
if [[ ! -d "$PROJECT_DIR" ]]; then
  die "Project directory not found: $PROJECT_DIR. Use --project-dir to override."
fi
say "Project dir: $PROJECT_DIR"

# Python interpreter exists?
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  die "Python interpreter not on PATH: $PYTHON_BIN. Install it first, e.g.: sudo apt install python3.12 python3.12-venv"
fi
PY_VER=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
say "Python: $PY_VER ($("$PYTHON_BIN" -V 2>&1))"

PY_MAJOR=${PY_VER%%.*}
PY_MINOR=${PY_VER##*.}
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -lt 10 || "$PY_MINOR" -ge 13 ]]; then
  die "Python $PY_VER is out of range. RAPIDS 24-26 requires Python 3.10-3.12. Use --python python3.12."
fi

# Python venv module available?
if ! "$PYTHON_BIN" -c 'import venv' >/dev/null 2>&1; then
  die "Python venv module missing. Install it: sudo apt install python${PY_VER}-venv"
fi

# NVIDIA driver / GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  say "GPU: ${GPU_NAME:-<unknown>}  driver: ${NVIDIA_DRIVER:-<unknown>}"
  DRIVER_OK=0
  if [[ "${NVIDIA_DRIVER:-}" =~ ^([0-9]+)\. ]]; then
    DRIVER_MAJOR=${BASH_REMATCH[1]}
    if [[ "$DRIVER_MAJOR" -ge 525 ]]; then
      DRIVER_OK=1
    fi
  fi
  if [[ "$DRIVER_OK" -eq 0 ]]; then
    warn "NVIDIA driver >= 525 required for CUDA 12.x. Found: ${NVIDIA_DRIVER:-unknown}."
    if [[ "$DRY_RUN" -eq 1 ]]; then
      say "[dry-run] continuing anyway"
    else
      read -r -p "Continue anyway? [y/N] " ANSWER
      if [[ "$ANSWER" != "y" && "$ANSWER" != "Y" ]]; then
        exit 1
      fi
    fi
  fi
else
  warn "nvidia-smi not found on PATH. If you have an NVIDIA GPU, install the WSL CUDA driver first:"
  warn "  https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    read -r -p "Continue without GPU? [y/N] " ANSWER
    if [[ "$ANSWER" != "y" && "$ANSWER" != "Y" ]]; then
      exit 1
    fi
  fi
fi

# Network sanity check
if command -v curl >/dev/null 2>&1; then
  if curl -sSf --max-time 8 -o /dev/null "$NVIDIA_INDEX" 2>/dev/null; then
    say "Network: pypi.nvidia.com reachable"
  else
    warn "pypi.nvidia.com unreachable. pip install will fail if the index is required."
  fi
fi

# --- 1. read defaults from requirements.txt ----------------------------------

section "1. Reading cuML/cupy versions from $REQUIREMENTS_FILE"

if [[ -z "$TARGET_CUML" || -z "$TARGET_CUPY" ]]; then
  if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    die "requirements.txt not found at $REQUIREMENTS_FILE. Use --target-cuml / --target-cupy."
  fi
  # Extract the first cuml-cu12== and cupy-cuda12x>= pins.
  if [[ -z "$TARGET_CUML" ]]; then
    TARGET_CUML=$(grep -E '^cuml-cu12==' "$REQUIREMENTS_FILE" | head -1 | sed -E 's/.*==([^ ;]+).*/\1/')
  fi
  if [[ -z "$TARGET_CUPY" ]]; then
    TARGET_CUPY=$(grep -E '^cupy-cuda12x' "$REQUIREMENTS_FILE" | head -1 | sed -E 's/.*>=([^ ;]+).*/\1/')
  fi
fi

if [[ -z "$TARGET_CUML" || -z "$TARGET_CUPY" ]]; then
  die "Could not determine cuML/cupy versions. Pass --target-cuml and --target-cupy explicitly."
fi
say "cuML target: $TARGET_CUML"
say "cupy target: >= $TARGET_CUPY"

# --- 2. venv ------------------------------------------------------------------

section "2. Python virtualenv at $VENV_DIR"

if [[ -d "$VENV_DIR" ]]; then
  if [[ "$DO_RECREATE_VENV" -eq 1 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      say "[dry-run] would rm -rf $VENV_DIR"
    else
      say "Recreating venv: removing $VENV_DIR"
      rm -rf "$VENV_DIR"
    fi
  else
    say "Existing venv found; leaving in place (use --recreate-venv to nuke it)"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    say "[dry-run] would run: $PYTHON_BIN -m venv $VENV_DIR"
  else
    say "Creating venv: $PYTHON_BIN -m venv $VENV_DIR"
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
      die "venv creation failed. On Ubuntu you may need: sudo apt install python${PY_VER}-venv python3.12-dev" 2
    fi
  fi
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [[ "$DRY_RUN" -eq 0 && ! -x "$VENV_PY" ]]; then
  die "venv Python not found at $VENV_PY"
fi
say "venv python: $VENV_PY"

# --- 3. install ---------------------------------------------------------------

if [[ "$DO_VERIFY_ONLY" -eq 1 ]]; then
  say "--verify-only set; skipping pip install."
  PIP_SECTION_SKIPPED=1
fi

if [[ "${PIP_SECTION_SKIPPED:-0}" -eq 0 ]]; then
  section "3. Upgrading pip + installing cuML/cupy/nvidia wheels"

  PIP_LOG=/tmp/setup-cuml-wsl-pip.log

  run_or_echo_capture() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '  [dry-run] %s\n' "$*"
      return 0
    else
      printf '  %s\n' "$*"
      "$@" &>"$PIP_LOG"
      return $?
    fi
  }

  if run_or_echo_capture "$VENV_PIP" install --upgrade pip wheel setuptools; then
    :
  else
    PIP_RC=$?
    warn "pip self-upgrade failed (rc=$PIP_RC). Tail of log:"
    tail -n 20 "$PIP_LOG" 2>/dev/null || true
    exit 3
  fi

  if run_or_echo_capture "$VENV_PIP" install \
      --extra-index-url "$NVIDIA_INDEX" \
      "cuml-cu12==$TARGET_CUML" \
      "cupy-cuda12x>=$TARGET_CUPY" \
      "nvidia-cuda-runtime-cu12>=12"; then
    :
  else
    PIP_RC=$?
    warn "cuML/cupy install failed (rc=$PIP_RC). Tail of log:"
    tail -n 40 "$PIP_LOG" 2>/dev/null || true
    echo
    echo "Common causes:"
    echo "  - Python $PY_VER has no cuML wheel (recreate venv on Python 3.10-3.12)."
    echo "  - Network blocking $NVIDIA_INDEX."
    echo "  - Mismatched CUDA toolkit vs RAPIDS version (driver >= 525 required)."
    exit 3
  fi

  # Install the rest of the project requirements (sklearn, pandas, etc.) if a
  # requirements file is present.
  if [[ -f "$REQUIREMENTS_FILE" ]]; then
    section "3b. Installing remaining requirements from $REQUIREMENTS_FILE"
    if run_or_echo_capture "$VENV_PIP" install -r "$REQUIREMENTS_FILE"; then
      :
    else
      PIP_RC=$?
      warn "Project requirements install failed (rc=$PIP_RC). Tail of log:"
      tail -n 40 "$PIP_LOG" 2>/dev/null || true
      exit 3
    fi
  fi
fi

# --- 4. verify ----------------------------------------------------------------

section "4. Post-install verification"

if [[ "$DRY_RUN" -eq 1 ]]; then
  say "[dry-run] skipping verification"
  exit 0
fi

if [[ ! -x "$VENV_PY" ]]; then
  die "Cannot run verification: $VENV_PY not found"
fi

# Probe cuML import + signature
PROBE_OUT=$("$VENV_PY" - <<'EOF' 2>&1
import json, sys
result = {
    "python": "%d.%d" % sys.version_info[:2],
    "cuml_imported": False,
    "cuml_version": None,
    "cupy_imported": False,
    "cupy_version": None,
    "rf_sample_weight": None,
    "lr_sample_weight": None,
    "cuda_runtime": None,
    "nvidia_driver": None,
    "errors": [],
}
try:
    import cuml
    result["cuml_imported"] = True
    result["cuml_version"] = cuml.__version__
except Exception as e:
    result["errors"].append(f"cuml import: {e}")
try:
    import cupy
    result["cupy_imported"] = True
    result["cupy_version"] = cupy.__version__
    try:
        result["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
    except Exception as e:
        result["errors"].append(f"cupy CUDA runtime: {e}")
except Exception as e:
    result["errors"].append(f"cupy import: {e}")
try:
    import inspect
    from cuml.ensemble import RandomForestClassifier as _cuRF
    rf = _cuRF(n_estimators=10, max_depth=4)
    result["rf_sample_weight"] = "sample_weight" in inspect.signature(rf.fit).parameters
except Exception as e:
    result["errors"].append(f"RF probe: {e}")
try:
    import inspect
    from cuml.linear_model import LogisticRegression as _cuLR
    lr = _cuLR()
    result["lr_sample_weight"] = "sample_weight" in inspect.signature(lr.fit).parameters
except Exception as e:
    result["errors"].append(f"LR probe: {e}")
print(json.dumps(result, indent=2))
EOF
)

echo "$PROBE_OUT"

# Summarize
RF_OK=$(echo "$PROBE_OUT" | grep -E '"rf_sample_weight":\s*true' | wc -l)
LR_OK=$(echo "$PROBE_OUT" | grep -E '"lr_sample_weight":\s*true' | wc -l)
CUML_IMPORTED=$(echo "$PROBE_OUT" | grep -E '"cuml_imported":\s*true' | wc -l)
CUPY_IMPORTED=$(echo "$PROBE_OUT" | grep -E '"cupy_imported":\s*true' | wc -l)

echo
echo "=== Summary ==="
if [[ "$CUML_IMPORTED" -eq 1 ]]; then
  CUML_VER=$(echo "$PROBE_OUT" | grep -E '"cuml_version":' | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
  echo "  cuML: $CUML_VER (imported OK)"
else
  echo "  cuML: NOT IMPORTED"
fi
if [[ "$CUPY_IMPORTED" -eq 1 ]]; then
  CUPY_VER=$(echo "$PROBE_OUT" | grep -E '"cupy_version":' | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
  CUDA_RT=$(echo "$PROBE_OUT" | grep -E '"cuda_runtime":' | head -1 | sed -E 's/.*"([0-9]+)".*/\1/' | head -c 4)
  echo "  cupy: $CUPY_VER (CUDA runtime $CUDA_RT)"
else
  echo "  cupy: NOT IMPORTED"
fi
echo "  cuML RF accepts sample_weight: $RF_OK"
echo "  cuML LR accepts sample_weight: $LR_OK"

if [[ "$CUML_IMPORTED" -ne 1 || "$CUPY_IMPORTED" -ne 1 ]]; then
  echo
  echo "FAIL: cuML or cupy did not import. enhanced_prediction.py will fall back to CPU sklearn."
  exit 4
fi

if [[ "$RF_OK" -eq 1 && "$LR_OK" -eq 1 ]]; then
  echo
  echo "SUCCESS: cuML is fully wired (RF and LR both accept sample_weight)."
  echo "enhanced_prediction.py will use the GPU path with class balancing."
  exit 0
fi

if [[ "$LR_OK" -eq 1 && "$RF_OK" -ne 1 ]]; then
  echo
  echo "PARTIAL: cuML LR accepts sample_weight but RF does not."
  echo "This is expected on RAPIDS 24.10-24.12 (LR was updated first; RF waits for 25.x+)."
  echo "enhanced_prediction.py will train RF on GPU without class balancing on that path."
  echo "Either upgrade to RAPIDS >= 25.x, or accept the partial balancing."
  exit 0
fi

echo
echo "FAIL: cuML neither accepts sample_weight on RF nor on LR."
echo "You are likely on RAPIDS < 24.10. Upgrade with:"
echo "  $VENV_PIP install --upgrade --extra-index-url $NVIDIA_INDEX cuml-cu12"
exit 4