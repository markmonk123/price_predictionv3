#!/usr/bin/env bash
# =============================================================================
# upgrade-cuml.sh -- upgrade cuML/cupy in the WSL .venv to a version that
# natively supports sample_weight on .fit() (RAPIDS >= 24.x).
#
# Run inside the WSL distro that holds the .venv. Default distro: Ubuntu-22.04.
# Target path assumed: /home/monki/projects/price_predictionv3/.venv
#
# Usage:
#   bash scripts/upgrade-cuml.sh                   # upgrade to requirements.txt pin (26.4.0)
#   bash scripts/upgrade-cuml.sh --to 24.4.0       # upgrade to a specific RAPIDS version
#   bash scripts/upgrade-cuml.sh --verify-only     # only run diagnostics, do not install
#   bash scripts/upgrade-cuml.sh --dry-run         # print pip commands without running them
#   bash scripts/upgrade-cuml.sh --help            # show this help
#
# Exit codes:
#   0  success (or upgrade not needed)
#   1  prerequisite failure (Python/cuML import / NVIDIA driver / etc.)
#   2  pip install failed
#   3  post-upgrade verification failed (signature probe still says sample_weight unsupported)
# =============================================================================

set -u
# We deliberately do NOT use 'set -e' -- we want to continue past non-fatal
# diagnostic checks and only exit on hard failures.

PROJECT_DIR="${PROJECT_DIR:-/home/monki/projects/price_predictionv3}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

# cuML wheel index lives on NVIDIA's PyPI -- the public index does not host it.
NVIDIA_INDEX="https://pypi.nvidia.com"

# Default target version -- must match requirements.txt's "cuml-cu12==<X>" pin.
TARGET_CUML="${TARGET_CUML:-26.4.0}"
TARGET_CUPY="${TARGET_CUPY:-13.0.0}"

DO_INSTALL=1
DO_VERIFY_ONLY=0
DRY_RUN=0

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to)        TARGET_CUML="$2"; shift 2 ;;
    --cupy-to)   TARGET_CUPY="$2"; shift 2 ;;
    --verify-only) DO_VERIFY_ONLY=1; DO_INSTALL=0; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --project-dir) PROJECT_DIR="$2"; VENV_DIR="$PROJECT_DIR/.venv"; PYTHON_BIN="$VENV_DIR/bin/python"; PIP_BIN="$VENV_DIR/bin/pip"; shift 2 ;;
    --venv)      VENV_DIR="$2"; PYTHON_BIN="$VENV_DIR/bin/python"; PIP_BIN="$VENV_DIR/bin/pip"; shift 2 ;;
    -h|--help)   usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

# --- helpers -----------------------------------------------------------------

section() {
  printf '\n=== %s ===\n' "$1"
}

run_or_echo() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  [dry-run] %s\n' "$*"
  else
    printf '  %s\n' "$*"
    "$@"
  fi
}

# --- 0. guard rails ----------------------------------------------------------

section "0. Prerequisites"

if ! command -v wsl.exe >/dev/null 2>&1 && [[ "${WSL_DISTRO:-}" == "" ]]; then
  # We are most likely already inside WSL; if wsl.exe is on PATH and WSL_DISTRO
  # is unset, this script is being run from Windows and should error out.
  echo "ERROR: wsl.exe found on PATH. This script must be run inside the WSL"
  echo "       distro that holds the .venv. Re-run from inside WSL:"
  echo "         wsl -d Ubuntu-22.04 -- bash scripts/upgrade-cuml.sh"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found at $PYTHON_BIN"
  echo "       Set PROJECT_DIR and VENV_DIR or pass --project-dir / --venv."
  exit 1
fi

PY_VER=$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "  Python: $PY_VER ($PYTHON_BIN)"

# RAPIDS 26.x supports 3.10 -- 3.12. Newer Python versions have no cuML wheels.
PY_MAJOR=${PY_VER%%.*}
PY_MINOR=${PY_VER##*.}
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -ge 13 ]]; then
  echo "ERROR: cuML wheels are not published for Python $PY_VER."
  echo "       Recreate the .venv on Python 3.12 first, e.g.:"
  echo "         rm -rf $VENV_DIR"
  echo "         python3.12 -m venv $VENV_DIR"
  echo "         $PIP_BIN install --upgrade pip"
  exit 1
fi

# --- 1. NVIDIA / CUDA sanity check -------------------------------------------

section "1. NVIDIA driver + CUDA runtime"

if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  echo "  nvidia-smi driver: ${NVIDIA_DRIVER:-<not available>}"
  DRIVER_OK=0
  if [[ "${NVIDIA_DRIVER:-}" =~ ^([0-9]+)\. ]]; then
    DRIVER_MAJOR=${BASH_REMATCH[1]}
    if [[ "$DRIVER_MAJOR" -ge 525 ]]; then
      DRIVER_OK=1
    fi
  fi
  if [[ "$DRIVER_OK" -eq 0 ]]; then
    echo "WARN: NVIDIA driver >= 525 required for CUDA 12.0+. Driver found: ${NVIDIA_DRIVER:-unknown}."
    echo "      Continue anyway? (y/N)"
    read -r ANSWER
    if [[ "$ANSWER" != "y" && "$ANSWER" != "Y" ]]; then
      exit 1
    fi
  fi
else
  echo "WARN: nvidia-smi not on PATH. Continuing without GPU probe."
fi

# --- 2. cuML / cupy current state --------------------------------------------

section "2. cuML / cupy versions currently installed"

CURRENT_CUML="(not installed)"
CURRENT_CUPY="(not installed)"
if "$PYTHON_BIN" -c 'import cuml' >/dev/null 2>&1; then
  CURRENT_CUML=$($PYTHON_BIN -c 'import cuml; print(cuml.__version__)' 2>/dev/null || echo '(import error)')
fi
if "$PYTHON_BIN" -c 'import cupy' >/dev/null 2>&1; then
  CURRENT_CUPY=$($PYTHON_BIN -c 'import cupy; print(cupy.__version__)' 2>/dev/null || echo '(import error)')
fi
echo "  cuml: $CURRENT_CUML"
echo "  cupy: $CURRENT_CUPY"

# --- 3. signature probe ------------------------------------------------------

section "3. sample_weight probe (pre-upgrade)"

probe_signature() {
  "$PYTHON_BIN" - <<'EOF' 2>&1
import inspect
result = {"cuml_imported": False, "rf_sample_weight": None, "lr_sample_weight": None}
try:
    from cuml.ensemble import RandomForestClassifier as _cuRF
    from cuml.linear_model import LogisticRegression as _cuLR
    result["cuml_imported"] = True
    try:
        rf = _cuRF(n_estimators=10, max_depth=4)
        result["rf_sample_weight"] = "sample_weight" in inspect.signature(rf.fit).parameters
    except Exception as exc:
        result["rf_sample_weight"] = f"probe-error: {exc}"
    try:
        lr = _cuLR()
        result["lr_sample_weight"] = "sample_weight" in inspect.signature(lr.fit).parameters
    except Exception as exc:
        result["lr_sample_weight"] = f"probe-error: {exc}"
except Exception as exc:
    result["error"] = str(exc)
import json
print(json.dumps(result, indent=2))
EOF
}

PRE_PROBE=$(probe_signature)
echo "$PRE_PROBE"

# --- 4. upgrade --------------------------------------------------------------

if [[ "$DO_VERIFY_ONLY" -eq 1 ]]; then
  echo
  echo "--verify-only set; skipping pip install."
  exit 0
fi

section "4. Upgrade cuML -> $TARGET_CUML, cupy -> >= $TARGET_CUPY"

# Use a function so we capture both stdout/stderr to a log AND the exit code.
# 'run_or_echo' already short-circuits on DRY_RUN; this wraps it.
PIP_LOG=/tmp/upgrade-cuml-pip.log
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

if run_or_echo_capture "$PIP_BIN" install --upgrade \
    --extra-index-url "$NVIDIA_INDEX" \
    "cuml-cu12==$TARGET_CUML" \
    "cupy-cuda12x>=$TARGET_CUPY"; then
  PIP_RC=0
else
  PIP_RC=$?
fi

if [[ "$PIP_RC" -ne 0 ]]; then
  echo
  echo "ERROR: pip install failed (rc=$PIP_RC). Tail of pip log:"
  tail -n 40 /tmp/upgrade-cuml-pip.log 2>/dev/null || true
  echo
  echo "Common causes:"
  echo "  - Python >= 3.13 (no cuML wheel). Recreate .venv on Python 3.12."
  echo "  - Network/proxy blocking pypi.nvidia.com."
  echo "  - Mismatched CUDA toolkit vs cuML RAPIDS version."
  exit 2
fi

# --- 5. post-upgrade probe + sanity check ------------------------------------

section "5. Post-upgrade probe"

POST_PROBE=$(probe_signature)
echo "$POST_PROBE"

NEW_CUML=$($PYTHON_BIN -c 'import cuml; print(cuml.__version__)' 2>/dev/null || echo '(import error)')
NEW_CUPY=$($PYTHON_BIN -c 'import cupy; print(cupy.__version__)' 2>/dev/null || echo '(import error)')
echo
echo "  cuml: $CURRENT_CUML -> $NEW_CUML"
echo "  cupy: $CURRENT_CUPY -> $NEW_CUPY"

RF_OK=$(echo "$POST_PROBE" | grep -E '"rf_sample_weight":\s*true' | wc -l)
LR_OK=$(echo "$POST_PROBE" | grep -E '"lr_sample_weight":\s*true' | wc -l)

if [[ "$RF_OK" -eq 1 && "$LR_OK" -eq 1 ]]; then
  echo
  echo "SUCCESS: cuML .fit() now accepts sample_weight on RF and LR."
  echo "The helper _fit_with_balanced_sample_weight() in enhanced_prediction.py"
  echo "will route balanced class weights to the GPU automatically -- no warning."
  exit 0
fi

echo
echo "ERROR: sample_weight still not in cuML .fit() signature after upgrade."
echo "       rf_sample_weight ok: $RF_OK  lr_sample_weight ok: $LR_OK"
echo "       This usually means the .venv is still pinned to an older cuML"
echo "       via some other constraint (conda env marker, platform pin, etc.)."
echo "       Try recreating the .venv on Python 3.12 and rerunning."
exit 3
