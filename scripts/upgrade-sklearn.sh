#!/usr/bin/env bash
# =============================================================================
# upgrade-sklearn.sh -- upgrade scikit-learn inside the WSL .venv to the
# newest compatible release on Python 3.12 (currently 1.9.x).
#
# cuML 26.4.0 declares no Python deps, so sklearn can move freely. The current
# requirements.txt pin is `>=1.4,<1.10`, which already permits the latest 1.9.x.
#
# Usage:
#   bash scripts/upgrade-sklearn.sh                  # upgrade to latest 1.9.x
#   bash scripts/upgrade-sklearn.sh --to 1.9.1       # upgrade to a specific version
#   bash scripts/upgrade-sklearn.sh --verify-only    # only report current version
#   bash scripts/upgrade-sklearn.sh --dry-run        # print pip command without running
#   bash scripts/upgrade-sklearn.sh --help           # show help
#
# Exit codes:
#   0  success
#   1  prerequisite check failed (Python < 3.10, .venv missing, etc.)
#   2  pip install failed
#   3  import or feature verification failed
# =============================================================================

set -u

PROJECT_DIR="${PROJECT_DIR:-/home/monki/projects/price_predictionv3}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

# Default target -- latest 1.9.x. Override with --to.
TARGET="${TARGET:-1.9.1}"
DO_VERIFY_ONLY=0
DRY_RUN=0

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to)          TARGET="$2"; shift 2 ;;
    --verify-only) DO_VERIFY_ONLY=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --project-dir) PROJECT_DIR="$2"; VENV_DIR="$PROJECT_DIR/.venv"; PYTHON_BIN="$VENV_DIR/bin/python"; PIP_BIN="$VENV_DIR/bin/pip"; shift 2 ;;
    --venv)        VENV_DIR="$2"; PYTHON_BIN="$VENV_DIR/bin/python"; PIP_BIN="$VENV_DIR/bin/pip"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

section() { printf '\n=== %s ===\n' "$1"; }
say()     { printf '  %s\n' "$*"; }
warn()    { printf '  WARN: %s\n' "$*" >&2; }
die()     { printf 'ERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

run_or_echo_capture() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  [dry-run] %s\n' "$*"
    return 0
  else
    printf '  %s\n' "$*"
    "$@" &>"$LOG_FILE"
    return $?
  fi
}

# --- 0. prerequisites --------------------------------------------------------

section "0. Prerequisites"

if [[ "$(uname -s)" != "Linux" ]]; then
  die "Must be run inside WSL/Linux (got $(uname -s))"
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  die "Project directory not found: $PROJECT_DIR"
fi
say "Project dir: $PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  die "Python not found at $PYTHON_BIN -- run scripts/setup-cuml-wsl.sh first"
fi

PY_VER=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
say "Python: $PY_VER"
PY_MAJOR=${PY_VER%%.*}
PY_MINOR=${PY_VER##*.}
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -lt 10 ]]; then
  die "Python $PY_VER is below 3.10 (sklearn 1.4+ requires 3.10+)"
fi

# --- 1. current state --------------------------------------------------------

section "1. Current sklearn version"

CURRENT=$("$PYTHON_BIN" -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || echo "(not installed)")
say "Currently installed: $CURRENT"
say "Target: $TARGET"

# --- 2. upgrade --------------------------------------------------------------

LOG_FILE=/tmp/upgrade-sklearn-pip.log

if [[ "$DO_VERIFY_ONLY" -eq 1 ]]; then
  say "--verify-only set; skipping pip install"
else
  section "2. Upgrade scikit-learn -> $TARGET"

  if run_or_echo_capture "$PIP_BIN" install --upgrade \
      "scikit-learn==$TARGET"; then
    :
  else
    PIP_RC=$?
    warn "sklearn upgrade failed (rc=$PIP_RC). Tail of log:"
    tail -n 40 "$LOG_FILE" 2>/dev/null || true
    echo
    echo "Common causes:"
    echo "  - Network blocked for pypi.org (proxy / firewall)"
    echo "  - cuML wheel conflicts (unlikely; cuML declares no Python deps)"
    echo "  - Pre-release wheel requested that doesn't exist (use --to with a stable version)"
    exit 2
  fi
fi

# --- 3. verify ----------------------------------------------------------------

section "3. Verification"

if [[ "$DRY_RUN" -eq 1 ]]; then
  say "[dry-run] skipping verification"
  exit 0
fi

NEW=$("$PYTHON_BIN" -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || echo "(still not installed)")
say "Now installed: $NEW"

# Verify import + VotingClassifier.sample_weight support
VERIFY=$("$PYTHON_BIN" - <<'EOF' 2>&1
import json
result = {"sklearn": None, "voting_sw": False, "voting_cv_sw": False, "errors": []}
try:
    import sklearn
    result["sklearn"] = sklearn.__version__
except Exception as e:
    result["errors"].append(f"import: {e}")
try:
    import inspect
    from sklearn.ensemble import VotingClassifier
    vc = VotingClassifier(estimators=[])
    params = inspect.signature(vc.fit).parameters
    result["voting_sw"] = "sample_weight" in params
except Exception as e:
    result["errors"].append(f"VotingClassifier.fit probe: {e}")
try:
    import inspect
    from sklearn.model_selection import cross_val_score
    params = inspect.signature(cross_val_score).parameters
    result["cross_val_score_has_sw"] = "sample_weight" in params
except Exception as e:
    result["errors"].append(f"cross_val_score probe: {e}")
print(json.dumps(result, indent=2))
EOF
)

echo "$VERIFY"

VOTING_OK=$(echo "$VERIFY" | grep -E '"voting_sw":\s*true' | wc -l)
SK_OK=$(echo "$VERIFY" | grep -E '"sklearn":' | wc -l)

echo
echo "=== Summary ==="
if [[ "$SK_OK" -ge 1 && "$VOTING_OK" -eq 1 ]]; then
  echo "SUCCESS: scikit-learn $NEW installed; VotingClassifier.fit() accepts sample_weight."
  echo "enhanced_prediction.py will route balanced weights through Calibration -> Voting -> sub-estimators."
  echo "The 'VotingClassifier does not appear to accept sample_weight' warning is now gone."
  exit 0
fi

echo "FAIL: Verification did not pass. Check the JSON output above."
exit 3