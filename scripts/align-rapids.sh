#!/usr/bin/env bash
# =============================================================================
# align-rapids.sh -- bring the WSL .venv back to a coherent RAPIDS release
# (default 26.4.0).
#
# pip's resolver can leave the .venv with a mix of RAPIDS minor versions
# (e.g. cuml-cu12==26.4.0 but ucxx-cu12==0.46.0 pulling in rmm-cu12==25.10.*).
# This script forces every RAPIDS package onto the same minor release so the
# dependency resolver stays consistent.
#
# Usage:
#   bash scripts/align-rapids.sh                 # align to 26.4.0 (the default)
#   bash scripts/align-rapids.sh --to 26.4.0     # same
#   bash scripts/align-rapids.sh --verify-only   # report current state only
#   bash scripts/align-rapids.sh --dry-run       # print pip commands, don't run
#   bash scripts/align-rapids.sh --help
#
# Exit codes:
#   0  success
#   1  prerequisites failed
#   2  pip install failed
#   3  post-alignment verification failed (versions still mismatched)
# =============================================================================

set -u

PROJECT_DIR="${PROJECT_DIR:-/home/monki/projects/price_predictionv3}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
NVIDIA_INDEX="https://pypi.nvidia.com"

TARGET="${TARGET:-26.4.0}"
UCXX_TARGET="${UCXX_TARGET:-0.45.0}"
NUMBA_CUDA_TARGET="${NUMBA_CUDA_TARGET:-0.19.1}"
CUDA_CORE_TARGET="${CUDA_CORE_TARGET:-0.5.*}"

DO_VERIFY_ONLY=0
DRY_RUN=0

usage() { sed -n '2,30p' "$0"; exit 0; }
section() { printf '\n=== %s ===\n' "$1"; }
say() { printf '  %s\n' "$*"; }
warn() { printf '  WARN: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to)            TARGET="$2"; shift 2 ;;
    --ucxx-to)       UCXX_TARGET="$2"; shift 2 ;;
    --numba-cuda-to) NUMBA_CUDA_TARGET="$2"; shift 2 ;;
    --cuda-core-to)  CUDA_CORE_TARGET="$2"; shift 2 ;;
    --verify-only)   DO_VERIFY_ONLY=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --project-dir)   PROJECT_DIR="$2"; VENV_DIR="$PROJECT_DIR/.venv"; PYTHON_BIN="$VENV_DIR/bin/python"; PIP_BIN="$VENV_DIR/bin/pip"; shift 2 ;;
    --venv)          VENV_DIR="$2"; PYTHON_BIN="$VENV_DIR/bin/python"; PIP_BIN="$VENV_DIR/bin/pip"; shift 2 ;;
    -h|--help)       usage ;;
    *) die "Unknown arg: $1" ;;
  esac
done

# --- 0. prerequisites --------------------------------------------------------

section "0. Prerequisites"

if [[ "$(uname -s)" != "Linux" ]]; then
  die "Must be run inside WSL/Linux"
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  die "Project dir not found: $PROJECT_DIR"
fi
say "Project dir: $PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  die "Python not found at $PYTHON_BIN"
fi

PY_VER=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_MAJOR=${PY_VER%%.*}
PY_MINOR=${PY_VER##*.}
say "Python: $PY_VER"
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -ge 13 ]]; then
  die "Python $PY_VER is out of range. RAPIDS 26.4 requires Python 3.10-3.12."
fi

# --- 1. probe current state ---------------------------------------------------

section "1. Current RAPIDS package versions"

PROBE_LOG=/tmp/align-rapids-probe.log
PROBE=$("$PYTHON_BIN" -m pip list 2>/dev/null | grep -E "^(cuml-cu12|cupy-cuda12x|rmm-cu12|pylibraft-cu12|libraft-cu12|librmm-cu12|libcuvs-cu12|libnvforest-cu12|nvforest-cu12|ucxx-cu12|distributed-ucxx-cu12|libucxx-cu12|cuda-core|numba-cuda|cudf-cu12|dask-cuda|dask-cudf-cu12|raft-dask-cu12|nvidia-cuda-runtime-cu12)" || echo "(none found)")
echo "$PROBE" | sed 's/^/    /'

# --- 2. alignment -------------------------------------------------------------

LOG_FILE=/tmp/align-rapids-pip.log

if [[ "$DO_VERIFY_ONLY" -eq 1 ]]; then
  say "--verify-only set; skipping pip"
else
  section "2. Align RAPIDS to release $TARGET (ucxx=$UCXX_TARGET, numba-cuda=$NUMBA_CUDA_TARGET)"

  # The full list of RAPIDS-related wheels we want aligned.
  # We do this in a single `pip install` so the resolver sees all constraints at once.
  declare -a RAPIDS_PINS=(
    "cuml-cu12==$TARGET"
    "cupy-cuda12x>=13"
    "rmm-cu12==$TARGET"
    "pylibraft-cu12==$TARGET"
    "libraft-cu12==$TARGET"
    "librmm-cu12==$TARGET"
    "libcuvs-cu12==$TARGET"
    "libnvforest-cu12==$TARGET"
    "nvforest-cu12==$TARGET"
    "ucxx-cu12==$UCXX_TARGET"
    "distributed-ucxx-cu12==$UCXX_TARGET"
    "libucxx-cu12==$UCXX_TARGET"
    "cuda-core==$CUDA_CORE_TARGET"
    "numba-cuda==$NUMBA_CUDA_TARGET"
    "cudf-cu12==$TARGET"
    "dask-cuda==$TARGET"
    "dask-cudf-cu12==$TARGET"
    "raft-dask-cu12==$TARGET"
    "nvidia-cuda-runtime-cu12>=12"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    say "[dry-run] would run:"
    printf '    %s install --upgrade \\\n' "$PIP_BIN"
    printf '        --extra-index-url %s \\\n' "$NVIDIA_INDEX"
    for pin in "${RAPIDS_PINS[@]}"; do
      printf '        "%s" \\\n' "$pin"
    done
    printf '\n'
  else
    printf '  %s install --upgrade \\\n' "$PIP_BIN"
    printf '      --extra-index-url %s \\\n' "$NVIDIA_INDEX"
    for pin in "${RAPIDS_PINS[@]}"; do
      printf '      "%s" \\\n' "$pin"
    done
    if "$PIP_BIN" install --upgrade --extra-index-url "$NVIDIA_INDEX" "${RAPIDS_PINS[@]}" &>"$LOG_FILE"; then
      :
    else
      PIP_RC=$?
      warn "Alignment install failed (rc=$PIP_RC). Tail of pip log:"
      tail -n 40 "$LOG_FILE" 2>/dev/null || true
      echo
      echo "If pip still complains about version conflicts:"
      echo "  1. Run --recreate-venv on setup-cuml-wsl.sh to start over."
      echo "  2. Or run 'pip install --upgrade --force-reinstall <conflicting-pkg>' for each error."
      exit 2
    fi
  fi
fi

# --- 3. verify ----------------------------------------------------------------

section "3. Post-alignment verification"

if [[ "$DRY_RUN" -eq 1 ]]; then
  say "[dry-run] skipping verification"
  exit 0
fi

VERIFY=$("$PYTHON_BIN" - <<'EOF' 2>&1
import json, subprocess
target_pkgs = [
    "cuml-cu12", "rmm-cu12", "pylibraft-cu12", "libraft-cu12", "librmm-cu12",
    "libcuvs-cu12", "libnvforest-cu12", "nvforest-cu12", "ucxx-cu12",
    "distributed-ucxx-cu12", "libucxx-cu12", "cuda-core", "numba-cuda",
    "cudf-cu12", "dask-cuda", "dask-cudf-cu12", "raft-dask-cu12",
]
versions = {}
out = subprocess.run(["pip", "list"], capture_output=True, text=True).stdout
for line in out.splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[0] in target_pkgs:
        versions[parts[0]] = parts[1]
# Group by minor version prefix (e.g. "26.4")
def major_minor(v):
    base = v.split("+")[0].split(".")[0:2]
    return ".".join(base)
groups = {}
for pkg, ver in versions.items():
    key = major_minor(ver)
    groups.setdefault(key, []).append(f"{pkg}=={ver}")
result = {
    "versions": versions,
    "groups": groups,
    "aligned": len(groups) <= 1,
    "group_count": len(groups),
}
print(json.dumps(result, indent=2))
EOF
)

echo "$VERIFY"

ALIGNED=$(echo "$VERIFY" | grep -E '"aligned":\s*true' | wc -l)
GROUP_COUNT=$(echo "$VERIFY" | grep -E '"group_count":\s*[0-9]+' | grep -oE '[0-9]+' | head -1)

echo
if [[ "$ALIGNED" -eq 1 ]]; then
  echo "SUCCESS: All RAPIDS packages on a single release. No more dependency conflicts."
  echo "Run 'pip check' inside the .venv to confirm."
  exit 0
fi

echo "FAIL: $GROUP_COUNT different RAPIDS release(s) detected:"
echo "$VERIFY" | grep -E '"[^"]+==[^"]+"' | sed 's/^/    /'
exit 3