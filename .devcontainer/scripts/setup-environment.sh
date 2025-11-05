#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_FOLDER:-/workspace/price_predictionv3}"
VENV_DIR="${WORKSPACE_DIR}/.venv"
REQUIREMENTS_FILE="${WORKSPACE_DIR}/python/requirements.txt"

python_bin="python3"
if command -v python &>/dev/null; then
  python_bin="python"
fi

# Create the virtual environment if it does not already exist
if [[ ! -d "${VENV_DIR}" ]]; then
  "${python_bin}" -m venv "${VENV_DIR}"
fi

# Activate the virtual environment and install dependencies
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"
fi

# Ensure quickfix is installed inside the virtual environment as well
pip install --no-cache-dir quickfix

deactivate

# Pre-warm the prediction script so the environment is ready for use
source "${VENV_DIR}/bin/activate"
cd "${WORKSPACE_DIR}"
python python/run_prediction.py --price 50000 --volume 100 --time "$(date --iso-8601=seconds)" >/tmp/run_prediction_warmup.json || true
rm -f /tmp/run_prediction_warmup.json

deactivate
