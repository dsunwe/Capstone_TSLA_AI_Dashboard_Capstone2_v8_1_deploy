#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${PYTHON_CMD:-}" ]]; then
  PYTHON_BIN="$PYTHON_CMD"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  echo "Python 3.11 or 3.12 is required."
  echo "Install Python 3.12, or rerun with PYTHON_CMD=/path/to/python3.12."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating Dashboard environment with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv .venv
fi

echo "Installing complete Dashboard and Sandbox dependencies..."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements-dashboard.txt

export TF_CPP_MIN_LOG_LEVEL=2
echo "Verifying Streamlit, LR/RF, XGBoost, LSTM, and Sandbox runtime..."
".venv/bin/python" -c \
  "import streamlit, pandas, numpy, matplotlib, seaborn, sklearn, xgboost, tensorflow, joblib, exchange_calendars, pyarrow; print('Dashboard runtime verified.')"

chmod +x run_dashboard.sh
echo "Setup complete. Run ./run_dashboard.sh"
