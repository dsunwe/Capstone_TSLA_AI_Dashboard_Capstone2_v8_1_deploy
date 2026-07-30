#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Dashboard environment is missing. Running setup first..."
  ./setup_macos_linux.sh
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/capstone-v8-dashboard-matplotlib"
export TF_CPP_MIN_LOG_LEVEL=2
echo "Starting the Capstone II v8.1 Dashboard at http://localhost:8501 ..."
exec ".venv/bin/python" -m streamlit run app.py --server.port 8501
