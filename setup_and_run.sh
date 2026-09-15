#!/usr/bin/env bash
#
# OmniBrain — one-shot install + run
#
# Run this from the REPO ROOT (the "Project-1-OmniBrain" folder,
# the one that contains "Backend Development/" and "Frontend Development/"):
#
#     chmod +x setup_and_run.sh
#     ./setup_and_run.sh
#
# What it does:
#   1. Creates/activates a local virtualenv (.venv)
#   2. Finds and installs EVERY requirements*.txt in the repo, wherever it lives
#   3. Installs packages that are imported in the code but missing from
#      every requirements file (see list below — update it if you add new
#      imports that aren't captured in a requirements.txt)
#   4. Starts the FastAPI backend with uvicorn
#
# Flags:
#   ./setup_and_run.sh --with-frontend   also launches the Streamlit UI
#   ./setup_and_run.sh --no-install      skip installs, just start the server
#   ./setup_and_run.sh --port 9000       run backend on a custom port

set -euo pipefail

BACKEND_DIR="Backend Development"
FRONTEND_DIR="Frontend Development"
PORT=8000
WITH_FRONTEND=0
DO_INSTALL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-frontend) WITH_FRONTEND=1; shift ;;
    --no-install)    DO_INSTALL=0; shift ;;
    --port)          PORT="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ ! -d "$BACKEND_DIR" ]]; then
  echo "ERROR: '$BACKEND_DIR' not found. Run this script from the repo root"
  echo "       (the folder that directly contains '$BACKEND_DIR')."
  exit 1
fi

# ---------------------------------------------------------------------
# 1. Virtualenv
# ---------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d ".venv" ]]; then
  echo "==> Creating virtualenv (.venv)..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

echo "==> Using $(python --version) at $(which python)"

if [[ "$DO_INSTALL" -eq 1 ]]; then
  python -m pip install --upgrade pip

  # -------------------------------------------------------------------
  # 2. Install every requirements*.txt found anywhere in the repo
  #    (Database/, Ingestion/, Guardrails/, or any new one added later)
  # -------------------------------------------------------------------
  echo "==> Discovering requirements files..."
  REQ_FILES=$(find . -iname "requirement*.txt" -not -path "./.venv/*" -not -path "./.git/*")

  if [[ -z "$REQ_FILES" ]]; then
    echo "   (none found)"
  else
    while IFS= read -r req; do
      echo "   installing from: $req"
      python -m pip install -r "$req"
    done <<< "$REQ_FILES"
  fi

  # -------------------------------------------------------------------
  # 3. Packages that are imported in the code but not captured in any
  #    requirements*.txt above. Update this list if you add new
  #    top-level imports that aren't tracked in a requirements file.
  # -------------------------------------------------------------------
  echo "==> Installing packages used in code but missing from requirements files..."
  EXTRA_PACKAGES=(
    "fastapi>=0.110.0"
    "uvicorn[standard]>=0.27.0"
    "python-multipart>=0.0.9"   # required by FastAPI UploadFile
    "websockets>=12.0"          # required by the /chat/stream WS route
    "python-jose[cryptography]>=3.3.0"
    "passlib[bcrypt]>=1.7.4"
    "langgraph>=0.2.0"
    "streamlit>=1.35.0"
    "requests>=2.31.0"
  )
  python -m pip install "${EXTRA_PACKAGES[@]}"

  echo "==> Install complete."
fi

# ---------------------------------------------------------------------
# 4. Start the backend
# ---------------------------------------------------------------------
echo "==> Starting FastAPI backend on port $PORT..."
(
  cd "$BACKEND_DIR"
  exec python -m uvicorn main:app --reload --host 0.0.0.0 --port "$PORT"
) &
BACKEND_PID=$!

if [[ "$WITH_FRONTEND" -eq 1 ]]; then
  echo "==> Starting Streamlit frontend..."
  (
    cd "$FRONTEND_DIR"
    exec streamlit run app.py
  ) &
fi

wait "$BACKEND_PID"