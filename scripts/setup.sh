#!/usr/bin/env bash
set -euo pipefail

VENV=${1:-.venv}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Activate with: source $VENV/bin/activate"
