#!/bin/bash
# Quick Matrix0 Orchestrator Run Script

source /Users/admin/Downloads/VSCode/Matrix0/.venv/bin/activate
cd /Users/admin/Downloads/VSCode/Matrix0

# Continuous run: cycles with increased self-play and deeper training
CONFIG=/Users/admin/Downloads/VSCode/Matrix0/config.yaml
WORKERS=3
SIMS=300
EVAL_GAMES=20
PROMOTE=0.55
DEVICE=auto

while true; do
  python -m azchess.orchestrator \
    --config "$CONFIG" \
    --workers "$WORKERS" \
    --sims "$SIMS" \
    --eval-games "$EVAL_GAMES" \
    --promotion-threshold "$PROMOTE" \
    --device "$DEVICE" \
    --tui table
  # brief pause between cycles to free resources
  sleep 10
done
