#!/bin/bash
# Quick Matrix0 Orchestrator Run Script

source /Users/admin/Downloads/VSCode/Matrix0/.venv/bin/activate
cd /Users/admin/Downloads/VSCode/Matrix0

# Full run: 3 workers × 70 games = 210 total, 300 sims, 20 eval games
python -m azchess.orchestrator \
  --config /Users/admin/Downloads/VSCode/Matrix0/config.yaml \
  --workers 3 \
  --games 100 \
  --sims 300 \
  --eval-games 20 \
  --promotion-threshold 0.52 \
  --device auto \
  --tui table
