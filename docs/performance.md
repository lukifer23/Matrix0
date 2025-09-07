# Performance Tuning (Apple Silicon)

This page captures practical knobs and defaults observed to work well on Apple Silicon (M1/M2/M3/M4) with the Matrix0 pipeline.

## Shared Inference Server (Self‑Play)

The orchestrator can launch a shared inference server used by self‑play workers. On Apple Silicon (MPS), the most impactful control is the per‑device target batch size:

- `MATRIX0_MPS_TARGET_BATCH` (default 4): preferred batch size used by the server when accumulating requests.
- Guidance:
  - Low concurrency or micro‑runs (≤ 10 games total): use 2–4 (avoid batch starvation).
  - Normal runs (3 workers, 200 sims): 4–6 is stable and fast.
  - High concurrency / long runs: 6–8 can improve throughput if heartbeat latency remains healthy.

Example:
```bash
MATRIX0_MPS_TARGET_BATCH=6 python -m azchess.orchestrator --workers 3 --games 300 --sims 200
```

If you see heartbeat ages creeping up or the table stalls, drop the target (e.g. 6 → 4). For micro‑runs you can also disable shared inference entirely:
```bash
python -m azchess.orchestrator --no-shared-infer ...
```

## Resignation / Aggressiveness

Resignation reduces long, lost games and improves data diversity. Recommended starting values:

- `--resign-threshold -0.6` (CLI) or `selfplay.resign_threshold: -0.6`
- In `config.yaml` under `selfplay`:
  - `min_resign_plies: 30`
  - `resign_consecutive_bad: 4`
  - `resign_value_margin: 0.03`
  - `resign_min_entropy: 0.25`
- Also shorten games slightly for faster cycles: `--max-game-length 160` and reduce opening random plies: `--opening-plies 4–6`.

## Draw Reduction

Configuration already includes `mcts.draw_penalty: -0.5`. If you need even fewer draws, you may try `-0.7` (with caution — can bias play).

## Tablebases

Syzygy tablebases are probed only at low piece counts (<= `tablebases.max_pieces`, default 7). They have modest overhead and improve endgame outcomes. If chasing raw speed for quick cycles, set `tablebases.enabled: false` or `max_pieces: 6`.

## Continuous Cycles

Simple shell loop for continuous runs that evaluate and promote, then start the next cycle:
```bash
MATRIX0_MPS_TARGET_BATCH=6 bash -lc 'while true; do \
  python -m azchess.orchestrator \
    --workers 3 --games 300 --sims 200 \
    --eval-games 40 --promotion-threshold 0.55 \
    --epochs 1 --steps-per-epoch 15000 \
    --opening-plies 6 --max-game-length 160 --resign-threshold -0.6; \
  sleep 10; done'
```

## WebUI Linkage

The WebUI tracks CLI runs via lightweight JSONL events written by the orchestrator to `logs/webui.jsonl`:
- `sp_start`, `sp_heartbeat`, `sp_game`, plus `training_start`, `eval_*`, `promotion`.
These events are intentionally tiny and have negligible overhead; keep them enabled.

## Troubleshooting

- ms/move too high at 200 sims in tiny runs: lower `MATRIX0_MPS_TARGET_BATCH` to 2–4 or run with `--no-shared-infer`.
- TUI table not updating during games: ensure you’re in a real TTY and table mode; the orchestrator now refreshes on heartbeats.
- Log spam “falling back to mixed” or missing datasets: loader now warns once per run. Add optional `.npz` datasets under `data/training/` at any time — they will be picked up automatically.

