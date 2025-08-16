# Matrix0: AlphaZero-Style Chess on Apple Silicon

Matrix0 is an efficient, small AlphaZero-style chess engine designed to train and run on consumer Apple Silicon (M3 Pro MacBook Pro). The project uses self-play, reinforcement learning, and Monte Carlo Tree Search (MCTS) with a compact CNN-ResNet backbone. Training runs on Metal (MPS) GPU with an option to export to Core ML for low-latency inference.

## Why this approach
- Small, efficient network (target 15–25M params) fits Apple GPU memory and trains in reasonable time.
- CNN residual tower is strong for 8x8 boards and simpler than attention for this scale.
- PyTorch on MPS for training; optional Core ML export for GUI/inference and potential ANE use.

## Stack
- Language: Python 3.11+
- Core libs: `torch` (MPS), `python-chess`, `numpy`, `PyYAML`, `tqdm`, `rich`, `tensorboard`
- Optional export: `coremltools` (macOS only)
- IDE: Cursor (VS Code)

## Hardware guidance (M3 Pro)
- Use MPS device for training: automatically selected when available.
- Mixed precision: autocast to `float16` on MPS to reduce memory and increase throughput.
- Batch size: start small (e.g., 256 positions/batch) and scale to memory.
- Data: on-disk replay buffer to avoid RAM pressure; stream in small shards.

## Project layout
```
azchess/
  __init__.py
  config.py
  logging_utils.py
  monitor.py
  encoding.py
  mcts.py
  selfplay.py
  train.py
  eval.py
  cli_play.py
  orchestrator.py
  model/
    __init__.py
    resnet.py
  data_manager.py
config.yaml
requirements.txt
```

## Pipeline overview
1) Self-Play: Multiple workers run MCTS guided by the network to generate games and training examples `(s, π, z)`.
2) Training: Supervised on self-play buffer using policy cross-entropy and value MSE + L2.
3) Evaluation: Periodically pit new checkpoints vs. previous best; promote on Elo margin.
4) Inference/GUI: Lightweight CLI to play; GUI or UCI bridge later. Optional Core ML export for ANE.

## Model choice (v1)
- Input: 19 planes (v1 minimal: pieces, side-to-move, castling, rule counters). Configurable.
- Backbone: Residual CNN tower, width 160, depth 14 (≈18–22M params depending on heads).
- Heads: Policy (4672-action space) and Value (scalar tanh).
- Precision: FP16 on MPS for training; BF16 if supported; export to Core ML in FP16.

## Move/action space
We use a 4672-sized action space following AlphaZero indexing (64 squares × 73 moves each): 56 ray moves, 8 knights, and 9 underpromotions (N/B/R × 3 forward directions). The canonical implementation lives in `azchess/encoding.py`. `azchess/move_encoding.py` is a thin compatibility shim; set env `MATRIX0_STRICT_ENCODING=1` or `strict_encoding: true` in `config.yaml` to emit deprecation warnings for legacy imports.

## Milestones
- M0: End-to-end skeleton (self-play, train loop, eval, play CLI). [This commit]
- M1: Functional self-play pipeline, stable training, TB logging.
- M2: Strength climb, evaluation matches, checkpoint promotion.
- M3: GUI/bridge (simple web board or UCI); Core ML export path.
- M4: Efficiency passes (channels/depth tuning), quantization-aware inference.

## Data & Logging
- Self-play NPZ files land in `data/selfplay/`; orchestrator compacts them into replay shards in `data/replays/` with rotation and limits using `DataManager`.
- Logs: human-readable logs with rotation in `logs/matrix0.log`, structured JSON lines in `logs/structured.jsonl`.
- Monitoring: Orchestrator reports directory sizes, disk free, and process memory usage.

## Quickstart
- Create a virtual env and install requirements: `pip install -r requirements.txt`
- Run a small self-play shard: `python -m azchess.selfplay --games 8`
- Train on buffer: `python -m azchess.train --config config.yaml`
- Play a game in terminal: `python -m azchess.cli_play`
 - Orchestrate a full cycle: `python -m azchess.orchestrator --config config.yaml`
 - Save an untrained baseline checkpoint: `python -m azchess.save_init`

## Setup
- Create venv: `make venv` then `source .venv/bin/activate`
- Install deps: `make install`
- Run: `make orchestrate`

## Training notes
- Asynchronous: self-play and training run concurrently on different processes; synchronize via disk buffer.
- Checkpointing: save every N steps; best model determined by evaluation.
- Tuning: adjust `channels`, `blocks`, `dirichlet_alpha`, `cpuct`, `num_simulations`, and `temperature` schedule.

## Export to Core ML (preview)
See `azchess/model/resnet.py` for `to_coreml()` helper to export a checkpoint. Inference engines can load `*.mlpackage` for GUI.

## License
TBD by project owner (default: private). No third-party model weights included.
