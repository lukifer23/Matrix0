# Matrix0: AlphaZero-Style Chess on Apple Silicon

Matrix0 is an efficient, small AlphaZero-style chess engine designed to train and run on consumer Apple Silicon (M3 Pro MacBook Pro). The project uses self-play, reinforcement learning, and Monte Carlo Tree Search (MCTS) with a compact CNN-ResNet backbone. Training runs on Metal (MPS) GPU with an option to export to Core ML for low-latency inference.

## Why this approach
- Small, efficient network (target 15–25M params) fits Apple GPU memory and trains in reasonable time.
- CNN residual tower is strong for 8x8 boards and simpler than attention for this scale.
- PyTorch on MPS for training; optional Core ML export for GUI/inference and potential ANE use.
- **NEW**: External engine integration for competitive training and evaluation.

## Stack
- Language: Python 3.11+
- Core libs: `torch` (MPS), `python-chess`, `numpy`, `PyYAML`, `tqdm`, `rich`, `tensorboard`
- Optional export: `coremltools` (macOS only)
- **NEW**: `psutil` for process management, external engine support
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
  arena.py
  cli_play.py
  orchestrator.py
  data_manager.py
  validate_moves.py
  save_init.py
  model/
    __init__.py
    resnet.py
  engines/                    # NEW: External engine integration
    __init__.py
    uci_bridge.py
    engine_manager.py
  selfplay/                  # NEW: Enhanced self-play system
    __init__.py
    internal.py
    external_engine_worker.py
  eval/                      # NEW: Multi-engine evaluation
    __init__.py
    multi_engine_evaluator.py
config.yaml
requirements.txt
roadmap.md
EXTERNAL_ENGINES.md          # NEW: External engine documentation
```

## Pipeline overview
1) **Self-Play**: Multiple workers run MCTS guided by the network to generate games and training examples `(s, π, z)`.
   - **NEW**: Support for external engines (Stockfish, LC0) as training partners
   - **NEW**: Mixed training data from internal self-play and external engine games
2) **Training**: Supervised on self-play buffer using policy cross-entropy and value MSE + L2.
   - **NEW**: Enhanced data management with SQLite metadata tracking
   - **NEW**: Improved checkpoint resumption and state persistence
3) **Evaluation**: Periodically pit new checkpoints vs. previous best; promote on Elo margin.
   - **NEW**: Multi-engine evaluation against external engines
   - **NEW**: Comprehensive strength benchmarking
4) **Inference/GUI**: Lightweight CLI to play; GUI or UCI bridge later. Optional Core ML export for ANE.

## Model choice (v1)
- Input: 19 planes (v1 minimal: pieces, side-to-move, castling, rule counters). Configurable.
- Backbone: Residual CNN tower, width 160, depth 14 (≈18–22M params depending on heads).
- Heads: Policy (4672-action space) and Value (scalar tanh).
- Precision: FP16 on MPS for training; BF16 if supported; export to Core ML in FP16.
- **NEW**: Optional Squeeze-and-Excitation (SE) blocks for enhanced performance.

## Move/action space
We use a 4672-sized action space following AlphaZero indexing (64 squares × 73 moves each): 56 ray moves, 8 knights, and 9 underpromotions (N/B/R × 3 forward directions). The canonical implementation lives in `azchess/encoding.py`. `azchess/move_encoding.py` is a thin compatibility shim; set env `MATRIX0_STRICT_ENCODING=1` or `strict_encoding: true` in `config.yaml` to emit deprecation warnings for legacy imports.

## **NEW: External Engine Integration**
Matrix0 now supports integration with external chess engines:
- **Stockfish**: World-class traditional chess engine
- **Leela Chess Zero (LC0)**: Neural network-based engine
- **Training Partners**: Generate games against external engines for diverse training data
- **Competitive Evaluation**: Measure strength against established engines
- **UCI Protocol**: Standard chess engine communication

See `EXTERNAL_ENGINES.md` for detailed usage instructions.

## Milestones
- **M0**: ✅ End-to-end skeleton (self-play, train loop, eval, play CLI)
- **M1**: ✅ Functional self-play pipeline, stable training, TB logging
- **M2**: ✅ Strength climb, evaluation matches, checkpoint promotion
- **M3**: 🚧 GUI/bridge (simple web board or UCI); Core ML export path
- **M4**: 🚧 Efficiency passes (channels/depth tuning), quantization-aware inference
- **NEW**: ✅ External engine integration for competitive training

## Data & Logging
- Self-play NPZ files land in `data/selfplay/`; orchestrator compacts them into replay shards in `data/replays/` with rotation and limits using `DataManager`.
- **NEW**: SQLite database for metadata tracking and data integrity
- **NEW**: External engine games stored in JSON format with comprehensive metadata
- Logs: human-readable logs with rotation in `logs/matrix0.log`, structured JSON lines in `logs/structured.jsonl`.
- Monitoring: Orchestrator reports directory sizes, disk free, and process memory usage.

## Quickstart
- Create a virtual env and install requirements: `pip install -r requirements.txt`
- Run a small self-play shard: `python -m azchess.selfplay --games 8`
- **NEW**: Use external engines: `python -m azchess.selfplay --external-engines --games 8`
- Train on buffer: `python -m azchess.train --config config.yaml`
- Play a game in terminal: `python -m azchess.cli_play`
- Evaluate two checkpoints (balanced colors, optional PGN): `python -m azchess.arena --ckpt_a checkpoints/best.pt --ckpt_b checkpoints/model.pt --games 20 --pgn-out out/pgns --pgn-sample 5`
- **NEW**: Evaluate against external engines: `python -m azchess.eval --external-engines --games 50`
- Orchestrate a full cycle: `python -m azchess.orchestrator --config config.yaml`
- **NEW**: With external engines: `python -m azchess.orchestrator --external-engines`
- Save an untrained baseline checkpoint: `python -m azchess.save_init`

## Setup
- Create venv: `make venv` then `source .venv/bin/activate`
- Install deps: `make install`
- Run: `make orchestrate`
- **NEW**: Install external engines (Stockfish, LC0) for full functionality

## Training notes
- Asynchronous: self-play and training run concurrently on different processes; synchronize via disk buffer.
- **NEW**: Mixed training data from internal self-play and external engine games
- **NEW**: Configurable external engine ratio for training diversity
- Checkpointing: save every N steps; best model determined by evaluation.
- **NEW**: Enhanced checkpoint resumption with full state persistence
- Tuning: adjust `channels`, `blocks`, `dirichlet_alpha`, `cpuct`, `num_simulations`, and `temperature` schedule.
- **NEW**: Advanced resignation logic with consecutive bad move detection

## Export to Core ML (preview)
See `azchess/model/resnet.py` for `to_coreml()` helper to export a checkpoint. Inference engines can load `*.mlpackage` for GUI.

## **NEW: Advanced Features**
- **Opening Diversity**: Random opening moves for training variety
- **Enhanced Resignation**: Smart resignation based on consecutive bad evaluations
- **Selection Jitter**: Configurable exploration in MCTS
- **Data Integrity**: Comprehensive corruption detection and recovery
- **Process Management**: Robust external engine lifecycle management

## License
TBD by project owner (default: private). No third-party model weights included.
