# Matrix0: AlphaZero-Style Chess Engine for Apple Silicon

Matrix0 is an efficient, AlphaZero-style chess engine designed specifically for Apple Silicon (M3 Pro MacBook Pro and similar). The project implements a complete self-play reinforcement learning pipeline with Monte Carlo Tree Search (MCTS) and a modern ResNet backbone featuring chess-specific attention mechanisms.

## 🎯 **Project Status: PRODUCTION READY**

**Current Version**: v1.0 - Fully functional training pipeline with advanced features
**Last Updated**: August 2025
**Status**: ✅ **READY FOR PRODUCTION TRAINING**

## 🚀 **Key Features**

- **Complete Training Pipeline**: Self-play → Training → Evaluation → Promotion
- **Modern Architecture**: ResNet with chess-specific attention and SSL heads
- **Apple Silicon Optimized**: MPS GPU acceleration with mixed precision
- **Robust Data Management**: SQLite metadata, corruption detection, backup system
- **External Engine Integration**: Train against Stockfish, LC0 for competitive data
- **Production Monitoring**: Rich TUI, comprehensive logging, performance metrics

## 🏗️ **Architecture Overview**

### **Neural Network**
- **Input**: 19 planes (pieces, castling, move counters)
- **Backbone**: ResNet-14 with 160 channels (~22M parameters)
- **Heads**: Policy (4672 actions), Value (scalar), SSL (piece prediction)
- **Features**: Squeeze-and-Excitation blocks, chess-specific attention

### **MCTS Engine**
- **Search**: Monte Carlo Tree Search with transposition tables
- **Optimizations**: LRU cache, memory management, early termination
- **Parameters**: Configurable cpuct, dirichlet noise, FPU

### **Training System**
- **Data Sources**: Self-play games, Lichess database, external engine games
- **Loss Function**: Policy cross-entropy + value MSE + SSL classification
- **Optimization**: Adam optimizer, learning rate scheduling, gradient clipping

## 📁 **Project Structure**

```
Matrix0/
├── azchess/                    # Core package
│   ├── __init__.py            # Exports config, data_manager, mcts, model
│   ├── config.py              # Configuration management
│   ├── model/                 # Neural network models
│   │   ├── resnet.py         # ResNet with attention & SSL
│   │   └── __init__.py
│   ├── mcts.py               # MCTS implementation
│   ├── encoding.py            # Board encoding & move mapping
│   ├── data_manager.py        # Data pipeline & integrity
│   ├── selfplay/              # Self-play generation
│   │   ├── internal.py       # Main self-play worker
│   │   ├── inference.py      # Shared inference server
│   │   └── external_engine_worker.py
│   ├── eval/                  # Evaluation system
│   │   └── multi_engine_evaluator.py
│   ├── tools/                 # Utility scripts
│   ├── arena.py               # Model vs model evaluation
│   ├── orchestrator.py        # Full training cycle coordination
│   └── cli_play.py           # Interactive play interface
├── config.yaml                # Main configuration
├── train_comprehensive.py     # Training script
├── requirements.txt           # Python dependencies
├── checkpoints/               # Model checkpoints
├── data/                      # Training data
│   ├── selfplay/             # Self-play games
│   ├── lichess/              # Lichess database
│   ├── replays/              # Training shards
│   └── backups/              # Data backups
├── logs/                      # Training logs
└── webui/                     # Web interface (eval-only)
```

The package re-exports its most commonly used modules, allowing direct access:

```python
from azchess import config, data_manager, mcts, model
```

## 🚀 **Quick Start**

### **1. Environment Setup**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Initial Checkpoint**
```bash
# Generate baseline model with random weights
python -m azchess.save_init
```

### **3. Training Cycle**
```bash
# Run complete training cycle (recommended)
python -m azchess.orchestrator --config config.yaml

# Or run components individually:
# Self-play only
python -m azchess.selfplay --games 64 --workers 4

# Training only
python train_comprehensive.py --config config.yaml

# Evaluation only
python -m azchess.arena --ckpt_a checkpoints/enhanced_best.pt --ckpt_b checkpoints/best.pt --games 20
```

### **4. Interactive Play**
```bash
# Play against trained model
python -m azchess.cli_play

# Web interface with real-time updates
uvicorn webui.server:app --host 127.0.0.1 --port 8000
```

Launch the server and open <http://127.0.0.1:8000> in your browser. The
frontend connects to `ws://localhost:8000/ws/{game_id}` for live FEN,
evaluation and result updates. Multiple concurrent games can be managed via
tabs at the top of the page.

### **5. Analyze Self-Play Games**
```bash
python analyze_games.py --directory data/selfplay --date-pattern "2025-08-16T15:*"
```

- `--directory` &mdash; Location of self-play game files (default: `data/selfplay`)
- `--date-pattern` &mdash; Glob fragment used to match game files, appended after
  `selfplay_w*_g*_` (default: `2025-08-16T15:*`)

When run without arguments the script behaves as before, analyzing games in
`data/selfplay` that match the default pattern.

## ⚙️ **Configuration**

The main configuration is in `config.yaml` with sections for:

- **Model**: Architecture parameters (channels, blocks, features)
- **MCTS**: Search parameters (simulations, cpuct, dirichlet)
- **Self-play**: Worker count, game settings, termination criteria
- **Training**: Batch size, learning rate, optimization settings
- **Evaluation**: Game count, engine settings, promotion thresholds

**Key Parameters**:
```yaml
mcts:
  num_simulations: 200      # MCTS search depth
  cpuct: 2.5                # Exploration constant
  dirichlet_alpha: 0.3      # Noise for exploration
  dirichlet_frac: 0.25      # Noise impact fraction
  fpu: 0.5                  # First-play urgency for unvisited nodes
  draw_penalty: -0.1        # Slight draw penalty in terminal evaluation

selfplay:
  num_workers: 4            # Parallel workers
  shared_inference: true    # GPU optimization
  max_game_len: 140        # Game length limit

training:
  batch_size: 512           # Training batch size
  ssl_weight: 0.1          # SSL loss weight
  warmup_steps: 500        # Learning rate warmup
```

### Additional Tips
- For evaluation, disable Dirichlet noise: set `eval.dirichlet_frac: 0.0`.
- Reduce eval marathon games: set `eval.max_moves` to ~220.
- To reduce drawish play, slightly increase `mcts.draw_penalty` magnitude (e.g., -0.2).
- In self-play, lower `selfplay.temperature_moves` (e.g., 20) to rein in midgame randomness.

## 📊 **Performance & Monitoring**

### **TUI Modes**
- **Table Mode** (recommended): Compact live table with per-worker stats
- **Bars Mode**: Traditional progress bars

### **Logging**
- **Training**: TensorBoard, JSONL summaries, checkpoint tracking
- **Self-play**: Game statistics, worker performance, data integrity
- **Evaluation**: Win rates, confidence intervals, promotion decisions

### **Memory Management**
- **MPS Optimization**: Automatic memory pressure handling
- **Data Streaming**: On-disk replay buffer with smart sharding
- **MCTS Cleanup**: Automatic tree pruning and transposition table management

## 🔧 **Advanced Usage**

### **External Engine Integration**
```bash
# Train against Stockfish/LC0
python -m azchess.selfplay --external-engines --games 32

# Evaluate against external engines
python -m azchess.eval --external-engines --games 50
```

### **Data Management**
```bash
# Check data integrity
python -m azchess.data_manager --action stats

# Compact self-play to training data
python -m azchess.data_manager --action compact
```

### **Model Analysis**
```bash
# Model information
python -m azchess.tools.model_info

# Performance benchmarks
python -m azchess.tools.bench_inference
python -m azchess.tools.bench_mcts
```

Use the MCTS benchmark to estimate sims/s and tune your `mcts.num_simulations` and batch sizes.
For evaluation stability, combine `eval.dirichlet_frac: 0.0` with a lower `eval.max_moves`.

## 📥 External Data → Training Shards

You can ingest external CSV datasets (openings, evaluations, puzzles) into NPZ shards compatible with training:

```bash
# FEN + best move (one-hot policy, optional win% → value)
python -m azchess.tools.convert_csv --csv openings_fen7.csv --format fen_bestmove --out data/replays --prefix openfen7 --shard-size 16384

# FEN + evaluation in centipawns (uniform policy, cp → value)
python -m azchess.tools.convert_csv --csv chessData.csv --format fen_eval --out data/replays --prefix chesseval --shard-size 16384

# Lichess puzzles (UCI move sequences; one-hot policy per step; value=+0.8)
python -m azchess.tools.convert_csv --csv lichess_db_puzzle.csv --format puzzles --out data/replays --prefix puzzles --shard-size 16384

# Auto-detect format from headers (fen/best_move, fen/evaluation, fen/moves)
python -m azchess.tools.convert_csv --csv openings.csv --format auto --out data/replays --prefix openings_auto --shard-size 16384
```

Notes:
- Value targets: `fen_eval` uses cp→value via `tanh(cp/300)`, flipped for Black-to-move. `fen_bestmove` uses `winning_percentage` when present (`2*p-1`), else 0.0. Puzzles default to `+0.8`.
- Policies: one-hot on provided best move when available; uniform over legals for eval-only rows.
- Shards are written under `training.replay_dir` (defaults to `data/replays`). Training will automatically pick them up.

## 🎯 **Training Pipeline**

### **1. Self-Play Phase**
- Multiple workers generate games using MCTS + neural network
- Shared inference server maximizes GPU utilization
- Early termination prevents draw loops and long games
- Games saved as NPZ files with metadata

### **2. Training Phase**
- Data loaded from replay buffer and external sources
- Combined loss: policy + value + SSL
- Mixed precision training on MPS
- Checkpointing every 1000 steps with EMA

### **3. Evaluation Phase**
- New model vs current best (balanced colors)
- Statistical significance with confidence intervals
- Promotion threshold: 55% win rate
- PGN export for game analysis

### **4. Promotion & Continuation**
- Successful models promoted to `best.pt`
- Training continues with new best model
- Full cycle repeats automatically

## 📈 **Expected Performance**

### **Hardware Requirements**
- **Minimum**: M1 MacBook Air (8GB RAM)
- **Recommended**: M3 Pro MacBook Pro (16GB+ RAM)
- **Optimal**: M3 Max with dedicated GPU

### **Training Times**
- **Self-play**: ~2-4 hours for 100 games (4 workers)
- **Training**: ~1-2 hours for 10,000 steps
- **Evaluation**: ~30 minutes for 20 games
- **Full cycle**: ~4-8 hours total

### **Model Strength**
- **Baseline**: Random play (~800 Elo)
- **After 1 cycle**: Basic tactics (~1200 Elo)
- **After 5 cycles**: Strategic play (~1600 Elo)
- **After 10+ cycles**: Advanced play (~2000+ Elo)

## 🐛 **Troubleshooting**

### **Common Issues**
- **MPS unavailable**: Ensure PyTorch 2.0+ and arm64 Python
- **Memory pressure**: Reduce batch size or worker count
- **Training stalls**: Check data integrity with `--doctor-fix`
- **Import errors**: Verify virtual environment activation

### **Debug Commands**
```bash
# Device diagnostics
python -m azchess.tools.diag_device

# Memory usage
python -m azchess.monitor

# Data validation
python -m azchess.data_manager --action validate
```

## 🤝 **Contributing**

This is a production system designed for serious chess AI training. Contributions should focus on:

- **Performance optimization** (MPS, memory, throughput)
- **Training stability** (convergence, regularization)
- **Data quality** (game generation, augmentation)
- **Evaluation accuracy** (strength measurement)

## 📄 **License**

Private project - no third-party model weights included.

---

**Matrix0 v1.0** - Ready for production training on Apple Silicon 🚀
