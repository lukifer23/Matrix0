# Matrix0: AlphaZero-Style Chess Engine for Apple Silicon

Matrix0 is an efficient, AlphaZero-style chess engine designed for Apple Silicon.
It provides a complete self-play reinforcement learning pipeline with Monte Carlo
Tree Search (MCTS) and a modern ResNet backbone with chess-specific attention.

## Project Status
Production ready. See the [status report](docs/status.md) and
[development roadmap](docs/roadmap.md) for details.

## Key Features

- Complete training pipeline: self-play → training → evaluation → promotion
- ResNet-14 architecture with chess-specific attention and SSL head
- Optimized for MPS with mixed precision
- Robust data management and monitoring
- External engine integration (Stockfish, LC0)

## Project Structure

```
Matrix0/
├── azchess/                    # Core package
├── config.yaml                 # Main configuration
├── train_comprehensive.py      # Training script
├── data/                       # Training data
├── checkpoints/                # Model checkpoints
└── webui/                      # Web interface (evaluation only)
```

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initial Checkpoint
```bash
python -m azchess.save_init
```

### 3. Training Cycle
```bash
python -m azchess.orchestrator --config config.yaml
```

### 4. Interactive Play
```bash
python -m azchess.cli_play

# Web interface (evaluation mode)
uvicorn webui.server:app --host 127.0.0.1 --port 8000
```

### Model Analysis
```bash
python -m azchess.tools.model_info
python -m azchess.tools.bench_inference
python -m azchess.tools.bench_mcts
```

## Documentation
- [Configuration guide](docs/configuration.md)
- [Web UI guide](docs/webui.md)
- [External engine integration](EXTERNAL_ENGINES.md)
- [Full documentation index](docs/index.md)

## Contributing
Contributions are welcome. Focus areas include performance optimization,
training stability, data quality and evaluation accuracy.

## License
Private project - no third-party model weights included.

---

**Matrix0 v1.0**
