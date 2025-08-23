# Matrix0: AlphaZero-Style Chess Engine for Apple Silicon

[![Code Quality](https://github.com/lukifer23/Matrix0/workflows/Code%20Quality/badge.svg)](https://github.com/lukifer23/Matrix0/actions)
[![Model Validation](https://github.com/lukifer23/Matrix0/workflows/Model%20Validation/badge.svg)](https://github.com/lukifer23/Matrix0/actions)
[![Training Pipeline Test](https://github.com/lukifer23/Matrix0/workflows/Training%20Pipeline%20Test/badge.svg)](https://github.com/lukifer23/Matrix0/actions)

Matrix0 is a production-ready AlphaZero-style chess engine designed for Apple Silicon.
It provides a complete self-play reinforcement learning pipeline with Monte Carlo
Tree Search (MCTS) and a modern ResNet backbone with advanced attention and SSL capabilities.

## 🎯 Project Overview

Matrix0 implements cutting-edge reinforcement learning concepts from AlphaZero and modern research, optimized for Apple Silicon (MPS). The project delivers:

- **Complete Training Pipeline**: Self-play → Training → Evaluation → Model Promotion
- **Advanced Architecture**: 53M parameter ResNet-24 with chess-specific attention
- **Multi-Task Learning**: Enhanced SSL with threat detection, pins, forks, and control
- **Apple Silicon Optimization**: MPS GPU acceleration with 14GB memory management
- **Production Features**: Robust data management, monitoring, and evaluation tools

## 📊 Project Status
**ACTIVE DEVELOPMENT** - Training pipeline operational, SSL enhanced, actively improving. See the
[status report](docs/status.md), the [development roadmap](docs/roadmap.md), and the
[Open Issues](docs/index.md#open-issues) section for current problem areas.

## ✨ Key Features

- **Complete Training Pipeline**: Self-play → Training → Evaluation → Model Promotion (✅ OPERATIONAL)
- **Advanced Architecture**: 53M parameter ResNet-24 with 320 channels and chess-specific attention
- **Multi-Task SSL**: Threat detection, pin detection, fork opportunities, square control
- **MPS Optimization**: Apple Silicon GPU acceleration with 14GB memory management
- **Robust Data Management**: SQLite metadata, corruption detection, automatic backup system
- **External Engine Integration**: Stockfish and LC0 support for competitive training
- **Rich Monitoring**: Rich TUI, comprehensive logging, real-time performance metrics
- **Web Interface**: FastAPI-based evaluation and analysis interface
- **Training Stability**: Branch normalization, gradient clipping, emergency checkpoints

## 🏗️ Project Structure

```
Matrix0/
├── azchess/                    # Core package (53M parameter model)
│   ├── model/                  # Neural network architecture
│   │   └── resnet.py          # ResNet-24 with attention and SSL
│   ├── selfplay/               # Self-play generation pipeline
│   ├── mcts/                   # Monte Carlo Tree Search engine
│   ├── training/               # Training pipeline and optimizers
│   ├── eval/                   # Model evaluation and tournaments
│   ├── tools/                  # Analysis and benchmarking tools
│   ├── data_manager.py        # SQLite metadata and backup system
│   ├── orchestrator.py        # Main training coordinator
│   └── config.py              # Configuration management
├── config.yaml                 # Main configuration (53M model, SSL enabled)
├── data/                       # Training data and replays
│   ├── backups/               # Automatic backup system
│   ├── selfplay/              # Self-play game data
│   └── data_metadata.db       # SQLite database for data integrity
├── checkpoints/                # Model checkpoints (v2_base.pt active)
│   ├── v2_base.pt             # Current stable checkpoint
│   └── model_step_5000.pt     # Latest training checkpoint
├── webui/                      # FastAPI web interface
├── logs/                       # Comprehensive logging system
├── docs/                       # Complete documentation suite
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ with virtual environment support
- 18GB+ RAM (14GB for model, 4GB for system)
- 100GB+ free disk space (50GB for checkpoints, 50GB for data)
- 16GB+ unified memory recommended

### 1. Environment Setup
```bash
git clone https://github.com/lukifer23/Matrix0.git
cd Matrix0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initial Model Checkpoint
```bash
python create_v2_checkpoint.py  # Creates optimized 53M parameter model
```

### 3. Start Training (Current Session)
```bash
source .venv/bin/activate
python -m azchess.orchestrator --workers 2 --sims 300 --lr 0.001 --batch-size 192 --epochs 10 --eval-games 50 --device mps
```

### 4. Monitor & Interact
```bash
# Interactive play against current model
python -m azchess.cli_play

# Web interface for analysis (evaluation mode)
uvicorn webui.server:app --host 127.0.0.1 --port 8000

# Check current training status
tail -f logs/matrix0.log
```

### Model Analysis & Benchmarks
```bash
# Model information and parameter count
python -m azchess.tools.model_info

# Inference performance benchmarking
python -m azchess.tools.bench_inference

# MCTS performance benchmarking
python -m azchess.tools.bench_mcts

# Training data analysis
python -m azchess.tools.process_lichess
```

## 📚 Documentation
- [Configuration guide](docs/configuration.md)
- [Web UI guide](docs/webui.md)
- [Model V2 Design](docs/model_v2.md)
- [External engine integration](EXTERNAL_ENGINES.md)
- [Full documentation index](docs/index.md)

## 🔧 Current Training Status

**Latest Update**: August 2025
- **Training Progress**: Step 5000+ completed ✅
- **Training Speed**: ~3-4 seconds per step
- **Model Size**: 53,217,919 parameters (53M)
- **Architecture**: ResNet-24 with 320 channels, 20 attention heads
- **SSL Status**: Multi-task learning enabled with curriculum progression
- **Training Stability**: Branch normalization, gradient clipping, emergency checkpoints
- **Memory Usage**: ~10.7-11.0GB MPS usage with automatic management

## 🔧 Development

### Testing & Validation
The project includes comprehensive validation and monitoring:
- **Code Quality**: Linting, formatting, and type checking workflows
- **Model Validation**: Architecture integrity and encoding verification
- **Training Pipeline**: End-to-end pipeline validation and stability tests
- **Performance Monitoring**: Real-time training metrics and memory usage

### Local Development
```bash
# Install development dependencies
pip install flake8 black mypy

# Run comprehensive tests
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
black --check --diff .
mypy azchess/ --ignore-missing-imports

# Model validation tests
python -m azchess.model.test_encoding
python -m azchess.model.test_attention

# Performance benchmarking
python -m azchess.tools.bench_inference
python -m azchess.tools.bench_mcts
```

## 🤝 Contributing
Contributions are welcome! Current focus areas include:

### High Priority
- **SSL Feature Enhancement**: Complete threat detection, pin detection algorithms
- **Training Stability**: Further improvements to numerical stability
- **Memory Optimization**: Reduce memory footprint for larger batch sizes
- **MCTS Improvements**: Enhanced tree search efficiency

### Medium Priority
- **Model Architecture**: Additional attention mechanisms and residual structures
- **Data Quality**: Enhanced position evaluation and game quality metrics
- **Performance Benchmarking**: Comprehensive speed and accuracy testing
- **Documentation**: Keep all docs current with latest features

### Development Guidelines
- Follow PEP 8 style guidelines with 88-character line limits
- Add comprehensive tests for new functionality
- Update documentation for all changes
- Use type hints and docstrings for all functions
- Ensure MPS compatibility for all new features

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This is a research project. No third-party model weights are included.

## 🙏 Acknowledgments
- Inspired by AlphaZero, Leela Chess Zero, and modern chess AI research
- Built with PyTorch and optimized for Apple Silicon MPS architecture
- Advanced SSL concepts from recent computer vision and NLP research
- Community contributions and feedback

## 📈 Current Achievements & Next Steps

### ✅ Completed Milestones
- **Training Pipeline**: Self-play → Training → Evaluation → Model Promotion
- **Model Architecture**: 53M parameter ResNet-24 with attention and SSL
- **Training Stability**: Branch normalization, gradient clipping, emergency checkpoints
- **Memory Management**: 14GB MPS optimization with automatic cleanup
- **SSL Implementation**: Multi-task learning with threat/pin/fork/control detection

### 🔄 Active Development
- **SSL Curriculum**: Progressive difficulty from basic to advanced concepts
- **Performance Optimization**: Memory efficiency and training throughput
- **Model Evaluation**: Enhanced tournament and strength estimation systems

See [docs/roadmap.md](docs/roadmap.md) and [docs/status.md](docs/status.md) for detailed development plans.

---

**Matrix0 v2.0 - Production Training Pipeline**

*Advanced chess AI research platform with 53M parameter model and multi-task SSL learning. Actively training at step 1000+ with stable performance.*
