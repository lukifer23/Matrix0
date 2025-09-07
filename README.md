# Matrix0: Advanced AlphaZero-Style Chess Engine with SSL Integration

[![Code Quality](https://github.com/lukifer23/Matrix0/workflows/Code%20Quality/badge.svg)](https://github.com/lukifer23/Matrix0/actions)
[![Model Validation](https://github.com/lukifer23/Matrix0/workflows/Model%20Validation/badge.svg)](https://github.com/lukifer23/Matrix0/actions)
[![Training Pipeline Test](https://github.com/lukifer23/Matrix0/workflows/Training%20Pipeline%20Test/badge.svg)](https://github.com/lukifer23/Matrix0/actions)
[![Bugbot Review](https://img.shields.io/badge/Bugbot-Review%20Ready-blue?logo=bug)](https://bugbot.dev)

Matrix0 is a **production-ready AlphaZero-style chess engine** featuring **complete SSL (Self-Supervised Learning) integration** designed for Apple Silicon. It provides a sophisticated multi-task learning pipeline combining policy/value optimization with advanced SSL capabilities for chess pattern recognition across **7 specialized tasks**.

## Project Overview

Matrix0 implements **cutting-edge multi-task learning** combining reinforcement learning from AlphaZero with advanced SSL (Self-Supervised Learning) for chess pattern recognition, optimized for Apple Silicon (MPS). The project delivers:

- **🔥 SSL Architecture Integration**: **ARCHITECTURE READY** - 7 specialized SSL heads for piece, threat, pin, fork, control, pawn structure, and king safety detection
- **Multi-Task Learning**: Simultaneous optimization of policy, value, and SSL objectives
- **🏗️ Advanced Architecture**: 53.2M parameter ResNet-24 with chess-specific attention and SSL foundation
- **🍎 Apple Silicon Optimization**: MPS GPU acceleration with 14GB memory management
- **📊 Enhanced WebUI**: Comprehensive monitoring platform with real-time SSL and training analytics
- **🏆 Advanced Benchmark System**: Multi-engine tournaments, SSL performance tracking, and comprehensive evaluation

## Project Status
**ACTIVE DEVELOPMENT** - **SSL architecture integration achieved**, training pipeline fully operational with SSL framework, **EX0Bench external engine benchmarking system** deployed. See the
[comprehensive status report](docs/CURRENT_STATUS_SUMMARY.md), the [enhanced WebUI guide](docs/webui.md), the
[EX0Bench documentation](benchmarks/EX0BENCH_README.md), and the
[development roadmap](docs/roadmap.md) for current achievements and next steps.

## ✨ Key Features

### SSL Integration (COMPLETE)
- **5 Specialized SSL Heads**: Piece recognition, threat detection, pin detection, fork detection, control detection
- **Multi-Task Learning**: Simultaneous optimization of policy, value, and SSL objectives
- **Dedicated SSL Parameters**: SSL capacity with weighted loss functions
- **Real-Time SSL Monitoring**: WebUI dashboard with SSL head performance tracking

### Advanced Architecture
- **53.2M Parameters**: ResNet-24 with 320 channels, 24 blocks, 20 attention heads
- **Chess-Specific Attention**: Optimized attention mechanisms for chess patterns
- **SSL Foundation**: Complete SSL integration with multi-head architecture
- **Memory Optimized**: 14GB MPS limit with efficient SSL processing

### Enhanced WebUI Platform
- **Interactive Chess Board**: Fully functional 8x8 board with proper alternating square colors
- **Multi-View Interface**: Game, Training, SSL, Tournament, and Analysis views
- **Real-Time Monitoring**: Live training status, SSL performance, and model analytics
- **Interactive Visualization**: Charts, progress bars, and performance metrics
- **SSL Dashboard**: Complete SSL head analysis and parameter tracking
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional interface with efficient space utilization

### Production Training Pipeline
- **Self-Play Generation**: 4 workers generating SSL-enhanced training data
- **Multi-Task Training**: Combined policy/value/SSL optimization with proper gradient accumulation
- **Model Evaluation**: Tournament system with SSL-aware strength estimation
- **Checkpoint Management**: Advanced checkpoint creation preserving SSL architecture

### Advanced Benchmark System
- **Multi-Engine Tournaments**: Round-robin, Swiss, and single-elimination formats
- **EX0Bench External Engine Battles**: Pure Stockfish vs LC0 comparisons with no neural network required
- **SSL Performance Tracking**: Real-time monitoring of SSL head effectiveness
- **Apple Silicon Optimization**: MPS memory monitoring and Metal backend support
- **Automated Engine Discovery**: Intelligent detection and configuration of installed engines
- **Comprehensive Analysis**: Statistical significance testing and performance regression analysis

### Apple Silicon Optimization
- **MPS GPU Acceleration**: Native Apple Silicon support with unified memory
- **14GB Memory Management**: Automatic cleanup and cache management
- **Mixed Precision Training**: FP16 optimization with MPS compatibility
- **Performance Monitoring**: Real-time MPS utilization and memory tracking

### 🔧 Enterprise Features
- **Robust Data Management**: SQLite metadata, corruption detection, automatic backup
- **External Engine Integration**: Stockfish and LC0 support for competitive evaluation
- **Comprehensive Logging**: Structured logging with SSL performance metrics
- **Training Stability**: Advanced error handling, gradient management, and recovery mechanisms

## Project Structure

```
Matrix0/
├── azchess/                    # Core package (53.2M parameter model with SSL)
│   ├── model/                  # Neural network architecture
│   │   └── resnet.py          # ResNet-24 with attention and complete SSL integration
│   ├── ssl_algorithms.py      # Advanced SSL algorithms (threat, pin, fork, control)
│   ├── selfplay/               # Self-play generation with SSL data augmentation
│   ├── mcts/                   # Monte Carlo Tree Search engine
│   ├── training/               # Multi-task training pipeline (policy/value/SSL)
│   ├── eval/                   # Model evaluation with SSL-aware metrics
│   ├── tools/                  # Analysis and benchmarking tools
│   ├── data_manager.py        # SQLite metadata and backup system
│   ├── orchestrator.py        # Main training coordinator
│   └── config.py              # Configuration management
├── config.yaml                 # Main configuration (SSL enabled, 7 SSL tasks, 3 workers)
├── data/                       # Training data and replays
│   ├── backups/               # Automatic backup system
│   ├── selfplay/              # SSL-enhanced self-play game data
│   └── data_metadata.db       # SQLite database for data integrity
├── checkpoints/                # Model checkpoints with SSL architecture
│   ├── best.pt                # Current best checkpoint
│   ├── model_step_1000.pt     # Step 1000 checkpoint
│   └── v2_base.pt             # V2 base checkpoint
├── webui/                      # Enhanced FastAPI monitoring platform
│   ├── server.py              # Backend with SSL/training endpoints
│   └── static/                # Multi-view frontend interface
├── logs/                       # Comprehensive logging with SSL metrics
├── docs/                       # Complete documentation suite
└── requirements.txt            # Python dependencies
```

## Quick Start

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
python -m azchess.orchestrator --workers 3 --sims 300 --lr 0.001 --batch-size 96 --epochs 1 --eval-games 10 --device mps
```

Recommended fast, stable Apple‑Silicon run (200 sims)
```bash
# Slightly more aggressive self‑play and shorter cycles
MATRIX0_MPS_TARGET_BATCH=6 \
python -m azchess.orchestrator \
  --tui table \
  --workers 3 \
  --games 300 \
  --sims 200 \
  --eval-games 40 \
  --promotion-threshold 0.55 \
  --epochs 1 \
  --steps-per-epoch 15000 \
  --opening-plies 6 \
  --max-game-length 160 \
  --resign-threshold -0.6
```

### 3b. Generate Stockfish Data (Optional)
```bash
# Example: generate 2k tactical positions with SSL targets
python tools/generate_stockfish_data.py \
  --domain tactical \
  --subcategory pins \
  --positions 2000 \
  --stockfish-depth 12

# Datasets are saved under data/stockfish_games/** and automatically ingested
# by training via training.extra_replay_dirs in config.yaml
```

Run the training loop directly without the orchestrator:

```bash
python -m azchess.training.train --config config.yaml
```

### 4. Monitor & Interact
```bash
# Interactive play against current SSL-integrated model
python -m azchess.cli_play

# Enhanced WebUI monitoring platform
python webui/server.py
# Then visit: http://127.0.0.1:8000

# Check current training status with SSL metrics
tail -f logs/matrix0.log

# View SSL performance and training analytics
# Access WebUI at http://127.0.0.1:8000
```

### Model Analysis & Benchmarks
```bash
# Model information and parameter count
python -m azchess.tools.model_info

# Inference performance benchmarking
python -m azchess.tools.bench_inference

# Export to CoreML for Apple Silicon optimization
python coreml_export.py --checkpoint checkpoints/best.pt --output matrix0.mlmodel --benchmark

# MCTS performance benchmarking
python -m azchess.tools.bench_mcts

# Enhanced Benchmark System (full Matrix0 capabilities)
python benchmarks/enhanced_runner.py --config benchmarks/configs/enhanced_scenarios.yaml

# EX0Bench - Pure external engine battles (Stockfish vs LC0, no neural network)
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 50 --time 60+0.6

# Quick benchmark against external engines
python benchmarks/benchmark.py --model checkpoints/v2_base.pt --engine stockfish --games 10

engine vs engine tournament (lc0 vs Stockfish)
```bash
python benchmarks/enhanced_runner.py --config benchmarks/configs/enhanced_scenarios.yaml --scenario Multi_Engine_Tournament
```

# Training data analysis
python -m azchess.tools.process_lichess
```

### Diagnostics & Evaluation Improvements
- Evaluation and benchmarks now report search diagnostics:
  - `mcts_empty_visits` (count of empty-search fallbacks)
  - average root policy entropy over legal moves (nats)
- PGN exports are validated: header `Result` is corrected if it mismatches the reconstructed board result.
- Evaluation fallbacks are policy-based (no random move injection) and exploration noise is disabled during eval.
- Recommended MCTS simulations for benchmarks: `--mcts-sims 800–1600`.

### Fast Iteration (Smaller Model)
- A smaller configuration for faster training/iteration is provided:
  - `config_small.yaml` (160 channels × 14 blocks, attention-enabled, SSL on)
- Example run:
```bash
python -m azchess.training.train --config config_small.yaml
```
Use this to iterate quickly on data/algorithms, then switch back to the main config for strength.

## 📚 Documentation
- [Configuration guide](docs/configuration.md)
- [Web UI guide](docs/webui.md)
- [Performance tuning](docs/performance.md)
- [Model V2 Design](docs/model_v2.md)
- [External engine integration](docs/EXTERNAL_ENGINES.md)
- [Full documentation index](docs/index.md)

## 🔧 Current Training Status

**Latest Update**: September 2025 - SSL Architecture Integration Complete + Data Pipeline Fixes
- **Training Progress**: **FULLY OPERATIONAL** with SSL architecture integration and data pipeline fixes
- **Training Speed**: ~1.58-2.5 seconds per step (optimized for SSL processing)
- **Model Size**: 53M parameter ResNet-24 with SSL heads
- **Architecture**: ResNet-24 with 320 channels, 24 blocks, 20 attention heads, **5 SSL heads**
- **SSL Status**: **ARCHITECTURE INTEGRATED** - All 5 SSL tasks integrated (piece, threat, pin, fork, control) - optimized data pipeline
- **Large-Scale Training**: 100K step pretraining run in progress using enhanced_best.pt checkpoint
- **SSL Parameters**: Dedicated SSL parameters with weighted loss functions
- **Training Stability**: 100% stable with proper gradient accumulation and scheduler stepping
- **Memory Usage**: ~10.7-11.0GB MPS usage with SSL processing optimization
- **Recent Enhancements**: SSL architecture integration, enhanced WebUI monitoring, EX0Bench external engine battles, data pipeline fixes, MPS stability improvements, CoreML export

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
- **SSL Performance Validation**: Measure and validate SSL learning effectiveness across all 5 tasks
- **SSL Task Balancing**: Optimize loss weights for balanced multi-task learning
- **Enhanced Evaluation**: Multi-engine tournaments with SSL-aware strength estimation
- **WebUI Refinement**: Enhance SSL visualization and monitoring features

### Medium Priority
- **SSL Learning Analytics**: Deep analysis of SSL contribution to policy/value learning
- **Model Interpretability**: SSL decision explanation and analysis tools
- **Performance Benchmarking**: Comprehensive SSL and training validation suites
- **Advanced SSL Features**: SSL curriculum progression and dynamic weighting

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
- **Bugbot**: Code review and quality assurance via [Bugbot](https://bugbot.dev) (14-day trial via Cursor)

## 📈 Current Achievements & Next Steps

### ✅ Major Milestones Completed (September 2025)
- **SSL Architecture Integration**: All 5 SSL tasks (piece, threat, pin, fork, control) integrated with optimized data pipeline
- **Data Pipeline Fixes**: Resolved SSL target concatenation issues, fixed control shape mismatches, corrected value targets
- **Multi-Task Learning**: Simultaneous training of policy, value, and SSL optimization working perfectly
- **EX0Bench System**: Pure external engine battles (Stockfish vs LC0) for fine-tuning decisions
- **Enhanced WebUI**: Complete monitoring platform with real-time SSL and training analytics
- **Training Stability**: 100% stable training with proper scheduler stepping and gradient management
- **MPS Stability Fixes**: Resolved Metal command buffer issues with comprehensive error recovery
- **Advanced Checkpoint Management**: SSL-preserving checkpoint creation and merging tools
- **Production Architecture**: 53M parameter ResNet-24 with complete SSL foundation
- **Apple Silicon Optimization**: 14GB MPS limit with SSL processing optimization
- **Real-Time Monitoring**: Live training status, SSL performance, and model analysis
- **Tournament System**: Advanced multi-format tournament system with Glicko-2 ratings
- **CoreML Export**: Apple Silicon optimization via CoreML for enhanced inference
- **GitHub Actions CI/CD**: Comprehensive testing with SSL validation and security scanning
- **Syzygy Tablebase Integration**: Complete 6-piece endgame tablebases for perfect play

### Active Development Priorities
- **SSL Performance Validation**: Comprehensive measurement of SSL learning effectiveness
- **SSL Task Balancing**: Optimization of loss weights for balanced multi-task learning
- **Enhanced Evaluation**: Multi-engine tournaments with SSL-aware strength estimation
- **SSL Visualization**: Advanced heatmaps and decision explanation tools

### 🎯 Immediate Next Steps
1. **SSL Learning Validation**: Measure effectiveness of all 5 SSL tasks
2. **Performance Benchmarking**: Establish SSL contribution baselines with EX0Bench
3. **WebUI Enhancement**: Add SSL-specific visualization features
4. **Model Analysis**: Deep-dive into SSL learning patterns and effectiveness

See [docs/CURRENT_STATUS_SUMMARY.md](docs/CURRENT_STATUS_SUMMARY.md), [docs/webui.md](docs/webui.md), and [docs/roadmap.md](docs/roadmap.md) for detailed development plans.

---

**Matrix0 v2.2 - SSL Architecture Integration + EX0Bench System**

*Advanced chess AI research platform with 53.4M parameter model and SSL architecture integration. Multi-task learning framework operational with comprehensive monitoring, EX0Bench external engine battles, and data pipeline fixes.*
