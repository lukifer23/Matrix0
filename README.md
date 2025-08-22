# Matrix0: AlphaZero-Style Chess Engine for Apple Silicon

[![Code Quality](https://github.com/lukifer23/Matrix0/workflows/Code%20Quality/badge.svg)](https://github.com/lukifer23/Matrix0/actions)
[![Model Validation](https://github.com/lukifer23/Matrix0/workflows/Model%20Validation/badge.svg)](https://github.com/lukifer23/Matrix0/actions)
[![Training Pipeline Test](https://github.com/lukifer23/Matrix0/workflows/Training%20Pipeline%20Test/badge.svg)](https://github.com/lukifer23/Matrix0/actions)

Matrix0 is an experimental AlphaZero-style chess engine designed for Apple Silicon.
It provides a self-play reinforcement learning pipeline with Monte Carlo
Tree Search (MCTS) and a modern ResNet backbone with chess-specific attention.

## 🎯 Project Overview

Matrix0 implements the core concepts from AlphaZero and related research papers, adapted for modern hardware and optimized for Apple Silicon (MPS). The project focuses on:

- **Self-Play Learning**: Autonomous improvement through self-play games
- **Modern Architecture**: ResNet with attention mechanisms and advanced SSL
- **Apple Silicon Optimization**: MPS GPU acceleration with mixed precision
- **Research Platform**: Extensible framework for chess AI research

## 📊 Project Status
**ACTIVE DEVELOPMENT** - Training pipeline operational, SSL enhanced, actively improving. See the [status report](docs/status.md) and
[development roadmap](docs/roadmap.md) for details.

## ✨ Key Features

- **Complete Training Pipeline**: self-play → training → evaluation → promotion
- **ResNet-14 Architecture**: 160 channels with chess-specific attention and advanced SSL
- **Enhanced SSL**: Multi-task learning with threat detection, pin detection, fork opportunities
- **MPS Optimization**: Apple Silicon GPU acceleration with mixed precision
- **Robust Data Management**: SQLite metadata, corruption detection, backup system
- **External Engine Integration**: Stockfish, LC0 support for competitive training
- **Rich Monitoring**: Rich TUI, comprehensive logging, performance metrics
- **Web Interface**: FastAPI-based evaluation and analysis interface

## 🏗️ Project Structure

```
Matrix0/
├── azchess/                    # Core package
│   ├── model/                  # Neural network architecture
│   ├── selfplay/               # Self-play generation
│   ├── mcts/                   # Monte Carlo Tree Search
│   ├── training/               # Training pipeline
│   └── tools/                  # Utility tools
├── config.yaml                 # Main configuration
├── data/                       # Training data and replays
├── checkpoints/                # Model checkpoints
├── webui/                      # Web interface
└── docs/                       # Documentation
```

## 🚀 Quick Start

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- 16GB+ RAM recommended
- 50GB+ free disk space

### 1. Environment Setup
```bash
git clone https://github.com/lukifer23/Matrix0.git
cd Matrix0
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

## 📚 Documentation
- [Configuration guide](docs/configuration.md)
- [Web UI guide](docs/webui.md)
- [Model V2 Design](docs/model_v2.md)
- [External engine integration](EXTERNAL_ENGINES.md)
- [Full documentation index](docs/index.md)

## 🔧 Development

### Testing
The project includes comprehensive GitHub Actions workflows:
- **Code Quality**: Linting, formatting, and type checking
- **Model Validation**: Architecture and encoding tests
- **Training Pipeline**: End-to-end pipeline validation

### Local Development
```bash
# Install development dependencies
pip install flake8 black mypy

# Run tests locally
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
black --check --diff .
mypy azchess/ --ignore-missing-imports
```

## 🤝 Contributing
Contributions are welcome! Focus areas include:
- Performance optimization for MPS
- Training stability improvements
- Data quality and evaluation accuracy
- Model architecture enhancements
- Documentation improvements

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation for changes
- Use type hints where appropriate

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This is a research project. No third-party model weights are included.

## 🙏 Acknowledgments
- Inspired by AlphaZero and related research
- Built with PyTorch and modern Python tooling
- Optimized for Apple Silicon architecture

## 📈 Roadmap
See [docs/roadmap.md](docs/roadmap.md) for detailed development plans including:
- Enhanced SSL capabilities and multi-task learning
- Performance optimization and memory management
- Advanced training techniques and curriculum learning
- LLM integration for strategic guidance
- Multi-modal learning support

---

**Matrix0 v1.1 - Active Development Version**

*Building the future of chess AI, one move at a time.*
