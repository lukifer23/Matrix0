# Matrix0 Documentation

Welcome to the Matrix0 documentation. This directory contains detailed guides for using and developing the engine.

- [Project Status](status.md)
- [Development Roadmap](roadmap.md)
- [Configuration Guide](configuration.md)
- [Web UI Guide](webui.md)
- [Model V2 Design](model_v2.md)
- [External Engine Integration](EXTERNAL_ENGINES.md)
- [Benchmark System Guide](BENCHMARK_SYSTEM.md)
- [EX0Bench External Engine Benchmarking](../benchmarks/EX0BENCH_README.md)
- [Changelog](../CHANGELOG.md)

## Current Project Status

**Matrix0 v2.2** - SSL Architecture Integration + Data Pipeline Fixes + EX0Bench System deployed.

### [x] What's Working
- **Training Pipeline**: Complete self-play → training → evaluation → promotion cycle with data pipeline fixes
- **Model Architecture**: 53M parameter ResNet-24 with attention and SSL foundation
- **SSL Foundation**: Complete 5-SSL-head integration (threat, pin, fork, control, piece detection) with optimized pipeline
- **EX0Bench System**: Pure external engine battles (Stockfish vs LC0) without neural network inference
- **Advanced Benchmark System**: Multi-engine tournaments, SSL performance tracking, Apple Silicon optimization
- **Data Pipeline**: Fixed SSL target concatenation, shape mismatches, and value target corrections
- **External Engines**: Stockfish and LC0 integration with automatic discovery and optimization
- **Apple Silicon**: MPS optimization with 14GB memory management and Metal backend support
- **Web Interface**: FastAPI-based evaluation and analysis interface with comprehensive monitoring

### Active Development
- **SSL Performance Validation**: Measure and validate SSL learning effectiveness across all 5 tasks
- **Tournament System Enhancement**: Multi-engine tournament analysis and ranking improvements
- **Performance Optimization**: Memory usage and training throughput improvements
- **Model Strength Improvement**: Continue training to close gap with top engines

### Documentation Status
- **Configuration Guide**: [x] Current and accurate (updated for SSL fixes)
- **Model V2 Design**: [x] Current and accurate
- **Web UI Guide**: [x] Current and accurate
- **External Engines**: [x] Current and accurate
- **Benchmark System**: [x] New comprehensive guide added
- **EX0Bench System**: [x] New comprehensive guide added
- **Changelog**: [x] New comprehensive changelog added
- **Status & Roadmap**: [x] Current and accurate (updated for v2.2)

## Open Issues

For a quick look at current problem areas, check the [status report](status.md).

### Current Priorities
1. **SSL Validation**: Test and validate advanced SSL algorithm effectiveness
2. **Training Enhancement**: Achieve stable training with full SSL capabilities
3. **Performance Optimization**: Memory usage and training throughput improvements
4. **Documentation**: Keep all technical docs current and comprehensive

## Quick Start

### Training Pipeline
```bash
# Start complete training pipeline
python -m azchess.orchestrator --config config.yaml

# Or run training directly
python -m azchess.training.train --config config.yaml
```

### Model Evaluation
```bash
# Interactive play
python -m azchess.cli_play

# Web interface
uvicorn webui.server:app --host 127.0.0.1 --port 8000
```

### External Engine Integration
```bash
# Evaluate against external engines
python -m azchess.eval --external-engines --games 50
```

## Development Guidelines

- **Code Quality**: Follow PEP 8 with comprehensive testing
- **Documentation**: Update docs for all changes
- **SSL Integration**: Focus on completing SSL algorithm integration
- **Performance**: Optimize for Apple Silicon MPS architecture

---

**Last Updated**: September 2025
**Status**: Production training pipeline operational, complete SSL architecture integration with data pipeline fixes, EX0Bench external benchmarking system deployed
