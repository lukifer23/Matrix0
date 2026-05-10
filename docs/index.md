# Matrix0 Documentation

Welcome to the Matrix0 documentation. This directory contains detailed guides for using and developing the engine.

- [Project Status](status.md)
- [Current Working Plan](current_working_plan.md)
- [Development Roadmap](roadmap.md)
- [Configuration Guide](configuration.md)
- [Local Loop Knob Guide](local_loop_knobs.md)
- [Web UI Guide](webui.md)
- [Model V2 Design](model_v2.md)
- [External Engine Integration](EXTERNAL_ENGINES.md)
- [Benchmark System Guide](BENCHMARK_SYSTEM.md)
- [EX0Bench External Engine Benchmarking](../benchmarks/EX0BENCH_README.md)
- [Changelog](../CHANGELOG.md)

## Current Project Status

**Matrix0 v2.2** - SSL Architecture Integration + hardened local-loop training diagnostics + EX0Bench System deployed. Active work is focused on reliable self-play improvement before checkpoint promotion.

### ✅ What's Working
- **Training Pipeline**: Complete self-play → training → evaluation cycle with strict diagnostics and promotion gates
- **Model Architecture**: 44.2M parameter ResNet-22 with attention and SSL foundation
- **SSL Foundation**: Complete 5-SSL-head integration (threat, pin, fork, control, piece detection) with optimized pipeline
- **EX0Bench System**: Pure external engine battles (Stockfish vs LC0) without neural network inference
- **Advanced Benchmark System**: Multi-engine tournaments, SSL performance tracking, Apple Silicon optimization
- **Data Pipeline**: Fixed SSL target concatenation, shape mismatches, and value target corrections
- **Local-Loop Diagnostics**: Fresh MCTS roots, current-search visit targets, exact zero-jitter ablations, final-position metadata, and legal-policy evaluation
- **External Engines**: Stockfish and LC0 integration with automatic discovery and optimization
- **Apple Silicon**: MPS optimization with 14GB memory management and Metal backend support
- **Web Interface**: FastAPI-based evaluation and analysis interface with comprehensive monitoring

### 🔄 Active Development
- **Reliable Model Improvement**: Validate self-play labels before training and promote only on stable heldout metrics plus candidate generator quality
- **Search/Data Tuning**: Diagnose capped games with final-position metadata; tune MCTS, draw adjudication, capped value weights, and anchor data mix
- **SSL Performance Validation**: Measure and validate SSL learning effectiveness across all 5 tasks after the local-loop promotion gate is trustworthy
- **Performance Optimization**: Memory usage and training throughput improvements

### 📚 Documentation Status
- **Current Working Plan**: ✅ Living document for active experiments
- **Local Loop Knob Guide**: ✅ Active guide for self-play/training/eval knobs
- **Configuration Guide**: ✅ Current and accurate (updated for SSL fixes)
- **Model V2 Design**: ✅ Current and accurate
- **Web UI Guide**: ✅ Current and accurate
- **External Engines**: ✅ Current and accurate
- **Benchmark System**: ✅ New comprehensive guide added
- **EX0Bench System**: ✅ New comprehensive guide added
- **Changelog**: ✅ New comprehensive changelog added
- **Status & Roadmap**: ✅ Current and accurate for active local-loop reliability work

## Open Issues

For a quick look at current problem areas, check the [status report](status.md).

### Current Priorities
1. **Reliable Training Signal**: Improve model only from verified self-play, heldout evaluation, and candidate generator checks
2. **Search/Outcome Mix**: Reduce capped-game dominance while preserving sharp policy labels
3. **Capped-Game Diagnosis**: Use final-position metadata to decide whether to fix search, draw adjudication, or value targets
4. **SSL Validation**: Test and validate advanced SSL algorithm effectiveness
5. **Performance Optimization**: Memory usage and training throughput improvements

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

**Last Updated**: 2026-05-10
**Status**: Production training pipeline operational with strict local-loop diagnostics, complete SSL architecture integration, EX0Bench external benchmarking, and active generator-quality gating before checkpoint promotion
