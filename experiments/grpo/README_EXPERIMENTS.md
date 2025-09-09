# GRPO Chess AI Experimental Framework

This experimental framework explores **Generalized Reward-based Policy Optimization (GRPO)** with a **transformer-based architecture** in chess. The goal is to push the boundaries of chess AI research while maintaining safety through complete isolation from the main Matrix0 codebase.

## Architecture Overview

```
experiments/grpo/
├── models/           # Transformer architecture
│   └── large_chess_transformer.py
├── mcts/             # MCTS integration
│   └── mcts_integration.py
├── training/         # GRPO algorithms
│   └── grpo_trainer.py
├── results/          # Experiment outputs
└── scripts/          # Experiment runners
    ├── grpo_orchestrator.py
    └── create_new_checkpoint.py
```

## Available Experiments

### Magnus Transformer + GRPO

```bash
python experiments/grpo/scripts/grpo_orchestrator.py --config magnus_transformer_grpo --games 100 --epochs 20
```

- **Architecture**: 12-layer transformer with 512-dimensional embeddings and 8 attention heads (~70M parameters).
- **GRPO**: Group size of 8, 200 MCTS simulations per move.

### Medium Transformer + GRPO

```bash
python experiments/grpo/scripts/grpo_orchestrator.py --config medium_transformer_grpo --games 75 --epochs 15
```

- **Architecture**: 6-layer transformer with 384-dimensional embeddings and 6 attention heads (~25M parameters).
- **GRPO**: Group size of 4, 150 MCTS simulations per move.

## Quick Start Guide

### 1. Create a new checkpoint

```bash
cd experiments/grpo
python scripts/create_new_checkpoint.py --model magnus
```

### 2. Run an experiment

```bash
python scripts/grpo_orchestrator.py --config magnus_transformer_grpo --games 50 --epochs 5 --device mps
```

## Configuration & Customization

Modify `configs/experiment_configs.yaml` to change experiment parameters:

```yaml
magnus_transformer_grpo:
  model:
    type: magnus_transformer
  grpo:
    group_size: 8
    learning_rate: 5e-5
    mcts_simulations: 200
  training:
    num_games_per_epoch: 100
```

## Future Research Directions

- **Meta-Learning:** Adaptive parameter learning based on game state.
- **Reward Shaping:** Incorporating domain-specific knowledge into the reward function.
- **Attention Mechanisms:** Exploring more advanced attention patterns for chess.