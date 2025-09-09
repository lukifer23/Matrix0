# GRPO Chess AI Experiments

This directory contains experimental implementations of GRPO (Generalized Reward-based Policy Optimization) with a transformer-based architecture for chess AI. The goal is to explore cutting-edge RL approaches while maintaining safety by keeping experiments separate from the main Matrix0 codebase.

## Architecture Overview

```
experiments/grpo/
├── models/           # Transformer architecture
│   └── large_chess_transformer.py
├── training/         # GRPO training system
│   └── grpo_trainer.py
├── mcts/             # MCTS trajectory generation
│   └── mcts_integration.py
├── configs/          # Experiment configurations
│   └── experiment_configs.yaml
├── scripts/          # Orchestration and checkpointing
│   ├── grpo_orchestrator.py
│   └── create_new_checkpoint.py
└── results/          # Training metrics & checkpoints
```

## Quick Start

### 1. Create a new checkpoint

```bash
python experiments/grpo/scripts/create_new_checkpoint.py --model magnus
```

### 2. Run an experiment

```bash
python experiments/grpo/scripts/grpo_orchestrator.py --config magnus_transformer_grpo --games 50 --epochs 5
```

## Configuration

Edit `experiments/grpo/configs/experiment_configs.yaml` to customize:

```yaml
magnus_transformer_grpo:
  device: "mps"             # Use MPS for Apple Silicon GPU acceleration
  model:
    type: "magnus_transformer"
  grpo:
    group_size: 8          # Trajectories per group
    clip_epsilon: 0.2      # PPO clipping
    learning_rate: 3e-5    # Learning rate
    mcts_simulations: 200  # MCTS simulations per move
  training:
    num_games_per_epoch: 50  # Games per training epoch
    max_epochs: 25          # Total training epochs
```

## Future Research Directions

- **Meta-Learning:** Adaptive parameter tuning based on game state and performance.
- **Reward Shaping:** Incorporating chess-specific domain knowledge into the reward function.
- **Hybrid Approaches:** Combining GRPO with other techniques like self-supervised learning (SSL).