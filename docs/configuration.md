# Configuration Guide

The Matrix0 project uses a comprehensive `config.yaml` file to manage all parameters for the training pipeline, MCTS, model architecture, and data management. This guide provides detailed documentation of the current configuration structure.

## 1. Configuration Overview

The `config.yaml` file is the single source of truth for all system parameters. It is loaded at startup by the orchestrator and passed to all components.

### Current Configuration Structure
```yaml
# Matrix0 V2 Production Configuration
# Optimized for performance and strong play

model:          # Neural network architecture (53M parameters)
selfplay:       # Self-play data generation
training:       # Training pipeline settings
orchestrator:   # Main training coordinator
eval:          # Model evaluation settings
mcts:          # Monte Carlo Tree Search parameters
```

### Example Current Configuration (Partial)
```yaml
# Model Architecture - 53M parameters
model:
  planes: 19                       # Input planes for chess board
  channels: 320                    # Number of channels in residual blocks
  blocks: 24                       # Number of residual blocks
  attention_heads: 20              # Number of attention heads
  policy_size: 4672                # Policy output size
  norm: "group"                    # GroupNorm instead of BatchNorm
  activation: "silu"               # SiLU activation instead of ReLU
  preact: true                     # Pre-activation residual blocks
  droppath: 0.1                    # DropPath regularization
  policy_factor_rank: 128          # Factorized policy head
  ssl_curriculum: true             # Self-supervised learning curriculum
  self_supervised: true            # Enable SSL
  ssl_tasks: ["piece"]  # SSL tasks (only piece task currently supported)
```

## 2. Key Configuration Sections

### `model` - Neural Network Architecture (53M Parameters)
- `planes`: Input planes for chess board representation (fixed: 19)
- `channels`: Number of channels in residual blocks (default: 320)
- `blocks`: Number of residual blocks (default: 24)
- `attention_heads`: Number of attention heads (default: 20)
- `policy_size`: Policy output size (fixed: 4672)
- `norm`: Normalization type (`"group"` or `"batch"`, default: `"group"`)
- `activation`: Activation function (`"silu"`, `"relu"`, `"gelu"`, default: `"silu"`)
- `preact`: Pre-activation residual blocks (default: `true`)
- `droppath`: DropPath regularization rate (default: `0.1`)
- `policy_factor_rank`: Factorized policy head rank (default: `128`)
- `ssl_curriculum`: Enable SSL progressive difficulty (default: `true`)
- `self_supervised`: Enable self-supervised learning (default: `true`)
- `ssl_tasks`: List of SSL tasks to train (currently only `["piece"]` is supported)

### `selfplay` - Self-Play Data Generation
- `num_workers`: Number of parallel self-play workers (default: `6`)
- `batch_size`: Self-play batch size (default: `128`)
- `max_games`: Maximum games per self-play cycle (default: `18`)
- `max_game_len`: Maximum game length in moves (default: `100`)
- `min_resign_plies`: Minimum plies before resignation (default: `20`)
- `resign_threshold`: Win probability threshold for resignation (default: `-0.90`)
- `num_simulations`: MCTS simulations per self-play move (default: `200`)

### `training` - Training Pipeline Settings
- `batch_size`: Training batch size (default: `192`)
- `epochs`: Number of training epochs per cycle (default: `3`)
- `learning_rate`: Initial learning rate (default: `0.001`)
- `weight_decay`: L2 regularization strength (default: `0.0001`)
- `gradient_accumulation_steps`: Effective batch size multiplier (default: `2`)
- `grad_clip_norm`: Gradient clipping threshold (default: `0.5`)
- `ssl_weight`: Weight of SSL loss in total loss (default: `0.05`)
- `ssl_warmup_steps`: Steps to gradually increase SSL weight (default: `200`)
- `precision`: Training precision (`"fp16"` or `"fp32"`, default: `"fp16"`)
- `memory_limit_gb`: MPS memory limit in GB (default: `14`)

### `orchestrator` - Training Coordination
- `initial_games`: Games for first training cycle (default: `12`)
- `subsequent_games`: Games for subsequent cycles (default: `12`)
- `games_per_cycle`: Total games per cycle (default: `12`)
- `train_epochs_per_cycle`: Training epochs per cycle (default: `2`)
- `eval_games_per_cycle`: Evaluation games per cycle (default: `6`)
- `keep_top_k`: Number of best models to keep (default: `1`)
- `continuous_mode`: Single run vs continuous training (default: `false`)

### `eval` - Model Evaluation Settings
- `games`: Number of evaluation games (default: `4`)
- `num_simulations`: MCTS simulations for evaluation (default: `800`)
- `max_moves`: Maximum moves per evaluation game (default: `80`)
- `external_engines`: List of engines to compare against (default: `["stockfish"]`)
- `tournament_rounds`: Tournament rounds (default: `4`)

### `mcts` - Monte Carlo Tree Search Parameters
- `num_simulations`: MCTS simulations per move (default: `300`)
- `cpuct`: Exploration-exploitation constant (default: `2.2`)
- `cpuct_start`: Initial cpuct value (default: `2.8`)
- `cpuct_end`: Final cpuct value (default: `1.8`)
- `cpuct_plies`: Plies for cpuct progression (default: `40`)
- `dirichlet_alpha`: Dirichlet noise alpha (default: `0.3`)
- `dirichlet_frac`: Fraction of Dirichlet noise (default: `0.25`)
- `dirichlet_plies`: Plies for Dirichlet noise (default: `16`)
- `fpu`: First Play Urgency reduction (default: `0.2`)
- `draw_penalty`: Penalty for draw positions (default: `-0.2`)

## 3. Configuration in Code

The configuration is loaded at startup by the orchestrator and passed to all components. The system uses a hierarchical configuration structure.

### Example Usage
```python
# In azchess/orchestrator.py
from azchess.config import Config

# Load configuration
config = Config.load('config.yaml')

# Access sections
model_config = config.model()
training_config = config.training()
mcts_config = config.mcts()

# Access parameters
print(f"Model channels: {model_config.channels}")
print(f"Training batch size: {training_config.batch_size}")
print(f"MCTS simulations: {mcts_config.num_simulations}")

# Get parameter counts
total_params = config.model().count_parameters()
print(f"Total parameters: {total_params:,}")
```

### Configuration Classes
The system uses dataclasses for type-safe configuration:

- `ModelConfig`: Neural network architecture parameters
- `TrainingConfig`: Training pipeline settings
- `MCTSConfig`: Monte Carlo Tree Search parameters
- `SelfPlayConfig`: Self-play data generation settings
- `EvalConfig`: Model evaluation settings

## 4. Configuration Management

### Best Practices
- **Single Source of Truth**: All parameters centralized in `config.yaml`
- **Type Safety**: Use configuration dataclasses for validation
- **Environment Adaptation**: Automatic device-specific optimizations
- **Memory Management**: Configurable memory limits for MPS training
- **Performance Tuning**: Batch sizes optimized for available hardware

### Validation and Safety
- **Startup Validation**: Configuration validated before training begins
- **Memory Safety**: MPS memory limits prevent out-of-memory crashes
- **Gradient Stability**: Configurable gradient clipping thresholds
- **SSL Safety**: Curriculum progression prevents learning instability

### Performance Optimization
- **Hardware Detection**: Automatic optimization for Apple Silicon
- **Mixed Precision**: FP16 training with stability safeguards
- **Memory Efficiency**: Gradient checkpointing and tensor cleanup
- **Training Throughput**: Optimized batch sizes and worker counts

## 5. Advanced Configuration

### SSL Configuration
```yaml
model:
  ssl_curriculum: true             # Progressive difficulty
  ssl_tasks: ["piece"]
  ssl_warmup_steps: 200            # Gradual SSL introduction

training:
  ssl_weight: 0.05                 # SSL loss weight
  ssl_warmup_steps: 200            # SSL warmup period
```

### Memory Management
```yaml
training:
  memory_limit_gb: 14              # MPS memory limit
  grad_clip_norm: 0.5              # Gradient clipping
  gradient_accumulation_steps: 2   # Effective batch size
```

### Hardware Optimization
```yaml
selfplay:
  num_workers: 6                   # Parallel data generation
  num_simulations: 200             # Optimized for speed/quality

training:
  batch_size: 192                  # MPS-optimized batch size
  precision: "fp16"                # Mixed precision training
```

### Performance Logging
Matrix0 includes optional performance diagnostics that log timing information for
training steps. These messages are prefixed with `PERF:` and are disabled by
default. To enable them, configure logging at the `DEBUG` level:

```python
import logging
from azchess.logging_utils import setup_logging

# Enable performance diagnostics
logger = setup_logging(level=logging.DEBUG)
```

With this configuration, the training scripts will emit detailed performance
metrics. Running at the default `INFO` level suppresses these messages.
