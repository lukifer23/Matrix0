# Configuration Guide

The Matrix0 project uses a central `config.yaml` file to manage all parameters for the training pipeline, MCTS, model architecture, and data management. This guide provides an overview of the key configuration sections.

## 1. Main Configuration (`config.yaml`)

The `config.yaml` file is the single source of truth for all system parameters. It is loaded at startup by the orchestrator and passed to all components.

### Example `config.yaml`
```yaml
# Main configuration for Matrix0
project_name: "Matrix0"
version: "1.0"

# System settings
device: "mps"  # "cpu", "mps", "cuda"
num_workers: 4 # Number of self-play workers

# Training parameters
training:
  batch_size: 256
  learning_rate: 1e-4
  weight_decay: 1e-4
  num_epochs: 5
  ssl_weight: 0.5
  max_grad_norm: 1.0

# MCTS parameters
mcts:
  num_simulations: 800
  cpuct: 2.5
  cpuct_start: 2.8
  cpuct_end: 1.6
  cpuct_plies: 40
  dirichlet_alpha: 0.3
  dirichlet_frac: 0.25
  dirichlet_plies: 16
  tt_capacity: 2000000
  tt_cleanup_frequency: 1000
  tt_memory_limit_mb: 2048
  selection_jitter: 0.01
  batch_size: 32
  fpu: 0.5
  parent_q_init: true
  draw_penalty: -0.1
  virtual_loss: 1.0
  value_from_white: false
  max_children: 0
  min_child_prior: 0.0
  legal_softmax: false
  encoder_cache: true
  tt_cleanup_interval_s: 5
  no_instant_backtrack: true

# Model architecture
model:
  resnet_channels: 160
  resnet_blocks: 14
  attention_heads: 8
  se_ratio: 0.25

# Data management
data:
  max_games: 10000
  max_replays: 50000
  backup_interval: 3600 # seconds
  compaction_delay: 600 # seconds
```

## 2. Key Configuration Sections

### `system`
- `device`: The compute device to use (`mps`, `cuda`, `cpu`).
- `num_workers`: Number of parallel self-play workers.

### `training`
- `batch_size`: Training batch size.
- `learning_rate`: Initial learning rate for the optimizer.
- `weight_decay`: L2 regularization strength.
- `num_epochs`: Number of training epochs per cycle.
- `ssl_weight`: Weight of the Self-Supervised Learning (SSL) loss.
- `max_grad_norm`: Gradient clipping threshold.

### `mcts`
- `num_simulations`: Number of MCTS simulations per move.
- `cpuct`: Exploration-exploitation trade-off constant.
- `dirichlet_alpha`: Alpha parameter for Dirichlet noise.
- `dirichlet_frac`: Fraction of noise to add to the root node.
- `fpu_reduction`: First Play Urgency (FPU) reduction factor.
- `draw_penalty`: Penalty applied to draw states.

### `model`
- `resnet_channels`: Number of channels in the ResNet backbone.
- `resnet_blocks`: Number of residual blocks.
- `attention_heads`: Number of heads in the attention mechanism.
- `se_ratio`: Squeeze-and-Excitation ratio.

### `data`
- `max_games`: Maximum number of self-play games to store.
- `max_replays`: Maximum number of replay positions to store.
- `backup_interval`: Interval for backing up data (in seconds).
- `compaction_delay`: Delay before compacting old data (in seconds).

## 3. Configuration in Code

The configuration is loaded into a Python object and accessed throughout the codebase.

### Example Usage
```python
# In orchestrator.py
import yaml
from azchess.config import MCTSConfig, ModelConfig

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mcts_config = MCTSConfig(**config["mcts"])
model_config = ModelConfig(**config["model"])

# Accessing parameters
print(f"Number of simulations: {mcts_config.num_simulations}")
print(f"ResNet channels: {model_config.resnet_channels}")
```

## 4. Validation and Best Practices

- **Single Source of Truth**: All configurable parameters should be in `config.yaml`.
- **Avoid Hardcoding**: Do not hardcode values in the Python code.
- **Validation**: The system should validate the configuration at startup to catch errors early.
- **Documentation**: Keep this guide updated with any changes to the configuration.