# Configuration Guide

The Matrix0 project uses a comprehensive `config.yaml` file to manage all parameters for the training pipeline, MCTS, model architecture, and data management. This guide provides detailed documentation of the current configuration structure.

## 1. Configuration Overview

The `config.yaml` file is the single source of truth for all system parameters. It is loaded at startup by the orchestrator and passed to all components.

### Recent Configuration Updates (v2.2)

**SSL Task Optimization**: Reduced from 7 to 5 SSL tasks for improved stability and performance
- ✅ **Active Tasks**: piece, threat, pin, fork, control detection
- ❌ **Removed Tasks**: pawn_structure, king_safety (due to data availability issues)

**Data Pipeline Fixes**: Resolved configuration conflicts and duplicate keys
- ✅ **Duplicate Key Removal**: All duplicate configuration entries eliminated
- ✅ **SSL Task Alignment**: Configuration matches actual implemented SSL tasks
- ✅ **Memory Optimization**: Settings optimized for stable SSL training within MPS limits

**EX0Bench Integration**: New external engine benchmarking configuration options
- ✅ **External-Only Mode**: CPU-only external engine battles without Matrix0
- ✅ **Automatic Detection**: Smart detection of pure external engine scenarios
- ✅ **Performance Benefits**: Faster startup, lower memory for external comparisons

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
  ssl_tasks: ["piece", "threat", "pin", "fork", "control"]  # SSL tasks (optimized 5-task configuration)
```

### Advanced Benchmark Configuration

The Matrix0 benchmark system supports comprehensive multi-engine evaluation with SSL performance tracking:

```yaml
# Benchmark Scenarios Configuration
scenarios:
  - name: "LC0_Matrix0_Showdown"
    description: "Direct competition with neural network engine"
    model_checkpoint: "checkpoints/v2_base.pt"
    engines: ["lc0_strong"]
    num_games: 50
    time_control: "30+0.3"
    max_moves: 150
    concurrency: 4
    ssl_tracking: true
    mcts_sims: 800

  - name: "Multi_Engine_Tournament"
    description: "Tournament-style evaluation"
    model_checkpoint: "checkpoints/v2_base.pt"
    engines: ["stockfish_medium", "stockfish_strong", "lc0_medium"]
    tournament_format: "round_robin"
    num_games_per_pairing: 10
    time_control: "45+0.5"
    ssl_tracking: true

# SSL Performance Tracking
ssl_config:
  enabled: true
  loss_weight: 0.04
  track_individual_heads: true
  heads_to_monitor: ["threat", "pin", "fork", "control", "piece"]
  learning_analysis: true
  convergence_tracking: true

# Apple Silicon Performance Monitoring
performance:
  track_cpu: true
  track_memory: true
  track_gpu: true
  track_mps: true          # MPS monitoring for Apple Silicon
  sample_interval: 0.5
  log_system_load: true
```

### Engine Configurations

#### Stockfish Configuration
```yaml
engines:
  stockfish_weak:
    command: "/opt/homebrew/bin/stockfish"
    options:
      Threads: "4"
      Hash: "512"
      Skill Level: "8"      # Club level (~1800 ELO)
      UCI_LimitStrength: "true"
      UCI_Elo: "1800"

  stockfish_strong:
    command: "/opt/homebrew/bin/stockfish"
    options:
      Threads: "4"
      Hash: "512"
      Skill Level: "20"     # Full strength
      UCI_LimitStrength: "false"
```

#### LC0 Configuration (Apple Silicon Optimized)
```yaml
engines:
  lc0_strong:
    command: "/opt/homebrew/bin/lc0"
    options:
      Threads: "4"
      NNCacheSize: "2000000"
      MinibatchSize: "32"
      Backend: "metal"        # Apple Silicon Metal optimization
      Blas: "true"           # Enable BLAS acceleration
      CPuct: "1.745000"      # Optimized for LC0
      MaxPrefetch: "32"
      RamLimitMb: "0"        # No RAM limit
      MoveOverheadMs: "200"
      TimeManager: "legacy"
      MultiPV: "1"
```

### External Engine & Stockfish Data Defaults

Add engines configuration and register Stockfish-generated datasets for training:

```yaml
engines:
  stockfish:
    command: /opt/homebrew/bin/stockfish  # Auto-detected path
    options:
      Threads: 4
      Hash: 512
      Skill Level: 20
      Hash: 256
      MultiPV: 1
    time_control: 100ms
    enabled: true
  matrix0:
    type: internal
    checkpoint: checkpoints/best.pt
    enabled: true

training:
  extra_replay_dirs:
    - data/stockfish_games
```

With `extra_replay_dirs` set as above, NPZ shards produced by `tools/generate_stockfish_data.py` under `data/stockfish_games/**` are automatically registered and sampled during training alongside self-play data.

## 2. Recent Configuration Enhancements (August 2025)

### Scheduler Robustness
- Fixed learning rate scheduler stepping issues with gradient accumulation
- Added scheduler error handling and logging
- Ensured scheduler steps only once per accumulation window

### Enhanced Memory Management
- Improved memory monitoring with configurable thresholds
- Added periodic proactive memory cleanup
- Enhanced MPS memory tracking and reporting

### SSL Algorithm Integration
Advanced SSL algorithms are now integrated into the training pipeline:
- **Piece Recognition**: Basic piece identification (working)
- **Threat Detection**: Identifies squares under attack
- **Pin Detection**: Identifies pinned pieces
- **Fork Detection**: Identifies fork opportunities
- **Square Control**: Determines who controls each square
- **Multi-task Training**: All SSL tasks trained simultaneously with configurable weights

The training pipeline now includes smart policy masking that automatically detects data source types:
- **External Data**: One-hot distributions (e.g., lichess puzzles) - no masking needed
- **Self-Play Data**: Soft MCTS distributions - applies policy masking for legal moves
- **Legal Mask Handling**: Improved alignment of legal masks with targets to prevent mask misalignment

### Intensive Self-Play Configuration
- **Workers**: Increased from 2 to 3 for parallel generation
- **Games per Cycle**: Increased from 12 to 750 for robust training dataset
- **Game Length**: Extended to 160 moves to reduce early draws
- **Resignation**: Stricter threshold (-0.98) and longer minimum plies (50)

### Training Pipeline Optimization
- **Batch Size**: Reduced to 128 for MPS memory headroom
- **Epochs**: Single epoch per cycle with extended steps (8000)
- **Gradient Accumulation**: Reduced to 1 step for MPS compatibility
- **SSL Chunk Size**: Reduced to 16 to prevent OOM

## 3. Key Configuration Sections

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
- `ssl_tasks`: List of SSL tasks to train (basic piece recognition + advanced algorithms: threat, pin, fork, control detection)

### `selfplay` - Self-Play Data Generation
- `num_workers`: Number of parallel self-play workers (default: `3`)
- `batch_size`: Self-play batch size (default: `128`)
- `max_games`: Maximum games per self-play cycle (default: `750`)
- `max_game_len`: Maximum game length in moves (default: `160`)
- `min_resign_plies`: Minimum plies before resignation (default: `50`)
- `resign_threshold`: Win probability threshold for resignation (default: `-0.98`)
- `num_simulations`: MCTS simulations per self-play move (default: `160`)

### `training` - Training Pipeline Settings
- `batch_size`: Training batch size (default: `128`)
- `epochs`: Number of training epochs per cycle (default: `1`)
- `learning_rate`: Initial learning rate (default: `0.001`)
- `weight_decay`: L2 regularization strength (default: `0.0001`)
- `gradient_accumulation_steps`: Effective batch size multiplier (default: `1`)
- `grad_clip_norm`: Gradient clipping threshold (default: `0.5`)
- `ssl_weight`: Weight of SSL loss in total loss (default: `0.05`)
- `ssl_warmup_steps`: Steps to gradually increase SSL weight (default: `200`)
- `ssl_chunk_size`: Process SSL in chunks to prevent OOM (default: `16`)
- `precision`: Training precision (`"fp16"` or `"fp32"`, default: `"fp16"`)
- `memory_limit_gb`: MPS memory limit in GB (default: `14`)
- `steps_per_epoch`: Extended training per cycle (default: `8000`)
- `policy_masking`: Smart policy masking (default: `true`)

### `orchestrator` - Training Coordination
- `initial_games`: Games for first training cycle (default: `750`)
- `subsequent_games`: Games for subsequent cycles (default: `750`)
- `games_per_cycle`: Total games per cycle (default: `750`)
- `train_epochs_per_cycle`: Training epochs per cycle (default: `1`)
- `eval_games_per_cycle`: Evaluation games per cycle (default: `12`)
- `keep_top_k`: Number of best models to keep (default: `1`)
- `continuous_mode`: Single run vs continuous training (default: `false`)

### `eval` - Model Evaluation Settings
- `games`: Number of evaluation games (default: `4`)
- `num_simulations`: MCTS simulations for evaluation (default: `800`)
- `max_moves`: Maximum moves per evaluation game (default: `80`)
- `external_engines`: List of engines to compare against (default: `["stockfish"]`)
- `tournament_rounds`: Tournament rounds (default: `4`)

### `mcts` - Monte Carlo Tree Search Parameters
- `num_simulations`: MCTS simulations per move (default: `96`)
- `cpuct`: Exploration-exploitation constant (default: `2.2`)
- `cpuct_start`: Initial cpuct value (default: `2.8`)
- `cpuct_end`: Final cpuct value (default: `1.8`)
- `batch_size`: Smaller per-worker inference batches on MPS (default: `8`)
- `num_threads`: Fewer threads reduces contention on MPS (default: `2`)
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
- **SSL Safety**: Basic SSL working, advanced algorithms ready for integration

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
  ssl_tasks: ["piece", "threat", "pin", "fork", "control"]  # All 5 SSL tasks enabled
  ssl_warmup_steps: 200            # Gradual SSL introduction

training:
  ssl_weight: 0.05                 # SSL loss weight
  ssl_warmup_steps: 200            # SSL warmup period
  ssl_chunk_size: 32               # Chunked SSL processing
```

**Note**: Currently only basic piece recognition is working. Advanced SSL algorithms (threat detection, pin detection, fork opportunities, square control) are implemented in `ssl_algorithms.py` but not yet integrated with the training pipeline.

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
  num_workers: 2                   # Parallel data generation
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

## 6. SSL Status and Future Enhancements

### Current SSL Implementation
- **Basic Piece Recognition**: ✅ Working and operational
- **Advanced SSL Algorithms**: ✅ Implemented in `ssl_algorithms.py`
- **Training Integration**: 🔄 Ready for integration
- **Multi-Task Learning**: 🔄 Ready for implementation

### Advanced SSL Tasks (Implemented, Ready for Integration)
- **Threat Detection**: Identify squares under attack/defense
- **Pin Detection**: Identify pinned pieces
- **Fork Detection**: Identify fork opportunities
- **Square Control**: Identify who controls each square
- **Pawn Structure**: Pawn chains, isolated pawns, passed pawns
- **King Safety**: Safe vs exposed king positions

### SSL Integration Roadmap
1. **Phase 1**: Integrate advanced SSL algorithms with training pipeline
2. **Phase 2**: Enable multi-task SSL learning with curriculum progression
3. **Phase 3**: Validate SSL effectiveness across all implemented tasks
4. **Phase 4**: Optimize SSL performance and memory usage

### Current SSL Configuration (Optimized)
```yaml
# Current SSL configuration (optimized 5-task setup)
model:
  ssl_tasks: ["piece", "threat", "pin", "fork", "control"]
  ssl_curriculum: true
  ssl_task_weights:
    piece: 1.0
    threat: 0.8
    pin: 0.6
    fork: 0.4
    control: 0.7

training:
  ssl_weight: 0.1                  # Increased SSL weight for multi-task learning
  ssl_warmup_steps: 500            # Longer warmup for complex SSL tasks
  ssl_chunk_size: 64               # Larger chunks for advanced SSL processing
```

This configuration will be enabled once the SSL integration is complete and all advanced SSL tasks are working with the training pipeline.
