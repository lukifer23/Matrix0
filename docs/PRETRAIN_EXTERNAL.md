# Pretrain External Tool Documentation

## Overview

The `pretrain_external.py` tool is designed for large-scale pretraining runs using external curated data (Stockfish games, tactical puzzles, openings) with full SSL architecture integration. This tool has been specifically configured for 100,000 step training runs with advanced memory management and monitoring.

## Features

### SSL Architecture Integration
- **7 SSL Tasks**: piece, threat, pin, fork, control, pawn_structure, king_safety
- **Individual Task Weights**: Configurable weights for each SSL task
- **SSL Warmup**: Gradual SSL weight increase over first 500 steps
- **Enhanced SSL Loss**: Multi-task SSL learning with proper loss computation

### Training Configuration
- **Batch Size**: 96 (optimized for memory efficiency)
- **Learning Rate**: 0.001 with cosine decay scheduling
- **Gradient Clipping**: 0.5 norm with every-step clipping
- **EMA**: 0.999 decay for model weight smoothing
- **Mixed Precision**: FP16 training with MPS optimization

### Memory Management
- **Advanced Monitoring**: Real-time memory usage tracking
- **Automatic Cleanup**: Periodic memory cache clearing every 1000 steps
- **Memory Alerts**: Warning and critical thresholds with automatic responses
- **MPS Optimization**: Apple Silicon specific optimizations

### Long-Term Training Support
- **Heartbeat Monitoring**: Progress logging every 5 minutes
- **Checkpoint Saving**: Intermediate saves every 5000 steps
- **Resume Capability**: Auto-resume from latest checkpoint
- **Progress Tracking**: Real-time ETA and loss monitoring

## Usage

### Quick 40K Step Training Run
```bash
python -m azchess.tools.pretrain_external \
  --config config.yaml \
  --steps 40000 \
  --batch-size 96 \
  --lr 0.001 \
  --weight-decay 1e-4 \
  --ema-decay 0.999 \
  --ssl-weight 0.15 \
  --ssl-warmup-steps 500 \
  --checkpoint-in checkpoints/enhanced_best.pt \
  --checkpoint-out checkpoints/pretrained_40k.pt \
  --checkpoint-prefix pretrained_40k \
  --save-every 5000 \
  --use-amp \
  --curriculum stockfish \
  --auto-resume \
  --progress-interval 100 \
  --grad-clip-norm 0.5 \
  --accum-steps 1 \
  --ssl-every-n 1 \
  --ssl-chunk-size 128
```

Teacher data is included by default as part of the curriculum phases (40% teacher when available).

### Basic 100K Step Training Run
```bash
python -m azchess.tools.pretrain_external \
  --config config.yaml \
  --steps 100000 \
  --batch-size 96 \
  --lr 0.001 \
  --weight-decay 1e-4 \
  --ema-decay 0.999 \
  --ssl-weight 0.15 \
  --ssl-warmup-steps 500 \
  --checkpoint-in checkpoints/enhanced_best.pt \
  --checkpoint-out checkpoints/pretrained_100k.pt \
  --save-every 5000 \
  --use-amp \
  --curriculum stockfish \
  --progress-interval 100 \
  --grad-clip-norm 0.5
```

### Key Parameters

#### SSL Configuration
- `--ssl-weight 0.15`: Overall SSL loss weight
- `--ssl-warmup-steps 500`: SSL weight warmup duration
- `--ssl-piece-weight 1.0`: Piece recognition task weight
- `--ssl-threat-weight 0.8`: Threat detection task weight
- `--ssl-pin-weight 0.7`: Pin detection task weight
- `--ssl-fork-weight 0.6`: Fork detection task weight
- `--ssl-control-weight 0.5`: Square control task weight
- `--ssl-pawn-structure-weight 0.4`: Pawn structure task weight
- `--ssl-king-safety-weight 0.4`: King safety task weight

#### Training Configuration
- `--steps 100000`: Total training steps
- `--batch-size 96`: Training batch size
- `--lr 0.001`: Learning rate
- `--weight-decay 1e-4`: Weight decay
- `--ema-decay 0.999`: EMA decay rate
- `--grad-clip-norm 0.5`: Gradient clipping norm

#### Monitoring and Saving
- `--save-every 5000`: Save intermediate checkpoints (now includes EMA)
- `--progress-interval 100`: Progress update frequency
- `--checkpoint-prefix pretrained_100k`: Checkpoint naming prefix

#### Advanced Throughput/Memory Controls
- `--accum-steps`: Gradient accumulation (effective batch size = batch × accum)
- `--ssl-every-n`: Compute SSL every N steps (e.g., 2 to halve SSL cost)
- `--ssl-chunk-size`: Chunk SSL computation to reduce peak memory

## Data Sources

### Stockfish Curriculum
- **Openings**: 39,286 steps (22% of training)
- **King Safety**: 21,429 steps (12% of training)
- **Endgames**: 39,285 steps (22% of training)
- **Tactical**: Mixed throughout phases
- **Positional**: Integrated in all phases

### External Data
- **Tactical Puzzles**: 10,000 samples
- **Opening Positions**: 5,000 samples
- **Stockfish Games**: 8 shards imported

## Performance Expectations

### Training Speed
- **Steps per Second**: ~0.65-0.67 (1.5-1.6 seconds per step)
- **Total Time**: ~42-44 hours for 100K steps
- **Memory Usage**: ~2.17GB MPS memory

### Loss Progression
- **Policy Loss**: Starts ~2.8, decreases to ~2.4 over first 500 steps
- **Value Loss**: Stable at 0.5 (Huber threshold) initially, decreases after 1000 steps
- **SSL Loss**: Fluctuates 2.2-2.6, shows active learning across all tasks

### Checkpoint Management
- **Intermediate Saves**: Every 5000 steps (includes EMA and optimizer/scheduler state)
- **Final Checkpoint**: `pretrained_100k.pt` (or `pretrained_40k.pt` for shorter runs)
- **Resume Support**: Automatic detection of latest checkpoint

## Monitoring

### Real-Time Metrics
- **Loss Components**: Policy, value, SSL losses
- **Memory Usage**: MPS memory consumption
- **Learning Rate**: Current LR with warmup progress
- **ETA**: Estimated time to completion

### Heartbeat Logging
Every 5 minutes, the system logs:
- Current step and progress
- All loss components
- Memory usage
- Learning rate
- Device information

### SSL Task Monitoring
- **Piece Recognition**: ~1.0 loss (13-class classification)
- **Threat Detection**: ~0.66 loss (binary classification)
- **Pin Detection**: ~0.003 loss (rare events)
- **Fork Detection**: ~0.02 loss (tactical patterns)
- **Control Detection**: ~0.8 loss (positional understanding)

## Troubleshooting

### Common Issues
1. **Memory Warnings**: Normal during SSL computation, automatic cleanup handles
2. **Value Loss at 0.5**: Expected behavior with Huber loss, decreases after 1000 steps
3. **SSL Loss Fluctuation**: Normal as model learns different tactical patterns

### Performance Optimization
- **Batch Size**: Reduce if memory issues occur
- **SSL Chunk Size**: Increase for better memory efficiency
- **Save Frequency**: Adjust based on checkpoint storage needs

## Integration

### With Main Training Pipeline
- **Checkpoint Compatibility**: Seamless integration with enhanced_best.pt
- **SSL Architecture**: Full compatibility with 7-task SSL system
- **Model Format**: Standard checkpoint format for orchestrator integration

### Evaluation
After training completion:
```bash
python -m azchess.tools.enhanced_eval \
  --model-a checkpoints/pretrained_100k.pt \
  --model-b checkpoints/enhanced_best.pt
```

## Status

**Current Status**: 100K step training run in progress
**Start Time**: December 2024
**Expected Completion**: ~42-44 hours
**Progress**: Real-time monitoring via heartbeat logs
