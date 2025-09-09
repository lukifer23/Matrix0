# GRPO Chess AI Experiments

This directory contains experimental implementations of GRPO (Generalized Reward-based Policy Optimization) for chess AI. The goal is to explore novel RL approaches while maintaining safety by keeping experiments separate from the main Matrix0 codebase.

## 🏗️ **Architecture Overview**

```
experiments/grpo/
├── models/           # Neural network architectures
│   ├── small_resnet.py      # Compact ResNet for quick iteration
│   └── chess_transformer.py # Transformer-based chess model
├── training/         # GRPO training implementations
│   └── grpo_trainer.py      # Core GRPO algorithm
├── data/            # Experiment data and results
├── configs/         # Configuration files
│   └── experiment_configs.yaml
├── scripts/         # Utility scripts
│   └── run_experiment.py
└── results/         # Performance metrics
```

## 🎯 **Key Innovations**

### **1. GRPO Algorithm**
- **Group-based reward normalization** for reduced variance
- **Sparse reward optimization** perfect for chess end-game rewards
- **Sample efficiency** through trajectory grouping

### **2. Architecture Experiments**
- **Small ResNet**: Lightweight CNN for rapid iteration
- **Chess Transformer**: Sequence-based model for long-range dependencies
- **Parameter counts**: 1-5M parameters (vs Matrix0's 53M)

### **3. Experimental Design**
- **Controlled testing** with baseline comparisons
- **Ablation studies** for group size, architecture, hyperparameters
- **Safety first**: Complete isolation from main codebase

## 🚀 **Quick Start**

### **Test Basic Functionality**
```bash
cd experiments/grpo
python scripts/run_experiment.py --experiment small_resnet_grpo --quick-test
```

### **Run Full Experiment**
```bash
python scripts/run_experiment.py --experiment small_transformer_grpo --max-epochs 5
```

### **Compare Architectures**
```bash
# Small ResNet
python scripts/run_experiment.py --experiment small_resnet_grpo --max-epochs 3

# Small Transformer
python scripts/run_experiment.py --experiment small_transformer_grpo --max-epochs 3
```

## 📊 **Available Experiments**

### **Model Size Experiments**
- `small_resnet_grpo`: 64-channel, 4-block ResNet (~1M params)
- `medium_resnet_grpo`: 128-channel, 6-block ResNet (~3M params)
- `small_transformer_grpo`: 128-dim, 4-layer transformer (~2M params)
- `medium_transformer_grpo`: 256-dim, 6-layer transformer (~5M params)

### **Ablation Studies**
- `grpo_group_ablation`: Test group sizes (2, 4, 8)
- `grpo_vs_ppo_comparison`: Compare GRPO vs PPO baselines

## 🔧 **Configuration**

Edit `configs/experiment_configs.yaml` to customize:

```yaml
small_resnet_grpo:
  model:
    base_channels: 64      # Model width
    num_blocks: 4         # Model depth

  grpo:
    group_size: 4         # Trajectories per group
    clip_epsilon: 0.2     # PPO clipping
    learning_rate: 1e-4   # Learning rate

  training:
    num_games_per_epoch: 50  # Games per training epoch
    max_epochs: 10          # Total training epochs
```

## 📈 **Expected Results**

### **Performance Targets**
- **Win Rate**: ≥ 25% (vs random baseline)
- **Sample Efficiency**: ≤ 50% more games than PPO for same performance
- **Training Stability**: ≤ 30% loss variance increase
- **Iteration Speed**: 5-10x faster than full Matrix0

### **Innovation Metrics**
- **Architecture Comparison**: ResNet vs Transformer performance
- **Group Size Impact**: Optimal trajectory grouping
- **Hyperparameter Sensitivity**: Robustness analysis

## 🔬 **Research Questions**

1. **Can GRPO achieve better sample efficiency than PPO in chess?**
2. **Do transformers capture chess patterns better than CNNs?**
3. **What's the optimal group size for chess trajectories?**
4. **How does GRPO scale with model size?**

## ⚠️ **Safety & Best Practices**

### **Isolation**
- ✅ **Separate directory** prevents main codebase contamination
- ✅ **Independent dependencies** avoid version conflicts
- ✅ **Isolated checkpoints** prevent model confusion

### **Monitoring**
- ✅ **Comprehensive logging** for debugging
- ✅ **Performance baselines** for comparison
- ✅ **Early stopping** if instability detected

### **Rollback**
- ✅ **Configurable experiments** for easy modification
- ✅ **Checkpoint system** for recovery
- ✅ **Baseline comparisons** for validation

## 🎪 **Fun Extensions**

### **Other Techniques to Explore**
1. **Reward Shaping**: Add intermediate rewards for piece development
2. **Curriculum Learning**: Start with simple positions, progress to complex
3. **Self-Play Curriculum**: Opponent strength progression
4. **Exploration Strategies**: Different MCTS exploration parameters
5. **Meta-Learning**: Learn-to-learn approaches

### **Hybrid Approaches**
1. **GRPO + SSL**: Reintroduce auxiliary tasks selectively
2. **GRPO + MCTS Variants**: Different tree search algorithms
3. **Multi-Task GRPO**: Policy + value + auxiliary objectives

## 📝 **Next Steps**

1. **Complete GRPO implementation** with MCTS integration
2. **Run baseline comparisons** (GRPO vs PPO)
3. **Test different architectures** (ResNet vs Transformer)
4. **Optimize group size** through ablation studies
5. **Scale up successful approaches**

## 🤝 **Contributing**

This experimental setup encourages:
- **Rapid prototyping** of new ideas
- **Comparative studies** of different approaches
- **Publication-ready research** on chess RL advances
- **Community contributions** to RL methodology

---

*Remember: This is experimental research! The goal is to push boundaries and discover new approaches, not to immediately replace the working Matrix0 system.*
