# Matrix0 Model V2 — Production Architecture & Implementation

Status: ACTIVE TRAINING (step 1000+ completed)
Owner: Matrix0 maintainers
Last updated: 2025-08-25

## 1) Current Production Architecture

Matrix0 V2 is a **53M parameter ResNet-24** model actively training and improving. The architecture is optimized for Apple Silicon MPS with advanced stability features and multi-task SSL learning.

### Key Specifications
- **Total Parameters**: 53,217,919 (53M)
- **Architecture**: ResNet-24 with chess-specific attention
- **Input**: 19×8×8 chess board representation
- **Policy Output**: 4,672 move logits (from-square × to-square)
- **Value Output**: Scalar win probability
- **SSL Output**: 13-class per-square predictions
- **Training Status**: Step 1000+ with stable 2.0s/step performance

### Production Achievements
- **Training Stability**: No NaN/Inf crashes with branch normalization
- **Memory Efficiency**: 14GB MPS limit with automatic management
- **Performance**: ~2.0s per training step on Apple Silicon
- **SSL Integration**: Multi-task learning with progressive curriculum
- **Data Pipeline**: Complete self-play → training → evaluation cycle

## 2) Implemented Architecture Features

### Core Architecture (53M Parameters)
- **Channels**: 320 (increased from 160 for better capacity)
- **Blocks**: 24 (increased from 14 for deeper learning)
- **Attention Heads**: 20 (optimized for chess patterns)
- **Normalization**: GroupNorm (more stable than BatchNorm for MPS)
- **Activation**: SiLU (better gradient flow than ReLU)
- **Residual Blocks**: Pre-activation style for improved training
- **DropPath**: 0.1 rate for regularization and ensemble effect

### Policy Head Architecture
- **Dual Branch Design**: Spatial convolution + dense factorization
- **Spatial Branch**: 73 channels per-square (8×8×73 = 4,672)
- **Dense Branch**: Factorized to 128-rank for parameter efficiency
- **Branch Normalization**: Independent LayerNorm before combination
- **Stability**: Gradient clipping and NaN/Inf detection

### SSL Implementation (Multi-Task Learning)
- **SSL Tasks**: Piece recognition, threat detection, pin identification, fork opportunities, square control
- **SSL Architecture**: Dedicated 13-class per-square prediction head
- **Curriculum Learning**: Progressive difficulty from basic to advanced patterns
- **Loss Integration**: Weighted SSL loss with policy/value learning
- **Training Stability**: SSL warmup and curriculum progression

### Training Stability Features
- **Branch Normalization**: Prevents magnitude differences between policy branches
- **Gradient Clipping**: Aggressive clipping at 0.5 to prevent exploding gradients
- **Memory Management**: 14GB MPS limit with automatic cleanup
- **Mixed Precision**: FP16 training with stability safeguards
- **Emergency Recovery**: Automatic checkpoint saving on errors

## 3) Architecture Blueprint (Current Implementation)

### Inputs (Standard Chess Representation)
- **Board Representation**: 19×8×8 planes (piece types, colors, special states)
- **Memory Format**: Standard contiguous for MPS compatibility
- **Data Type**: FP16 for mixed precision training

### Backbone (ResNet-24 Trunk)
- **Input Channels**: 19 → 320 (stem convolution)
- **Residual Blocks**: 24 blocks with pre-activation design
- **Block Structure**: [GN→SiLU→Conv3×3]×2 + identity skip connection
- **Normalization**: GroupNorm with 16 groups (more stable than BatchNorm)
- **Activation**: SiLU throughout (better gradient flow than ReLU)
- **DropPath**: 0.1 rate, linearly increasing across depth
- **Total Parameters**: ~48M in trunk

### Chess-Specific Attention
- **Attention Heads**: 20 heads (320/16 = 20) every 4th block
- **ChessAttention**: Line-of-sight masking for spatial relationships
- **Relative Bias**: Enhanced positional relationships
- **Unmasked Mix**: 0.1-0.2 blend for knight/tactical patterns
- **Cadence**: Every 4th residual block for computational efficiency
- **Parameters**: ~2M in attention components

### SSL Architecture (Multi-Task Learning)
- **SSL Head**: Dedicated 13-class per-square prediction (13×8×8 output)
- **SSL Tasks**: Piece recognition, threat detection, pin identification, fork opportunities, square control
- **SSL Curriculum**: Progressive difficulty (0.0 → 0.9 over training)
- **Loss Integration**: Weighted SSL loss with policy/value learning
- **Training Stability**: SSL warmup period and curriculum progression
- **Parameters**: ~500K in SSL head

### Policy Head (Dual Branch Design)
- **Input Features**: 320 channels from trunk
- **Branch A (Spatial)**: 1×1 conv → 73 channels per-square → permute/reshape → 4,672 logits
- **Branch B (Dense)**: Global average pool → 320 → FC → 128 → FC → 4,672 logits
- **Branch Normalization**: Independent LayerNorm before combination
- **Combination**: logits = A + B with NaN/Inf protection
- **Stability**: Gradient clipping and numerical safeguards
- **Parameters**: ~2.5M total

### Value Head
- **Architecture**: 1×1 conv (320→64) → flatten → FC → tanh
- **Output**: Scalar win probability (-1 to 1)
- **Parameters**: ~200K

### SSL Head (Multi-Task Learning)
- **Architecture**: 1×1 conv (320→64) → 13×8×8 output
- **SSL Tasks**: 13-class prediction per-square (empty + 12 piece types)
- **Loss Function**: Cross-entropy with SSL curriculum
- **Training**: Progressive difficulty increase during training
- **Parameters**: ~500K

### Outputs (Production Interface)
- **Policy**: (B, 4672) - move logits (from-square × to-square)
- **Value**: (B,) - win probability (-1 to 1)
- **SSL**: (B, 13, 8, 8) - piece predictions per-square
- **Memory Usage**: ~14GB MPS during training

### Training Configuration
- **Batch Size**: 192 (effective 384 with gradient accumulation)
- **Learning Rate**: 0.001 with warmup
- **Precision**: FP16 mixed precision
- **Gradient Clipping**: 0.5 norm threshold
- **SSL Weight**: 0.05 in total loss
- **SSL Warmup**: 200 steps

## 4) Parameter Budget and Efficiency

### Current Model Analysis (53M Parameters)
- **Total Parameters**: 53,217,919 (53.2M) - production model
- **Trunk (ResNet-24)**: ~48M (90.3%) - deep learning capacity
- **Policy Head**: ~2.5M (4.7%) - dual branch efficiency
- **Value Head**: ~200K (0.4%) - lightweight evaluation
- **SSL Head**: ~500K (0.9%) - multi-task learning
- **Chess Attention**: ~2M (3.7%) - spatial relationships

### Production Efficiency
- **Policy Head**: Dual branch design with factorization saves parameters
- **SSL Integration**: Multi-task learning adds minimal overhead
- **Memory Optimized**: 14GB MPS limit enables full training
- **Training Stable**: No NaN/Inf issues with current safeguards

### Efficiency Improvements
- **Parameter Redistribution**: Move compute from policy FC to trunk capacity
- **Multi-Task Learning**: Shared representations across SSL, policy, and value
- **Cross-Modal Efficiency**: Leverage visual and symbolic complementarity

## 5) Enhanced Configuration (Backward-Compatible)

```yaml
model:
  # Core shape
  channels: 192         # default (keep old value to stay on V1)
  blocks: 16            # default (keep old value to stay on V1)
  
  # Norms/activations/layout
  norm: group           # group|batch
  activation: silu      # silu|relu|gelu
  preact: true
  droppath: 0.07
  
  # Attention
  attention_heads: 12
  attention_every_k: 4
  attention_relbias: true
  attention_unmasked_mix: 0.15
  
  # Policy head
  policy_factor_rank: 256         # enable factorized dense; if 0 use legacy FC
  aux_policy_from_square: true
  aux_policy_move_type: true
  
  # Multi-modal features
  enable_visual: false            # enable visual board processing
  visual_encoder_channels: 64     # visual feature channels
  cross_modal_attention: true     # enable cross-modal attention
  
  # Enhanced SSL/SSRL
  ssl_tasks: ["piece"]  # only piece recognition supported currently
  ssl_curriculum: true            # enable progressive difficulty
  ssrl_tasks: ["masked_prediction", "contrastive", "rotation_invariance"]
  
  # LLM integration
  enable_llm_tutor: false        # enable LLM chess tutor
  llm_model_path: "models/gemma3_270m_chess"  # path to fine-tuned model
  llm_context_length: 512        # context window for position analysis

training:
  # SSL enhancements
  ssl_label_smoothing: 0.05
  ssl_task_weights:              # weights for supported SSL tasks
    piece: 1.0
  
  # SSRL configuration
  ssrl_loss_weight: 0.3
  masked_prediction_ratio: 0.15  # fraction of pieces to mask
  
  # Multi-modal training
  visual_loss_weight: 0.2
  cross_modal_consistency: 0.1
  
  # LLM integration
  llm_guidance_weight: 0.1       # weight for LLM strategic insights
  active_learning: false          # enable LLM-guided data generation
  
  # Standard training
  autocast_dtype: bf16            # bf16|fp16
  aux_policy_weights:
    from_square: 0.05
    move_type: 0.05

mcts:
  # Performance improvements
  encoder_cache: true
  legal_softmax: false            # if true, softmax only over legal indices
  tt_cleanup_interval_s: 5        # periodic LRU trims and memory checks
  max_branch_depth: 64            # deep-branch pruning safeguard
  no_instant_backtrack: true
  
  # LLM integration
  llm_move_analysis: false        # use LLM for move quality assessment
  llm_position_evaluation: false  # use LLM for strategic position assessment

# New section for LLM configuration
llm_tutor:
  model_type: "gemma3_270m"
  fine_tuned_path: "models/gemma3_270m_chess"
  max_tokens: 256
  temperature: 0.7
  chess_knowledge_base: "data/chess_literature/"
  training_annotations: true      # generate training data annotations
  strategic_guidance: true        # provide strategic insights during training
```

## 6) Implementation Plan (Enhanced Step-by-Step)

### Phase 1: Core V2 Architecture (Weeks 1-2)
1. **Config Scaffolding**: Add all new configuration options with V1 defaults
2. **Model V2 Trunk**: Implement pre-activation blocks, GroupNorm, SiLU, DropPath
3. **Enhanced Attention**: Parameterize cadence, heads, relative bias
4. **Factorized Policy**: Implement low-rank policy head with fallback to legacy
5. **Basic Testing**: Unit tests and mini integration tests

### Phase 2: Enhanced SSL/SSRL (Weeks 3-4)
1. **Multi-Task SSL**: Implement expanded SSL task set with curriculum
2. **SSRL Tasks**: Masked prediction, contrastive learning, rotation invariance
3. **SSL Curriculum**: Progressive difficulty system
4. **Multi-Task Losses**: Weighted combination of SSL objectives
5. **Validation**: Test SSL accuracy on known positions

### Phase 3: Multi-Modal Learning (Weeks 5-6)
1. **Visual Encoder**: Implement board image processing pipeline
2. **Cross-Modal Attention**: Attention between symbolic and visual features
3. **Multi-Modal Fusion**: Combine visual and symbolic representations
4. **Data Pipeline**: Board image generation and augmentation
5. **Training Integration**: Multi-modal training loop

### Phase 4: LLM Chess Tutor (Weeks 7-8)
1. **Gemma 3 Fine-tuning**: Fine-tune on chess literature and game annotations
2. **Position Encoding**: Convert chess positions to LLM-readable format
3. **Strategic Analysis**: LLM provides move quality and strategic insights
4. **Training Integration**: LLM guidance in training pipeline
5. **Active Learning**: LLM identifies training needs and generates scenarios

### Phase 5: Advanced Features (Weeks 9-10)
1. **Curriculum Learning**: Progressive difficulty from openings to complex positions
2. **Active Learning**: Intelligent data generation based on model uncertainty
3. **Strategic Context**: Integrate LLM insights into model predictions
4. **Performance Optimization**: MPS-specific optimizations and profiling
5. **Comprehensive Testing**: Full pipeline validation

### Phase 6: Rollout and A/B Testing (Weeks 11-12)
1. **V2 Preset**: Enable all V2 features via configuration preset
2. **A/B Testing**: Compare V1 vs V2 on openings phase (1-2k steps)
3. **Metrics Analysis**: Throughput, memory, non-finite occurrences, policy entropy
4. **Promotion**: Promote V2 as default if stable
5. **Documentation**: Update user guides and examples

## 7) Novel Feature Deep-Dive

### Enhanced SSL/SSRL System
**Multi-Task SSL Tasks:**
- **Piece Recognition**: Basic piece identification (current)
- **Piece Relationships**: Which pieces control which squares
- **Threat Detection**: Identify pieces under attack/defense
- **Pawn Structure**: Pawn chains, isolated pawns, passed pawns
- **King Safety**: Recognize safe vs exposed king positions

**SSRL Learning Objectives:**
- **Masked Position Prediction**: Hide random pieces, predict what should be there
- **Contrastive Learning**: Similar positions should have similar representations
- **Rotation/Flip Invariance**: Board orientation shouldn't change game state
- **Temporal Consistency**: Adjacent moves should have similar representations

**SSL Curriculum Progression:**
1. **Level 1**: Basic piece recognition and board state
2. **Level 2**: Piece relationships and basic threats
3. **Level 3**: Pawn structure and king safety
4. **Level 4**: Complex tactical patterns
5. **Level 5**: Strategic concepts and long-term planning

### LLM Chess Tutor Integration
**Fine-tuning Strategy:**
- **Base Model**: Gemma 3 270M (good balance of capability vs size)
- **Training Data**: Chess literature, game annotations, strategic explanations
- **Tasks**: Move quality assessment, strategic analysis, position evaluation
- **Output Format**: Structured analysis for easy integration

**Integration Points:**
- **Training Data Generation**: LLM annotates self-play games
- **Strategic Guidance**: LLM provides insights during training
- **Active Learning**: LLM identifies positions needing more training
- **Move Quality**: LLM rates move quality for training feedback

**Benefits:**
- **Human-Like Understanding**: Strategic thinking beyond pattern recognition
- **Quality Training Data**: Focus on positions with learning value
- **Continuous Improvement**: LLM adapts training based on model weaknesses
- **Strategic Context**: Long-term planning and positional understanding

### Multi-Modal Learning
**Visual Processing Pipeline:**
- **Input**: RGB board images (8×8×3)
- **Encoder**: Convolutional network to extract visual features
- **Fusion**: Cross-attention with symbolic representation
- **Output**: Enhanced features for policy/value/SSL heads

**Cross-Modal Learning:**
- **Symbolic-to-Visual**: Given symbolic board, predict visual appearance
- **Visual-to-Symbolic**: Given visual board, predict symbolic representation
- **Consistency Learning**: Ensure both modalities align
- **Weak Supervision**: Use visual data to improve symbolic understanding

**Data Sources:**
- **Generated Images**: Synthetic board images with different pieces
- **Real Game Screenshots**: Lichess, Chess.com, chess books
- **Augmentation**: Rotation, cropping, lighting adjustments
- **Style Transfer**: Different chess set styles for robustness

## 8) Risk Assessment and Mitigation

### Technical Risks
**MPS Stability Issues:**
- **Risk**: GroupNorm + SiLU combination may have numeric issues
- **Mitigation**: Extensive testing on MPS, fallback to BatchNorm + ReLU
- **Monitoring**: Track non-finite occurrences, implement early detection

**Multi-Modal Complexity:**
- **Risk**: Visual processing adds training complexity and potential instability
- **Mitigation**: Gradual integration, extensive validation, configurable enable/disable
- **Fallback**: Symbolic-only mode if visual processing fails

**LLM Integration Overhead:**
- **Risk**: LLM processing may slow training significantly
- **Mitigation**: Async processing, caching, configurable usage levels
- **Optimization**: Batch LLM processing, efficient position encoding

### Training Risks
**Over-regularization:**
- **Risk**: DropPath + enhanced SSL may over-regularize
- **Mitigation**: Conservative DropPath values, monitor validation loss
- **Adjustment**: Dynamic regularization based on training progress

**Multi-Task Learning Conflicts:**
- **Risk**: SSL tasks may interfere with policy/value learning
- **Mitigation**: Careful task weighting, curriculum progression
- **Monitoring**: Track individual task performance and overall metrics

**Data Quality Issues:**
- **Risk**: LLM-generated annotations may be incorrect
- **Mitigation**: Validation pipeline, human oversight, quality metrics
- **Fallback**: Traditional self-play if LLM quality degrades

## 9) Reversion and Fallback Plans

### Quick Reversion (V1 Compatibility)
```yaml
model:
  channels: 160
  blocks: 14
  norm: batch
  activation: relu
  preact: false
  droppath: 0.0
  policy_factor_rank: 0
  enable_visual: false
  enable_llm_tutor: false
  ssl_tasks: ["piece"]  # basic SSL only (currently the only supported task)
```

### Gradual Fallback Options
1. **Disable Multi-Modal**: Keep V2 trunk but disable visual processing
2. **Disable LLM Tutor**: Keep enhanced SSL but disable LLM integration
3. **Disable Enhanced SSL**: Keep V2 trunk but use basic SSL
4. **Disable V2 Trunk**: Fall back to V1 architecture with factorized policy

### Emergency Reversion
- **Instant Revert**: Set `model.v1_compat: true` in config
- **Checkpoint Rollback**: Load last known good V1 checkpoint
- **Data Rollback**: Use V1-compatible data format

## 10) Success Metrics and Validation

### Technical Metrics
- **Training Stability**: No increase in non-finite occurrences
- **Performance**: Maintain or improve training throughput (steps/s)
- **Memory Usage**: Stay within MPS memory constraints
- **Parameter Efficiency**: Better performance per parameter

### Chess-Specific Metrics
- **Policy Quality**: Improved move prediction accuracy
- **Value Calibration**: Better evaluation of positions
- **SSL Accuracy**: Enhanced understanding of chess concepts
- **Strategic Play**: Improved long-term planning and positional understanding

### Novel Capability Metrics
- **Multi-Modal Learning**: Visual and symbolic representation alignment
- **LLM Integration**: Quality of strategic insights and training guidance
- **Curriculum Learning**: Progressive improvement across difficulty levels
- **Active Learning**: Efficiency of targeted data generation

## 11) Implementation Checklist

### Core V2 Architecture
- [ ] Config scaffolding and documentation
- [ ] Pre-activation residual blocks with GroupNorm + SiLU
- [ ] Enhanced attention system with configurable cadence
- [ ] Factorized policy head with fallback
- [ ] DropPath regularization implementation
- [ ] Unit tests for all new components

### Enhanced SSL/SSRL
- [ ] Multi-task SSL task implementation
- [ ] SSRL learning objectives (masked prediction, contrastive)
- [ ] SSL curriculum progression system
- [ ] Multi-task loss weighting and combination
- [ ] SSL validation and testing framework

### Multi-Modal Learning
- [ ] Visual board encoder implementation
- [ ] Cross-modal attention mechanism
- [ ] Multi-modal fusion layer
- [ ] Data pipeline for board images
- [ ] Training integration and validation

### LLM Chess Tutor
- [ ] Gemma 3 fine-tuning pipeline
- [ ] Position encoding for LLM consumption
- [ ] Strategic analysis integration
- [ ] Training guidance system
- [ ] Active learning implementation

### Advanced Features
- [ ] Curriculum learning system
- [ ] Active learning pipeline
- [ ] Strategic context integration
- [ ] Performance optimization for MPS
- [ ] Comprehensive testing suite

### Rollout and Validation
- [ ] V2 preset configuration
- [ ] A/B testing framework
- [ ] Metrics collection and analysis
- [ ] Documentation updates
- [ ] User guide and examples

## 12) Quick Start for Maintainers

### Enable Full V2 Features
```yaml
model:
  channels: 192
  blocks: 16
  norm: group
  activation: silu
  preact: true
  droppath: 0.07
  attention_heads: 12
  attention_every_k: 4
  attention_relbias: true
  policy_factor_rank: 256
  aux_policy_from_square: true
  aux_policy_move_type: true
  enable_visual: true
  enable_llm_tutor: true
  ssl_tasks: ["piece"]
  ssl_curriculum: true
  ssrl_tasks: ["masked_prediction", "contrastive", "rotation_invariance"]

training:
  ssl_label_smoothing: 0.05
  ssl_task_weights:
    piece: 1.0
  ssrl_loss_weight: 0.3
  visual_loss_weight: 0.2
  llm_guidance_weight: 0.1
  active_learning: true
  autocast_dtype: bf16
  aux_policy_weights: { from_square: 0.05, move_type: 0.05 }

mcts:
  encoder_cache: true
  tt_cleanup_interval_s: 5
  no_instant_backtrack: true
  llm_move_analysis: true
  llm_position_evaluation: true

llm_tutor:
  enable_llm_tutor: true
  model_type: "gemma3_270m"
  fine_tuned_path: "models/gemma3_270m_chess"
  training_annotations: true
  strategic_guidance: true
```

### Testing and Validation
1. **Short Test Run**: 1-2k training steps on openings phase
2. **Monitor Metrics**: Throughput, memory, non-finite occurrences
3. **Validate Outputs**: Policy entropy, value calibration, SSL accuracy
4. **Check Novel Features**: Multi-modal learning, LLM integration
5. **Performance Analysis**: Compare with V1 baseline

### Troubleshooting
- **SSL Issues**: Disable enhanced SSL tasks, fall back to basic
- **Multi-Modal Problems**: Disable visual processing, use symbolic only
- **LLM Integration Failures**: Disable LLM tutor, use traditional training
- **Performance Degradation**: Reduce model complexity, adjust hyperparameters

---

This enhanced V2 design represents a significant evolution of Matrix0, combining architectural improvements with novel learning capabilities. The phased implementation approach ensures stability while enabling cutting-edge features like LLM-guided training and multi-modal learning. All features are configurable and can be enabled/disabled independently, providing maximum flexibility and risk mitigation.

