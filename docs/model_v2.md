# Matrix0 Model V2 — Enhanced Design, Novel Features, and Strategic Evolution

Status: Enhanced Vision (approved to implement)
Owner: Matrix0 maintainers
Last updated: 2025-08-20

## 1) Goals and Constraints

- **Core Stability**: Improve stability and throughput on Apple Silicon (MPS) without changing external interfaces
- **Parameter Efficiency**: Keep total parameters < 30M, maintain self-play/training/eval compatibility
- **Interface Preservation**: Preserve inputs (19×8×8), outputs (policy 4672, value scalar, SSL 13×8×8)
- **Backward Compatibility**: Maintain SSL, attention, and MPS autocast; keep data and move mapping unchanged
- **Risk Mitigation**: Implement behind config toggles with safe defaults and instant revert path
- **Novel Capabilities**: Integrate LLM chess tutor, enhanced SSL/SSRL, and multi-modal learning

## 2) Summary of Key Improvements

### Core Architecture Enhancements
- **Norm/Activation**: GroupNorm + SiLU (configurable) for better AMP/MPS stability
- **Residual Layout**: Pre-activation blocks for improved gradient flow
- **Attention**: Heads tuned to head_dim=16; cadence every 4th block; relative bias on
- **Policy Head**: Factorized dense (low-rank) to replace brittle huge FC; retain per-square conv branch
- **Optional Auxiliary Policy Heads**: from-square (64) and move-type (73) with small loss weights
- **SSL**: Enhanced with label smoothing; logits ensured standard-contiguous for CE backward on MPS
- **Regularization**: Mild DropPath (stochastic depth) across residual blocks
- **Numerics**: Train in standard contiguous memory format; channels_last reserved for inference

### Novel Learning Capabilities
- **Enhanced SSL/SSRL**: Multi-task self-supervised learning with curriculum progression
- **LLM Chess Tutor**: Integration with fine-tuned Gemma 3 270M for strategic guidance
- **Multi-Modal Learning**: Visual board processing combined with symbolic representation
- **Active Learning**: Intelligent data generation based on model uncertainty and LLM insights
- **Curriculum Learning**: Progressive difficulty from openings to complex middlegame positions

### MCTS Improvements
- **Encoder Caching**: Optimized move encoding for performance
- **Optional Legal-Only Softmax**: Improved move selection with validation
- **TT Cleanup Cadence**: Periodic LRU trims and memory checks
- **Backtracking Avoidance**: Prevent instant backtracking at root for stability

## 3) V2 Architecture Blueprint

### Inputs: Enhanced Multi-Modal
- **Primary**: 19×8×8 symbolic planes (unchanged)
- **Optional Visual**: RGB board images (8×8×3) for multi-modal training
- **LLM Context**: Strategic annotations and position analysis

### Backbone (Trunk)
- **Channels**: 192 (from 160) - increased capacity for multi-modal learning
- **Blocks**: 16 (from 14) - deeper network for complex pattern recognition
- **Residual Blocks**: Pre-activation style: [GN→SiLU→Conv3×3]×2 + identity
- **Normalization**: GroupNorm(groups=16) or BatchNorm (config `model.norm: group|batch`)
- **Activation**: SiLU (config `model.activation: silu|relu|gelu`)
- **DropPath**: 0.07 linearly across depth (config `model.droppath`)

### Chess Features
- **PST**: 1×1 + interaction conv (3×3) + learnable 8×8 positional embedding
- **Multi-Modal Fusion**: Visual encoder branch with cross-attention to symbolic features

### Enhanced SSL/SSRL Architecture
- **Multi-Task SSL**: Piece recognition, relationships, threats, pawn structure, king safety
- **SSRL Tasks**: Masked position prediction, contrastive learning, rotation invariance
- **SSL Curriculum**: Progressive difficulty from basic to advanced concepts
- **Cross-Modal SSL**: Symbolic-to-visual and visual-to-symbolic prediction tasks

### Attention System
- **ChessAttention**: Line-of-sight mask retained with enhanced capabilities
- **Configuration**: `attention_heads: 12` (C=192 => head_dim=16), `attention_relbias: true`
- **Cadence**: Every 4th block; blend unmasked mix ≈ 0.1–0.2
- **Multi-Modal Attention**: Cross-attention between symbolic and visual representations

### Policy Head
- **Shared Trunk**: 1×1 conv trunk: C→64; GN + SiLU + dropout
- **Branch A (Spatial)**: Conv1×1→73 per square; flatten → 4672
- **Branch B (Dense)**: Low-rank factorization 4096→K→4672 (K=256–384; default 256)
- **Combine**: logits = A + B (shape 4672), unchanged externally
- **Optional Auxiliary Heads**: From-square (64 logits) and move-type (73 logits), CE losses with small weights (0.05–0.1)

### Value Head
- **Architecture**: 1×1 conv (C→64) → flatten (4096) → MLP: 4096→C→C/2→1 with SiLU + dropout; tanh at end
- **Multi-Modal Input**: Enhanced with visual and strategic context features

### SSL Head
- **Enhanced Output**: 13-class per-square logits (13×8×8) with expanded task set
- **Label Smoothing**: 0.05–0.1 for improved training stability
- **Multi-Task Losses**: Weighted combination of all SSL tasks
- **Standard Contiguous**: Ensured before CE for MPS compatibility

### LLM Integration Layer
- **Chess Tutor**: Fine-tuned Gemma 3 270M for strategic analysis
- **Input Processing**: Position encoding for LLM consumption
- **Output Integration**: Strategic insights fed back into training pipeline
- **Active Learning**: LLM identifies training needs and generates targeted scenarios

### Outputs (Enhanced)
- **Policy**: (B, 4672) - unchanged externally
- **Value**: (B,) - enhanced with strategic context
- **SSL**: (B, 13, 8, 8) - expanded task set
- **Strategic Context**: (B, C) - LLM-derived strategic features

## 4) Parameter Budget and Efficiency

### Current Model Analysis
- **Total Parameters**: ~27.3M
- **Policy FC Dominance**: 19.1M (70% of total) - major bottleneck
- **Trunk Capacity**: 160×14 = 2,240 channels - limited for complex learning

### V2 Parameter Distribution
- **Factorized Policy**: K=256 saves ~14.6M parameters
- **Enhanced Trunk**: 192×16 = 3,072 channels (+37% capacity)
- **Multi-Modal Encoder**: ~2M parameters for visual processing
- **LLM Integration**: ~1M parameters for strategic context
- **Total Estimate**: 22–26M parameters (within 30M constraint)

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
  ssl_tasks: ["piece", "relationship", "threat", "pawn_structure", "king_safety"]
  ssl_curriculum: true            # enable progressive difficulty
  ssrl_tasks: ["masked_prediction", "contrastive", "rotation_invariance"]
  
  # LLM integration
  enable_llm_tutor: false        # enable LLM chess tutor
  llm_model_path: "models/gemma3_270m_chess"  # path to fine-tuned model
  llm_context_length: 512        # context window for position analysis

training:
  # SSL enhancements
  ssl_label_smoothing: 0.05
  ssl_task_weights:              # weights for multi-task SSL
    piece: 1.0
    relationship: 0.8
    threat: 0.9
    pawn_structure: 0.7
    king_safety: 0.8
  
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
  ssl_tasks: ["piece"]  # basic SSL only
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
  ssl_tasks: ["piece", "relationship", "threat", "pawn_structure", "king_safety"]
  ssl_curriculum: true
  ssrl_tasks: ["masked_prediction", "contrastive", "rotation_invariance"]

training:
  ssl_label_smoothing: 0.05
  ssl_task_weights:
    piece: 1.0
    relationship: 0.8
    threat: 0.9
    pawn_structure: 0.7
    king_safety: 0.8
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

