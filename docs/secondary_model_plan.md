# Secondary Optimized Model Plan

## Overview

This document outlines the plan for creating a **secondary optimized model** trained exclusively on curated Stockfish games with perfect annotations. This specialist model will complement the primary generalist Matrix0 model, creating a hybrid system with both broad chess knowledge and perfect tactical accuracy.

## Strategic Rationale

### Why a Secondary Model Makes Sense

1. **Perfect Training Data**: Stockfish provides optimal moves and evaluations
2. **Specialized Expertise**: Focus on tactical patterns where engines excel
3. **Computational Efficiency**: Smaller model for specific domains
4. **Knowledge Distillation**: Transfer perfect understanding to primary model

### Hybrid Architecture Benefits

- **Primary Model (53M)**: General chess knowledge, self-play + external data
- **Secondary Model (24M)**: Perfect tactical analysis, Stockfish games only
- **Ensemble Evaluation**: Combine strengths for maximum performance
- **Adaptive Selection**: Use specialist for complex tactical positions

## Model Architecture Specifications

### Secondary Model Configuration

```yaml
# Secondary Model - Tactical Specialist
model_secondary:
  # Smaller, focused architecture
  channels: 192                    # Reduced from 320 (40% smaller)
  blocks: 16                       # Reduced from 24 (33% smaller)
  attention_heads: 12              # Reduced from 20 (40% smaller)

  # Optimized for tactical analysis
  planes: 19                       # Same input representation
  policy_size: 4672                # Same output size
  norm: "group"                    # Same normalization
  activation: "silu"               # Same activation

  # Focused SSL tasks (tactical only)
  attention: true                  # Enable attention for patterns
  self_supervised: true
  ssl_tasks: ["threat", "pin", "fork"]  # Tactical focus only
  ssl_curriculum: true

  # Memory optimized
  droppath: 0.1                   # Higher regularization
  preact: true                     # Pre-activation blocks

  # Tactical-specific features
  chess_features: true
  piece_square_tables: true
  aux_policy_from_square: true     # Critical for tactics
  aux_policy_move_type: true       # Move classification
```

### Expected Model Size

- **Parameters**: ~24M (55% smaller than primary)
- **Memory Footprint**: ~0.09 GB FP16 (55% smaller)
- **Training Speed**: ~80% faster convergence
- **Inference Speed**: ~40% faster evaluation

## Training Strategy

### Phase 1: Foundation (Easy Stockfish Games)
**Duration**: 1-2 weeks
**Data**: Stockfish games with depth 8-12 analysis
**Focus**: Learn basic tactical patterns

```yaml
training_secondary_phase1:
  batch_size: 512                  # Large batches for efficiency
  learning_rate: 0.002             # Higher LR for fast learning
  ssl_weight: 0.1                  # Heavy SSL focus
  curriculum_phases:
    - { name: easy_threats, steps: 2000, description: "Basic threat detection" }
    - { name: easy_pins,    steps: 2000, description: "Pin recognition" }
    - { name: easy_forks,   steps: 2000, description: "Fork patterns" }
```

### Phase 2: Intermediate (Complex Positions)
**Duration**: 2-3 weeks
**Data**: Stockfish games with depth 15-20 analysis
**Focus**: Complex tactical combinations

```yaml
training_secondary_phase2:
  batch_size: 256                  # Moderate batch size
  learning_rate: 0.001             # Standard learning rate
  ssl_weight: 0.08                 # Balanced SSL focus
  curriculum_phases:
    - { name: complex_threats, steps: 3000, description: "Multi-threat positions" }
    - { name: tactical_combos, steps: 3000, description: "Combined tactics" }
    - { name: critical_moments, steps: 2000, description: "Game-deciding positions" }
```

### Phase 3: Mastery (Expert Positions)
**Duration**: 1-2 weeks
**Data**: Curated critical tactical positions
**Focus**: Perfect tactical accuracy

```yaml
training_secondary_phase3:
  batch_size: 128                  # Smaller for precision
  learning_rate: 0.0005            # Fine-tuning rate
  ssl_weight: 0.05                 # Precise SSL focus
  curriculum_phases:
    - { name: master_threats, steps: 2000, description: "Expert threat analysis" }
    - { name: perfect_defense, steps: 2000, description: "Perfect defense patterns" }
    - { name: winning_combos, steps: 2000, description: "Winning tactical sequences" }
```

## Data Pipeline

### Stockfish Game Generation

#### Script: `tools/generate_stockfish_games.py`

```python
class StockfishGameGenerator:
    def __init__(self, stockfish_path: str = "stockfish",
                 min_depth: int = 8, max_depth: int = 20):

        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Threads": 8, "Hash": 1024})

    def generate_tactical_dataset(self, num_games: int = 10000,
                                difficulty: str = "intermediate"):

        positions = []
        difficulties = {
            "easy": (8, 12),
            "intermediate": (12, 16),
            "hard": (16, 20),
            "expert": (18, 25)
        }

        min_d, max_d = difficulties[difficulty]

        for game_id in range(num_games):
            # Generate or load game
            game = self.generate_game(min_d, max_d)

            # Extract critical positions
            critical_positions = self.extract_critical_positions(game)

            for pos in critical_positions:
                # Analyze with Stockfish
                analysis = self.analyze_position(pos, depth=max_d)

                # Create training sample
                sample = self.create_training_sample(pos, analysis)
                positions.append(sample)

        return positions
```

#### Data Categories

1. **Opening Positions**: Critical opening decisions
2. **Middlegame Tactics**: Complex tactical battles
3. **Endgame Technique**: Perfect endgame play
4. **Defensive Mastery**: Perfect defense patterns
5. **Attacking Brilliance**: Winning attack sequences

### Data Format

#### NPZ Structure
```python
sample = {
    's': board_state,              # (19, 8, 8) board representation
    'pi': policy_targets,          # (4672,) move probabilities
    'z': value_target,             # Scalar game outcome
    'legal_mask': legal_moves,     # (4672,) legal move mask

    # Perfect SSL annotations
    'ssl_threat': threat_map,      # (8, 8) threat detection
    'ssl_pin': pin_map,            # (8, 8) pin detection
    'ssl_fork': fork_map,          # (8, 8) fork opportunities

    # Metadata
    'stockfish_eval': eval_cp,     # Centipawn evaluation
    'depth': analysis_depth,       # Stockfish analysis depth
    'difficulty': difficulty_score, # 0.0-1.0 difficulty
    'category': position_category, # opening/middlegame/endgame
    'tactical_theme': theme        # pin/fork/threat/defense/attack
}
```

## Integration Strategy

### Ensemble Evaluation System

#### Implementation: `azchess/ensemble_evaluator.py`

```python
class EnsembleEvaluator:
    def __init__(self, primary_model, secondary_model):
        self.primary = primary_model
        self.secondary = secondary_model

    def evaluate_position(self, board: chess.Board,
                         position_complexity: float) -> dict:

        # Primary model evaluation (always)
        primary_result = self.primary.evaluate(board)

        # Secondary model evaluation (for complex positions)
        if position_complexity > 0.7:
            secondary_result = self.secondary.evaluate(board)
            combined_result = self.combine_results(
                primary_result, secondary_result, position_complexity
            )
            return combined_result

        return primary_result

    def combine_results(self, primary, secondary, complexity):
        """Combine primary and secondary model results"""

        # Weight secondary model more for complex positions
        secondary_weight = min(complexity * 2, 1.0)
        primary_weight = 1.0 - secondary_weight

        # Weighted policy combination
        combined_policy = (
            primary_weight * primary['policy'] +
            secondary_weight * secondary['policy']
        )

        # Take best value estimate
        combined_value = max(primary['value'], secondary['value'])

        return {
            'policy': combined_policy,
            'value': combined_value,
            'from_primary': primary_weight,
            'from_secondary': secondary_weight
        }
```

### Adaptive Model Selection

#### Implementation: `azchess/adaptive_selector.py`

```python
class AdaptiveSelector:
    def __init__(self):
        self.complexity_detector = PositionComplexityDetector()

    def should_use_secondary(self, board: chess.Board,
                           move_history: List[chess.Move]) -> bool:

        # Analyze position complexity
        complexity = self.complexity_detector.analyze(board, move_history)

        # Tactical complexity indicators
        has_threats = self.has_tactical_threats(board)
        has_pins = self.has_pins_or_forks(board)
        is_critical = self.is_critical_position(board, move_history)

        # Use secondary model if position is complex
        return complexity > 0.6 or has_threats or has_pins or is_critical
```

## Performance Expectations

### Training Performance

#### Phase 1 Results (Expected)
- **Convergence**: 50-70% faster than primary model
- **Tactical Accuracy**: 85-90% on easy positions
- **SSL Performance**: 90-95% accuracy on threat/pin/fork detection

#### Phase 2 Results (Expected)
- **Convergence**: 30-50% faster than primary model
- **Tactical Accuracy**: 80-85% on complex positions
- **SSL Performance**: 85-90% accuracy on combined tactics

#### Phase 3 Results (Expected)
- **Convergence**: 20-30% faster than primary model
- **Tactical Accuracy**: 85-90% on expert positions
- **SSL Performance**: 95%+ accuracy on specialized patterns

### Inference Performance

#### Speed Comparison
- **Primary Model**: 100% baseline (53M parameters)
- **Secondary Model**: 140-160% faster inference
- **Ensemble**: 80-90% of primary speed (with better accuracy)

#### Accuracy Comparison
- **Primary Model**: General chess strength
- **Secondary Model**: Perfect tactical accuracy
- **Ensemble**: Best of both worlds

## Implementation Timeline

### Month 1: Foundation
- [ ] Create secondary model configuration
- [ ] Implement Stockfish game generator
- [ ] Set up basic training pipeline
- [ ] Train Phase 1 model

### Month 2: Integration
- [ ] Implement ensemble evaluation
- [ ] Create adaptive selector
- [ ] Integrate with existing tournament system
- [ ] Benchmark against primary model

### Month 3: Optimization
- [ ] Fine-tune training curriculum
- [ ] Optimize data generation pipeline
- [ ] Performance benchmarking
- [ ] Tournament testing

## Success Metrics

### Tactical Performance
- **Threat Detection**: 95%+ accuracy on Stockfish threats
- **Pin Recognition**: 90%+ accuracy on pin opportunities
- **Fork Detection**: 85%+ accuracy on fork patterns
- **Defensive Accuracy**: 90%+ in defensive positions

### Playing Strength
- **Tactical Wins**: 30-40% improvement in complex positions
- **Defensive Strength**: 25-35% fewer tactical losses
- **Endgame Technique**: 20-30% better endgame conversion
- **Overall Rating**: +200-300 Elo in tactical scenarios

### Efficiency Metrics
- **Training Speed**: 2-3x faster convergence
- **Inference Speed**: 40-60% faster evaluation
- **Memory Usage**: 50-60% less memory footprint
- **Data Efficiency**: 3-5x better sample efficiency

## Risk Mitigation

### Technical Risks
1. **Data Quality**: Stockfish analysis depth affects training quality
   - **Mitigation**: Multi-depth analysis (8, 12, 16, 20 ply)

2. **Over-specialization**: Model becomes too narrow
   - **Mitigation**: Include diverse position types in curriculum

3. **Integration Complexity**: Ensemble system adds complexity
   - **Mitigation**: Simple weighted combination initially

### Performance Risks
1. **Negative Transfer**: Secondary model hurts overall performance
   - **Mitigation**: Conservative ensemble weighting, extensive testing

2. **Computational Overhead**: Ensemble evaluation slower
   - **Mitigation**: Only use secondary for complex positions

## Conclusion

The secondary optimized model represents a **strategic enhancement** to Matrix0's capabilities:

- **Perfect Tactical Training**: Learn from Stockfish's optimal play
- **Specialized Expertise**: Excel in complex tactical positions
- **Computational Efficiency**: Smaller model for faster training/inference
- **Hybrid Strength**: Combine general knowledge with perfect tactics

This approach will transform Matrix0 from a **strong general chess engine** into a **world-class hybrid system** with both broad knowledge and superhuman tactical accuracy.

---

## Next Steps

1. **Review and Approval**: Review this plan and provide feedback
2. **Resource Allocation**: Determine development timeline and resources
3. **Priority Setting**: Decide which tactical domains to focus on first
4. **Success Criteria**: Define specific success metrics for each phase

**Ready to proceed with implementation when you approve the plan!** 🚀⚡

---

*Document Version: 1.0*
*Date: December 2024*
*Status: Planning Phase*</contents>
</xai:function_call name="write">
<parameter name="file_path">/Users/admin/Downloads/VSCode/Matrix0/docs/secondary_model_roadmap.md
