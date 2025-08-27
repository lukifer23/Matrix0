# Stockfish-Generated Training Data System

## Overview

This system generates high-quality, domain-specific training datasets using Stockfish to target specific weaknesses in Matrix0's play. Each dataset includes proper SSL annotations, policy targets, value estimates, and metadata for curriculum learning.

## Directory Structure

```
data/stockfish_games/
├── openings/           # Opening repertoire development
│   ├── london_system/
│   ├── sicilian_defense/
│   ├── kings_indian/
│   └── ...
├── tactical/           # Tactical pattern recognition
│   ├── pins/
│   ├── forks/
│   ├── discovered_attacks/
│   ├── skewers/
│   └── ...
├── endgames/           # Endgame perfection
│   ├── king_and_pawn/
│   ├── rook_endings/
│   ├── queen_endings/
│   └── ...
├── puzzles/            # Puzzle solving
│   ├── mate_in_2/
│   ├── mate_in_3/
│   ├── mate_in_4/
│   └── ...
└── weaknesses/         # Targeted weakness remediation
    ├── hanging_pieces/
    ├── undefended_squares/
    ├── back_rank_weakness/
    └── ...
```

## Data Format

Each dataset follows the existing NPZ format with SSL extensions:

### Core Fields (existing):
- `s`: Board state (19 planes: pieces, castling, en passant, move count)
- `pi`: Policy targets (move probabilities)
- `z`: Value targets (game outcome)
- `legal_mask`: Legal move mask

### SSL Extensions (new):
- `ssl_piece`: Piece detection targets
- `ssl_threat`: Threat detection targets
- `ssl_pin`: Pin detection targets
- `ssl_fork`: Fork detection targets
- `ssl_control`: Square control targets

### Metadata:
- `difficulty`: Normalized difficulty score (0.0-1.0)
- `stockfish_eval`: Stockfish evaluation in centipawns
- `optimal_depth`: Depth to optimal play
- `category`: Domain category (opening/tactical/endgame/puzzle)
- `subcategory`: Specific pattern type

## Generation Strategy

### 1. Position Selection
- **Openings**: Use established repertoire positions
- **Tactical**: Generate positions with clear tactical patterns
- **Endgames**: Use tablebase positions with known outcomes
- **Puzzles**: Curate positions requiring specific tactical solutions

### 2. Stockfish Analysis
- Multi-depth analysis (5, 10, 15, 20 ply)
- Best move identification
- Alternative move evaluation
- Pattern recognition

### 3. SSL Annotation
- Piece placement analysis
- Threat detection
- Pin identification
- Fork opportunities
- Square control evaluation

### 4. Curriculum Integration
- Difficulty-based sorting
- Domain-specific weighting
- Progressive complexity
- Weakness-targeted sampling

## Usage Examples

### Generate Opening Dataset
```bash
python tools/generate_stockfish_data.py \
  --domain openings \
  --subdomain london_system \
  --positions 10000 \
  --stockfish-depth 15
```

### Generate Tactical Dataset
```bash
python tools/generate_stockfish_data.py \
  --domain tactical \
  --subdomain pins \
  --positions 5000 \
  --stockfish-depth 10
```

### Generate Weakness Dataset
```bash
python tools/generate_stockfish_data.py \
  --domain weaknesses \
  --subdomain hanging_pieces \
  --positions 3000 \
  --stockfish-depth 8
```

## Integration with Training

### Curriculum Phases
- **opening_focus**: 80% openings, 20% mixed
- **tactical_focus**: 80% tactics, 20% mixed
- **endgame_focus**: 80% endgames, 20% mixed
- **weakness_remediation**: Targeted weakness positions

### SSL Weighting
- **Early training**: Higher SSL weight (0.3-0.5)
- **Mid training**: Balanced SSL weight (0.1-0.2)
- **Late training**: Lower SSL weight (0.05-0.1)

## Benefits

### 🎯 Targeted Improvement
- Address specific weaknesses identified in benchmarks
- Focus training on problematic patterns
- Accelerate learning in weak areas

### 📊 Perfect Labels
- Stockfish provides optimal move selection
- Precise value estimation
- Accurate pattern recognition

### 🔄 Curriculum Learning
- Progressive difficulty increase
- Domain-specific skill development
- Balanced overall improvement

### 🚀 SSL Enhancement
- Perfect SSL annotations from Stockfish
- Rich tactical pattern examples
- Complex position analysis

## Implementation Priority

### Phase 1: Core Infrastructure
- [x] Basic Stockfish integration
- [x] Position generation framework
- [x] SSL annotation system

### Phase 2: Domain Generators
- [ ] Opening repertoire generator
- [ ] Tactical pattern generator
- [ ] Endgame position generator

### Phase 3: Advanced Features
- [ ] Weakness detection system
- [ ] Adaptive difficulty scaling
- [ ] Performance tracking integration

### Phase 4: Training Integration
- [x] Core ingestion via training.extra_replay_dirs (see below)
- [ ] Curriculum system extension
- [ ] Metadata tracking
- [ ] Performance analytics

## Training Ingestion (Matrix0)

Stockfish datasets are automatically imported by the training pipeline when placed under `data/stockfish_games/`.

The default `config.yaml` includes:

```yaml
training:
  extra_replay_dirs:
    - data/stockfish_games
```

This registers all NPZ shards found in `data/stockfish_games/**` with the replay buffer so training can sample them alongside self-play data.

Notes:
- The generator uses Matrix0's canonical encoders: `encode_board` (19 planes) and `move_to_index` (4672 actions).
- `pi` is one-hot on the best move from Stockfish; `legal_mask` covers legal moves for the position.
- SSL targets (`ssl_*`) are included for future use and can be ignored safely by the training loop.

## Success Metrics

### Model Improvement
- **Tactical accuracy**: % of correct tactical moves
- **Opening strength**: Performance in opening positions
- **Endgame conversion**: Endgame win rate improvement
- **SSL accuracy**: SSL head prediction accuracy

### Data Quality
- **Label accuracy**: Stockfish analysis depth coverage
- **Position diversity**: Unique position coverage
- **Curriculum balance**: Difficulty distribution
- **SSL completeness**: SSL annotation coverage

This system will transform Matrix0 from a general chess player into a specialized, weakness-targeted learning system that continuously improves through structured, perfect-play training data.
