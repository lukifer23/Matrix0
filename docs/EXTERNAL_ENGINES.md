# External Engine Integration for Matrix0

This document describes how to integrate and use external chess engines (Stockfish, Leela Chess Zero) with Matrix0 for training, evaluation, and competition.

See [docs/configuration.md](docs/configuration.md) for general configuration details including draw adjudication settings.

## Overview

Matrix0 now supports advanced integration with external chess engines through the UCI protocol, enabling:

- **Training Partner Integration**: Generate games between Matrix0 and external engines
- **Multi-Engine Tournaments**: Round-robin, Swiss, and single-elimination formats
- **SSL Performance Tracking**: Monitor SSL effectiveness during engine competitions
- **Apple Silicon Optimization**: Metal backend support for LC0 neural networks
- **Automated Engine Discovery**: Intelligent detection and configuration of installed engines
- **Comprehensive Analysis**: Statistical significance testing and performance evaluation

## Supported Engines

### Stockfish
- **Path**: `/opt/homebrew/bin/stockfish` (auto-detected)
- **Parameters**: Threads, Hash, Skill Level, ELO limits
- **Apple Silicon**: Native x86_64 emulation support
- **Time Control**: Configurable (default: 30+0.3)
- **Status**: [x] Fully integrated with automated discovery

### Leela Chess Zero (LC0)
- **Path**: `/opt/homebrew/bin/lc0` (auto-detected)
- **Parameters**: Threads, NNCacheSize, MinibatchSize, Backend
- **Apple Silicon**: Metal backend optimization for M1/M2/M3
- **Neural Network**: Full neural evaluation with CUDA/Metal acceleration
- **Time Control**: Configurable (default: 30+0.3)
- **Status**: [x] Fully integrated with Apple Silicon optimization

### Matrix0 (Internal)
- **Type**: Internal neural network model with SSL integration
- **Checkpoint**: `checkpoints/v2_base.pt` (53M parameters)
- **Evaluation**: MCTS with SSL-enhanced neural network guidance
- **SSL Heads**: Threat, pin, fork, control, piece detection
- **Status**: [x] Enhanced with comprehensive benchmark system

### Additional Engines
The system supports automatic discovery of additional UCI-compliant engines:
- **Komodo**: Commercial engine with strong positional play
- **Houdini**: High-performance tactical engine
- **Fire**: Open-source engine with good endgame strength
- **Custom Engines**: Any UCI-compliant chess engine

## New Features in v2.1

### Advanced Benchmark System
- **Multi-Engine Tournaments**: Automated competitive evaluation
- **SSL Performance Tracking**: Real-time SSL effectiveness monitoring
- **Apple Silicon Optimization**: Metal backend for neural engines
- **Automated Discovery**: Intelligent engine detection and configuration
- **Comprehensive Analysis**: Statistical significance and regression testing

### Enhanced Engine Management
- **Process Isolation**: Robust engine process management
- **Health Monitoring**: Automatic engine health checks
- **Resource Management**: CPU and memory optimization
- **Configuration Optimization**: Engine-specific parameter tuning

## Configuration

### Basic Engine Configuration

Add engine configurations to your `config.yaml`:

```yaml
engines:
  stockfish:
    path: /usr/local/bin/stockfish
    parameters:
      Threads: 2
      Hash: 128
      MultiPV: 1
    time_control: 100ms
    enabled: true
  
  lc0:
    path: /usr/local/bin/lc0
    parameters:
      threads: 2
      minibatch-size: 256
      backend: cuda
    time_control: 100ms
    enabled: true
  
  matrix0:
    type: internal
    checkpoint: checkpoints/v2_base.pt
    enabled: true
```

### Training Configuration

Configure external engine training parameters:

```yaml
selfplay:
  external_engine_ratio: 0.3  # 30% games vs external engines
  engine_strength_curriculum: true

training:
  external_engine_ratio: 0.3
  engine_strength_curriculum: true
  adversarial_training: true
```

### Evaluation Configuration

Configure external engine evaluation:

```yaml
eval:
  external_engines: ["stockfish", "lc0"]
  tournament_rounds: 100
  strength_estimation_games: 50
```

### Orchestrator Configuration

External engine integration is disabled by default. Enable when engines are installed:

```yaml
orchestrator:
  external_engine_integration: true
```

Notes (macOS):
- Matrix0 uses spawn-safe process targets for external workers on macOS.
- The orchestrator sets MPS stability env vars when device=mps:
  - `PYTORCH_ENABLE_MPS_FALLBACK=1`
  - `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8`
  - `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`
  to avoid memory watermark errors in child processes.

## Usage

### 1. Self-Play with External Engines

Generate games between Matrix0 and external engines:

```bash
# Generate games using external engines
python -m azchess.selfplay --external-engines --games 32

# Or use the orchestrator with external engines enabled
python -m azchess.orchestrator --external-engines
```

### 2. Evaluate Against External Engines

Evaluate Matrix0's strength against external engines:

```bash
# Evaluate against all configured external engines (engines must be installed and enabled)
python -m azchess.eval --external-engines --games 50

# Or use the multi-engine evaluator directly
python -m azchess.eval.multi_engine_evaluator --config config.yaml --games 50
```

### 3. Multi-Engine Evaluation

Run comprehensive evaluation against multiple engines:

```python
from azchess.eval.multi_engine_evaluator import evaluate_matrix0_against_engines

# Evaluate against specific engines
results = await evaluate_matrix0_against_engines(
    "config.yaml", 
    engine_names=["stockfish", "lc0"], 
    games_per_engine=50
)

# Print results
for engine_name, result in results.items():
    print(f"{engine_name}: {result.win_rate:.3f} win rate")
```

### 4. Advanced Benchmark System
Use the enhanced benchmark system for comprehensive evaluation:

```bash
# Discover and validate all installed engines
python benchmarks/enhanced_runner.py --config benchmarks/configs/enhanced_scenarios.yaml --discover-engines

# Run LC0 vs Matrix0 tournament
python benchmarks/enhanced_runner.py --config benchmarks/configs/enhanced_scenarios.yaml --scenario LC0_Matrix0_Showdown

# Run complete benchmark suite with SSL tracking
python benchmarks/enhanced_runner.py --config benchmarks/configs/enhanced_scenarios.yaml
```

### 5. Tournament Mode
Run multi-engine tournaments with advanced analysis:

```python
from benchmarks.tournament import Tournament, TournamentConfig, TournamentFormat

# Configure tournament
config = TournamentConfig(
    name="Championship Tournament",
    format=TournamentFormat.ROUND_ROBIN,
    engines=["matrix0", "stockfish", "lc0"],
    num_games_per_pairing=20,
    time_control="60+0.6"
)

# Run tournament
tournament = Tournament(config)
results = await tournament.run_tournament()

# Get comprehensive analysis
rankings = results.get("final_rankings", [])
statistics = results.get("statistics", {})
```

### 4. External Engine Self-Play (advanced)

Generate training data using external engines:

```python
from azchess.selfplay.external_engine_worker import ExternalEngineSelfPlay
from azchess.engines import EngineManager

# Initialize
config = Config.load("config.yaml")
engine_manager = EngineManager(config.to_dict())
selfplay = ExternalEngineSelfPlay(config, engine_manager)

# Generate games
games = await selfplay.generate_games(100, "data/external_games")
```

## Engine Management

### Starting and Stopping Engines

```python
from azchess.engines import EngineManager

# Initialize and start all engines
engine_manager = EngineManager(config.to_dict())
await engine_manager.start_all_engines()

# Start specific engine
await engine_manager.start_engine("stockfish")

# Stop specific engine
await engine_manager.stop_engine("stockfish")

# Cleanup all engines
await engine_manager.cleanup()
```

### Health Monitoring

```python
# Check engine health
health_results = await engine_manager.check_all_engines_health()

# Get engine information
info = engine_manager.get_engine_info("stockfish")
print(f"Status: {info['health']['status']}")
print(f"Last error: {info['health']['last_error']}")
```

### Engine Selection

```python
# Select training partner based on target strength
partner = engine_manager.select_training_partner(target_strength=1800.0)
print(f"Selected partner: {partner}")
```

## File Structure

```
azchess/
├── engines/
│   ├── __init__.py
│   ├── uci_bridge.py          # UCI protocol communication
│   └── engine_manager.py      # Engine coordination
├── selfplay/
│   ├── __init__.py            # Package exports
│   ├── internal.py            # Internal self-play worker
│   └── external_engine_worker.py  # External engine self-play
└── eval/
    ├── __init__.py            # Package exports
    └── multi_engine_evaluator.py  # Multi-engine evaluation
```

## Data Formats

### External Engine Games

Games against external engines are saved in JSON format:

```json
{
  "metadata": {
    "white_engine": "matrix0",
    "black_engine": "stockfish",
    "moves": 45,
    "result": 1.0,
    "time_seconds": 12.5,
    "timestamp": 1234567890.123
  },
  "game_data": {
    "moves": ["e2e4", "e7e5", "g1f3", ...],
    "positions": [...],
    "evaluations": [...],
    "final_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  }
}
```

### Evaluation Results

Evaluation results include comprehensive statistics:

```python
@dataclass
class EvaluationResult:
    matrix0_wins: int
    matrix0_losses: int
    matrix0_draws: int
    total_games: int
    win_rate: float
    engine_name: str
    time_control: str
    games: List[Dict[str, Any]]
```

## **NEW: Recent Improvements** ### Enhanced Self-Play System
- **Modular Architecture**: Separated internal and external self-play into distinct modules
- **Improved Resignation Logic**: Smart resignation based on consecutive bad evaluations
- **Opening Diversity**: Configurable random opening moves for training variety
- **Selection Jitter**: Configurable exploration in MCTS for better training diversity

### Robust Engine Management
- **Process Isolation**: Each external engine runs in its own process
- **Automatic Recovery**: Engines automatically restart on failures
- **Health Monitoring**: Comprehensive engine status tracking
- **Resource Management**: Proper cleanup and memory management

### Quality Assurance
- **Move Validation**: All external engine moves are validated for legality
- **Game Quality Filtering**: Automatic filtering of corrupted or invalid games
- **Metadata Tracking**: Comprehensive game metadata for analysis
- **Error Handling**: Graceful fallback when external engines fail

## Troubleshooting

### Common Issues

1. **Engine Not Found** - Verify engine path in configuration
   - Ensure engine executable has proper permissions
   - Check if engine is installed and accessible

2. **Engine Communication Errors** - Verify UCI protocol compatibility
   - Check engine parameters and time controls
   - Review engine logs for errors

3. **Performance Issues** - Adjust time controls for faster games
   - Reduce engine thread count
   - Monitor system resources

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Run engine health checks:

```python
# Check all engines
health = await engine_manager.check_all_engines_health()

# Check specific engine
is_healthy = await engine_manager.check_engine_health("stockfish")
```

## Performance Considerations

### Resource Management
- **Memory**: External engines consume additional memory
- **CPU**: Engine processes compete with Matrix0 training
- **Disk**: Game data storage requirements increase

### Optimization Tips
- Use appropriate time controls for your use case
- Limit concurrent engine processes
- Implement engine rotation for load balancing
- Monitor system resources during operation

## Future Enhancements

### Planned Features
- **Engine Strength Estimation**: Automatic rating calculation
- **Adaptive Training**: Dynamic opponent selection
- **Opening Book Integration**: Engine-specific opening strategies
- **Performance Analytics**: Detailed engine comparison metrics

### Extensibility
The system is designed to easily add new engines:
1. Add engine configuration to `config.yaml`
2. Ensure UCI protocol compatibility
3. Test engine integration
4. Configure training and evaluation parameters

## Examples

### Complete Training Cycle

```bash
# 1. Generate games with external engines
python -m azchess.selfplay --external-engines --games 64

# 2. Train on combined data
python -m azchess.train --config config.yaml

# 3. Evaluate against external engines
python -m azchess.eval --external-engines --games 50

# 4. Or run complete cycle
python -m azchess.orchestrator --external-engines
```

### Custom Engine Configuration

```yaml
engines:
  custom_engine:
    path: /path/to/custom/engine
    parameters:
      custom_param: value
    time_control: 200ms
    enabled: true
    estimated_rating: 2000.0
```

### Advanced Training Configuration

```yaml
selfplay:
  external_engine_ratio: 0.5  # 50% external engine games
  engine_strength_curriculum: true
  opening_random_plies: 4     # Random opening moves
  selection_jitter: 0.1       # MCTS exploration parameter

```

For draw adjudication and resignation settings, see [Configuration Guide](docs/configuration.md)

## Current Project Status

### Training Pipeline
- **Status**: [x] Operational and stable
- **SSL Foundation**: Basic piece recognition working
- **Advanced SSL**: Algorithms implemented and integrated
- **External Engines**: [x] Fully integrated and tested

### Development Priorities
1. **SSL Validation**: Test and validate advanced SSL algorithm effectiveness
2. **Performance Optimization**: Memory usage and training throughput
3. **Enhanced Evaluation**: Multi-engine tournament and strength estimation
4. **External Engine Training**: Leverage external engines for competitive training

## Support

For issues or questions about external engine integration:

1. Check the troubleshooting section
2. Review engine-specific documentation
3. Verify configuration syntax
4. Test with minimal configuration first

The external engine integration is designed to be robust and maintainable, providing Matrix0 with access to world-class chess engines for training and evaluation. The system has been thoroughly tested and is production-ready for competitive training and evaluation workflows.

**Current focus**: SSL algorithm validation and training pipeline enhancement with external engine support.
