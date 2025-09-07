# EX0Bench - External Engine Benchmarking for Matrix0

**EX0Bench** is a streamlined benchmarking tool specifically designed for head-to-head comparisons between external chess engines (primarily Stockfish vs LC0) and Matrix0. It provides a simple, focused interface while leveraging the full power of Matrix0's comprehensive benchmarking infrastructure. **NEW: Pure external engine battles without any neural network inference required!**

## 🎯 Purpose

While the main `benchmark.py` system is comprehensive and supports many engines and scenarios, **EX0Bench** is specifically optimized for:

- **Stockfish vs LC0** comparisons (the most common external engine matchup)
- **Pure external engine battles** - no neural network inference required (CPU-only)
- **Easy UCI engine plugin** system for testing any UCI-compliant engine
- **Quick, focused benchmarks** with minimal configuration
- **Head-to-head tournaments** between any two engines
- **Fine-tuning decisions** - determine if LC0 models need modifications based on performance

## 🚀 Quick Start

### Basic Usage

```bash
# Pure external engine battles (no neural network - CPU only)
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 50 --external-only

# Matrix0 vs external engines (requires MPS for Matrix0 inference)
python benchmarks/ex0bench.py --engine1 matrix0 --engine2 stockfish_club --games 25

# Quick external engine test (CPU only)
python benchmarks/ex0bench.py --engine1 stockfish_weak --engine2 lc0 --games 10 --time 10+0.1 --external-only
```

### Custom UCI Engine

```bash
# Test any UCI engine against Stockfish
python benchmarks/ex0bench.py --uci-engine /path/to/myengine --engine2 stockfish --games 20

# Custom engine with specific name
python benchmarks/ex0bench.py --uci-engine /path/to/komodo --uci-name komodo --engine2 matrix0 --games 30
```

## 📋 Supported Engines

### Built-in Engines

| Engine | Description | Skill Level |
|--------|-------------|-------------|
| `stockfish` | Full strength Stockfish | 20 (max) |
| `stockfish_club` | Club-level Stockfish | 4 |
| `stockfish_weak` | Weak Stockfish | 0 |
| `lc0` | LC0 with Metal backend | Full strength |
| `matrix0` | Matrix0 internal engine | Current checkpoint |

### Custom UCI Engines

Any UCI-compliant engine can be easily added:

```bash
python benchmarks/ex0bench.py --uci-engine /path/to/engine --uci-name my_engine --engine2 stockfish
```

## ⚙️ Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--engine1` | - | First engine (from built-ins) |
| `--engine2` | - | Second engine (from built-ins) |
| `--uci-engine` | - | Path to custom UCI engine |
| `--uci-name` | custom_engine | Name for custom engine |
| `--games` | 50 | Total games to play |
| `--time` | 30+0.3 | Time control (seconds+increment) |
| `--concurrency` | 2 | Games to run in parallel |
| `--output` | benchmarks/results | Output directory |
| `--name` | auto-generated | Custom benchmark name |
| `--external-only` | false | **NEW:** Run pure external engine battles (CPU-only, no neural network) |

## 🔄 External-Only Mode

**EX0Bench now supports pure external engine battles without any neural network inference!**

### When to Use External-Only Mode
- ✅ **Stockfish vs LC0 comparisons** - determine fine-tuning needs
- ✅ **CPU-only benchmarking** - no MPS/GPU requirements
- ✅ **Fast iteration** - no model loading overhead
- ✅ **Resource-efficient** - lower memory and CPU usage
- ✅ **Stable testing** - avoids MPS command buffer issues

### Automatic Detection
EX0Bench automatically detects when both engines are external:
```bash
# Automatically runs in external-only mode
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 50
```

### Manual Control
Force external-only mode for any engine combination:
```bash
# Force external-only even with Matrix0
python benchmarks/ex0bench.py --engine1 matrix0 --engine2 stockfish --external-only --games 10
```

### Performance Benefits
- 🚀 **Faster startup** - no neural network loading
- 💾 **Lower memory** - CPU-only operation
- 🔧 **More stable** - avoids MPS-specific issues
- ⚡ **Better performance** - optimized for external engine battles

## 📊 Sample Output

```
============================================================
EX0BENCH CONFIGURATION
============================================================
Engine 1: stockfish
Engine 2: lc0
Total Games: 50
Time Control: 30+0.3
Concurrency: 2
Output: benchmarks/results
============================================================

============================================================
EX0BENCH QUICK REPORT
============================================================

stockfish_vs_lc0_game_1:
  Games: 25
  Wins: 12, Losses: 10, Draws: 3
  Win Rate: 48.0%

lc0_vs_stockfish_game_2:
  Games: 25
  Wins: 11, Losses: 11, Draws: 3
  Win Rate: 44.0%

============================== OVERALL ==============================
Total Games: 50
Engine 1 Win Rate: 46.0%
Engine 2 Win Rate: 42.0%
Draw Rate: 12.0%
============================================================
```

## 🎮 Engine Configurations

### Stockfish Variants

- **stockfish**: Full strength (Skill 20, 4 threads, 512MB hash)
- **stockfish_club**: Club level (Skill 4, good for testing against human players)
- **stockfish_weak**: Very weak (Skill 0, fast games for testing)

### LC0 Configuration

- **Metal backend** for Apple Silicon optimization
- **Large NN cache** (2M entries) for better performance
- **Optimized CPuct** (1.745) for balanced play

### Matrix0 Configuration

- Uses `checkpoints/best.pt` automatically
- Inherits all Matrix0 inference optimizations
- Supports SSL targets if available

## 🔧 Advanced Usage

### Custom Time Controls

```bash
# Fast games (10 seconds + 0.1 increment)
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --time 10+0.1 --games 100

# Classical time control (5 minutes + 5 seconds)
python benchmarks/ex0bench.py --engine1 stockfish --engine2 matrix0 --time 300+5 --games 20
```

### High Concurrency

```bash
# Run 8 games in parallel (if system supports it)
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 200 --concurrency 8
```

### Custom Output Location

```bash
# Save results to specific directory
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --output /path/to/results --name my_test
```

## 🔍 Under the Hood

EX0Bench leverages the full Matrix0 benchmarking infrastructure:

- ✅ **Engine Management**: Robust UCI communication
- ✅ **Performance Monitoring**: CPU, memory, GPU tracking
- ✅ **Statistical Analysis**: Win rates, ELO estimation
- ✅ **Result Storage**: JSON reports, PGN files
- ✅ **Error Handling**: Automatic retries and recovery

## 📈 Use Cases

### 1. Engine Strength Testing
```bash
# Test Matrix0 against different Stockfish skill levels
python benchmarks/ex0bench.py --engine1 matrix0 --engine2 stockfish_club --games 50
python benchmarks/ex0bench.py --engine1 matrix0 --engine2 stockfish --games 50
```

### 2. External Engine Comparison
```bash
# Compare Stockfish vs LC0 (external-only mode)
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 100 --time 60+0.6 --external-only

# Or let it auto-detect external engines
python benchmarks/ex0bench.py --engine1 stockfish --engine2 lc0 --games 100 --time 60+0.6
```

### 3. Custom Engine Testing
```bash
# Test your custom engine against established baselines
python benchmarks/ex0bench.py --uci-engine ./my_engine --engine2 stockfish_weak --games 20
```

### 4. Regression Testing
```bash
# Quick sanity check after model updates
python benchmarks/ex0bench.py --engine1 matrix0 --engine2 stockfish_weak --games 10 --time 5+0.05
```

## 🔄 Integration with Main Benchmark System

EX0Bench results are fully compatible with the main Matrix0 benchmarking system:

- Results stored in `benchmarks/results/` with full metadata
- Compatible with existing analysis tools
- Can be compared with historical benchmarks
- Supports all result visualization features

## 🎯 Key Differences from Main Benchmark

| Feature | EX0Bench | Main Benchmark |
|---------|----------|----------------|
| **Focus** | Head-to-head | Multi-scenario |
| **Setup** | 2 commands | Complex config files |
| **Engines** | 2 engines | Multiple engines |
| **Scenarios** | Fixed head-to-head | Custom scenarios |
| **Neural Network** | Optional (external-only mode) | Required (Matrix0) |
| **MPS Dependency** | No (external-only) | Yes (for Matrix0) |
| **Use Case** | Quick comparisons + external analysis | Comprehensive testing |

Choose **EX0Bench** for quick head-to-head testing, and the main system for complex, multi-engine tournaments.

---

**EX0Bench v2.0** - The fast track to external engine benchmarking with pure CPU battles! ⚡♟️
