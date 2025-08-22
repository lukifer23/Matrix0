# Matrix0 Benchmark System

A comprehensive benchmarking system for evaluating Matrix0 models against UCI-compliant chess engines like Stockfish, lc0, and Komodo.

## 🚀 Features

- **Multi-Engine Support**: Test against Stockfish, lc0, Komodo, and other UCI engines
- **Performance Metrics**: CPU, memory, GPU usage, and timing analysis
- **Statistical Analysis**: Win/loss/draw rates, confidence intervals, ELO estimation
- **Flexible Configuration**: Multiple test scenarios with different time controls
- **Comprehensive Reporting**: JSON results, HTML visualizations, summary reports
- **Resource Monitoring**: System load tracking during benchmark runs

## 📁 Project Structure

```
benchmarks/
├── __init__.py              # Package initialization
├── benchmark.py             # Main benchmark script
├── config.py               # Configuration management
├── uci_bridge.py           # UCI engine communication
├── metrics.py              # Performance monitoring
├── results.py              # Results analysis and reporting
├── configs/                # Configuration files
│   └── default.yaml       # Default benchmark config
├── results/                # Generated results and reports
├── engines/                # Engine-specific configurations
└── README.md              # This file
```

## 🛠️ Quick Start

### 1. Basic Usage
```bash
# Run with default configuration
python benchmarks/benchmark.py

# Run with custom configuration
python benchmarks/benchmark.py --config benchmarks/configs/default.yaml

# Quick benchmark from command line
python benchmarks/benchmark.py --model checkpoints/v2_base.pt --engine stockfish --games 10
```

### 2. Configuration
Edit `benchmarks/configs/default.yaml` to customize:
- Engine configurations and paths
- Test scenarios and time controls
- Performance monitoring settings
- Output directories

### 3. Prerequisites
- **Matrix0 Model**: Trained model in `checkpoints/`
- **UCI Engines**: Stockfish, lc0, or other UCI engines installed
- **Python Dependencies**: All Matrix0 dependencies + `psutil`

## 📊 Configuration Options

### UCI Engine Setup
```yaml
engines:
  stockfish:
    command: "stockfish"
    options:
      Threads: "4"
      Hash: "512"
      Skill Level: "20"
```

### Test Scenarios
```yaml
scenarios:
  - name: "Stockfish_Medium"
    engine: "stockfish"
    model_checkpoint: "checkpoints/v2_base.pt"
    num_games: 50
    time_control: "60+0.6"    # 60s + 0.6s increment
    concurrency: 2            # Games to run in parallel
    max_moves: 200           # Maximum game length
```

### Performance Monitoring
```yaml
performance:
  track_cpu: true
  track_memory: true
  track_gpu: true
  sample_interval: 0.1        # Sample every 100ms
```

## 📈 Output and Results

### Generated Files
- **`benchmark_results.json`**: Complete benchmark data
- **`benchmark_summary.json`**: Statistical summary
- **`benchmark_config.yaml`**: Configuration backup
- **`benchmark_report.txt`**: Human-readable report
- **`benchmark_results.html`**: Interactive visualizations (if plotly available)

### Key Metrics
- **Win/Loss/Draw Rates**: Performance against each engine
- **ELO Estimation**: Calculated strength rating
- **Timing Analysis**: Move time statistics and variance
- **Resource Usage**: CPU, memory, GPU utilization
- **Game Statistics**: Length, duration, complexity metrics

## 🎯 Advanced Usage

### Custom Engine Configuration
```python
from benchmarks.config import UCIEngineConfig, TestScenario

# Configure custom engine
engine = UCIEngineConfig(
    name="CustomEngine",
    command="/path/to/engine",
    options={"CustomOption": "value"}
)
```

### Performance Analysis
```python
from benchmarks.results import ResultsAnalyzer

# Analyze results
report = ResultsAnalyzer.analyze_results("benchmarks/results/benchmark_results.json")
print(f"Win Rate: {report.summary_stats['results']['win_rate']:.1%}")
```

### Multi-Engine Comparison
```python
# Compare multiple engines
results_files = [
    "stockfish_results.json",
    "lc0_results.json",
    "komodo_results.json"
]

comparison = generate_comparison_report(results_files)
```

## 🔧 Troubleshooting

### Common Issues

#### Engine Not Found
```
Error: Engine 'stockfish' failed to start
Solution: Install Stockfish or update engine path in config
```

#### Model Loading Error
```
Error: No checkpoints found
Solution: Ensure model file exists in checkpoints directory
```

#### Memory Issues
```
Error: MPS memory limit exceeded
Solution: Reduce batch sizes or increase memory limit in config
```

#### Port Conflicts
```
Error: UCI engine communication failed
Solution: Ensure no other UCI processes are running
```

### Performance Tips
- **Use appropriate time controls** for meaningful results
- **Monitor system resources** during long benchmark runs
- **Start with small game counts** for initial testing
- **Use concurrency carefully** to avoid resource contention

## 📚 API Reference

### Core Classes

#### `BenchmarkRunner`
Main benchmark execution engine.

```python
runner = BenchmarkRunner(config)
results = runner.run_benchmark()
```

#### `EngineManager`
Manages UCI engine lifecycle.

```python
manager = EngineManager()
manager.add_engine(config)
engine = manager.get_engine("stockfish")
```

#### `MetricsCollector`
Collects system performance metrics.

```python
collector = MetricsCollector()
collector.start_collection()
# ... run benchmark ...
metrics = collector.stop_collection()
```

## 🤝 Contributing

The benchmark system is designed to be extensible:

1. **Add new UCI engines** by extending `UCIEngineConfig`
2. **Create custom metrics** by extending `MetricsCollector`
3. **Add analysis tools** by extending `ResultsAnalyzer`
4. **Support new formats** by extending `ReportGenerator`

## 📋 Roadmap

- [ ] **Multi-model comparison** - Compare different Matrix0 checkpoints
- [ ] **Tournament mode** - Round-robin tournaments between engines
- [ ] **Real-time monitoring** - Live dashboard during benchmark runs
- [ ] **Cloud integration** - Run benchmarks on remote instances
- [ ] **Database storage** - Store historical benchmark results
- [ ] **Automated testing** - Continuous benchmark regression testing

---

**Benchmark System v1.0** - Comprehensive evaluation framework for Matrix0 chess AI.
