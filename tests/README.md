# Matrix0 Test Suite

Comprehensive test suite for Matrix0 covering performance, error handling, integration, and stress testing.

## Overview

The test suite is organized into several categories:

- **Performance Tests** (`test_performance.py`): Latency benchmarks, throughput measurements, memory profiling
- **Error Handling Tests** (`test_error_handling.py`): Robustness tests, error detection, recovery mechanisms
- **Integration Tests** (`test_integration.py`): End-to-end workflow tests, component integration
- **Stress Tests** (`test_stress.py`): Extended operations, concurrent execution, memory pressure

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_performance.py

# Run specific test class
pytest tests/test_performance.py::TestInferenceLatency

# Run specific test function
pytest tests/test_performance.py::TestInferenceLatency::test_single_inference_latency
```

### Using Markers

```bash
# Run only performance tests
pytest -m performance

# Run only integration tests
pytest -m integration

# Run only stress tests
pytest -m stress

# Run only error handling tests
pytest -m error_handling

# Run tests excluding slow ones
pytest -m "not slow"

# Run tests requiring GPU/MPS
pytest -m requires_gpu
```

### Using Test Runner

```bash
# Run all tests with reporting
python tests/test_runner.py

# Run performance tests only
python tests/test_runner.py --performance

# Run with verbose output
python tests/test_runner.py -v

# Run specific test paths
python tests/test_runner.py tests/test_performance.py

# Run with custom output directory
python tests/test_runner.py -o test_results/
```

## Test Categories

### Performance Tests

Tests for latency, throughput, and memory usage:

- **Inference Latency**: Single and batched inference timing
- **MCTS Latency**: Search performance, batched vs sequential comparison
- **Training Performance**: Training step timing, SSL overhead
- **Memory Performance**: Memory usage profiling, scaling analysis

**Example:**
```bash
pytest -m performance -v
```

### Error Handling Tests

Tests for robustness and error recovery:

- **Inference Errors**: NaN/Inf detection, shape validation, timeout handling
- **MCTS Errors**: Empty visit counts, invalid board states, terminal positions
- **Data Errors**: Invalid policy targets, missing legal masks, corrupted shards
- **Robustness**: Concurrent operations, memory cleanup, stability

**Example:**
```bash
pytest -m error_handling -v
```

### Integration Tests

End-to-end workflow tests:

- **Self-Play Integration**: Game generation, SSL target generation
- **Training Integration**: Data loading, training step execution
- **MCTS Integration**: Inference backend integration, tree consistency
- **Data Pipeline**: Save/load operations, SSL target handling

**Example:**
```bash
pytest -m integration -v
```

### Stress Tests

Extended and concurrent operation tests:

- **MCTS Stress**: Extended searches, many positions, rapid searches
- **Inference Stress**: Large batches, continuous inference
- **Memory Stress**: Sustained load, memory leak detection
- **Concurrency Stress**: Concurrent MCTS, concurrent inference

**Example:**
```bash
pytest -m stress -v
```

## Test Fixtures

Common fixtures available in `conftest.py`:

- `dummy_model`: Lightweight model for testing
- `dummy_model_mps`: Model on MPS if available
- `test_board`: Standard chess board
- `encoded_board`: Encoded board position
- `constant_backend`: Constant inference backend
- `device`: Test device (MPS/CPU)
- `temp_data_dir`: Temporary directory for test data
- `sample_batch_data`: Sample training batch
- `sample_ssl_batch_data`: Sample batch with SSL targets

## Performance Benchmarks

Performance tests measure:

- **Latency**: Time per operation (inference, MCTS search, training step)
- **Throughput**: Operations per second
- **Memory**: Peak memory usage, memory scaling
- **Scalability**: Performance with different batch sizes

Results are logged and can be collected for analysis.

## Requirements

- pytest
- pytest-html (for HTML reports)
- pytest-xdist (optional, for parallel execution)

Install with:
```bash
pip install pytest pytest-html pytest-xdist
```

## Parallel Execution

Run tests in parallel for faster execution:

```bash
# Run with 4 workers
pytest -n 4 tests/

# Auto-detect CPU count
pytest -n auto tests/
```

## Continuous Integration

Example CI configuration:

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt pytest pytest-html
      - run: pytest -m "not slow" --junitxml=junit.xml
      - uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: junit.xml
```

## Best Practices

1. **Mark slow tests**: Use `@pytest.mark.slow` for tests taking >5 seconds
2. **Skip when appropriate**: Use `pytest.skip()` for unsupported configurations
3. **Clear assertions**: Use descriptive assert messages
4. **Clean up resources**: Use fixtures for temporary resources
5. **Profile selectively**: Performance tests can be run separately

## Troubleshooting

### Tests failing on MPS

Some tests may fail on MPS due to hardware limitations. Skip with:
```bash
pytest -m "not requires_mps"
```

### Memory errors

If tests fail with OOM errors, reduce batch sizes or skip memory-intensive tests:
```bash
pytest -m "not stress"
```

### Timeout issues

Some tests may timeout on slower hardware. Increase timeout or skip:
```bash
pytest --timeout=300 -m slow
```

## Contributing

When adding new tests:

1. Place in appropriate test file
2. Add appropriate markers (`@pytest.mark.performance`, etc.)
3. Use fixtures from `conftest.py`
4. Add docstrings explaining what is tested
5. Include assertions with clear error messages

