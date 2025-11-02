"""Performance and latency benchmarks for Matrix0 components."""

import time
import statistics
from typing import List, Dict, Any, Tuple
import logging
import sys

import chess
import numpy as np
import pytest
import torch

from azchess.mcts import MCTS, MCTSConfig
from azchess.model.resnet import PolicyValueNet, NetConfig
from azchess.encoding import encode_board


logger = logging.getLogger(__name__)


def print_table(rows: List[List[str]], headers: List[str], title: str = ""):
    """Print a formatted table."""
    if title:
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(row_str)
    
    print()


class PerformanceProfiler:
    """Utility for profiling performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, name: str, duration: float):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        values = self.metrics[name]
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
    
    def print_summary(self):
        """Print summary of all metrics."""
        logger.info("Performance Summary:")
        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            if stats:
                logger.info(
                    f"  {name}: mean={stats['mean']:.4f}s, "
                    f"median={stats['median']:.4f}s, "
                    f"min={stats['min']:.4f}s, max={stats['max']:.4f}s "
                    f"(n={stats['count']})"
                )


@pytest.fixture
def profiler():
    """Create a performance profiler."""
    return PerformanceProfiler()


@pytest.mark.performance
class TestInferenceLatency:
    """Test inference latency for various scenarios."""
    
    def test_single_inference_latency(self, dummy_model, test_board, device, profiler):
        """Measure single inference latency."""
        model = dummy_model.to(device)
        model.eval()
        
        encoded = encode_board(test_board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(tensor)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        num_iterations = 50
        latencies = []
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(tensor)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            latencies.append(latency)
            profiler.record("single_inference", latency)
        
        mean_latency_ms = statistics.mean(latencies) * 1000
        median_latency_ms = statistics.median(latencies) * 1000
        p95_latency_ms = np.percentile(latencies, 95) * 1000
        p99_latency_ms = np.percentile(latencies, 99) * 1000
        min_latency_ms = min(latencies) * 1000
        max_latency_ms = max(latencies) * 1000
        
        results = [[
            f"{mean_latency_ms:.2f}",
            f"{median_latency_ms:.2f}",
            f"{p95_latency_ms:.2f}",
            f"{p99_latency_ms:.2f}",
            f"{min_latency_ms:.2f}",
            f"{max_latency_ms:.2f}",
        ]]
        
        print_table(
            results,
            headers=["Mean", "Median", "P95", "P99", "Min", "Max"],
            title="Single Inference Latency (ms)"
        )
        
        print(f"Single inference: mean={mean_latency_ms:.2f}ms, "
              f"p95={p95_latency_ms:.2f}ms, p99={p99_latency_ms:.2f}ms", file=sys.stderr)
        
        # Assert reasonable latency (adjust thresholds based on hardware)
        assert mean_latency_ms < 1000.0, f"Mean inference latency too high: {mean_latency_ms:.2f}ms"
        assert p95_latency_ms < 2000.0, f"P95 inference latency too high: {p95_latency_ms:.2f}ms"
    
    def test_batched_inference_latency(self, dummy_model, test_board, device, profiler):
        """Measure batched inference latency."""
        model = dummy_model.to(device)
        model.eval()
        
        batch_sizes = [1, 8, 16, 32, 64, 96]
        encoded = encode_board(test_board)
        
        results = []
        
        for batch_size in batch_sizes:
            batch_tensor = torch.from_numpy(np.repeat(encoded[None, :], batch_size, axis=0)).to(device)
            
            # Warmup
            with torch.no_grad():
                _ = model(batch_tensor)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            
            # Measure
            num_iterations = 20
            latencies = []
            
            for _ in range(num_iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(batch_tensor)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                latencies.append(latency)
            
            mean_latency = statistics.mean(latencies)
            mean_latency_ms = mean_latency * 1000
            throughput = batch_size / mean_latency
            latency_per_sample_ms = mean_latency_ms / batch_size
            
            profiler.record(f"batched_inference_b{batch_size}", mean_latency)
            
            results.append([
                str(batch_size),
                f"{mean_latency_ms:.2f}",
                f"{latency_per_sample_ms:.2f}",
                f"{throughput:.1f}",
            ])
            
            print(f"Batch {batch_size}: {mean_latency_ms:.2f}ms total, "
                  f"{latency_per_sample_ms:.2f}ms/sample, "
                  f"{throughput:.1f} samples/sec", file=sys.stderr)
        
        print_table(
            results,
            headers=["Batch Size", "Total Latency (ms)", "Latency/Sample (ms)", "Throughput (samples/sec)"],
            title="Batched Inference Performance"
        )
    
    def test_amp_inference_latency(self, dummy_model, test_board, device, profiler):
        """Measure inference latency with AMP enabled."""
        if device not in ("mps", "cuda"):
            pytest.skip("AMP only supported on MPS/CUDA")
        
        model = dummy_model.to(device)
        model.eval()
        
        encoded = encode_board(test_board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
        
        # Warmup
        with torch.no_grad(), torch.autocast(device_type=device, enabled=True):
            _ = model(tensor)
        
        # Measure
        num_iterations = 50
        latencies = []
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=device, enabled=True):
                _ = model(tensor)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            latencies.append(latency)
            profiler.record("amp_inference", latency)
        
        mean_latency = statistics.mean(latencies)
        logger.info(f"AMP inference latency: mean={mean_latency*1000:.2f}ms")


@pytest.mark.performance
class TestMCTSLatency:
    """Test MCTS search latency with detailed metrics."""
    
    def test_mcts_search_latency_multiple_sims(self, dummy_model, test_board, device, profiler):
        """Measure MCTS search latency across different simulation counts."""
        model = dummy_model.to(device)
        
        simulation_counts = [50, 100, 200, 400, 800]
        batch_size = 96
        
        # Expected targets (ms/move) - adjust based on hardware expectations
        expected_targets = {
            50: 200.0,   # 200ms per move with 50 sims
            100: 400.0,  # 400ms per move with 100 sims
            200: 800.0,  # 800ms per move with 200 sims
            400: 1600.0, # 1.6s per move with 400 sims
            800: 3200.0, # 3.2s per move with 800 sims
        }
        
        results = []
        
        for num_sims in simulation_counts:
            cfg = MCTSConfig(
                num_simulations=num_sims,
                inference_batch_size=batch_size,
                cpuct=2.5,
                parallel_simulations=True,
            )
            mcts = MCTS(cfg, model, device=device)
            
            # Warmup
            _ = mcts.run(test_board, num_simulations=min(10, num_sims))
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            
            # Measure
            num_runs = 5 if num_sims >= 400 else 10
            latencies = []
            
            for _ in range(num_runs):
                start = time.perf_counter()
                visits, policy, value = mcts.run(test_board, num_simulations=num_sims)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                latencies.append(latency)
                profiler.record(f"mcts_{num_sims}sims", latency)
            
            mean_latency_ms = statistics.mean(latencies) * 1000
            median_latency_ms = statistics.median(latencies) * 1000
            min_latency_ms = min(latencies) * 1000
            max_latency_ms = max(latencies) * 1000
            sims_per_sec = num_sims / statistics.mean(latencies)
            expected_ms = expected_targets.get(num_sims, 0.0)
            diff_pct = ((mean_latency_ms - expected_ms) / expected_ms * 100) if expected_ms > 0 else 0.0
            
            results.append([
                str(num_sims),
                f"{mean_latency_ms:.1f}",
                f"{median_latency_ms:.1f}",
                f"{min_latency_ms:.1f}",
                f"{max_latency_ms:.1f}",
                f"{sims_per_sec:.1f}",
                f"{expected_ms:.1f}",
                f"{diff_pct:+.1f}%",
            ])
            
            print(f"[{num_sims} sims] Mean: {mean_latency_ms:.1f}ms, "
                  f"Sims/sec: {sims_per_sec:.1f}, "
                  f"Expected: {expected_ms:.1f}ms, "
                  f"Diff: {diff_pct:+.1f}%", file=sys.stderr)
        
        print_table(
            results,
            headers=["Sims", "Mean (ms)", "Median (ms)", "Min (ms)", "Max (ms)", 
                    "Sims/sec", "Expected (ms)", "Diff %"],
            title="MCTS Latency by Simulation Count"
        )
        
        # Assert reasonable performance (mean latency should be within 2x of expected)
        for i, num_sims in enumerate(simulation_counts):
            mean_ms = float(results[i][1])
            expected_ms = expected_targets.get(num_sims, float('inf'))
            assert mean_ms < expected_ms * 2.0, \
                f"MCTS with {num_sims} sims too slow: {mean_ms:.1f}ms (expected <{expected_ms*2:.1f}ms)"
    
    def test_mcts_move_generation_throughput(self, dummy_model, test_board, device, profiler):
        """Measure self-play move generation throughput (moves/second)."""
        model = dummy_model.to(device)
        
        # Test configurations: (sims, batch_size, expected_moves_per_sec)
        test_configs = [
            (100, 96, 2.5),   # 100 sims, batch 96, expect 2.5 moves/sec
            (200, 96, 1.25),  # 200 sims, batch 96, expect 1.25 moves/sec
            (400, 96, 0.625), # 400 sims, batch 96, expect 0.625 moves/sec
        ]
        
        results = []
        
        for num_sims, batch_size, expected_mps in test_configs:
            cfg = MCTSConfig(
                num_simulations=num_sims,
                inference_batch_size=batch_size,
                cpuct=2.5,
                parallel_simulations=True,
            )
            mcts = MCTS(cfg, model, device=device)
            
            # Warmup
            _ = mcts.run(test_board, num_simulations=min(10, num_sims))
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            
            # Measure 10 moves
            num_moves = 10
            move_times = []
            
            board = chess.Board()
            for move_idx in range(num_moves):
                if board.is_game_over():
                    board = chess.Board()
                
                start = time.perf_counter()
                visits, policy, value = mcts.run(board, num_simulations=num_sims)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                move_time = time.perf_counter() - start
                move_times.append(move_time)
                
                # Make a move to continue the game
                if visits:
                    move = max(visits.items(), key=lambda kv: kv[1])[0]
                    board.push(move)
                else:
                    # Fallback: pick first legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(legal_moves[0])
            
            mean_move_time = statistics.mean(move_times)
            moves_per_sec = 1.0 / mean_move_time
            mean_ms_per_move = mean_move_time * 1000
            diff_pct = ((moves_per_sec - expected_mps) / expected_mps * 100) if expected_mps > 0 else 0.0
            
            results.append([
                str(num_sims),
                str(batch_size),
                f"{mean_ms_per_move:.1f}",
                f"{moves_per_sec:.3f}",
                f"{expected_mps:.3f}",
                f"{diff_pct:+.1f}%",
            ])
            
            profiler.record(f"selfplay_throughput_{num_sims}sims", mean_move_time)
            
            print(f"[{num_sims} sims, batch {batch_size}] "
                  f"{mean_ms_per_move:.1f}ms/move, "
                  f"{moves_per_sec:.3f} moves/sec, "
                  f"Expected: {expected_mps:.3f} moves/sec, "
                  f"Diff: {diff_pct:+.1f}%", file=sys.stderr)
        
        print_table(
            results,
            headers=["Sims", "Batch", "ms/move", "Moves/sec", "Expected (moves/sec)", "Diff %"],
            title="Self-Play Move Generation Throughput"
        )
        
        # Assert minimum throughput requirements
        for i, (num_sims, _, expected_mps) in enumerate(test_configs):
            actual_mps = float(results[i][3])
            assert actual_mps >= expected_mps * 0.5, \
                f"Throughput too low for {num_sims} sims: {actual_mps:.3f} moves/sec (expected >={expected_mps*0.5:.3f})"
    
    def test_mcts_batched_vs_sequential(self, dummy_model, test_board, device, profiler):
        """Compare batched vs sequential MCTS inference."""
        model = dummy_model.to(device)
        
        # Batched configuration
        cfg_batched = MCTSConfig(
            num_simulations=200,
            inference_batch_size=96,
            parallel_simulations=True,
        )
        mcts_batched = MCTS(cfg_batched, model, device=device)
        
        # Sequential configuration
        cfg_seq = MCTSConfig(
            num_simulations=200,
            inference_batch_size=1,
            parallel_simulations=False,
        )
        mcts_seq = MCTS(cfg_seq, model, device=device)
        
        # Warmup
        _ = mcts_batched.run(test_board, num_simulations=10)
        _ = mcts_seq.run(test_board, num_simulations=10)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        # Measure batched (5 runs)
        batched_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = mcts_batched.run(test_board, num_simulations=200)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            batched_times.append(time.perf_counter() - start)
        
        # Measure sequential (5 runs)
        sequential_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = mcts_seq.run(test_board, num_simulations=200)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            sequential_times.append(time.perf_counter() - start)
        
        batched_mean = statistics.mean(batched_times)
        sequential_mean = statistics.mean(sequential_times)
        speedup = sequential_mean / batched_mean if batched_mean > 0 else 0
        
        batched_ms = batched_mean * 1000
        sequential_ms = sequential_mean * 1000
        
        profiler.record("mcts_batched", batched_mean)
        profiler.record("mcts_sequential", sequential_mean)
        
        results = [
            ["Batched (batch=96)", f"{batched_ms:.1f}", f"{200/batched_mean:.2f}"],
            ["Sequential (batch=1)", f"{sequential_ms:.1f}", f"{200/sequential_mean:.2f}"],
            ["Speedup", f"{speedup:.2f}x", ""],
        ]
        
        print_table(
            results,
            headers=["Method", "Mean (ms)", "Sims/sec"],
            title="MCTS Batched vs Sequential Comparison"
        )
        
        print(f"Batched: {batched_ms:.1f}ms, Sequential: {sequential_ms:.1f}ms, "
              f"Speedup: {speedup:.2f}x", file=sys.stderr)
        
        # Batched should be faster (or at least not much slower)
        assert speedup >= 0.5, f"Batched MCTS slower than sequential: {speedup:.2f}x"


@pytest.mark.performance
class TestTrainingPerformance:
    """Test training step performance."""
    
    def test_train_step_latency(self, dummy_model, sample_batch_data, device, profiler):
        """Measure training step latency."""
        from azchess.training.train import train_step
        from torch.optim import AdamW
        
        model = dummy_model.to(device)
        optimizer = AdamW(model.parameters(), lr=0.001)
        scaler = None
        
        batch_size = sample_batch_data["s"].shape[0]
        
        # Warmup
        _ = train_step(
            model, optimizer, scaler, sample_batch_data, device,
            accum_steps=1, augment=False, enable_ssl=False,
        )
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        num_steps = 20
        latencies = []
        
        for _ in range(num_steps):
            start = time.perf_counter()
            _ = train_step(
                model, optimizer, scaler, sample_batch_data, device,
                accum_steps=1, augment=False, enable_ssl=False,
            )
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            latencies.append(latency)
            profiler.record("train_step", latency)
        
        mean_latency = statistics.mean(latencies)
        mean_latency_ms = mean_latency * 1000
        steps_per_sec = 1.0 / mean_latency
        samples_per_sec = batch_size * steps_per_sec
        
        results = [[
            f"{mean_latency_ms:.1f}",
            f"{steps_per_sec:.2f}",
            f"{samples_per_sec:.1f}",
        ]]
        
        print_table(
            results,
            headers=["Latency (ms)", "Steps/sec", "Samples/sec"],
            title=f"Training Step Performance (batch_size={batch_size})"
        )
        
        print(f"Training step: {mean_latency_ms:.1f}ms, "
              f"{steps_per_sec:.2f} steps/sec, "
              f"{samples_per_sec:.1f} samples/sec", file=sys.stderr)
    
    def test_train_step_with_ssl(self, dummy_model, sample_ssl_batch_data, device, profiler):
        """Measure training step latency with SSL."""
        from azchess.training.train import train_step
        from torch.optim import AdamW
        
        model = dummy_model.to(device)
        optimizer = AdamW(model.parameters(), lr=0.001)
        scaler = None
        
        batch_size = sample_ssl_batch_data["s"].shape[0]
        
        # Warmup
        _ = train_step(
            model, optimizer, scaler, sample_ssl_batch_data, device,
            accum_steps=1, augment=False, enable_ssl=True, ssl_weight=0.1,
        )
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        num_steps = 10
        latencies = []
        
        for _ in range(num_steps):
            start = time.perf_counter()
            _ = train_step(
                model, optimizer, scaler, sample_ssl_batch_data, device,
                accum_steps=1, augment=False, enable_ssl=True, ssl_weight=0.1,
            )
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            latencies.append(latency)
            profiler.record("train_step_ssl", latency)
        
        mean_latency = statistics.mean(latencies)
        mean_latency_ms = mean_latency * 1000
        steps_per_sec = 1.0 / mean_latency
        samples_per_sec = batch_size * steps_per_sec
        
        results = [[
            f"{mean_latency_ms:.1f}",
            f"{steps_per_sec:.2f}",
            f"{samples_per_sec:.1f}",
        ]]
        
        print_table(
            results,
            headers=["Latency (ms)", "Steps/sec", "Samples/sec"],
            title=f"Training Step Performance with SSL (batch_size={batch_size})"
        )
        
        print(f"Training step with SSL: {mean_latency_ms:.1f}ms, "
              f"{steps_per_sec:.2f} steps/sec, "
              f"{samples_per_sec:.1f} samples/sec", file=sys.stderr)


@pytest.mark.performance
@pytest.mark.requires_gpu
class TestMemoryPerformance:
    """Test memory usage and profiling."""
    
    def test_inference_memory_usage(self, dummy_model, test_board, device):
        """Measure memory usage during inference."""
        if device not in ("mps", "cuda"):
            pytest.skip("Memory profiling only on MPS/CUDA")
        
        model = dummy_model.to(device)
        model.eval()
        
        encoded = encode_board(test_board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
        
        # Clear cache before measurement
        if device == "mps":
            torch.mps.empty_cache()
            initial_memory = torch.mps.current_allocated_memory()
        elif device == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            pytest.skip("Memory profiling not supported")
        
        # Run inference
        with torch.no_grad():
            _ = model(tensor)
        
        if device == "mps":
            torch.mps.synchronize()
            peak_memory = torch.mps.current_allocated_memory()
        elif device == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
        
        memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # MB
        
        results = [[
            f"{memory_used:.2f}",
            f"{memory_used * 1024:.2f}",
        ]]
        
        print_table(
            results,
            headers=["Memory (MB)", "Memory (KB)"],
            title="Single Inference Memory Usage"
        )
        
        print(f"Inference memory usage: {memory_used:.2f} MB", file=sys.stderr)
        
        # Assert reasonable memory usage (adjust threshold)
        assert memory_used < 500, f"Inference uses too much memory: {memory_used:.2f} MB"
    
    def test_batched_inference_memory_scaling(self, dummy_model, test_board, device):
        """Test memory usage scales reasonably with batch size."""
        if device not in ("mps", "cuda"):
            pytest.skip("Memory profiling only on MPS/CUDA")
        
        model = dummy_model.to(device)
        model.eval()
        
        encoded = encode_board(test_board)
        batch_sizes = [1, 8, 16, 32, 64]
        memory_per_sample = []
        results = []
        
        for batch_size in batch_sizes:
            if device == "mps":
                torch.mps.empty_cache()
                initial = torch.mps.current_allocated_memory()
            elif device == "cuda":
                torch.cuda.empty_cache()
                initial = torch.cuda.memory_allocated()
            
            batch_tensor = torch.from_numpy(
                np.repeat(encoded[None, :], batch_size, axis=0)
            ).to(device)
            
            with torch.no_grad():
                _ = model(batch_tensor)
            
            if device == "mps":
                torch.mps.synchronize()
                peak = torch.mps.current_allocated_memory()
            elif device == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
            
            memory_mb = (peak - initial) / (1024 ** 2)
            per_sample = memory_mb / batch_size
            memory_per_sample.append(per_sample)
            
            results.append([
                str(batch_size),
                f"{memory_mb:.2f}",
                f"{per_sample:.2f}",
            ])
            
            print(f"Batch {batch_size}: {memory_mb:.2f} MB total, {per_sample:.2f} MB/sample", file=sys.stderr)
        
        # Memory per sample should be relatively constant (within 2x)
        max_per_sample = max(memory_per_sample)
        min_per_sample = min(memory_per_sample)
        ratio = max_per_sample / min_per_sample if min_per_sample > 0 else float('inf')
        
        print_table(
            results,
            headers=["Batch Size", "Total Memory (MB)", "Memory/Sample (MB)"],
            title="Batched Inference Memory Scaling"
        )
        
        ratio_row = [["Max/Min Ratio", f"{ratio:.2f}x", ""]]
        print_table(
            ratio_row,
            headers=["Metric", "Value", ""],
            title="Memory Scaling Efficiency"
        )
        
        print(f"Memory per sample ratio: {ratio:.2f}x", file=sys.stderr)
        assert ratio < 3.0, f"Memory scaling not linear: {ratio:.2f}x"

