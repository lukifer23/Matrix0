"""Stress tests for system robustness under load."""

import logging
import threading
import time
from typing import List

import chess
import numpy as np
import pytest
import torch

from azchess.mcts import MCTS, MCTSConfig
from azchess.encoding import encode_board


logger = logging.getLogger(__name__)


@pytest.mark.stress
@pytest.mark.slow
class TestStressMCTS:
    """Stress tests for MCTS under various conditions."""
    
    def test_extended_mcts_search(self, dummy_model, test_board, device):
        """Test MCTS with extended search (many simulations)."""
        cfg = MCTSConfig(
            num_simulations=1000,
            inference_batch_size=32,
            cpuct=2.5,
        )
        mcts = MCTS(cfg, dummy_model, device=device)
        
        start_time = time.time()
        moves, policy, value = mcts.run(test_board, num_simulations=1000)
        elapsed = time.time() - start_time
        
        assert isinstance(moves, dict), "Moves not a dict"
        assert len(moves) > 0, "No moves found"
        assert policy.shape == (4672,), f"Policy wrong shape: {policy.shape}"
        assert isinstance(value, float), "Value not float"
        assert -1.0 <= value <= 1.0, f"Value out of range: {value}"
        
        logger.info(f"Extended search (1000 sims) completed in {elapsed:.2f}s")
    
    def test_mcts_many_positions(self, dummy_model, device):
        """Test MCTS on many different positions."""
        cfg = MCTSConfig(num_simulations=100, inference_batch_size=16)
        mcts = MCTS(cfg, dummy_model, device=device)
        
        positions = [
            chess.Board(),  # Starting position
            chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),  # Standard
            chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),  # Complex
        ]
        
        results = []
        for i, board in enumerate(positions):
            try:
                moves, policy, value = mcts.run(board, num_simulations=100)
                results.append((i, True, value))
            except Exception as e:
                results.append((i, False, str(e)))
        
        successful = sum(1 for _, success, _ in results if success)
        assert successful == len(positions), f"Only {successful}/{len(positions)} positions succeeded: {results}"
    
    def test_mcts_rapid_searches(self, dummy_model, test_board, device):
        """Test rapid consecutive MCTS searches."""
        cfg = MCTSConfig(num_simulations=50, inference_batch_size=8)
        mcts = MCTS(cfg, dummy_model, device=device)
        
        num_searches = 20
        start_time = time.time()
        
        for i in range(num_searches):
            moves, policy, value = mcts.run(test_board, num_simulations=50)
            assert isinstance(moves, dict)
            assert policy.shape == (4672,)
            assert isinstance(value, float)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_searches
        
        logger.info(f"Rapid searches: {num_searches} searches in {elapsed:.2f}s (avg {avg_time:.3f}s/search)")
        assert avg_time < 5.0, f"Average search time too high: {avg_time:.3f}s"


@pytest.mark.stress
@pytest.mark.slow
class TestStressInference:
    """Stress tests for inference performance."""
    
    def test_large_batch_inference(self, dummy_model, test_board, device):
        """Test inference with very large batches."""
        model = dummy_model.to(device)
        model.eval()
        
        encoded = encode_board(test_board)
        large_batch_sizes = [128, 256, 512]
        
        for batch_size in large_batch_sizes:
            try:
                batch_tensor = torch.from_numpy(
                    np.repeat(encoded[None, :], batch_size, axis=0)
                ).to(device)
                
                start = time.time()
                with torch.no_grad():
                    policy, value = model(batch_tensor)
                if device == "mps":
                    torch.mps.synchronize()
                elif device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                
                assert policy.shape[0] == batch_size, f"Wrong batch size: {policy.shape[0]}"
                assert value.shape[0] == batch_size, f"Wrong value batch size: {value.shape[0]}"
                
                throughput = batch_size / elapsed
                logger.info(f"Batch {batch_size}: {elapsed:.3f}s, {throughput:.1f} samples/s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at batch size {batch_size}: {e}")
                    break
                raise
    
    def test_continuous_inference(self, dummy_model, test_board, device):
        """Test continuous inference for extended period."""
        model = dummy_model.to(device)
        model.eval()
        
        encoded = encode_board(test_board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
        
        num_iterations = 1000
        start_time = time.time()
        errors = []
        
        for i in range(num_iterations):
            try:
                with torch.no_grad():
                    _ = model(tensor)
                if i % 100 == 0 and device == "mps":
                    torch.mps.empty_cache()
            except Exception as e:
                errors.append((i, str(e)))
        
        elapsed = time.time() - start_time
        throughput = num_iterations / elapsed
        
        logger.info(f"Continuous inference: {num_iterations} iterations in {elapsed:.2f}s ({throughput:.1f} iter/s)")
        
        assert len(errors) == 0, f"Inference errors: {errors}"


@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.requires_gpu
class TestStressMemory:
    """Stress tests for memory management."""
    
    def test_memory_under_load(self, dummy_model, test_board, device):
        """Test memory usage under sustained load."""
        if device not in ("mps", "cuda"):
            pytest.skip("Memory profiling only on MPS/CUDA")
        
        cfg = MCTSConfig(num_simulations=200, inference_batch_size=32)
        mcts = MCTS(cfg, dummy_model, device=device)
        
        # Clear cache
        if device == "mps":
            torch.mps.empty_cache()
            initial = torch.mps.current_allocated_memory()
        elif device == "cuda":
            torch.cuda.empty_cache()
            initial = torch.cuda.memory_allocated()
        
        # Run many searches
        peak_memory = initial
        for i in range(20):
            _ = mcts.run(test_board, num_simulations=200)
            
            if device == "mps":
                current = torch.mps.current_allocated_memory()
            elif device == "cuda":
                current = torch.cuda.memory_allocated()
            
            peak_memory = max(peak_memory, current)
            
            # Periodic cleanup
            if i % 5 == 0:
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
        
        # Final cleanup
        if device == "mps":
            torch.mps.empty_cache()
            final = torch.mps.current_allocated_memory()
        elif device == "cuda":
            torch.cuda.empty_cache()
            final = torch.cuda.memory_allocated()
        
        memory_growth = (final - initial) / (1024 ** 2)  # MB
        peak_growth = (peak_memory - initial) / (1024 ** 2)  # MB
        
        logger.info(f"Memory: initial={initial/(1024**2):.1f}MB, peak={peak_memory/(1024**2):.1f}MB, final={final/(1024**2):.1f}MB")
        logger.info(f"Memory growth: {memory_growth:.1f}MB, peak growth: {peak_growth:.1f}MB")
        
        # Check for memory leaks (should be within 200MB)
        assert abs(memory_growth) < 200, f"Memory leak detected: {memory_growth:.1f}MB"


@pytest.mark.stress
@pytest.mark.slow
class TestStressConcurrency:
    """Stress tests for concurrent operations."""
    
    def test_concurrent_mcts_searches(self, dummy_model, device):
        """Test many concurrent MCTS searches."""
        import threading
        
        cfg = MCTSConfig(num_simulations=100, inference_batch_size=16)
        results = []
        errors = []
        lock = threading.Lock()
        
        def run_search(search_id):
            try:
                mcts = MCTS(cfg, dummy_model, device=device)
                board = chess.Board()
                moves, policy, value = mcts.run(board, num_simulations=100)
                
                with lock:
                    results.append((search_id, True, value))
            except Exception as e:
                with lock:
                    errors.append((search_id, str(e)))
                    results.append((search_id, False, None))
        
        num_threads = 8
        threads = []
        
        start_time = time.time()
        for i in range(num_threads):
            t = threading.Thread(target=run_search, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        elapsed = time.time() - start_time
        
        successful = sum(1 for _, success, _ in results if success)
        logger.info(f"Concurrent searches: {successful}/{num_threads} succeeded in {elapsed:.2f}s")
        
        assert successful == num_threads, f"Only {successful}/{num_threads} searches succeeded. Errors: {errors}"
        
        # Check all values are valid
        for search_id, success, value in results:
            if success:
                assert isinstance(value, float)
                assert -1.0 <= value <= 1.0, f"Invalid value: {value}"
    
    def test_concurrent_inference(self, dummy_model, test_board, device):
        """Test concurrent inference operations."""
        import threading
        
        model = dummy_model.to(device)
        model.eval()
        
        encoded = encode_board(test_board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def run_inference(iter_id):
            try:
                with torch.no_grad():
                    policy, value = model(tensor)
                
                with lock:
                    results.append((iter_id, True))
            except Exception as e:
                with lock:
                    errors.append((iter_id, str(e)))
                    results.append((iter_id, False))
        
        num_threads = 10
        num_iterations_per_thread = 20
        
        threads = []
        for t_id in range(num_threads):
            for i in range(num_iterations_per_thread):
                iter_id = t_id * num_iterations_per_thread + i
                t = threading.Thread(target=run_inference, args=(iter_id,))
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join()
        
        successful = sum(1 for _, success in results if success)
        total = len(results)
        
        logger.info(f"Concurrent inference: {successful}/{total} succeeded")
        
        # Allow some failures due to thread safety, but most should succeed
        success_rate = successful / total if total > 0 else 0
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"

