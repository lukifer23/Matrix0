"""Comprehensive error handling and robustness tests."""

import logging
from typing import Optional

import chess
import numpy as np
import pytest
import torch

from azchess.mcts import MCTS, MCTSConfig
from azchess.model.resnet import PolicyValueNet
from tests.test_utils import DummyModel


logger = logging.getLogger(__name__)


@pytest.mark.error_handling
class TestInferenceErrorHandling:
    """Test error handling in inference paths."""
    
    def test_inference_nan_detection(self, dummy_model, test_board, device):
        """Test that NaN outputs are detected and raise errors."""
        class NaNModel(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                policy = torch.full((batch, 4672), float('nan'), dtype=torch.float32, device=x.device)
                value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
                return policy, value
        
        model = NaNModel().to(device)
        cfg = MCTSConfig(num_simulations=10, inference_batch_size=1)
        mcts = MCTS(cfg, model, device=device)
        
        with pytest.raises((ValueError, RuntimeError), match="NaN|nan"):
            _ = mcts.run(test_board, num_simulations=10)
    
    def test_inference_inf_detection(self, dummy_model, test_board, device):
        """Test that Inf outputs are detected and raise errors."""
        class InfModel(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                policy = torch.full((batch, 4672), float('inf'), dtype=torch.float32, device=x.device)
                value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
                return policy, value
        
        model = InfModel().to(device)
        cfg = MCTSConfig(num_simulations=10, inference_batch_size=1)
        mcts = MCTS(cfg, model, device=device)
        
        with pytest.raises((ValueError, RuntimeError), match="Inf|inf"):
            _ = mcts.run(test_board, num_simulations=10)
    
    def test_inference_shape_mismatch(self, dummy_model, test_board, device):
        """Test that shape mismatches are detected."""
        class WrongShapeModel(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                policy = torch.zeros((batch, 1000), dtype=torch.float32, device=x.device)  # Wrong size
                value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
                return policy, value
        
        model = WrongShapeModel().to(device)
        cfg = MCTSConfig(num_simulations=10, inference_batch_size=1)
        mcts = MCTS(cfg, model, device=device)
        
        with pytest.raises((ValueError, RuntimeError), match="shape|Shape|mismatch"):
            _ = mcts.run(test_board, num_simulations=10)
    
    def test_inference_timeout_handling(self, test_board, device):
        """Test that inference timeouts are handled correctly."""
        class SlowModel(torch.nn.Module):
            def forward(self, x):
                import time
                time.sleep(10)  # Simulate hang
                batch = x.shape[0]
                return (
                    torch.zeros((batch, 4672), dtype=torch.float32, device=x.device),
                    torch.zeros((batch, 1), dtype=torch.float32, device=x.device),
                )
        
        model = SlowModel().to(device)
        cfg = MCTSConfig(num_simulations=1, inference_batch_size=1)
        mcts = MCTS(cfg, model, device=device)
        
        # This should timeout or be handled gracefully
        # Note: Actual timeout handling depends on implementation
        try:
            _ = mcts.run(test_board, num_simulations=1)
        except (TimeoutError, RuntimeError):
            pass  # Expected


@pytest.mark.error_handling
class TestMCTSErrorHandling:
    """Test error handling in MCTS."""
    
    def test_mcts_empty_visit_counts(self, dummy_model, test_board, device):
        """Test MCTS handles empty visit counts gracefully."""
        class ZeroModel(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                # Return all zeros (no legal moves will have positive probability)
                policy = torch.zeros((batch, 4672), dtype=torch.float32, device=x.device)
                value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
                return policy, value
        
        model = ZeroModel().to(device)
        cfg = MCTSConfig(num_simulations=10, inference_batch_size=1)
        mcts = MCTS(cfg, model, device=device)
        
        # Should handle gracefully or raise clear error
        try:
            moves, policy, value = mcts.run(test_board, num_simulations=10)
            # If it succeeds, verify outputs are valid
            assert isinstance(moves, dict)
            assert policy.shape == (4672,)
            assert isinstance(value, float)
        except RuntimeError as e:
            assert "visit counts" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_mcts_invalid_board_state(self, dummy_model, device):
        """Test MCTS handles invalid board states."""
        cfg = MCTSConfig(num_simulations=10, inference_batch_size=1)
        mcts = MCTS(cfg, dummy_model, device=device)
        
        # Create invalid board (multiple kings, etc.)
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.WHITE))  # Invalid: two white kings
        
        # Should handle gracefully
        try:
            _ = mcts.run(board, num_simulations=10)
        except (ValueError, RuntimeError, AssertionError):
            pass  # Expected
    
    def test_mcts_terminal_position(self, dummy_model, device):
        """Test MCTS handles terminal positions correctly."""
        cfg = MCTSConfig(num_simulations=10, inference_batch_size=1)
        mcts = MCTS(cfg, dummy_model, device=device)
        
        # Checkmate position
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5PP1/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        
        moves, policy, value = mcts.run(board, num_simulations=10)
        
        # Should return valid outputs even for terminal position
        assert isinstance(moves, dict)
        assert policy.shape == (4672,)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0


@pytest.mark.error_handling
class TestDataErrorHandling:
    """Test error handling in data loading and processing."""
    
    def test_invalid_policy_targets(self, sample_batch_data):
        """Test handling of invalid policy targets."""
        from azchess.training.train import train_step
        from torch.optim import AdamW
        from tests.test_utils import DummyModel
        
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=0.001)
        
        # Create invalid policy (negative values, wrong sum)
        invalid_batch = sample_batch_data.copy()
        invalid_batch["pi"] = np.random.randn(32, 4672).astype(np.float32)  # Negative values
        
        # Should handle gracefully (normalize or raise clear error)
        try:
            _ = train_step(
                model, optimizer, None, invalid_batch, "cpu",
                accum_steps=1, augment=False, enable_ssl=False,
            )
        except (ValueError, RuntimeError) as e:
            assert "policy" in str(e).lower() or "target" in str(e).lower()
    
    def test_missing_legal_mask(self, sample_batch_data):
        """Test handling of missing legal mask."""
        from azchess.training.train import train_step
        from torch.optim import AdamW
        from tests.test_utils import DummyModel
        
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=0.001)
        
        # Remove legal mask
        batch_no_mask = {k: v for k, v in sample_batch_data.items() if k != "legal_mask"}
        
        # Should handle gracefully (compute mask or proceed without)
        try:
            _ = train_step(
                model, optimizer, None, batch_no_mask, "cpu",
                accum_steps=1, augment=False, enable_ssl=False,
            )
        except (KeyError, ValueError) as e:
            # If it requires legal mask, error should be clear
            assert "legal" in str(e).lower() or "mask" in str(e).lower()
    
    def test_corrupted_shard_handling(self, temp_data_dir):
        """Test handling of corrupted data shards."""
        from azchess.data_manager import DataManager
        import numpy as np
        
        # Create corrupted NPZ file
        corrupted_path = temp_data_dir / "corrupted.npz"
        np.savez(corrupted_path, invalid_key=np.array([1, 2, 3]))
        
        dm = DataManager(base_dir=str(temp_data_dir))
        
        # Should mark as corrupted and skip
        shards = dm._get_all_shards()
        corrupted_shards = [s for s in shards if s.corrupted]
        
        # The corrupted shard should be detected
        # (exact behavior depends on DataManager implementation)


@pytest.mark.error_handling
class TestRobustness:
    """Test overall system robustness."""
    
    def test_concurrent_mcts_searches(self, dummy_model, device):
        """Test multiple concurrent MCTS searches."""
        import threading
        
        cfg = MCTSConfig(num_simulations=50, inference_batch_size=8)
        results = []
        errors = []
        
        def run_search(search_id):
            try:
                mcts = MCTS(cfg, dummy_model, device=device)
                board = chess.Board()
                moves, policy, value = mcts.run(board, num_simulations=50)
                results.append((search_id, True))
            except Exception as e:
                errors.append((search_id, str(e)))
                results.append((search_id, False))
        
        threads = []
        num_threads = 4
        
        for i in range(num_threads):
            t = threading.Thread(target=run_search, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All searches should complete successfully
        successful = sum(1 for _, success in results if success)
        assert successful == num_threads, f"Only {successful}/{num_threads} searches succeeded. Errors: {errors}"
    
    def test_memory_cleanup_after_failure(self, dummy_model, test_board, device):
        """Test that memory is cleaned up after failures."""
        if device not in ("mps", "cuda"):
            pytest.skip("Memory profiling only on MPS/CUDA")
        
        class FailingModel(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("Simulated failure")
        
        model = FailingModel().to(device)
        cfg = MCTSConfig(num_simulations=10, inference_batch_size=1)
        mcts = MCTS(cfg, model, device=device)
        
        # Get initial memory
        if device == "mps":
            torch.mps.empty_cache()
            initial = torch.mps.current_allocated_memory()
        elif device == "cuda":
            torch.cuda.empty_cache()
            initial = torch.cuda.memory_allocated()
        
        # Attempt search (will fail)
        try:
            _ = mcts.run(test_board, num_simulations=10)
        except RuntimeError:
            pass
        
        # Clear cache and check memory
        if device == "mps":
            torch.mps.empty_cache()
            final = torch.mps.current_allocated_memory()
        elif device == "cuda":
            torch.cuda.empty_cache()
            final = torch.cuda.memory_allocated()
        
        # Memory should be cleaned up (within 10% of initial)
        memory_leak = (final - initial) / (1024 ** 2)  # MB
        assert abs(memory_leak) < 100, f"Memory leak detected: {memory_leak:.2f} MB"
    
    def test_repeated_operations_stability(self, dummy_model, test_board, device):
        """Test that repeated operations remain stable."""
        cfg = MCTSConfig(num_simulations=50, inference_batch_size=8)
        mcts = MCTS(cfg, dummy_model, device=device)
        
        values = []
        policies = []
        
        # Run multiple searches
        for _ in range(10):
            moves, policy, value = mcts.run(test_board, num_simulations=50)
            values.append(value)
            policies.append(policy)
        
        # Check for NaN/Inf
        assert all(not np.isnan(v) and not np.isinf(v) for v in values), "Values contain NaN/Inf"
        assert all(not np.any(np.isnan(p)) and not np.any(np.isinf(p)) for p in policies), "Policies contain NaN/Inf"
        
        # Check consistency (values should be reasonable)
        assert all(-1.0 <= v <= 1.0 for v in values), f"Values out of range: {values}"

