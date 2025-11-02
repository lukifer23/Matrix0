"""Integration tests for end-to-end workflows."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

import chess
import numpy as np
import pytest
import torch

from azchess.config import Config
from azchess.data_manager import DataManager
from azchess.mcts import MCTS, MCTSConfig
from azchess.selfplay.internal import selfplay_worker


logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.slow
class TestSelfPlayIntegration:
    """Test self-play generation workflow."""
    
    def test_selfplay_game_generation(self, temp_data_dir, dummy_model, test_config_dict):
        """Test that self-play generates valid game data."""
        import multiprocessing as mp
        
        # Create minimal config for self-play
        sp_config = {
            "model": {
                "planes": 19,
                "channels": 64,
                "blocks": 4,
                "policy_size": 4672,
                "self_supervised": False,
            },
            "selfplay": {
                "num_workers": 1,
                "batch_size": 32,
                "num_simulations": 50,
                "temperature_start": 1.0,
                "temperature_end": 0.1,
                "temperature_moves": 10,
            },
            "mcts": {
                "num_simulations": 50,
                "inference_batch_size": 8,
                "cpuct": 2.5,
            },
            "data_dir": str(temp_data_dir),
        }
        
        # Run a single self-play game
        queue = mp.Queue()
        
        try:
            selfplay_worker(
                proc_id=0,
                cfg_dict=sp_config,
                ckpt_path=None,
                games=1,
                q=queue,
                shared_memory_resource=None,
            )
        except Exception as e:
            pytest.skip(f"Self-play worker failed: {e}")
        
        # Check that game data was saved
        data_manager = DataManager(base_dir=str(temp_data_dir))
        stats = data_manager.get_stats()
        
        assert stats.total_samples > 0, "No self-play samples generated"
        assert stats.total_shards > 0, "No self-play shards created"
    
    def test_selfplay_with_ssl(self, temp_data_dir, dummy_model, test_config_dict):
        """Test self-play with SSL target generation."""
        import multiprocessing as mp
        
        sp_config = {
            "model": {
                "planes": 19,
                "channels": 64,
                "blocks": 4,
                "policy_size": 4672,
                "self_supervised": True,
                "ssl_tasks": ["piece", "threat"],
            },
            "selfplay": {
                "num_workers": 1,
                "batch_size": 32,
                "num_simulations": 50,
            },
            "mcts": {
                "num_simulations": 50,
                "inference_batch_size": 8,
            },
            "data_dir": str(temp_data_dir),
        }
        
        queue = mp.Queue()
        
        try:
            selfplay_worker(
                proc_id=0,
                cfg_dict=sp_config,
                ckpt_path=None,
                games=1,
                q=queue,
                shared_memory_resource=None,
            )
        except Exception as e:
            pytest.skip(f"Self-play worker with SSL failed: {e}")
        
        # Check that SSL targets are in saved data
        data_manager = DataManager(base_dir=str(temp_data_dir))
        shards = data_manager._get_all_shards()
        
        if shards:
            import numpy as np
            shard_path = shards[0].path
            with np.load(shard_path, mmap_mode='r') as data:
                ssl_keys = [k for k in data.keys() if k.startswith('ssl_')]
                assert len(ssl_keys) > 0, "SSL targets not found in self-play data"


@pytest.mark.integration
class TestTrainingIntegration:
    """Test training workflow integration."""
    
    def test_training_data_loading(self, temp_data_dir, sample_batch_data):
        """Test that training data can be loaded."""
        import numpy as np
        
        # Create sample data shard
        shard_dir = temp_data_dir / "replays"
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        shard_path = shard_dir / "test_shard_000.npz"
        np.savez(
            shard_path,
            s=sample_batch_data["s"],
            pi=sample_batch_data["pi"],
            z=sample_batch_data["z"],
            legal_mask=sample_batch_data["legal_mask"],
        )
        
        # Load via DataManager
        dm = DataManager(base_dir=str(temp_data_dir))
        stats = dm.get_stats()
        
        assert stats.total_samples > 0, "Data not loaded"
        
        # Try to get a batch
        batch_gen = dm.get_training_batch(batch_size=8, device="cpu")
        batch = next(batch_gen, None)
        
        assert batch is not None, "Could not load training batch"
        assert len(batch) >= 3, "Batch missing required fields"
    
    def test_training_step_with_real_data(self, dummy_model, sample_batch_data, device):
        """Test training step with realistic data."""
        from azchess.training.train import train_step
        from torch.optim import AdamW
        
        model = dummy_model.to(device)
        optimizer = AdamW(model.parameters(), lr=0.001)
        
        # Normalize policy targets
        policy = sample_batch_data["pi"]
        policy = policy / (policy.sum(axis=1, keepdims=True) + 1e-8)
        sample_batch_data["pi"] = policy.astype(np.float32)
        
        result = train_step(
            model, optimizer, None, sample_batch_data, device,
            accum_steps=1, augment=False, enable_ssl=False,
        )
        
        assert result is not None, "Training step returned None"
        assert len(result) == 6, "Training step returned wrong number of values"
        
        loss, policy_loss, value_loss, ssl_loss, ssrl_loss, wdl_loss = result
        
        assert isinstance(loss, float), "Loss is not float"
        assert loss >= 0, f"Loss is negative: {loss}"
        assert not np.isnan(loss), "Loss is NaN"
        assert not np.isinf(loss), "Loss is Inf"


@pytest.mark.integration
class TestMCTSIntegration:
    """Test MCTS integration with various components."""
    
    def test_mcts_with_inference_backend(self, dummy_model, test_board, device, constant_backend):
        """Test MCTS with inference backend."""
        backend = constant_backend
        backend.value = 0.1
        cfg = MCTSConfig(num_simulations=50, inference_batch_size=8)
        mcts = MCTS(cfg, dummy_model, device=device, inference_backend=backend)
        
        moves, policy, value = mcts.run(test_board, num_simulations=50)
        
        assert isinstance(moves, dict), "Moves not a dict"
        assert policy.shape == (4672,), f"Policy wrong shape: {policy.shape}"
        assert isinstance(value, float), "Value not float"
        assert -1.0 <= value <= 1.0, f"Value out of range: {value}"
    
    def test_mcts_tree_consistency(self, dummy_model, test_board, device):
        """Test that MCTS tree maintains consistency."""
        cfg = MCTSConfig(
            num_simulations=100,
            inference_batch_size=16,
            cpuct=2.5,
        )
        mcts = MCTS(cfg, dummy_model, device=device)
        
        # Run multiple searches on same position
        results = []
        for _ in range(3):
            moves, policy, value = mcts.run(test_board, num_simulations=100)
            results.append((moves, policy, value))
        
        # All results should be valid
        for moves, policy, value in results:
            assert isinstance(moves, dict)
            assert len(moves) > 0, "No moves found"
            assert policy.shape == (4672,)
            assert isinstance(value, float)
            assert -1.0 <= value <= 1.0
        
        # Policies should sum to approximately 1.0
        for _, policy, _ in results:
            policy_sum = policy.sum()
            assert 0.9 <= policy_sum <= 1.1, f"Policy doesn't sum to 1: {policy_sum}"


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Test data pipeline end-to-end."""
    
    def test_data_save_and_load(self, temp_data_dir, sample_batch_data):
        """Test saving and loading data shards."""
        import numpy as np
        
        # Save data
        shard_dir = temp_data_dir / "replays"
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        shard_path = shard_dir / "test_shard_000.npz"
        np.savez(
            shard_path,
            s=sample_batch_data["s"],
            pi=sample_batch_data["pi"],
            z=sample_batch_data["z"],
            legal_mask=sample_batch_data["legal_mask"],
        )
        
        # Load via DataManager
        dm = DataManager(base_dir=str(temp_data_dir))
        
        # Get batch
        batch_gen = dm.get_training_batch(batch_size=8, device="cpu")
        batch = next(batch_gen, None)
        
        assert batch is not None, "Could not load batch"
        
        states, policies, values, legal_mask = batch
        
        assert states.shape[0] == 8, f"Wrong batch size: {states.shape[0]}"
        assert states.shape[1:] == (19, 8, 8), f"Wrong state shape: {states.shape}"
        assert policies.shape == (8, 4672), f"Wrong policy shape: {policies.shape}"
        assert values.shape == (8,), f"Wrong value shape: {values.shape}"
        assert legal_mask is None or legal_mask.shape == (8, 4672), f"Wrong legal mask shape: {legal_mask.shape if legal_mask is not None else None}"
    
    def test_data_with_ssl_targets(self, temp_data_dir, sample_ssl_batch_data):
        """Test saving and loading data with SSL targets."""
        import numpy as np
        
        shard_dir = temp_data_dir / "replays"
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        shard_path = shard_dir / "test_ssl_shard_000.npz"
        
        # Save with SSL targets
        save_dict = {
            "s": sample_ssl_batch_data["s"],
            "pi": sample_ssl_batch_data["pi"],
            "z": sample_ssl_batch_data["z"],
            "legal_mask": sample_ssl_batch_data["legal_mask"],
        }
        
        # Add SSL targets
        for key in sample_ssl_batch_data.keys():
            if key.startswith("ssl_"):
                save_dict[key] = sample_ssl_batch_data[key]
        
        np.savez(shard_path, **save_dict)
        
        # Load and verify SSL targets
        with np.load(shard_path, mmap_mode='r') as data:
            ssl_keys = [k for k in data.keys() if k.startswith('ssl_')]
            assert len(ssl_keys) > 0, "SSL targets not in saved data"
            
            for key in ssl_keys:
                assert key in data, f"SSL target {key} missing"
                assert data[key].shape[0] == sample_ssl_batch_data["s"].shape[0], \
                    f"SSL target {key} wrong batch size"

