"""Pytest configuration and shared fixtures for Matrix0 test suite."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import chess
import numpy as np
import pytest
import torch
import yaml

from azchess.config import Config
from azchess.encoding import encode_board
from azchess.model.resnet import PolicyValueNet, NetConfig
from tests.test_utils import DummyModel, ConstantBackend


# Configure test logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "performance: Performance and latency benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for end-to-end workflows"
    )
    config.addinivalue_line(
        "markers", "stress: Stress tests for system robustness"
    )
    config.addinivalue_line(
        "markers", "error_handling: Error handling and robustness tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: Tests that require GPU/MPS"
    )
    config.addinivalue_line(
        "markers", "requires_cuda: Tests that require CUDA"
    )
    config.addinivalue_line(
        "markers", "requires_mps: Tests that require MPS"
    )


@pytest.fixture(scope="session")
def test_config_dict() -> Dict[str, Any]:
    """Load test configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


@pytest.fixture(scope="session")
def test_config(test_config_dict) -> Config:
    """Create Config instance for testing."""
    return Config(test_config_dict)


@pytest.fixture(scope="function")
def temp_data_dir(tmp_path) -> Path:
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope="function")
def dummy_model() -> DummyModel:
    """Create a dummy model for testing."""
    return DummyModel()


@pytest.fixture(scope="function")
def dummy_model_mps() -> DummyModel:
    """Create a dummy model on MPS if available, else CPU."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DummyModel()
    model.device_str = device
    return model.to(device) if hasattr(model, 'to') else model


@pytest.fixture(scope="function")
def test_board() -> chess.Board:
    """Create a standard chess board for testing."""
    return chess.Board()


@pytest.fixture(scope="function")
def encoded_board(test_board) -> np.ndarray:
    """Encode a standard board position."""
    return encode_board(test_board)


@pytest.fixture(scope="function")
def constant_backend() -> ConstantBackend:
    """Create a constant inference backend."""
    return ConstantBackend(value=0.0)


@pytest.fixture(scope="function")
def device() -> str:
    """Get test device (MPS if available, else CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="function", autouse=True)
def clear_torch_cache():
    """Clear PyTorch cache before each test."""
    yield
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def minimal_model_config() -> Dict[str, Any]:
    """Minimal model configuration for fast tests."""
    return {
        "planes": 19,
        "channels": 64,
        "blocks": 4,
        "policy_size": 4672,
        "attention": False,
        "self_supervised": False,
    }


@pytest.fixture(scope="function")
def full_model_config(test_config_dict) -> Dict[str, Any]:
    """Full model configuration from config.yaml."""
    model_cfg = test_config_dict.get("model", {})
    if not model_cfg:
        return {
            "planes": 19,
            "channels": 192,
            "blocks": 16,
            "policy_size": 4672,
            "attention": True,
            "self_supervised": True,
            "ssl_tasks": ["piece", "threat", "pin"],
        }
    return model_cfg


@pytest.fixture(scope="function")
def sample_batch_data() -> Dict[str, np.ndarray]:
    """Create sample batch data for training tests."""
    batch_size = 32
    # Create valid probability distributions by normalizing random values
    pi_raw = np.random.rand(batch_size, 4672).astype(np.float32)
    pi = pi_raw / pi_raw.sum(axis=1, keepdims=True)
    return {
        "s": np.random.randn(batch_size, 19, 8, 8).astype(np.float32),
        "pi": pi,
        "z": np.random.uniform(-1.0, 1.0, batch_size).astype(np.float32),
        "legal_mask": np.ones((batch_size, 4672), dtype=np.uint8),
    }


@pytest.fixture(scope="function")
def sample_ssl_batch_data(sample_batch_data) -> Dict[str, np.ndarray]:
    """Add SSL targets to sample batch data."""
    batch_size = sample_batch_data["s"].shape[0]
    sample_batch_data["ssl_piece"] = np.random.randint(0, 13, (batch_size, 8, 8)).astype(np.float32)
    sample_batch_data["ssl_threat"] = np.random.rand(batch_size, 8, 8).astype(np.float32)
    return sample_batch_data

