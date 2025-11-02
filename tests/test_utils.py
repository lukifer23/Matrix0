"""Test utilities and helper classes."""

import numpy as np
import torch


class DummyModel(torch.nn.Module):
    """Dummy model for testing that returns valid shapes."""
    
    def __init__(self, policy_size: int = 4672, device: str = "cpu"):
        super().__init__()
        self.policy_size = policy_size
        self.device_str = device
        # Add cfg attribute for compatibility
        from azchess.model.resnet import NetConfig
        self.cfg = NetConfig(planes=19, channels=64, blocks=4, policy_size=policy_size)
        
        # Add minimal learnable parameters so optimizers work
        # Use a simple linear layer to map input to policy/value
        self.policy_head = torch.nn.Linear(19 * 8 * 8, policy_size)
        self.value_head = torch.nn.Linear(19 * 8 * 8, 1)
    
    def forward(self, x, return_ssl=False):
        batch_size = x.shape[0]
        # Flatten input: (B, 19, 8, 8) -> (B, 19*8*8)
        x_flat = x.view(batch_size, -1)
        
        # Generate outputs through learnable layers
        policy = self.policy_head(x_flat)
        value = self.value_head(x_flat)
        
        if return_ssl:
            ssl = {'piece': torch.zeros((batch_size, 13, 8, 8), dtype=torch.float32, device=x.device)}
            return policy, value, ssl
        return policy, value


class ConstantBackend:
    """Inference backend that returns constant values for testing."""
    
    def __init__(self, value: float = 0.0, policy_size: int = 4672):
        self.value = float(value)
        self.policy_size = policy_size
        self.logits = np.zeros((policy_size,), dtype=np.float32)
    
    def infer_np(self, batch: np.ndarray):
        batch_size = batch.shape[0]
        logits = np.repeat(self.logits[None, :], batch_size, axis=0)
        values = np.full((batch_size,), self.value, dtype=np.float32)
        return logits, values

