#!/usr/bin/env python3
"""
Small ResNet Model for GRPO Experiments

A compact ResNet architecture optimized for quick iteration and experimentation.
Maintains core chess understanding while being lightweight for rapid prototyping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SmallResNetBlock(nn.Module):
    """Simplified ResNet block for efficiency"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SmallResNet(nn.Module):
    """Compact ResNet for chess position evaluation"""

    def __init__(self, input_channels: int = 19, base_channels: int = 64, num_blocks: int = 4):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Residual blocks
        self.blocks = nn.ModuleList([
            SmallResNetBlock(base_channels) for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(base_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)  # 8x8 board, 4672 legal moves

        # Value head
        self.value_conv = nn.Conv2d(base_channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Attention mask support
        self.attention_mask = None

    def set_attention_mask(self, mask):
        """Set attention mask for legal moves"""
        self.attention_mask = mask

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Apply attention mask if available
        if self.attention_mask is not None:
            policy_logits = policy_logits + self.attention_mask

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value


class ChessSmallResNet:
    """Factory class for creating small ResNet models"""

    @staticmethod
    def create(input_channels: int = 19, base_channels: int = 64, num_blocks: int = 4):
        """Create a small ResNet model"""
        return SmallResNet(input_channels, base_channels, num_blocks)

    @staticmethod
    def get_parameter_count(model):
        """Get total parameter count"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_info(model):
        """Get model information"""
        params = ChessSmallResNet.get_parameter_count(model)
        return {
            'architecture': 'SmallResNet',
            'parameters': params,
            'parameter_count': f"{params:,}",
            'base_channels': model.conv1.out_channels,
            'num_blocks': len(model.blocks)
        }


if __name__ == "__main__":
    # Test the model
    model = ChessSmallResNet.create()
    print("=== Small ResNet Model Test ===")
    print(f"Model: {ChessSmallResNet.get_model_info(model)}")

    # Test forward pass
    x = torch.randn(1, 19, 8, 8)  # Batch size 1, 19 input channels, 8x8 board
    with torch.no_grad():
        policy, value = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")
    print("✅ Model test passed!")
