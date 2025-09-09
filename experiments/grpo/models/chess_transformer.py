#!/usr/bin/env python3
"""
Chess Transformer Model for GRPO Experiments

A transformer-based architecture for chess that treats the board as a sequence
of positions. This could capture long-range dependencies better than CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for board positions"""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class ChessTransformerBlock(nn.Module):
    """Transformer encoder block adapted for chess"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class ChessTransformer(nn.Module):
    """Transformer-based chess model"""

    def __init__(self, input_channels: int = 19, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = d_model

        # Convert board to sequence: flatten 8x8 board into 64 positions
        self.board_to_sequence = nn.Linear(input_channels, d_model)

        # Positional encoding for board positions
        self.pos_encoder = PositionalEncoding(d_model, max_len=64)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            ChessTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 4672)  # 4672 legal moves
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        # Attention mask for legal moves
        self.attention_mask = None

    def set_attention_mask(self, mask):
        """Set attention mask for legal moves"""
        self.attention_mask = mask

    def board_to_tokens(self, board):
        """
        Convert board representation to token sequence
        Input: (batch, channels, 8, 8)
        Output: (batch, 64, d_model)
        """
        batch_size, channels, height, width = board.shape
        # Flatten spatial dimensions: (batch, channels, 64)
        board_flat = board.view(batch_size, channels, -1)
        # Transpose to (batch, 64, channels)
        board_flat = board_flat.transpose(1, 2)
        # Project to model dimension: (batch, 64, d_model)
        tokens = self.board_to_sequence(board_flat)
        return tokens

    def forward(self, x):
        # Convert board to token sequence
        tokens = self.board_to_tokens(x)  # (batch, 64, d_model)

        # Add positional encoding
        tokens = self.pos_encoder(tokens)

        # Apply transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Global average pooling across positions
        # (batch, 64, d_model) -> (batch, d_model)
        representation = tokens.mean(dim=1)

        # Policy head
        policy_logits = self.policy_head(representation)

        # Apply attention mask if available
        if self.attention_mask is not None:
            policy_logits = policy_logits + self.attention_mask

        # Value head
        value = self.value_head(representation)

        return policy_logits, value


class ChessTransformerFactory:
    """Factory for creating chess transformer models"""

    @staticmethod
    def create_small(input_channels: int = 19, d_model: int = 128, nhead: int = 4,
                    num_layers: int = 4, dim_feedforward: int = 512):
        """Create a small transformer for quick iteration"""
        return ChessTransformer(input_channels, d_model, nhead, num_layers, dim_feedforward)

    @staticmethod
    def create_medium(input_channels: int = 19, d_model: int = 256, nhead: int = 8,
                     num_layers: int = 6, dim_feedforward: int = 1024):
        """Create a medium transformer with more capacity"""
        return ChessTransformer(input_channels, d_model, nhead, num_layers, dim_feedforward)

    @staticmethod
    def get_parameter_count(model):
        """Get total parameter count"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_info(model):
        """Get model information"""
        params = ChessTransformerFactory.get_parameter_count(model)
        return {
            'architecture': 'ChessTransformer',
            'parameters': params,
            'parameter_count': f"{params:,}",
            'd_model': model.d_model,
            'num_layers': len(model.transformer_layers),
            'nhead': model.transformer_layers[0].self_attn.num_heads
        }


if __name__ == "__main__":
    # Test both small and medium models
    print("=== Chess Transformer Model Test ===")

    # Small model for quick iteration
    small_model = ChessTransformerFactory.create_small()
    print(f"Small Model: {ChessTransformerFactory.get_model_info(small_model)}")

    # Medium model for more capacity
    medium_model = ChessTransformerFactory.create_medium()
    print(f"Medium Model: {ChessTransformerFactory.get_model_info(medium_model)}")

    # Test forward pass
    x = torch.randn(1, 19, 8, 8)  # Batch size 1, 19 input channels, 8x8 board

    for name, model in [("Small", small_model), ("Medium", medium_model)]:
        with torch.no_grad():
            policy, value = model(x)
        print(f"{name} - Input: {x.shape}, Policy: {policy.shape}, Value: {value.shape}")

    print("✅ Transformer models test passed!")
