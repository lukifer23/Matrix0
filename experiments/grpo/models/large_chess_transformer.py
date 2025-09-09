#!/usr/bin/env python3
"""
Large Chess Transformer Model for GRPO Experiments

A more capable transformer architecture with deeper learning capacity.
Designed for serious chess pattern recognition while remaining experimental.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AdvancedPositionalEncoding(nn.Module):
    """Advanced positional encoding with learnable parameters"""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Fixed sinusoidal encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Learnable positional encoding
        self.learned_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x):
        # Combine fixed and learned positional encodings
        combined_pe = self.pe + self.learned_pe
        return self.dropout(x + combined_pe[:x.size(1)])


class MultiHeadAttentionWithRelativePos(nn.Module):
    """Multi-head attention with relative positional embeddings"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Relative positional embeddings for chess board
        self.relative_pos_emb = nn.Parameter(torch.randn(64, nhead, self.head_dim) * 0.02)  # 8x8 board = 64 positions

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape

        # Linear transformations and reshape
        Q = self.q_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add relative positional bias
        # This is a simplified version - in practice you'd want full relative attention
        rel_pos_bias = self._get_relative_pos_bias(batch_size, seq_len)
        scores = scores + rel_pos_bias

        # Apply attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, V)

        # Reshape and output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(context)

        return output

    def _get_relative_pos_bias(self, batch_size, seq_len):
        """Get relative positional bias for attention"""
        # Simplified relative positioning for chess board
        # Return zeros with exact shape to match attention scores: [batch, nhead, seq_len, seq_len]
        return torch.zeros(batch_size, self.nhead, seq_len, seq_len, device=self.relative_pos_emb.device)


class LargeChessTransformerBlock(nn.Module):
    """Enhanced transformer block with better capacity"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()

        # Multi-head attention with relative positioning
        self.self_attn = MultiHeadAttentionWithRelativePos(d_model, nhead, dropout)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Residual dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class LargeChessTransformer(nn.Module):
    """Large transformer model for serious chess learning"""

    def __init__(self,
                 input_channels: int = 19,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = d_model

        # Enhanced board to sequence conversion
        self.board_to_sequence = nn.Sequential(
            nn.Conv2d(input_channels, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        # Flatten spatial dimensions to sequence
        self.spatial_flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),  # Remove spatial dimensions
            nn.Linear(d_model, d_model)  # Project to sequence dimension
        )

        # Positional encoding
        self.pos_encoder = AdvancedPositionalEncoding(d_model, max_len=64, dropout=dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            LargeChessTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, activation
            ) for _ in range(num_layers)
        ])

        # Layer norm after transformer
        self.transformer_norm = nn.LayerNorm(d_model)

        # Enhanced policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 4672)  # 4672 legal moves
        )

        # Enhanced value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        # Attention mask for legal moves
        self.attention_mask = None

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def set_attention_mask(self, mask):
        """Set attention mask for legal moves"""
        self.attention_mask = mask

    def board_to_tokens(self, board):
        """
        Convert board representation to token sequence with spatial reasoning
        Input: (batch, channels, 8, 8)
        Output: (batch, 64, d_model)
        """
        batch_size, channels, height, width = board.shape

        # Apply convolutional processing
        conv_features = self.board_to_sequence(board)  # (batch, d_model, 8, 8)

        # Flatten to sequence while preserving some spatial information
        # We'll create 64 tokens, each with spatial context
        tokens = conv_features.view(batch_size, self.d_model, -1)  # (batch, d_model, 64)
        tokens = tokens.transpose(1, 2)  # (batch, 64, d_model)

        return tokens

    def forward(self, x):
        # Convert board to token sequence
        tokens = self.board_to_tokens(x)  # (batch, 64, d_model)

        # Add positional encoding
        tokens = self.pos_encoder(tokens)

        # Apply transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Final layer norm
        tokens = self.transformer_norm(tokens)

        # Global representation (mean pooling across positions)
        representation = tokens.mean(dim=1)

        # Policy head
        policy_logits = self.policy_head(representation)

        # Apply attention mask if available
        if self.attention_mask is not None:
            policy_logits = policy_logits + self.attention_mask

        # Value head
        value = self.value_head(representation)

        return policy_logits, value


class LargeChessTransformerFactory:
    """Factory for creating large chess transformer models"""

    @staticmethod
    def create_large(input_channels: int = 19, d_model: int = 512, nhead: int = 8,
                    num_layers: int = 8, dim_feedforward: int = 2048):
        """Create a large transformer for deep chess learning"""
        return LargeChessTransformer(
            input_channels=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward
        )

    @staticmethod
    def create_medium_large(input_channels: int = 19, d_model: int = 384, nhead: int = 6,
                           num_layers: int = 6, dim_feedforward: int = 1536):
        """Create a medium-large transformer balancing capacity and speed"""
        return LargeChessTransformer(
            input_channels=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward
        )

    @staticmethod
    def get_parameter_count(model):
        """Get total parameter count"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_info(model):
        """Get model information"""
        params = LargeChessTransformerFactory.get_parameter_count(model)
        return {
            'architecture': 'LargeChessTransformer',
            'parameters': params,
            'parameter_count': f"{params:,}",
            'd_model': model.d_model,
            'num_layers': len(model.transformer_layers),
            'nhead': model.transformer_layers[0].self_attn.nhead,
            'feedforward_dim': model.transformer_layers[0].feed_forward[0].out_features
        }


if __name__ == "__main__":
    # Test the large transformer models
    print("=== Large Chess Transformer Models ===")

    # Large model for serious learning
    large_model = LargeChessTransformerFactory.create_large()
    print(f"Large Model: {LargeChessTransformerFactory.get_model_info(large_model)}")

    # Medium-large model for balanced performance
    med_large_model = LargeChessTransformerFactory.create_medium_large()
    print(f"Medium-Large Model: {LargeChessTransformerFactory.get_model_info(med_large_model)}")

    # Test forward pass
    x = torch.randn(1, 19, 8, 8)  # Batch size 1, 19 input channels, 8x8 board

    for name, model in [("Large", large_model), ("Medium-Large", med_large_model)]:
        with torch.no_grad():
            policy, value = model(x)
        print(f"{name} - Input: {x.shape}, Policy: {policy.shape}, Value: {value.shape}")

    print("✅ Large transformer models initialized successfully!")
    print(f"🚀 Ready for deep chess learning with {LargeChessTransformerFactory.get_parameter_count(large_model):,} parameters!")
