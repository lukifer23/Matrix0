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
import logging

logger = logging.getLogger(__name__)


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
        self.relative_pos_emb = nn.Parameter(torch.randn(2 * 64 - 1, self.head_dim))

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape

        # Linear transformations and reshape
        Q = self.q_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add relative positional bias
        rel_pos_bias = self._get_relative_pos_bias(seq_len, self.device)
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

    def _get_relative_pos_bias(self, seq_len, device):
        """Get relative positional bias for attention"""
        q_pos = torch.arange(seq_len, device=device)[:, None]
        k_pos = torch.arange(seq_len, device=device)[None, :]
        rel_pos = k_pos - q_pos + seq_len - 1
        return self.relative_pos_emb[rel_pos].unsqueeze(0).unsqueeze(0)


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


class MagnusChessTransformer(nn.Module):
    """Magnus - Advanced transformer model for chess learning (~70M parameters)
    Named after Magnus Carlsen, the world champion, representing peak chess performance.

    Architecture: 12-layer transformer with 512d embeddings, 8 attention heads
    Total parameters: ~70M (38M transformer, 17M features, 13M policy head)
    Designed for serious chess pattern recognition with efficient training."""

    def __init__(self,
                 input_channels: int = 19,
                 d_model: int = 512,   # Optimized for 70M total parameters
                 nhead: int = 8,       # 8 attention heads
                 num_layers: int = 12, # 12 transformer layers
                 dim_feedforward: int = 2048,  # Efficient feedforward
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = d_model

        # Enhanced multi-scale board to sequence conversion
        self.board_encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, d_model // 4, 3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.GELU(),
            nn.Conv2d(d_model // 4, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            # Second conv block with residual
            nn.Conv2d(d_model // 2, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        # Simplified feature extraction (single scale for efficiency)
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # Fixed 8x8 output
            nn.Flatten(),
            nn.Linear(d_model * 64, d_model),  # 64 = 8*8
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Advanced positional encoding with learnable parameters
        self.pos_encoder = AdvancedPositionalEncoding(d_model, max_len=64, dropout=dropout)

        # Deep transformer layers with fixed capacity (not progressive scaling)
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            # Fixed dimensions for all layers (much more reasonable parameter count)
            self.transformer_layers.append(
                LargeChessTransformerBlock(
                    d_model, nhead,  # Fixed dimensions
                    dim_feedforward,  # Fixed feedforward
                    dropout, activation
                )
            )

        # Layer norm after transformer
        self.transformer_norm = nn.LayerNorm(d_model)

        # Ultra-large policy head with multiple branches
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 4672)  # 4672 legal moves
        )

        # Enhanced value head with multiple prediction branches
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        # Auxiliary heads for multi-task learning
        self.material_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)  # Material advantage prediction
        )

        self.threat_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64)  # Threat detection (8x8 board)
        )

        # Attention mask for legal moves
        self.attention_mask = None

        # Initialize weights with advanced schemes
        self._initialize_weights_advanced()

    def _initialize_weights_advanced(self):
        """Advanced weight initialization for ultra-large model"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use T5-style initialization for better transformer training
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model ** -0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def board_to_tokens(self, board):
        """
        Convert board representation to token sequence with spatial reasoning
        Input: (batch, channels, 8, 8)
        Output: (batch, 64, d_model)
        """
        batch_size, channels, height, width = board.shape

        # Apply convolutional processing
        conv_features = self.board_encoder(board)  # (batch, d_model, 8, 8)

        # Flatten to sequence while preserving some spatial information
        # We'll create 64 tokens, each with spatial context
        tokens = conv_features.reshape(batch_size, self.d_model, -1)  # (batch, d_model, 64)
        tokens = tokens.transpose(1, 2)  # (batch, 64, d_model)

        return tokens

    def set_attention_mask(self, mask):
        """Set attention mask for legal moves"""
        self.attention_mask = mask

    def forward(self, x, return_aux=False):
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

        if return_aux:
            # Auxiliary predictions for multi-task learning
            material_advantage = self.material_head(representation)
            threat_map = self.threat_head(representation).view(-1, 8, 8)
            return policy_logits, value, material_advantage, threat_map
        else:
            return policy_logits, value


class MagnusChessTransformerFactory:
    """Factory for creating Magnus ultra-large chess transformer models"""

    @staticmethod
    def create_magnus_chess(input_channels: int = 19):
        """Create optimized Magnus transformer for chess (~70M parameters)
        Optimized configuration: 512d embeddings, 8 heads, 12 layers
        Total: ~70M parameters with excellent chess learning capacity"""
        return MagnusChessTransformer(
            input_channels=input_channels,
            d_model=512,       # Optimized embedding dimension
            nhead=8,           # 8 attention heads for balance
            num_layers=12,     # 12 transformer layers (deep but efficient)
            dim_feedforward=2048,  # Efficient feedforward dimension
            dropout=0.1,       # Standard dropout
            activation="gelu"  # GELU activation
        )

    @staticmethod
    def create_medium_transformer(input_channels: int = 19):
        """Create a medium-large transformer balancing capacity and speed"""
        return MagnusChessTransformer(
            input_channels=input_channels,
            d_model=384,
            nhead=6,
            num_layers=6,
            dim_feedforward=1536
        )

    @staticmethod
    def get_parameter_count(model):
        """Get total parameter count"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_info(model):
        """Get model information"""
        params = MagnusChessTransformerFactory.get_parameter_count(model)
        return {
            'architecture': 'MagnusChessTransformer',
            'parameters': params,
            'parameter_count': f"{params:,}",
            'd_model': model.d_model,
            'num_layers': len(model.transformer_layers),
            'nhead': model.transformer_layers[0].self_attn.nhead,
            'feedforward_dim': model.transformer_layers[0].feed_forward[0].out_features
        }


if __name__ == "__main__":
    # Test the large transformer models
    print("=== Large Chess Transformer Models ====")

    # Large model for serious learning
    large_model = MagnusChessTransformerFactory.create_magnus_chess()
    print(f"Large Model: {MagnusChessTransformerFactory.get_model_info(large_model)}")

    # Medium-large model for balanced performance
    med_large_model = MagnusChessTransformerFactory.create_medium_transformer()
    print(f"Medium-Large Model: {MagnusChessTransformerFactory.get_model_info(med_large_model)}")

    # Test forward pass
    x = torch.randn(1, 19, 8, 8)  # Batch size 1, 19 input channels, 8x8 board

    for name, model in [("Large", large_model), ("Medium-Large", med_large_model)]:
        with torch.no_grad():
            policy, value = model(x)
        print(f"{name} - Input: {x.shape}, Policy: {policy.shape}, Value: {value.shape}")

    print("✅ Large transformer models initialized successfully!")
    print(f"🚀 Ready for deep chess learning with {MagnusChessTransformerFactory.get_parameter_count(large_model):,} parameters!")