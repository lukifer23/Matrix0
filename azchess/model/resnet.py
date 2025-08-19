from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.9)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se: bool = False, se_ratio: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = _norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = _norm(channels)
        self.use_se = se
        if se:
            hidden = max(8, int(channels * se_ratio))
            self.se_fc1 = nn.Linear(channels, hidden)
            self.se_fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            # Squeeze
            w = F.adaptive_avg_pool2d(out, 1).flatten(1)
            w = F.relu(self.se_fc1(w), inplace=True)
            w = torch.sigmoid(self.se_fc2(w)).unsqueeze(-1).unsqueeze(-1)
            out = out * w
        out = out + x
        out = F.relu(out, inplace=True)
        return out


class ChessAttention(nn.Module):
    """Chess-specific attention mechanism for capturing spatial relationships."""
    
    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        assert channels % heads == 0, "Channels must be divisible by heads"
        
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        # Precompute chess attention mask for 8x8 boards (N=64) and register as buffer
        n = 8
        rows = torch.arange(n).repeat_interleave(n)
        cols = torch.arange(n).repeat(n)
        same_row = rows[:, None] == rows[None, :]
        same_col = cols[:, None] == cols[None, :]
        diag = (rows[:, None] - rows[None, :]).abs() == (cols[:, None] - cols[None, :]).abs()
        mask = (same_row | same_col | diag).to(torch.bool)  # (64, 64)
        mask = mask.view(1, 1, n * n, n * n)  # (1,1,N,N)
        self.register_buffer("attn_mask", mask, persistent=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        n = height * width  # 64 on chess boards

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, 3, self.heads, self.head_dim, n)
        qkv = qkv.permute(1, 0, 2, 4, 3).contiguous()  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, D)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)

        # Apply chess-specific attention mask (broadcast over batch and heads)
        scores = scores.masked_fill(self.attn_mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, H, N, D)

        # Merge heads -> (B, N, C) -> (B, C, H, W)
        # Ensure tensors are contiguous before view operations
        out = out.transpose(1, 2).contiguous().reshape(batch_size, n, channels)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, channels, height, width)

        # Project and add residual
        out = self.proj(out)
        out = out + x

        # Layer norm over channels
        out = out.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out
    
    def _create_chess_attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Deprecated: mask is precomputed; kept for compatibility."""
        return self.attn_mask


class ChessSpecificFeatures(nn.Module):
    """Chess-specific feature extraction and enhancement."""
    
    def __init__(self, channels: int, piece_square_tables: bool = True):
        super().__init__()
        self.piece_square_tables = piece_square_tables
        
        if piece_square_tables:
            # Piece-square table features
            self.pst_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.pst_norm = _norm(channels)
            
        # Chess-specific convolutions for piece interactions
        self.interaction_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.interaction_norm = _norm(channels)
        
        # Position encoding for chess board (8x8 for chess)
        self.position_encoding = nn.Parameter(torch.randn(1, channels, 8, 8))
        
        # Initialize position encoding properly
        nn.init.normal_(self.position_encoding, mean=0.0, std=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add position encoding
        x = x + self.position_encoding
        
        if self.piece_square_tables:
            # Apply PST features
            pst_features = F.relu(self.pst_norm(self.pst_conv(x)))
            x = x + pst_features
        
        # Apply interaction features
        interaction_features = F.relu(self.interaction_norm(self.interaction_conv(x)))
        x = x + interaction_features
        
        return x


@dataclass
class NetConfig:
    planes: int = 19
    channels: int = 160
    blocks: int = 14
    policy_size: int = 4672
    se: bool = True  # Enable SE blocks by default
    se_ratio: float = 0.25
    attention: bool = True  # Enable attention mechanisms
    attention_heads: int = 8
    chess_features: bool = True  # Enable chess-specific features
    self_supervised: bool = True  # Enable self-supervised learning
    piece_square_tables: bool = True  # Enable piece-square table features


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        C = cfg.channels
        self.cfg = cfg
        
        # Enhanced stem with chess-specific features
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.planes, C, kernel_size=3, padding=1, bias=False),
            _norm(C),
            nn.ReLU(inplace=True),
        )
        
        # Add chess-specific features if enabled
        if cfg.chess_features:
            self.chess_features = ChessSpecificFeatures(C, cfg.piece_square_tables)
        else:
            self.chess_features = None
        
        # Enhanced tower with attention and SE blocks
        tower_layers = []
        for i in range(cfg.blocks):
            # Add residual block with SE
            tower_layers.append(ResidualBlock(C, se=cfg.se, se_ratio=cfg.se_ratio))
            
            # Add attention every few blocks for efficiency
            if cfg.attention and i % 3 == 2:  # Every 3rd block
                tower_layers.append(ChessAttention(C, cfg.attention_heads))
        
        self.tower = nn.Sequential(*tower_layers)
        
        # Self-supervised learning head (if enabled)
        if cfg.self_supervised:
            self.ssl_head = nn.Sequential(
                nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                _norm(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, 13, kernel_size=1, bias=False),  # 12 pieces + 1 empty
            )
        else:
            self.ssl_head = None

        # Enhanced policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=1, bias=False),  # Increased from 32
            _norm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Add dropout for regularization
        )
        self.policy_fc = nn.Linear(64 * 8 * 8, cfg.policy_size)

        # Enhanced value head
        self.value_head = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=1, bias=False),  # Increased from 32
            _norm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Add dropout for regularization
        )
        self.value_fc1 = nn.Linear(64 * 8 * 8, C)
        self.value_fc2 = nn.Linear(C, C // 2)
        self.value_fc3 = nn.Linear(C // 2, 1)

    def forward(self, x: torch.Tensor, return_ssl: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.stem(x)
        
        # Apply chess-specific features if enabled
        if self.chess_features is not None:
            x = self.chess_features(x)
        
        x = self.tower(x)

        # Policy head
        p = self.policy_head(x)
        p = p.contiguous().reshape(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = self.value_head(x)
        v = v.contiguous().reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = F.relu(self.value_fc2(v), inplace=True)
        v = torch.tanh(self.value_fc3(v))
        
        # Self-supervised learning head (if enabled and requested)
        ssl_output = None
        if self.ssl_head is not None and return_ssl:
            ssl_output = self.ssl_head(x) # (B, 13, 8, 8)
        
        if return_ssl:
            return p, v.squeeze(-1), ssl_output
        else:
            return p, v.squeeze(-1)

    @staticmethod
    def from_config(d: dict) -> "PolicyValueNet":
        return PolicyValueNet(NetConfig(**d))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_ssl_loss(self, x: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Compute self-supervised learning loss for piece prediction."""
        if self.ssl_head is None:
            raise RuntimeError("SSL head not enabled in model configuration")
        
        # Get SSL output directly from the SSL head
        x_processed = self.stem(x)
        if self.chess_features is not None:
            x_processed = self.chess_features(x_processed)
        x_processed = self.tower(x_processed)
        
        ssl_output = self.ssl_head(x_processed)
        ssl_output = ssl_output.contiguous().reshape(ssl_output.size(0), -1)  # (B, 64)
        
        # Binary cross-entropy loss for piece presence prediction
        loss = F.binary_cross_entropy(ssl_output, target_mask, reduction='mean')
        return loss
    
    def get_chess_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract chess-specific features from the model."""
        if self.chess_features is None:
            raise RuntimeError("Chess features not enabled in model configuration")
        
        x = self.stem(x)
        return self.chess_features(x)

    def to_coreml(self, example_input: torch.Tensor, out_path: str) -> None:
        try:
            import coremltools as ct
        except Exception as e:
            raise RuntimeError("coremltools not available") from e

        self.eval()
        traced = torch.jit.trace(self, example_input)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=example_input.shape)],
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.save(out_path)
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory during training."""
        for module in self.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
    
    def quantize_model(self, dtype: torch.dtype = torch.float16) -> None:
        """Quantize model to specified dtype for memory efficiency."""
        self.to(dtype)
        for module in self.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data = module.weight.data.to(dtype)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.to(dtype)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics for the model."""
        total_params = 0
        trainable_params = 0
        buffer_size = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        for buffer in self.buffers():
            buffer_size += buffer.numel()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'buffer_size': buffer_size,
            'total_size_mb': (total_params * 4 + buffer_size * 4) / (1024 * 1024)  # Assuming float32
        }
    
    def create_ssl_targets(self, board_states: torch.Tensor) -> torch.Tensor:
        """Create self-supervised learning targets from board states."""
        # Target shape: (B, 13, 8, 8)
        # Planes 0-11 for pieces, plane 12 for empty squares
        target = torch.zeros(board_states.size(0), 13, 8, 8, device=board_states.device)
        
        # Copy piece planes
        target[:, :12, :, :] = board_states[:, :12, :, :]
        
        # Create empty square mask
        empty_mask = (target[:, :12, :, :].sum(dim=1) == 0).float()
        target[:, 12, :, :] = empty_mask
        
        # Convert to class indices for cross-entropy
        return torch.argmax(target, dim=1)
    
    def get_enhanced_ssl_loss(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute enhanced self-supervised learning loss with multiple tasks."""
        if self.ssl_head is None:
            raise RuntimeError("SSL head not enabled in model configuration")
        
        _, _, ssl_output = self.forward(x, return_ssl=True)
        
        # Piece presence prediction loss
        presence_loss = F.binary_cross_entropy(ssl_output, targets['piece_presence'], reduction='mean')
        
        # Add other SSL tasks here as needed
        total_loss = presence_loss
        
        return total_loss
