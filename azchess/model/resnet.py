from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


def _norm(channels: int, norm_type: str = "batch") -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.9)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=max(1, channels // 16), num_channels=channels)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se: bool = False, se_ratio: float = 0.25, 
                 activation: str = "relu", preact: bool = False, droppath: float = 0.0):
        super().__init__()
        self.use_preact = preact
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = _norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = _norm(channels)
        self.use_se = se
        if se:
            hidden = max(8, int(channels * se_ratio))
            self.se_fc1 = nn.Linear(channels, hidden)
            self.se_fc2 = nn.Linear(hidden, channels)
        self.activation = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)
        self.droppath = droppath

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_preact:
            out = self.bn1(x)
            out = self.activation(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.activation(out)
            out = self.conv2(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)
            out = self.conv2(out)
            out = self.bn2(out)

        if self.use_se:
            # Squeeze (avoid view-related stride issues by using reshape)
            pool = F.adaptive_avg_pool2d(out, 1)
            w = pool.reshape(pool.size(0), -1).contiguous()
            w = self.activation(self.se_fc1(w))
            w = torch.sigmoid(self.se_fc2(w)).unsqueeze(-1).unsqueeze(-1)
            out = out * w
        out = out + x
        if not self.use_preact:
            out = self.activation(out)
        
        # Apply DropPath regularization
        if self.droppath > 0.0 and self.training:
            if torch.rand(1) < self.droppath:
                return x  # Drop the entire residual path
            else:
                out = out / (1.0 - self.droppath)  # Scale up to maintain expectation
        
        return out


class ChessAttention(nn.Module):
    """Chess-specific attention mechanism for capturing spatial relationships."""
    
    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1, unmasked_mix: float = 0.2, relbias: bool = False):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        assert channels % heads == 0, "Channels must be divisible by heads"

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        # Blend ratio between masked and unmasked attention (0.0 → only unmasked, 1.0 → only masked)
        self.unmasked_mix = float(unmasked_mix)

        # Precompute enhanced chess attention mask for 8x8 boards (N=64)
        n = 8
        rows = torch.arange(n).repeat_interleave(n)
        cols = torch.arange(n).repeat(n)
        same_row = rows[:, None] == rows[None, :]
        same_col = cols[:, None] == cols[None, :]
        diag = (rows[:, None] - rows[None, :]).abs() == (cols[:, None] - cols[None, :]).abs()

        # Enhanced chess-specific attention patterns
        # Include knight moves (L-shapes) and adjacent squares for tactical awareness
        knight_moves = (
            ((rows[:, None] - rows[None, :]) == 2) & ((cols[:, None] - cols[None, :]).abs() == 1) |
            ((rows[:, None] - rows[None, :]) == 1) & ((cols[:, None] - cols[None, :]).abs() == 2) |
            ((rows[:, None] - rows[None, :]) == -2) & ((cols[:, None] - cols[None, :]).abs() == 1) |
            ((rows[:, None] - rows[None, :]) == -1) & ((cols[:, None] - cols[None, :]).abs() == 2)
        )

        # Adjacent squares (king moves) for local tactical awareness
        adjacent = (
            ((rows[:, None] - rows[None, :]).abs() <= 1) &
            ((cols[:, None] - cols[None, :]).abs() <= 1)
        )

        # Enhanced chess attention mask
        mask = (same_row | same_col | diag | knight_moves | adjacent).to(torch.bool)
        mask = mask.view(1, 1, n * n, n * n)  # (1,1,N,N)
        self.register_buffer("attn_mask", mask, persistent=False)

        # Optional learnable relative bias over (N,N)
        self.use_relbias = bool(relbias)
        if self.use_relbias:
            self.rel_bias = nn.Parameter(torch.zeros(1, self.heads, n * n, n * n))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        n = height * width  # 64 on chess boards

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, 3, self.heads, self.head_dim, n)
        qkv = qkv.permute(1, 0, 2, 4, 3).contiguous()  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, D)

        # Compute attention scores
        scores_base = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)
        if self.use_relbias:
            scores_base = scores_base + self.rel_bias
        # CRITICAL: Clamp attention scores to prevent softmax NaN/Inf
        scores_base = torch.clamp(scores_base, -50.0, 50.0)

        # Masked attention branch (line-of-sight) - use large negative value instead of -inf
        scores_masked = scores_base.masked_fill(self.attn_mask == 0, -1e4)
        attn_masked = F.softmax(scores_masked, dim=-1)
        attn_masked = self.dropout(attn_masked)
        out_masked = torch.matmul(attn_masked, v)
        # Unmasked branch for knight-like/tactical patterns
        if self.unmasked_mix > 0.0 and self.unmasked_mix < 1.0:
            attn_unmasked = F.softmax(scores_base, dim=-1)
            attn_unmasked = self.dropout(attn_unmasked)
            out_unmasked = torch.matmul(attn_unmasked, v)
            # Blend outputs: majority masked by default (1 - unmasked_mix is masked weight)
            blend = float(1.0 - self.unmasked_mix)
            out = blend * out_masked + (1.0 - blend) * out_unmasked
        elif self.unmasked_mix >= 1.0:
            out = out_masked
        else:
            # Only unmasked - also clamp this branch
            attn_unmasked = F.softmax(scores_base, dim=-1)
            attn_unmasked = self.dropout(attn_unmasked)
            out = torch.matmul(attn_unmasked, v)

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
    attention_unmasked_mix: float = 0.2
    attention_relbias: bool = True
    attention_every_k: int = 3
    chess_features: bool = True  # Enable chess-specific features
    self_supervised: bool = True  # Enable self-supervised learning
    piece_square_tables: bool = True  # Enable piece-square table features
    wdl: bool = False  # Optional WDL auxiliary head
    # V2 toggles (safe defaults keep legacy behavior)
    policy_factor_rank: int = 0
    norm: str = "batch"  # batch|group
    activation: str = "relu"  # relu|silu
    preact: bool = False
    droppath: float = 0.0 # DropPath regularization
    aux_policy_from_square: bool = False # Auxiliary from-square head
    aux_policy_move_type: bool = False # Auxiliary move-type head
    enable_visual: bool = False # Enable visual encoder
    visual_encoder_channels: int = 64 # Channels for visual encoder
    ssl_tasks: List[str] = field(default_factory=lambda: ["piece"]) # Basic piece recognition only
    ssl_curriculum: bool = False # Progressive difficulty
    ssrl_tasks: List[str] = field(default_factory=list) # No SSRL tasks by default
    enable_llm_tutor: bool = False # LLM integration
    llm_model_path: str = "" # Path to LLM model


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        C = cfg.channels
        self.cfg = cfg
        
        activation = nn.SiLU(inplace=True) if cfg.activation == "silu" else nn.ReLU(inplace=True)

        # Enhanced stem with chess-specific features
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.planes, C, kernel_size=3, padding=1, bias=False),
            _norm(C, cfg.norm),
            activation,
        )
        
        # Add chess-specific features if enabled
        if cfg.chess_features:
            self.chess_features = ChessSpecificFeatures(C, cfg.piece_square_tables)
        else:
            self.chess_features = None
            
        # Add visual encoder if enabled
        if getattr(cfg, 'enable_visual', False):
            visual_channels = getattr(cfg, 'visual_encoder_channels', 64)
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, visual_channels, kernel_size=3, padding=1, bias=False),
                _norm(visual_channels, cfg.norm),
                activation,
                nn.Conv2d(visual_channels, C, kernel_size=1, bias=False),
                _norm(C, cfg.norm),
                activation,
            )
        else:
            self.visual_encoder = None
        
        # Enhanced tower with attention and SE blocks
        tower_layers = []
        _att_every = int(getattr(cfg, 'attention_every_k', 3))
        for i in range(cfg.blocks):
            # Add residual block with SE
            tower_layers.append(ResidualBlock(C, se=cfg.se, se_ratio=cfg.se_ratio, activation=cfg.activation, preact=cfg.preact, droppath=cfg.droppath))
            
            # Add attention every few blocks for efficiency
            if cfg.attention and _att_every > 0 and (i % _att_every) == (_att_every - 1):
                tower_layers.append(ChessAttention(C, cfg.attention_heads, unmasked_mix=cfg.attention_unmasked_mix, relbias=getattr(cfg, 'attention_relbias', False)))
        
        self.tower = nn.Sequential(*tower_layers)
        
        # Self-supervised learning head (if enabled)
        if cfg.self_supervised:
            # Enhanced SSL with multiple tasks support
            ssl_tasks = getattr(cfg, 'ssl_tasks', ['piece'])
            if 'piece' in ssl_tasks:
                self.ssl_piece_head = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 13, kernel_size=1, bias=False),  # 12 pieces + 1 empty
                )
                logger.info(f"SSL piece head created with {sum(p.numel() for p in self.ssl_piece_head.parameters())} parameters")
            else:
                self.ssl_piece_head = None
                logger.warning("SSL piece head not created - no piece task in ssl_tasks")
            
            # Additional SSL tasks can be added here
            self.ssl_head = self.ssl_piece_head  # For backward compatibility
        else:
            self.ssl_head = None
            self.ssl_piece_head = None
            logger.warning("SSL head not created - self_supervised=False")

        # Enhanced policy head trunk shared by both branches
        self.policy_head = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=1, bias=False),
            _norm(64, cfg.norm),
            activation,
            nn.Dropout(0.1),
        )
        
        # Auxiliary policy heads for enhanced training
        if getattr(cfg, 'aux_policy_from_square', False):
            self.aux_from_square = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=1, bias=False),
                _norm(32, cfg.norm),
                activation,
                nn.Conv2d(32, 64, kernel_size=1, bias=False),  # 64 squares
            )
        else:
            self.aux_from_square = None
            
        if getattr(cfg, 'aux_policy_move_type', False):
            self.aux_move_type = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=1, bias=False),
                _norm(32, cfg.norm),
                activation,
                nn.Conv2d(32, 12, kernel_size=1, bias=False),  # 12 move types
            )
        else:
            self.aux_move_type = None
        # Spatial conv branch: per-square 73 logits
        self.policy_conv_out = nn.Conv2d(64, 73, kernel_size=1, bias=True)
        
        # CRITICAL: Add normalization layers for branch stability
        # Note: We'll normalize after reshaping to match the expected dimensions
        self.policy_conv_norm = nn.LayerNorm(73)  # Normalize spatial branch (per-square)
        self.policy_fc_norm = nn.LayerNorm(cfg.policy_size)  # Normalize dense branch
        
        # Dense branch: preserves original capacity (4096 → 4672) or factorized when enabled
        _rank = int(getattr(cfg, 'policy_factor_rank', 0))
        if _rank and _rank > 0:
            self.policy_fc1 = nn.Linear(64 * 8 * 8, _rank)
            self.policy_fc2 = nn.Linear(_rank, cfg.policy_size)
            self.policy_fc = None
        else:
            self.policy_fc = nn.Linear(64 * 8 * 8, cfg.policy_size)
            self.policy_fc1 = None
            self.policy_fc2 = None
        
        # Enhanced value head
        self.value_head = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=1, bias=False),  # Increased from 32
            _norm(64, cfg.norm),
            activation,
            nn.Dropout(0.1),  # Add dropout for regularization
        )
        self.value_fc1 = nn.Linear(64 * 8 * 8, C)
        self.value_fc2 = nn.Linear(C, C // 2)
        self.value_fc3 = nn.Linear(C // 2, 1)

        # Optional WDL auxiliary head
        if cfg.wdl:
            self.wdl_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
                nn.Flatten(),             # (B, C)
                nn.Linear(C, max(32, C // 2)),
                nn.ReLU(inplace=True),
                nn.Linear(max(32, C // 2), 3),  # [loss, draw, win]
            )
        else:
            self.wdl_head = None
            
        # SSRL tasks if enabled
        ssrl_tasks = getattr(cfg, 'ssrl_tasks', [])
        if ssrl_tasks:
            self.ssrl_heads = {}
            for task in ssrl_tasks:
                if task == 'position':
                    # Position prediction head
                    self.ssrl_heads[task] = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(C, C // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(C // 2, 64),  # 64 squares
                    )
                elif task == 'material':
                    # Material count prediction
                    self.ssrl_heads[task] = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(C, C // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(C // 2, 12),  # 12 piece types
                    )
        else:
            self.ssrl_heads = {}
        
        # Initialize weights properly for training
        self._init_weights()
        
        # LLM tutor integration if enabled
        if getattr(cfg, 'enable_llm_tutor', False):
            llm_path = getattr(cfg, 'llm_model_path', '')
            if llm_path:
                try:
                    # This would be implemented in a separate LLM tutor module
                    # For now, just store the path
                    self.llm_tutor_path = llm_path
                    self.llm_tutor_enabled = True
                except Exception as e:
                    print(f"Warning: Could not load LLM tutor from {llm_path}: {e}")
                    self.llm_tutor_enabled = False
            else:
                self.llm_tutor_enabled = False
        else:
            self.llm_tutor_enabled = False
            
        # SSL curriculum support
        self.ssl_curriculum = getattr(cfg, 'ssl_curriculum', False)
        if self.ssl_curriculum:
            self.ssl_difficulty = 0.0  # Start with easy tasks
            self.ssl_difficulty_step = 0.1  # Increase difficulty gradually

    def _init_weights(self):
        """Initialize weights properly for chess policy learning."""
        # Policy head initialization - CRITICAL: More conservative initialization
        nn.init.kaiming_normal_(self.policy_head[0].weight, mode='fan_out', nonlinearity='relu')
        # CRITICAL: Use much smaller initialization for policy conv to prevent NaN explosion
        nn.init.xavier_uniform_(self.policy_conv_out.weight, gain=0.05)  # Further reduced from 0.1 to 0.05
        nn.init.constant_(self.policy_conv_out.bias, 0.0)
        
        # Initialize policy head dropout properly
        if hasattr(self.policy_head, 'dropout'):
            logger.info("Policy head dropout initialized")
        if self.policy_fc is not None:
            nn.init.xavier_uniform_(self.policy_fc.weight, gain=1.0)
            nn.init.constant_(self.policy_fc.bias, 0.0)
        if self.policy_fc1 is not None and self.policy_fc2 is not None:
            nn.init.xavier_uniform_(self.policy_fc1.weight, gain=1.0)
            nn.init.constant_(self.policy_fc1.bias, 0.0)
            nn.init.xavier_uniform_(self.policy_fc2.weight, gain=1.0)
            nn.init.constant_(self.policy_fc2.bias, 0.0)
        
        # Value head initialization
        nn.init.kaiming_normal_(self.value_head[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.value_fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.value_fc2.weight, gain=1.0)
        nn.init.xavier_uniform_(self.value_fc3.weight, gain=1.0)
        
        # Initialize biases to reasonable values
        nn.init.constant_(self.value_fc1.bias, 0.0)
        nn.init.constant_(self.value_fc2.bias, 0.0)
        nn.init.constant_(self.value_fc3.bias, 0.0)
        
        # CRITICAL: Scale policy output weights to prevent NaN explosion
        with torch.no_grad():
            # Much more conservative scaling for policy conv
            self.policy_conv_out.weight.data *= 0.05  # Further reduced from 0.1 to 0.05
            if self.policy_fc is not None:
                self.policy_fc.weight.data *= 0.3  # Further reduced from 0.5 to 0.3
            logger.info("Policy head weights scaled down for numerical stability")
            
        # CRITICAL: Initialize normalization layers for stability
        # LayerNorm has no learnable parameters, but we ensure it's properly set up
        logger.info("Policy head branch normalization layers initialized for stability")

        # SSL head initialization - CRITICAL: Was missing!
        if self.ssl_piece_head is not None:
            for m in self.ssl_piece_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)  # Conservative initialization
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            logger.info("SSL piece head weights initialized with conservative gains")

        # WDL head init
        if self.wdl_head is not None:
            for m in self.wdl_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)

    def _forward_features(self, x: torch.Tensor, visual_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.stem(x)
        if self.chess_features is not None:
            x = self.chess_features(x)
        
        # Integrate visual features if available
        if self.visual_encoder is not None and visual_input is not None:
            visual_features = self.visual_encoder(visual_input)
            x = x + visual_features  # Add visual features to chess features
        
        x = self.tower(x)
        return x

    def forward(self, x: torch.Tensor, return_ssl: bool = False, visual_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self._forward_features(x, visual_input)

        # CRITICAL: Simplified policy head for stability
        pfeat = self.policy_head(x)

        # CRITICAL: Clamp features before any operations
        pfeat = torch.clamp(pfeat, -5.0, 5.0)

        # Single unified approach: flatten and process through FC layers
        # This avoids the scale mismatch issues of dual-branch architecture
        p = pfeat.contiguous().reshape(pfeat.size(0), -1)

        # Apply FC layers with proper activation and clamping
        if self.policy_fc is not None:
            p = self.policy_fc(p)
            p = torch.clamp(p, -10.0, 10.0)  # Clamp after FC
        else:
            p = F.relu(self.policy_fc1(p), inplace=True)
            p = torch.clamp(p, -10.0, 10.0)  # Clamp after ReLU
            p = self.policy_fc2(p)
            p = torch.clamp(p, -10.0, 10.0)  # Final clamp

        # Apply final normalization for unified policy head
        p = self.policy_fc_norm(p)  # Use the FC normalization layer
        
        # CRITICAL: Handle NaN/Inf gracefully instead of crashing
        if torch.isnan(p).any() or torch.isinf(p).any():
            logger.warning(f"Policy output contains NaN/Inf: total={torch.isnan(p).sum() + torch.isinf(p).sum()}")
            logger.warning("Replacing NaN/Inf with safe values to continue training")
            # Replace NaN/Inf with zeros (safe fallback)
            p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
        
        # CRITICAL: More aggressive clamping to prevent gradient explosion
        p = torch.clamp(p, -5.0, 5.0)  # Reduced from -10.0, 10.0 to -5.0, 5.0

        # Value head - ensure contiguity throughout
        v = self.value_head(x)
        v = v.contiguous().reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = F.relu(self.value_fc2(v), inplace=True)
        v = torch.tanh(self.value_fc3(v))
        
        # Self-supervised learning head (if enabled and requested)
        ssl_output = None
        if self.ssl_piece_head is not None and return_ssl:
            ssl_output = self.ssl_piece_head(x).contiguous() # (B, 13, 8, 8)
        # Return (policy, value); if return_ssl is True, also return SSL output
        if return_ssl:
            return p, v.squeeze(-1).contiguous(), ssl_output
        return p, v.squeeze(-1).contiguous()

    def forward_with_features(self, x: torch.Tensor, return_ssl: bool = False, visual_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        feats = self._forward_features(x, visual_input)
        # Policy - ensure contiguity throughout
        pfeat = self.policy_head(feats)
        
        # Spatial conv branch - normalize BEFORE reshaping (same as forward)
        p_conv = self.policy_conv_out(pfeat)  # (B, 73, 8, 8)
        p_conv = p_conv.permute(0, 2, 3, 1).contiguous()  # (B, 8, 8, 73)
        p_conv = self.policy_conv_norm(p_conv)  # Normalize last dimension (73)
        p_conv = p_conv.reshape(p_conv.size(0), -1)  # (B, 64*73) = (B, 4672)
        
        _pflat = pfeat.contiguous().reshape(pfeat.size(0), -1)
        if self.policy_fc is not None:
            p_fc = self.policy_fc(_pflat)
        else:
            p_fc = self.policy_fc2(F.relu(self.policy_fc1(_pflat), inplace=True))
        
        # CRITICAL: Apply same normalization for consistency
        p_fc_norm = self.policy_fc_norm(p_fc)
        p = (p_conv + p_fc_norm).contiguous()
        # Value - ensure contiguity throughout
        v = self.value_head(feats)
        v = v.contiguous().reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = F.relu(self.value_fc2(v), inplace=True)
        v = torch.tanh(self.value_fc3(v))
        # SSL - ensure contiguity
        ssl_output = None
        if self.ssl_piece_head is not None and return_ssl:
            ssl_output = self.ssl_piece_head(feats).contiguous()
        return p, v.squeeze(-1).contiguous(), ssl_output, feats

    def compute_wdl_logits(self, feats: torch.Tensor) -> Optional[torch.Tensor]:
        if self.wdl_head is None:
            return None
        return self.wdl_head(feats)

    @staticmethod
    def from_config(d: dict) -> "PolicyValueNet":
        return PolicyValueNet(NetConfig(**d))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_ssl_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute self-supervised learning loss for piece prediction - optimized for speed."""
        if self.ssl_piece_head is None:
            return torch.tensor(0.0, device=x.device, requires_grad=False)

        # Get SSL output directly from the SSL head
        x_processed = self.stem(x)
        if self.chess_features is not None:
            x_processed = self.chess_features(x_processed)
        x_processed = self.tower(x_processed)

        ssl_output = self.ssl_piece_head(x_processed)

        # Fast path: Use reshape for compatibility with permuted tensors
        ssl_output = ssl_output.permute(0, 2, 3, 1).reshape(-1, 13)  # (B*64, 13)
        targets = targets.reshape(-1).long()  # (B*64,)

        # FORCE DEBUG: Always log first few calls to see what's happening
        if not hasattr(self, '_ssl_debug_count'):
            self._ssl_debug_count = 0

        if self._ssl_debug_count < 5:  # Log first 5 calls
            self._ssl_debug_count += 1
            logger.info(f"SSL DEBUG #{self._ssl_debug_count}: targets_shape={targets.shape}, targets_min={targets.min().item()}, targets_max={targets.max().item()}, targets_sum={targets.sum().item()}, all_zeros={torch.all(targets == 0).item()}")

        # Validate targets efficiently - allow 13-class targets (0-12)
        if targets.min() < 0 or targets.max() > 12:  # Changed from >= 13 to > 12
            if self._ssl_debug_count <= 5:
                logger.error(f"SSL TARGETS INVALID: min={targets.min().item()}, max={targets.max().item()} - returning 0.0")
            return torch.tensor(0.0, device=x.device, requires_grad=False)

        # Only return 0 if ALL targets are 0 (not just if sum is 0)
        if torch.all(targets == 0):
            if self._ssl_debug_count <= 5:
                logger.warning("SSL TARGETS ARE ALL ZEROS - returning 0.0 loss")
            return torch.tensor(0.0, device=x.device, requires_grad=False)

        # Compute loss with optimized cross-entropy
        loss = F.cross_entropy(ssl_output, targets, reduction='mean')

        # FORCE DEBUG: Always log first few loss computations
        if self._ssl_debug_count <= 5:
            valid_targets = (targets >= 0) & (targets <= 12)
            class_counts = torch.bincount(targets[valid_targets], minlength=13)
            logger.info(f"SSL LOSS COMPUTED #{self._ssl_debug_count}: loss={loss:.6f}, targets_sum={targets.sum().item()}, class_dist={class_counts.tolist()}")

        # Debug: Log SSL loss statistics (more frequent during debugging)
        if torch.rand(1).item() < 0.1:  # 10% chance to log (increased for debugging)
            valid_targets = (targets >= 0) & (targets <= 12)
            class_counts = torch.bincount(targets[valid_targets], minlength=13)
            logger.info(f"SSL Loss Debug: loss={loss:.6f}, targets_sum={targets.sum().item()}, targets_range=[{targets.min().item()}, {targets.max().item()}], class_dist={class_counts.tolist()}")

        # TEMPORARY: Log SSL loss every time it's computed (remove after debugging)
        if loss > 0.0:
            logger.info(f"SSL LOSS > 0 DETECTED: {loss:.6f}")
        elif torch.rand(1).item() < 0.05:  # 5% chance when loss is 0
            logger.warning(f"SSL LOSS IS 0.0 - targets_sum={targets.sum().item()}, all_zeros={torch.all(targets == 0).item()}")

        return loss
    
    def get_enhanced_ssl_loss(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute enhanced SSL loss for multiple tasks."""
        total_loss = 0.0
        
        # Piece prediction task
        if 'piece' in targets and self.ssl_piece_head is not None:
            piece_loss = self.get_ssl_loss(x, targets['piece'])
            total_loss += piece_loss
        
        # Additional SSL tasks can be added here
        # For now, just return the piece loss
        return total_loss
    
    def get_ssrl_loss(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute SSRL loss for various tasks."""
        if not self.ssrl_heads:
            raise RuntimeError("SSRL heads not enabled in model configuration")
        
        total_loss = 0.0
        x_processed = self._forward_features(x)
        
        for task, target in targets.items():
            if task in self.ssrl_heads:
                output = self.ssrl_heads[task](x_processed)
                if task == 'position':
                    # Position prediction loss
                    loss = F.cross_entropy(output, target.long(), reduction='mean')
                elif task == 'material':
                    # Material count loss
                    loss = F.mse_loss(output, target.float(), reduction='mean')
                else:
                    continue
                total_loss += loss
        
        return total_loss
    
    def get_auxiliary_policy_loss(self, x: torch.Tensor, aux_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute auxiliary policy losses for enhanced training."""
        total_loss = 0.0
        x_processed = self._forward_features(x)
        
        if self.aux_from_square is not None and 'from_square' in aux_targets:
            from_square_output = self.aux_from_square(x_processed)
            from_square_output = from_square_output.contiguous().reshape(from_square_output.size(0), -1)
            loss = F.cross_entropy(from_square_output, aux_targets['from_square'].long(), reduction='mean')
            total_loss += loss
        
        if self.aux_move_type is not None and 'move_type' in aux_targets:
            move_type_output = self.aux_move_type(x_processed)
            move_type_output = move_type_output.contiguous().reshape(move_type_output.size(0), -1)
            loss = F.cross_entropy(move_type_output, aux_targets['move_type'].long(), reduction='mean')
            total_loss += loss
        
        return total_loss
    
    def update_ssl_curriculum(self, step: int, max_steps: int) -> None:
        """Update SSL curriculum difficulty based on training progress."""
        if self.ssl_curriculum:
            # Gradually increase difficulty from 0.0 to 0.9 over training
            self.ssl_difficulty = min(0.9, step / max_steps * 0.9)
    
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

    def enable_memory_optimization(self):
        """Enable memory optimization techniques for MPS."""
        # Use gradient checkpointing for memory efficiency
        if hasattr(self, 'enable_gradient_checkpointing'):
            self.enable_gradient_checkpointing()

        # Enable eval mode during forward pass for SSL (no gradients needed)
        # This is handled in get_ssl_loss method

        # Use more memory-efficient attention implementation if available
        for module in self.modules():
            if hasattr(module, 'memory_efficient') and hasattr(module, 'memory_efficient'):
                module.memory_efficient = True

    def create_ssl_targets(self, board_states: torch.Tensor) -> torch.Tensor:
        """Create sophisticated SSL targets for meaningful chess learning with enhanced performance."""
        # Target shape: (B, 13, 8, 8) -> flattened to (B, 64)
        # Planes 0-11 for pieces, plane 12 for enhanced multi-task learning
        batch_size = board_states.size(0)
        device = board_states.device

        # Initialize enhanced targets with vectorized operations where possible
        targets = torch.zeros(batch_size, 13, 8, 8, device=device)

        # Planes 0-11: Enhanced piece recognition with relationships
        targets[:, :12, :, :] = board_states[:, :12, :, :]

        # Plane 12: Multi-task learning target - compute all masks vectorized
        enhanced_plane = torch.zeros(batch_size, 8, 8, device=device)

        # Vectorized empty square detection
        empty_mask = (board_states[:, :12, :, :].sum(dim=1) == 0).float()

        # For each position in the batch - process chess logic
        for b in range(batch_size):
            # Create a python-chess board from the tensor
            board = chess.Board()
            board.clear()
            for i in range(12):
                piece = chess.PIECE_TYPES[i % 6]
                color = chess.WHITE if i < 6 else chess.BLACK
                for r in range(8):
                    for c in range(8):
                        if board_states[b, i, r, c] == 1:
                            board.set_piece_at(chess.square(c, 7 - r), chess.Piece(piece, color))

            # Task 2: Threat detection (pieces under attack) - optimized
            threat_mask = torch.zeros(8, 8, device=device)
            for square in chess.SQUARES:
                if board.is_attacked_by(not board.turn, square):
                    r, c = divmod(square, 8)
                    threat_mask[7 - r, c] = 1

            # Task 3: Pin detection (pinned pieces) - optimized
            pin_mask = torch.zeros(8, 8, device=device)
            for square in chess.SQUARES:
                if board.is_pinned(board.turn, square):
                    r, c = divmod(square, 8)
                    pin_mask[7 - r, c] = 1

            # Task 4: Fork opportunities (pieces attacking multiple targets) - optimized
            fork_mask = torch.zeros(8, 8, device=device)
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == board.turn:
                    attacks = list(board.attacks(square))
                    if len(attacks) >= 2:  # Need at least 2 attacks for fork
                        valuable_targets = []
                        for attack_square in attacks:
                            target_piece = board.piece_at(attack_square)
                            if target_piece and target_piece.color != board.turn:
                                # Higher value pieces are more valuable targets
                                value = 1  # Pawn
                                if target_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                                    value = 3
                                elif target_piece.piece_type == chess.ROOK:
                                    value = 5
                                elif target_piece.piece_type == chess.QUEEN:
                                    value = 9
                                valuable_targets.append(value)

                        # Consider it a fork if attacking 2+ pieces worth at least 6 points total
                        if len(valuable_targets) >= 2 and sum(valuable_targets) >= 6:
                            r, c = divmod(square, 8)
                            fork_mask[7 - r, c] = 1

            # Task 5: Square control (controlled squares) - optimized with vectorized operations
            control_mask = torch.zeros(8, 8, device=device)
            for square in chess.SQUARES:
                attackers = list(board.attackers(board.turn, square))
                defenders = list(board.attackers(not board.turn, square))

                # Weight attackers by piece value
                attacker_value = 0
                for attacker_square in attackers:
                    attacker_piece = board.piece_at(attacker_square)
                    if attacker_piece:
                        if attacker_piece.piece_type == chess.PAWN:
                            attacker_value += 1
                        elif attacker_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                            attacker_value += 3
                        elif attacker_piece.piece_type == chess.ROOK:
                            attacker_value += 5
                        elif attacker_piece.piece_type == chess.QUEEN:
                            attacker_value += 9
                        # King attacks don't count for control

                defender_value = len(defenders) * 2  # Defenders have slight defensive advantage

                if attacker_value > defender_value:
                    r, c = divmod(square, 8)
                    control_mask[7 - r, c] = 1

            # Combine all tasks into plane 12 with optimized weights
            enhanced_plane[b] = (
                empty_mask[b] * 0.2 +      # Empty squares (20% weight)
                threat_mask * 0.3 +        # Threats (30% weight)
                pin_mask * 0.2 +           # Pins (20% weight)
                fork_mask * 0.15 +         # Forks (15% weight)
                control_mask * 0.15        # Control (15% weight)
            )

        targets[:, 12, :, :] = enhanced_plane

        # Convert to class indices and flatten to match SSL output shape (B, 64)
        # Use more efficient argmax with dim=1 for better performance
        targets = torch.argmax(targets, dim=1)  # (B, 8, 8)
        targets = targets.reshape(batch_size, -1)  # (B, 64) - each position gets a class index 0-12

        # FORCE DEBUG: Always log first few target creations
        if not hasattr(self, '_ssl_target_debug_count'):
            self._ssl_target_debug_count = 0

        if self._ssl_target_debug_count < 5:  # Log first 5 creations
            self._ssl_target_debug_count += 1
            class_counts = torch.bincount(targets.flatten(), minlength=13)
            logger.info(f"SSL TARGETS CREATED #{self._ssl_target_debug_count}: distribution={class_counts.tolist()}, sum={targets.sum().item()}, all_zeros={torch.all(targets == 0).item()}")

        # Debug: Log SSL target distribution (more frequent for debugging)
        if torch.rand(1).item() < 0.1:  # 10% chance to log
            class_counts = torch.bincount(targets.flatten(), minlength=13)
            logger.info(f"SSL Targets: distribution={class_counts.tolist()}, total={targets.numel()}")

        # ENSURE we always have meaningful targets - even if argmax gives all zeros
        if targets.sum() == 0:
            logger.warning("SSL targets argmax resulted in all zeros, generating piece-based targets")
            # Generate targets based on actual piece positions (planes 0-11)
            piece_targets = torch.zeros_like(targets)
            for b in range(batch_size):
                # Use the piece planes (0-11) to create meaningful targets
                piece_data = board_states[b, :12, :, :]  # Only piece planes
                if piece_data.sum() > 0:  # There are pieces on the board
                    # Find positions with pieces and assign appropriate class
                    for i in range(12):  # 12 piece types
                        piece_mask = (piece_data[i, :, :] == 1)
                        if piece_mask.any():
                            positions = torch.nonzero(piece_mask, as_tuple=False)
                            for pos in positions:
                                r, c = pos[0], pos[1]
                                idx = r * 8 + c
                                piece_targets[b, idx] = i
                    # If we still have zeros, fill with enhanced plane values
                    if piece_targets[b].sum() == 0:
                        enhanced_data = targets[b].view(8, 8)
                        piece_targets[b] = enhanced_data.flatten()
            targets = piece_targets
            logger.info("Generated SSL targets from piece positions")

        # Final validation - ensure we never return all zeros
        if targets.sum() == 0:
            logger.error("CRITICAL: Unable to generate any SSL targets - returning random targets to prevent training halt")
            # Last resort: generate random but valid targets
            targets = torch.randint(0, 12, (batch_size, 64), device=device, dtype=torch.long)
        
        # Debug: Log SSL target statistics (occasionally to avoid spam)
        if torch.rand(1).item() < 0.005:  # 0.5% chance to log
            target_distribution = torch.bincount(targets.reshape(-1), minlength=13)
            logger.info(f"SSL Targets: distribution={target_distribution.tolist()}, total={targets.sum()}")
        
        return targets

    def load_state_dict(self, state_dict, strict=False):
        """Handle V1 to V2 migration by mapping old keys to new ones."""
        # Map old V1 keys to new V2 keys
        key_mapping = {
            'policy_fc.weight': 'policy_fc1.weight',
            'policy_fc.bias': 'policy_fc1.bias',
        }
        
        # Create new state dict with mapped keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key in key_mapping:
                new_state_dict[key_mapping[key]] = value
            else:
                new_state_dict[key] = value
        
        # Initialize new V2 layers with sensible defaults if they don't exist
        missing_keys = []
        for name, module in self.named_modules():
            if name not in new_state_dict and hasattr(module, 'weight'):
                if 'aux_' in name or 'policy_fc1' in name:
                    # Initialize new V2 layers with small random weights
                    if hasattr(module, 'weight'):
                        if module.weight.dim() >= 2:
                            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                        else:
                            torch.nn.init.normal_(module.weight, std=0.01)
                    if hasattr(module, 'bias') and module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
        
        return super().load_state_dict(new_state_dict, strict=False)
