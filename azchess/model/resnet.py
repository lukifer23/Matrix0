from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 activation: str = "relu", preact: bool = False, droppath: float = 0.0, norm: str = "batch"):
        super().__init__()
        self.use_preact = preact
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = _norm(channels, norm)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = _norm(channels, norm)
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
            # Ensure SE weights match feature dtype under autocast (fix MPS type ops)
            if w.dtype != out.dtype:
                w = w.to(dtype=out.dtype)
            out = out * w

        # Standard DropPath: drop residual branch only and scale kept paths
        res = out
        if self.droppath > 0.0 and self.training:
            keep_prob = 1.0 - float(self.droppath)
            b = res.size(0)
            # Per-sample mask with matching dtype/device (MPS-safe)
            mask = torch.rand(b, 1, 1, 1, device=res.device, dtype=res.dtype) < res.new_tensor(keep_prob)
            mask = mask.to(dtype=res.dtype)
            res = res * mask / res.new_tensor(keep_prob)

        out = x + res
        if not self.use_preact:
            out = self.activation(out)

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

        # Compute attention scores (use dtype-safe scaling constant for MPS)
        inv_sqrt = (x.new_tensor(1.0) / x.new_tensor(math.sqrt(self.head_dim)))
        scores_base = torch.matmul(q, k.transpose(-2, -1)) * inv_sqrt  # (B, H, N, N)
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
            # Blend outputs: ensure scalars match tensor dtype/device (MPS-safe)
            one = out_masked.new_tensor(1.0)
            blend = one - out_masked.new_tensor(self.unmasked_mix)
            out = blend * out_masked + (one - blend) * out_unmasked
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
    value_activation: str = "silu"  # relu|silu|leaky_relu
    preact: bool = False
    droppath: float = 0.0 # DropPath regularization
    aux_policy_from_square: bool = False # Auxiliary from-square head
    aux_policy_move_type: bool = False # Auxiliary move-type head
    enable_visual: bool = False # Enable visual encoder
    visual_encoder_channels: int = 64 # Channels for visual encoder
    ssl_tasks: List[str] = field(default_factory=lambda: ["piece"]) # Basic piece recognition only (other tasks ignored)
    ssl_curriculum: bool = False # Progressive difficulty
    ssrl_tasks: List[str] = field(default_factory=list) # No SSRL tasks by default
    enable_llm_tutor: bool = False # LLM integration
    llm_model_path: str = "" # Path to LLM model


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        C = cfg.channels
        self.cfg = cfg
        # Track running SSL statistics for diagnostic logging
        self._ssl_loss_stats: Dict[str, Dict[str, float]] = {}
        # Guard: Only legacy 4672 policy is supported until 1858 path is implemented
        try:
            _ps = int(getattr(cfg, 'policy_size', 4672))
        except Exception:
            _ps = 4672
        if _ps != 4672:
            raise ValueError(
                f"Unsupported policy_size={_ps}. Matrix0 currently supports legacy 4672 only; "
                "AZ1858 mapping is not implemented yet."
            )
        
        activation = nn.SiLU(inplace=True) if cfg.activation == "silu" else nn.ReLU(inplace=True)

        self._value_activation_name = getattr(cfg, 'value_activation', 'silu')
        logger.info(f"Value head activation set to {self._value_activation_name}")

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
            # Add residual block with SE; respect cfg.norm
            tower_layers.append(ResidualBlock(C, se=cfg.se, se_ratio=cfg.se_ratio, activation=cfg.activation, preact=cfg.preact, droppath=cfg.droppath, norm=cfg.norm))
            
            # Add attention every few blocks for efficiency
            if cfg.attention and _att_every > 0 and (i % _att_every) == (_att_every - 1):
                tower_layers.append(ChessAttention(C, cfg.attention_heads, unmasked_mix=cfg.attention_unmasked_mix, relbias=getattr(cfg, 'attention_relbias', False)))
        
        self.tower = nn.Sequential(*tower_layers)
        
        # Self-supervised learning head (if enabled)
        if cfg.self_supervised:
            # Enhanced SSL with multiple tasks support
            ssl_tasks = getattr(cfg, 'ssl_tasks', ['piece'])
            # Register SSL heads as ModuleDict so .to(device) moves them correctly
            self.ssl_heads = nn.ModuleDict()

            # Piece recognition SSL (basic task)
            if 'piece' in ssl_tasks:
                self.ssl_heads['piece'] = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 13, kernel_size=1, bias=False),  # 12 pieces + 1 empty
                )
                logger.info(f"SSL piece head created with {sum(p.numel() for p in self.ssl_heads['piece'].parameters())} parameters")

            # Threat detection SSL
            if 'threat' in ssl_tasks:
                self.ssl_heads['threat'] = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 1, kernel_size=1, bias=False),  # Binary: under threat or not
                )
                logger.info(f"SSL threat head created with {sum(p.numel() for p in self.ssl_heads['threat'].parameters())} parameters")

            # Pin detection SSL
            if 'pin' in ssl_tasks:
                self.ssl_heads['pin'] = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 1, kernel_size=1, bias=False),  # Binary: pinned or not
                )
                logger.info(f"SSL pin head created with {sum(p.numel() for p in self.ssl_heads['pin'].parameters())} parameters")

            # Fork detection SSL
            if 'fork' in ssl_tasks:
                self.ssl_heads['fork'] = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 1, kernel_size=1, bias=False),  # Binary: forking or not
                )
                logger.info(f"SSL fork head created with {sum(p.numel() for p in self.ssl_heads['fork'].parameters())} parameters")

            # Square control SSL (3-class: black, neutral, white)
            if 'control' in ssl_tasks:
                self.ssl_heads['control'] = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 3, kernel_size=1, bias=False),  # 3-class control logits
                )
                logger.info(f"SSL control head created with {sum(p.numel() for p in self.ssl_heads['control'].parameters())} parameters")

            # Pawn structure SSL
            if 'pawn_structure' in ssl_tasks:
                self.ssl_heads['pawn_structure'] = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 8, kernel_size=1, bias=False),  # 8 pawn structure features
                )
                logger.info(f"SSL pawn_structure head created with {sum(p.numel() for p in self.ssl_heads['pawn_structure'].parameters())} parameters")

            # King safety SSL
            if 'king_safety' in ssl_tasks:
                self.ssl_heads['king_safety'] = nn.Sequential(
                    nn.Conv2d(C, C // 2, kernel_size=1, bias=False),
                    _norm(C // 2, cfg.norm),
                    activation,
                    nn.Conv2d(C // 2, 3, kernel_size=1, bias=False),  # 3 safety levels
                )
                logger.info(f"SSL king_safety head created with {sum(p.numel() for p in self.ssl_heads['king_safety'].parameters())} parameters")

            # For backward compatibility, keep ssl_head pointing to piece head if it exists
            self.ssl_head = self.ssl_heads['piece'] if 'piece' in self.ssl_heads else None
            self.ssl_piece_head = self.ssl_head  # Maintain compatibility

            total_ssl_params = sum(sum(p.numel() for p in head.parameters()) for head in self.ssl_heads.values())
            logger.info(f"Total SSL parameters: {total_ssl_params}")
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

        # Learnable logit scale for the dense policy branch (replaces LN on logits)
        self._policy_logit_scale_eps = 1e-3
        init_scale = float(getattr(cfg, 'policy_logit_init_scale', 0.2))
        safe_init = max(init_scale - self._policy_logit_scale_eps, 1e-6)
        raw_init = math.log(math.expm1(safe_init))
        self._policy_logit_scale_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        
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

    def _value_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._value_activation_name == "silu":
            return F.silu(tensor)
        if self._value_activation_name == "leaky_relu":
            return F.leaky_relu(tensor, negative_slope=0.05)
        return F.relu(tensor)

    def _init_weights(self):
        """Initialize weights properly for chess policy learning."""
        # Policy head initialization - CRITICAL: More conservative initialization
        nn.init.kaiming_normal_(self.policy_head[0].weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize policy head dropout properly
        if isinstance(self.policy_head[-1], nn.Dropout):
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
            if self.policy_fc is not None:
                self.policy_fc.weight.data *= 0.8  # Increased from 0.3 to allow more learning capacity
            elif self.policy_fc2 is not None:
                self.policy_fc2.weight.data *= 0.8  # Apply same scaling in factorized case
            logger.info("Policy head weights scaled for balanced learning capacity")

        # Logit scale initialized conservatively to avoid large logits after LN removal
        logger.info("Policy head logit scale initialized to 0.2")

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
        # Run stem in fp32 regardless of autocast to satisfy MPS conv dtype rules
        try:
            device_type = x.device.type
        except Exception:
            device_type = 'cpu'
        x_fp32 = x.float()
        with torch.autocast(device_type=device_type, enabled=False):
            x = self.stem(x_fp32)
        # Run potentially fragile blocks in fp32 as well for stability on MPS/AMP
        with torch.autocast(device_type=device_type, enabled=False):
            if self.chess_features is not None:
                x = self.chess_features(x)
        
        # Integrate visual features if available
        if self.visual_encoder is not None and visual_input is not None:
            visual_features = self.visual_encoder(visual_input)
            x = x + visual_features  # Add visual features to chess features
        
        # Run the tower in fp32 to avoid half-precision normalization/softmax instabilities
        with torch.autocast(device_type=device_type, enabled=False):
            x = self.tower(x)

        # Sanitize features if any non-finite values slipped through
        if not torch.isfinite(x).all():
            logger.warning("Feature tensor contains NaN/Inf; sanitizing to zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def _compute_policy_value(self, feats: torch.Tensor, return_ssl: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Policy - ensure contiguity throughout
        pfeat = self.policy_head(feats)
        p = pfeat.contiguous().reshape(pfeat.size(0), -1)

        if self.policy_fc is not None:
            p = self.policy_fc(p)
        else:
            p = F.relu(self.policy_fc1(p), inplace=True)
            p = self.policy_fc2(p)

        # Apply learnable logit scale; keep dtype consistent
        logit_scale = F.softplus(self._policy_logit_scale_raw) + self._policy_logit_scale_eps
        logit_scale = torch.clamp(logit_scale, max=5.0)
        p = p * logit_scale.to(dtype=p.dtype, device=p.device)

        if torch.isnan(p).any() or torch.isinf(p).any():
            logger.warning(f"Policy output contains NaN/Inf: total={torch.isnan(p).sum() + torch.isinf(p).sum()}")
            logger.warning("Replacing NaN/Inf with safe values to continue training")
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        # Avoid over-constraining logits; let CE handle stability

        # Value - ensure contiguity throughout
        v = self.value_head(feats)
        v = v.contiguous().reshape(v.size(0), -1)
        v = self._value_activation(self.value_fc1(v))
        if torch.isnan(v).any() or torch.isinf(v).any():
            logger.warning("Value head intermediate contains NaN/Inf after fc1; sanitizing")
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = self._value_activation(self.value_fc2(v))
        if torch.isnan(v).any() or torch.isinf(v).any():
            logger.warning("Value head intermediate contains NaN/Inf after fc2; sanitizing")
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = torch.tanh(self.value_fc3(v))

        # SSL - compute outputs for all enabled SSL heads
        ssl_output = None
        if return_ssl and self.ssl_heads:
            ssl_output = {}
            for task_name, head in self.ssl_heads.items():
                task_output = head(feats).contiguous()
                if torch.isnan(task_output).any() or torch.isinf(task_output).any():
                    logger.warning(f"SSL {task_name} head output contains NaN/Inf; sanitizing")
                    task_output = torch.nan_to_num(task_output, nan=0.0, posinf=0.0, neginf=0.0)
                ssl_output[task_name] = task_output
        elif self.ssl_piece_head is not None and return_ssl:
            # Backward compatibility: return piece head output directly
            ssl_output = self.ssl_piece_head(feats).contiguous()
            if torch.isnan(ssl_output).any() or torch.isinf(ssl_output).any():
                logger.warning("SSL head output contains NaN/Inf; sanitizing")
                ssl_output = torch.nan_to_num(ssl_output, nan=0.0, posinf=0.0, neginf=0.0)

        return p, v.squeeze(-1).contiguous(), ssl_output

    def forward(self, x: torch.Tensor, return_ssl: bool = False, visual_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        feats = self._forward_features(x, visual_input)
        p, v, ssl_output = self._compute_policy_value(feats, return_ssl)
        if return_ssl:
            return p, v, ssl_output
        return p, v

    def forward_with_features(self, x: torch.Tensor, return_ssl: bool = False, visual_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        feats = self._forward_features(x, visual_input)
        p, v, ssl_output = self._compute_policy_value(feats, return_ssl)
        return p, v, ssl_output, feats

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
        """Compute self-supervised learning loss for piece prediction - memory optimized."""
        if self.ssl_piece_head is None:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=False)

        # Memory optimization: process SSL in smaller chunks to avoid OOM
        batch_size = x.shape[0]
        device = x.device

        # For large batches, process in chunks to reduce memory usage
        # Use configurable chunk size or default to 32
        default_chunk_size = getattr(self.cfg, 'ssl_chunk_size', 32)
        chunk_size = min(default_chunk_size, batch_size)
        # Accumulate as a tensor so gradients flow through SSL head
        total_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
        valid_samples = 0

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            chunk_x = x[start_idx:end_idx]
            chunk_targets = targets[start_idx:end_idx]

            # Process chunk through the model
            try:
                # Option: avoid backprop through tower for SSL to reduce memory on MPS
                backprop_tower = bool(getattr(self.cfg, 'ssl_backprop_through_tower', False))
                if not backprop_tower:
                    # Compute features without grad; SSL head still trains
                    device_type = chunk_x.device.type
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, enabled=False):
                            feats = self.stem(chunk_x.float())
                        if self.chess_features is not None:
                            feats = self.chess_features(feats)
                        feats = self.tower(feats)
                    x_processed = feats
                else:
                    # Run SSL stem in fp32 regardless of autocast to satisfy MPS conv dtype rules
                    device_type = chunk_x.device.type
                    with torch.autocast(device_type=device_type, enabled=False):
                        x_processed = self.stem(chunk_x.float())
                    if self.chess_features is not None:
                        x_processed = self.chess_features(x_processed)

                    # Use gradient checkpointing for the tower to save memory
                    checkpoint_strategy = getattr(self, 'checkpoint_strategy', 'tower_only')

                    if checkpoint_strategy == "full" and hasattr(torch, 'checkpoint') and x_processed.requires_grad:
                        x_processed = torch.utils.checkpoint.checkpoint(self.tower, x_processed)
                    elif checkpoint_strategy == "tower_only" and hasattr(torch, 'checkpoint') and x_processed.requires_grad:
                        x_processed = torch.utils.checkpoint.checkpoint(self.tower, x_processed)
                    elif checkpoint_strategy == "adaptive" and hasattr(torch, 'checkpoint') and x_processed.requires_grad:
                        try:
                            import psutil
                            memory_percent = psutil.virtual_memory().percent
                            if memory_percent > 80:
                                x_processed = torch.utils.checkpoint.checkpoint(self.tower, x_processed)
                            else:
                                x_processed = self.tower(x_processed)
                        except ImportError:
                            x_processed = self.tower(x_processed)
                    else:
                        x_processed = self.tower(x_processed)

                ssl_output = self.ssl_piece_head(x_processed)

                # Prepare logits and targets for cross-entropy: (N, 13) logits and (N,) class indices
                ssl_output = ssl_output.permute(0, 2, 3, 1).reshape(-1, 13)  # (chunk_size*64, 13)

                # Accept targets as either one-hot planes (B,13,8,8) or class map (B,8,8) or already flattened
                if chunk_targets.dim() == 4 and chunk_targets.size(1) == 13:
                    # One-hot → class indices per square
                    chunk_targets = torch.argmax(chunk_targets, dim=1)
                elif chunk_targets.dim() == 1:
                    # Already flattened indices (N,)
                    pass
                elif chunk_targets.dim() == 3:
                    # Class map (B,8,8)
                    pass
                else:
                    raise ValueError(f"Unexpected SSL target shape: {tuple(chunk_targets.shape)}")

                chunk_targets = chunk_targets.reshape(-1).long()  # (chunk_size*64,)

                # Validate targets efficiently - allow 13-class targets (0-12)
                if chunk_targets.min() < 0 or chunk_targets.max() > 12:
                    continue

                # Only process if not all targets are 0
                if torch.all(chunk_targets == 0):
                    continue

                # Compute loss for this chunk (sum, then normalize at end)
                chunk_loss = F.cross_entropy(ssl_output, chunk_targets, reduction='sum')
                total_loss = total_loss + chunk_loss
                valid_samples += int(chunk_targets.numel())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"SSL chunk {start_idx}:{end_idx} failed due to memory, skipping")
                    # Clear cache and continue
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    continue
                else:
                    raise

        # Return average loss per sample; keep as tensor to allow gradients
        if valid_samples > 0:
            return total_loss / float(valid_samples)
        else:
            return total_loss.detach()  # zero tensor
    
    def get_enhanced_ssl_loss(self, x: torch.Tensor, targets: Dict[str, torch.Tensor], feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute SSL loss for supported tasks with advanced algorithms.

        Enhanced implementation with proper multi-task loss computation and error handling.
        """
        total_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        device = x.device

        # Get configuration for SSL tasks and weights
        ssl_tasks = getattr(self.cfg, 'ssl_tasks', ['piece'])

        # Log SSL task configuration
        if torch.rand(1).item() < 0.01:  # Log 1% of the time
            logger.info(f"SSL TASKS CONFIG: enabled={ssl_tasks}, available_heads={list(self.ssl_heads.keys())}")

        # Compute shared features once for all tasks that need them
        feats_shared: Optional[torch.Tensor] = feats if feats is not None else None
        active_advanced_tasks = [t for t in ['threat', 'pin', 'fork', 'control', 'pawn_structure', 'king_safety']
                                if t in targets and t in ssl_tasks and t in self.ssl_heads]
        piece_task_enabled = (
            'piece' in targets
            and self.ssl_piece_head is not None
            and 'piece' in ssl_tasks
        )

        needs_shared_features = bool(active_advanced_tasks) or piece_task_enabled
        if feats_shared is None and needs_shared_features:
            feats_shared = self._forward_features(x, None)

        # Process each SSL task
        task_losses = {}
        task_scalar_losses: Dict[str, float] = {}

        # 1. Piece recognition (primary SSL task)
        if piece_task_enabled:
            try:
                # Handle different target formats
                piece_targets = targets['piece']
                if isinstance(piece_targets, torch.Tensor) and piece_targets.dim() == 4:
                    # One-hot format: convert to class indices for cross-entropy
                    piece_targets = torch.argmax(piece_targets, dim=1)
                piece_targets = piece_targets.long()
                piece_targets = piece_targets.clamp_(0, 12)

                if feats_shared is None:
                    feats_shared = self._forward_features(x, None)

                chunk_size_cfg = int(getattr(self.cfg, 'ssl_chunk_size', 0))
                piece_feat_batches = [(feats_shared, piece_targets)]
                use_chunking = chunk_size_cfg > 0 and feats_shared.size(0) > chunk_size_cfg
                if use_chunking:
                    piece_feat_batches = []
                    for start_idx in range(0, feats_shared.size(0), chunk_size_cfg):
                        end_idx = min(start_idx + chunk_size_cfg, feats_shared.size(0))
                        piece_feat_batches.append(
                            (feats_shared[start_idx:end_idx], piece_targets[start_idx:end_idx])
                        )

                piece_loss_accum = None
                total_positions = 0
                for feat_chunk, target_chunk in piece_feat_batches:
                    logits_chunk = self.ssl_piece_head(feat_chunk)
                    logits_flat = logits_chunk.permute(0, 2, 3, 1).reshape(-1, logits_chunk.size(1))
                    targets_flat = target_chunk.reshape(-1).long()
                    loss_chunk = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
                    piece_loss_accum = loss_chunk if piece_loss_accum is None else (piece_loss_accum + loss_chunk)
                    total_positions += targets_flat.numel()

                if piece_loss_accum is None or total_positions == 0:
                    piece_loss = torch.zeros((), device=device, dtype=total_loss.dtype)
                else:
                    piece_loss = piece_loss_accum / float(total_positions)
                if torch.isfinite(piece_loss) and piece_loss > 0:
                    task_losses['piece'] = piece_loss
                    task_scalar_losses['piece'] = float(piece_loss.detach())
                    total_loss += piece_loss
            except Exception as e:
                logger.warning(f"Piece SSL loss failed: {e}")

        # 2. Advanced SSL tasks using dedicated heads
        for task in active_advanced_tasks:
            try:
                task_head = self.ssl_heads[task]
                task_targets = targets[task]

                # Compute task output using shared features
                task_output = task_head(feats_shared)

                # Validate output dimensions
                if task_output.dim() != 4:
                    logger.warning(f"SSL {task} output has wrong dimensions: {task_output.shape}")
                    continue

                # Compute loss based on task type
                task_loss = self._compute_task_loss(task, task_output, task_targets, device)

                if task_loss is not None and torch.isfinite(task_loss) and task_loss > 0:
                    # Apply task-specific weight
                    task_weight = getattr(self.cfg, f'ssl_{task}_weight', 1.0)
                    weighted_loss = task_weight * task_loss
                    task_losses[task] = weighted_loss
                    task_scalar_losses[task] = float(weighted_loss.detach())
                    total_loss += weighted_loss
                else:
                    logger.debug(f"SSL {task} loss invalid or zero: {task_loss}")

            except Exception as e:
                logger.warning(f"{task} SSL loss computation failed: {e}")
                continue

        # Log summary if any SSL loss was computed
        if len(task_losses) > 0:
            total_scalar = float(total_loss.detach())
            stats_records: Dict[str, Dict[str, float]] = {}

            def _update_stats(name: str, value: float, decay: float = 0.95) -> Dict[str, float]:
                entry = self._ssl_loss_stats.setdefault(name, {"ema": value, "count": 0.0, "max": value})
                if entry["count"] == 0:
                    entry["ema"] = value
                    entry["max"] = value
                else:
                    entry["ema"] = decay * entry["ema"] + (1.0 - decay) * value
                    entry["max"] = max(entry.get("max", value), value)
                entry["count"] += 1
                entry["last"] = value
                ratio = value / (entry["ema"] + 1e-6)
                stats_records[name] = {"value": value, "ema": entry["ema"], "ratio": ratio, "count": entry["count"]}
                return stats_records[name]

            # Update stats for tasks and total
            for name, value in task_scalar_losses.items():
                _update_stats(f"task:{name}", value)
            total_stats = _update_stats("total", total_scalar)

            # Detect spikes versus the running EMA
            spike_threshold = 1.25
            spike_notes = []
            for name, record in stats_records.items():
                if record["count"] <= 10:
                    continue  # allow EMA to warm up
                if record["ratio"] >= spike_threshold:
                    label = name.replace("task:", "")
                    spike_notes.append(f"{label} x{record['ratio']:.2f} (ema {record['ema']:.4f})")

            random_sample = random.random() < 0.02
            should_log = random_sample or spike_notes
            if should_log:
                task_summary = ", ".join(
                    [
                        f"{name}: {stats_records[f'task:{name}']['value']:.4f} (ema {stats_records[f'task:{name}']['ema']:.4f})"
                        for name in sorted(task_scalar_losses.keys())
                    ]
                )
                msg = (
                    f"SSL LOSS SUMMARY: total={total_scalar:.6f} (ema {total_stats['ema']:.6f})"
                    + (f", tasks=[{task_summary}]" if task_summary else "")
                )
                if spike_notes:
                    msg += f" | spikes={'; '.join(spike_notes)}"
                logger.info(msg)

        return total_loss

    def _compute_task_loss(self, task: str, output: torch.Tensor, targets: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
        """Compute loss for a specific SSL task with proper error handling."""
        try:
            if task in ['threat', 'pin', 'fork']:
                # Binary classification tasks
                if output.size(1) != 1:
                    logger.warning(f"SSL {task} output channels != 1: {output.shape}")
                    return None

                output_flat = output.permute(0, 2, 3, 1).reshape(-1, 1)
                targets_flat = targets.reshape(-1, 1).float()

                # Ensure targets are in valid range for BCE
                targets_flat = torch.clamp(targets_flat, 0.0, 1.0)

                return F.binary_cross_entropy_with_logits(output_flat, targets_flat, reduction='mean')

            elif task == 'control':
                # Ternary classification: -1 (black), 0 (neutral), 1 (white) -> map to 0,1,2
                if output.size(1) != 3:
                    logger.warning(f"SSL control output channels != 3: {output.shape}")
                    return None

                output_flat = output.permute(0, 2, 3, 1).reshape(-1, 3)
                targets_flat = targets.reshape(-1).float()

                # Map targets to class indices
                targets_3class = torch.where(targets_flat < -0.5, torch.tensor(0, device=device),
                                           torch.where(targets_flat > 0.5, torch.tensor(2, device=device),
                                                      torch.tensor(1, device=device)))

                return F.cross_entropy(output_flat, targets_3class.long(), reduction='mean')

            elif task == 'pawn_structure':
                # 8-class classification for pawn structure features
                if output.size(1) != 8:
                    logger.warning(f"SSL pawn_structure output channels != 8: {output.shape}")
                    return None

                # Accept either class-index targets (B,H,W) or one-hot (B,8,H,W)
                if torch.is_tensor(targets) and targets.dim() == 4 and targets.size(1) == 8:
                    targets = torch.argmax(targets, dim=1)

                output_flat = output.permute(0, 2, 3, 1).reshape(-1, 8)
                targets_flat = targets.reshape(-1).long()

                # Ensure targets are in valid range
                targets_flat = torch.clamp(targets_flat, 0, 7)

                return F.cross_entropy(output_flat, targets_flat, reduction='mean')

            elif task == 'king_safety':
                # 3-class classification for king safety levels
                if output.size(1) != 3:
                    logger.warning(f"SSL king_safety output channels != 3: {output.shape}")
                    return None

                # Accept either class-index targets (B,H,W) or one-hot (B,3,H,W)
                if torch.is_tensor(targets) and targets.dim() == 4 and targets.size(1) == 3:
                    targets = torch.argmax(targets, dim=1)

                output_flat = output.permute(0, 2, 3, 1).reshape(-1, 3)
                targets_flat = targets.reshape(-1).long()

                # Ensure targets are in valid range
                targets_flat = torch.clamp(targets_flat, 0, 2)

                return F.cross_entropy(output_flat, targets_flat, reduction='mean')

            else:
                logger.warning(f"Unknown SSL task: {task}")
                return None

        except Exception as e:
            logger.warning(f"Error computing {task} loss: {e}")
            return None
    
    def get_ssrl_loss(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute SSRL (Self-Supervised Representation Learning) loss for general tasks.

        SSRL focuses on general representation learning tasks:
        - position: Predict board positions (rotation invariance)
        - material: Predict material counts (counting skills)

        This differs from SSL (chess-specific tactical tasks) in that SSRL learns
        general visual and counting abilities rather than chess-specific tactics.
        """
        if not self.ssrl_heads:
            raise RuntimeError("SSRL heads not enabled in model configuration")

        total_loss = 0.0
        x_processed = self._forward_features(x)

        for task, target in targets.items():
            if task in self.ssrl_heads:
                output = self.ssrl_heads[task](x_processed)
                if task == 'position':
                    # Position prediction loss - encourages rotation invariance
                    loss = F.cross_entropy(output, target.long(), reduction='mean')
                elif task == 'material':
                    # Material count loss - encourages counting skills
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
    
    def enable_gradient_checkpointing(self, strategy: str = "tower_only") -> None:
        """Enable gradient checkpointing to save memory during training.

        Args:
            strategy: Checkpointing strategy
                - "tower_only": Only checkpoint the tower (default)
                - "full": Checkpoint all major modules
                - "adaptive": Use adaptive checkpointing based on memory
        """
        self.checkpoint_strategy = strategy

        if strategy == "full":
            # Enable checkpointing for all modules that support it
            for module in self.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                    logger.info(f"Enabled gradient checkpointing for {module.__class__.__name__}")

        elif strategy == "adaptive":
            # Adaptive checkpointing - only enable when memory is low
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:  # High memory usage threshold
                    self.enable_gradient_checkpointing("full")
                    logger.info("Adaptive checkpointing enabled due to high memory usage")
            except ImportError:
                # Fallback to tower-only if psutil not available
                self.enable_gradient_checkpointing("tower_only")

        # Default is tower_only which is handled in forward pass
        logger.info(f"Gradient checkpointing enabled with strategy: {strategy}")
    
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
        # Preserve any previously selected checkpointing strategy; do not override
        if hasattr(self, 'checkpoint_strategy'):
            strategy = getattr(self, 'checkpoint_strategy')
            logger.debug(f"Memory optimization: preserving checkpoint strategy '{strategy}'")
        else:
            strategy = None
        # Only (re)apply if an explicit strategy was already set elsewhere
        if strategy is not None and hasattr(self, 'enable_gradient_checkpointing'):
            self.enable_gradient_checkpointing(strategy=strategy)

        # Enable eval mode during forward pass for SSL (no gradients needed)
        # This is handled in get_ssl_loss method

        # Use more memory-efficient attention implementation if available
        for module in self.modules():
            if hasattr(module, 'memory_efficient') and hasattr(module, 'memory_efficient'):
                module.memory_efficient = True

    def create_ssl_targets(self, board_states: torch.Tensor):
        """Create SSL targets efficiently with optimized computation.

        Enhanced implementation that prioritizes speed and memory efficiency.
        """
        ssl_tasks = getattr(self.cfg, 'ssl_tasks', ['piece'])

        # Fast path for piece-only tasks
        if ssl_tasks == ['piece']:
            from ..ssl_algorithms import get_ssl_algorithms
            targets = get_ssl_algorithms()._create_piece_targets(board_states)
            if hasattr(self, 'ssl_target_weight') and self.ssl_target_weight != 1.0:
                targets = targets * self.ssl_target_weight
            return targets

        # Full SSL target generation for multiple tasks
        from ..ssl_algorithms import get_ssl_algorithms
        ssl_targets = get_ssl_algorithms().create_enhanced_ssl_targets(board_states)

        # Add pawn structure and king safety if needed (optimized computation)
        try:
            need_aug = any(t in ssl_tasks for t in ('pawn_structure', 'king_safety'))
            if need_aug:
                # Convert to numpy efficiently (only when needed)
                from azchess.training.ssl_targets import generate_ssl_targets_from_states
                s_np = board_states.detach().to('cpu').numpy()
                simple = generate_ssl_targets_from_states(s_np)

                for k in ('pawn_structure', 'king_safety'):
                    if k in ssl_tasks and k not in ssl_targets and k in simple:
                        ssl_targets[k] = torch.from_numpy(simple[k]).to(device=board_states.device, dtype=torch.float32)
                        logger.debug(f"Added {k} SSL targets via simple provider")
        except Exception as e:
            logger.debug(f"SSL simple augmentation skipped: {e}")

        # Apply target weight scaling
        if hasattr(self, 'ssl_target_weight') and self.ssl_target_weight != 1.0:
            for key in ssl_targets:
                if ssl_targets[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    ssl_targets[key] = ssl_targets[key] * self.ssl_target_weight

        logger.debug(f"Generated SSL targets for tasks: {list(ssl_targets.keys())}")
        return ssl_targets

    def create_ssl_targets_batch(self, board_states: torch.Tensor, batch_size: int = 32):
        """Create SSL targets in batches to manage memory usage.

        Args:
            board_states: (B, 19, 8, 8) tensor of board states
            batch_size: Batch size for processing to avoid memory issues

        Returns:
            Dict of SSL targets with same batch dimension as input
        """
        total_samples = board_states.size(0)
        all_targets = {}

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_states = board_states[start_idx:end_idx]

            # Generate targets for this batch
            batch_targets = self.create_ssl_targets(batch_states)

            # Accumulate results
            for key, target_tensor in batch_targets.items():
                if key not in all_targets:
                    # Pre-allocate full tensor
                    all_targets[key] = torch.zeros((total_samples,) + target_tensor.shape[1:],
                                                 device=target_tensor.device, dtype=target_tensor.dtype)

                # Copy batch results to full tensor
                all_targets[key][start_idx:end_idx] = target_tensor

        return all_targets

    def ensure_dtype_consistency(self, target_dtype: torch.dtype):
        """Ensure all model parameters and buffers have consistent dtype for MPS operations."""
        if not hasattr(self, '_original_dtypes'):
            # Store original dtypes for restoration
            self._original_dtypes = {}
            for name, param in self.named_parameters():
                self._original_dtypes[name] = param.dtype

        # Convert all parameters to target dtype
        for name, param in self.named_parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(dtype=target_dtype)

        # Convert buffers (like BatchNorm running stats) to target dtype
        for name, buffer in self.named_buffers():
            if buffer.dtype != target_dtype:
                buffer.data = buffer.data.to(dtype=target_dtype)

    def restore_original_dtypes(self):
        """Restore original parameter dtypes after MPS operations."""
        if hasattr(self, '_original_dtypes'):
            for name, param in self.named_parameters():
                if name in self._original_dtypes:
                    original_dtype = self._original_dtypes[name]
                    if param.dtype != original_dtype:
                        param.data = param.data.to(dtype=original_dtype)
            # Clean up stored dtypes
            delattr(self, '_original_dtypes')

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

        # Initialize parameters that are missing from the loaded state dict
        missing_keys = []
        for key, _ in self.state_dict().items():
            if key not in new_state_dict:
                missing_keys.append(key)

                # Walk the module hierarchy to get the actual parameter tensor
                module_name, _, param_name = key.rpartition('.')
                module = self
                if module_name:
                    for attr in module_name.split('.'):
                        module = getattr(module, attr)

                param_tensor = getattr(module, param_name, None)
                if param_tensor is not None:
                    # Initialize weights with small random values and biases with zeros
                    if param_name == 'weight':
                        if param_tensor.dim() >= 2:
                            torch.nn.init.xavier_uniform_(param_tensor, gain=0.1)
                        else:
                            torch.nn.init.normal_(param_tensor, std=0.01)
                    elif param_name == 'bias':
                        torch.nn.init.zeros_(param_tensor)

        return super().load_state_dict(new_state_dict, strict=strict)
