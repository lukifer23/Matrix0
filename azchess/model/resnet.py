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
            ssl_tasks = getattr(cfg, 'ssl_tasks', ['piece'])  # Only 'piece' task currently implemented
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

        # Normalization for the dense policy branch
        self.policy_fc_norm = nn.LayerNorm(cfg.policy_size)
        
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
                self.policy_fc.weight.data *= 0.3  # Further reduced from 0.5 to 0.3
            elif self.policy_fc2 is not None:
                self.policy_fc2.weight.data *= 0.3  # Apply same scaling in factorized case
            logger.info("Policy head weights scaled down for numerical stability")

        # CRITICAL: Initialize normalization layer for stability
        # LayerNorm has no learnable parameters, but we ensure it's properly set up
        logger.info("Policy head normalization layer initialized for stability")

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

    def _compute_policy_value(self, feats: torch.Tensor, return_ssl: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Policy - ensure contiguity throughout
        pfeat = self.policy_head(feats)
        pfeat = torch.clamp(pfeat, -5.0, 5.0)
        p = pfeat.contiguous().reshape(pfeat.size(0), -1)

        if self.policy_fc is not None:
            p = self.policy_fc(p)
            p = torch.clamp(p, -10.0, 10.0)
        else:
            p = F.relu(self.policy_fc1(p), inplace=True)
            p = torch.clamp(p, -10.0, 10.0)
            p = self.policy_fc2(p)
            p = torch.clamp(p, -10.0, 10.0)

        p = self.policy_fc_norm(p)

        if torch.isnan(p).any() or torch.isinf(p).any():
            logger.warning(f"Policy output contains NaN/Inf: total={torch.isnan(p).sum() + torch.isinf(p).sum()}")
            logger.warning("Replacing NaN/Inf with safe values to continue training")
            p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))

        p = torch.clamp(p, -5.0, 5.0)

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
        total_loss = 0.0
        valid_samples = 0

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            chunk_x = x[start_idx:end_idx]
            chunk_targets = targets[start_idx:end_idx]

            # Process chunk through the model
            try:
                x_processed = self.stem(chunk_x)
                if self.chess_features is not None:
                    x_processed = self.chess_features(x_processed)

                # Use gradient checkpointing for the tower to save memory
                if hasattr(torch, 'checkpoint') and x_processed.requires_grad:
                    x_processed = torch.utils.checkpoint.checkpoint(self.tower, x_processed)
                else:
                    x_processed = self.tower(x_processed)

                ssl_output = self.ssl_piece_head(x_processed)

                # Fast path: Use reshape for compatibility with permuted tensors
                ssl_output = ssl_output.permute(0, 2, 3, 1).reshape(-1, 13)  # (chunk_size*64, 13)
                chunk_targets = chunk_targets.reshape(-1).long()  # (chunk_size*64,)

                # Validate targets efficiently - allow 13-class targets (0-12)
                if chunk_targets.min() < 0 or chunk_targets.max() > 12:
                    continue

                # Only process if not all targets are 0
                if torch.all(chunk_targets == 0):
                    continue

                # Compute loss for this chunk
                chunk_loss = F.cross_entropy(ssl_output, chunk_targets, reduction='sum')
                total_loss += chunk_loss.item()
                valid_samples += chunk_targets.numel()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"SSL chunk {start_idx}:{end_idx} failed due to memory, skipping")
                    # Clear cache and continue
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    continue
                else:
                    raise

        # Return average loss per sample with consistent dtype
        if valid_samples > 0:
            return torch.tensor(total_loss / valid_samples, device=device, dtype=x.dtype, requires_grad=False)
        else:
            return torch.tensor(0.0, device=device, dtype=x.dtype, requires_grad=False)
    
    def get_enhanced_ssl_loss(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute SSL loss for supported tasks.

        Currently only the 'piece' task is implemented; additional tasks
        will be added in future updates.
        """
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
        """Create SSL targets using GPU vectorized operations - FAST version."""
        # Target shape: (B, 13, 8, 8) -> flattened to (B, 64)
        # Planes 0-11 for pieces, plane 12 for enhanced multi-task learning
        batch_size = board_states.size(0)
        device = board_states.device

        # Initialize targets efficiently
        targets = torch.zeros(batch_size, 13, 8, 8, device=device, dtype=board_states.dtype)

        # Planes 0-11: Direct copy of piece positions (already vectorized)
        targets[:, :12, :, :] = board_states[:, :12, :, :]

        # Plane 12: Multi-task learning target using vectorized operations
        # Get side-to-move information (plane 12, position 0,0)
        side_to_move = board_states[:, 12:13, 0:1, 0:1]  # Shape: (B, 1, 1, 1)

        # Create simplified SSL targets using tensor operations only
        # This avoids all the slow python-chess library calls

        # 1. Empty squares (where no pieces exist)
        empty_squares = (board_states[:, :12, :, :].sum(dim=1) == 0).float()

        # 2. Center squares (simple heuristic for positional value)
        center_mask = torch.zeros(batch_size, 8, 8, device=device, dtype=board_states.dtype)
        center_mask[:, 2:6, 2:6] = 1.0  # Inner 4x4 center squares

        # 3. Edge squares (opposite of center)
        edge_mask = 1.0 - center_mask

        # 4. Pawn advancement opportunities (simplified)
        # White pawns on ranks 2-6, black pawns on ranks 1-5
        white_pawns = board_states[:, 0, :, :]  # White pawns
        black_pawns = board_states[:, 6, :, :]  # Black pawns

        # Pawn advancement mask (squares in front of pawns)
        pawn_advance = torch.zeros_like(center_mask)
        # White pawn advancement (move up the board)
        pawn_advance[:, 1:7, :] = white_pawns[:, :6, :]  # Can advance to next rank
        # Black pawn advancement (move down the board)
        pawn_advance[:, 1:7, :] += black_pawns[:, 2:8, :]  # Can advance to previous rank

        # 5. King safety zones (squares around kings)
        king_safety = torch.zeros_like(center_mask)
        white_kings = board_states[:, 5, :, :]  # White king
        black_kings = board_states[:, 11, :, :]  # Black king

        # Simple king safety (3x3 area around kings)
        for b in range(min(batch_size, 32)):  # Limit to prevent memory issues
            if white_kings[b].sum() > 0:
                king_pos = white_kings[b].nonzero(as_tuple=True)
                if len(king_pos[0]) > 0:
                    r, c = king_pos[0][0], king_pos[1][0]
                    r1, r2 = max(0, r-1), min(7, r+2)
                    c1, c2 = max(0, c-1), min(7, c+2)
                    king_safety[b, r1:r2, c1:c2] = 1.0

            if black_kings[b].sum() > 0:
                king_pos = black_kings[b].nonzero(as_tuple=True)
                if len(king_pos[0]) > 0:
                    r, c = king_pos[0][0], king_pos[1][0]
                    r1, r2 = max(0, r-1), min(7, r+2)
                    c1, c2 = max(0, c-1), min(7, c+2)
                    king_safety[b, r1:r2, c1:c2] = 1.0

        # Combine all SSL features into plane 12
        # Weight different features based on importance
        targets[:, 12, :, :] = (
            0.3 * empty_squares +      # Empty squares for development
            0.2 * center_mask +        # Center control
            0.2 * pawn_advance +       # Pawn advancement
            0.3 * king_safety          # King safety zones
        )

        # Apply SSL target weight scaling if specified
        if hasattr(self, 'ssl_target_weight') and self.ssl_target_weight != 1.0:
            targets = targets * self.ssl_target_weight

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

        return super().load_state_dict(new_state_dict, strict=False)
