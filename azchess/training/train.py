#!/usr/bin/env python3
"""
Comprehensive Training Script for Matrix0
- Integrated DataManager for robust data handling.
- Advanced training features: mixed precision, LR scheduling, and data augmentation.
- Progress bars for all phases with ETA.
- Proper checkpointing and TensorBoard logging.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging

from azchess.config import Config, select_device
from azchess.model import PolicyValueNet
from azchess.data_manager import DataManager
from azchess.encoding import encode_board, move_to_index, POLICY_SHAPE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow:
                p.copy_(self.shadow[k])

def train_step(model, optimizer, scaler, batch, device: str, accum_steps: int = 1, augment: bool = True,
               augment_rotate180: bool = True,
               ssl_weight: float = 0.1, enable_ssl: bool = True,
               label_smoothing: float = 0.0, value_loss_type: str = 'mse', huber_delta: float = 1.0,
               policy_masking: bool = True, ssl_warmup_steps: int = 0, current_step: int = 0, ssl_target_weight: float = 1.0,
               use_wdl: bool = False, wdl_weight: float = 0.0, wdl_margin: float = 0.25, precision: str = "fp16"):
    """Single training step with augmentation, mixed precision, and self-supervised learning."""
    model.train()
    s, pi, z = batch
    
    # Convert numpy arrays to PyTorch tensors and ensure contiguity immediately
    s = torch.from_numpy(s).to(device).contiguous()
    pi = torch.from_numpy(pi).to(device).contiguous()
    z = torch.from_numpy(z).to(device).contiguous()
    # Normalize common shape variant for values: (N,) or (N,1) → (N,)
    if z.dim() == 2 and z.size(1) == 1:
        z = z.reshape(z.size(0)).contiguous()
    
    # Validate tensor properties to catch issues early
    if not s.is_contiguous():
        raise RuntimeError(f"States tensor is not contiguous after conversion. Shape: {s.shape}, strides: {s.stride()}")
    if not pi.is_contiguous():
        raise RuntimeError(f"Policy tensor is not contiguous after conversion. Shape: {pi.shape}, strides: {pi.stride()}")
    if not z.is_contiguous():
        raise RuntimeError(f"Value tensor is not contiguous after conversion. Shape: {z.shape}, strides: {z.stride()}")
    
    # Validate tensor shapes with detailed error messages
    try:
        if s.shape[0] != pi.shape[0] or s.shape[0] != z.shape[0]:
            raise RuntimeError(f"Batch size mismatch: states={s.shape[0]}, policy={pi.shape[0]}, values={z.shape[0]}")
        if pi.shape[1] != np.prod(POLICY_SHAPE):
            raise RuntimeError(f"Policy tensor shape mismatch: expected {np.prod(POLICY_SHAPE)}, got {pi.shape[1]}")
        # Value tensor can be 1D (batch_size,) or 2D (batch_size, 1) - both are valid
        if len(z.shape) == 1:
            # 1D tensor: (batch_size,) - this is correct
            pass
        elif len(z.shape) == 2 and z.shape[1] == 1:
            # 2D tensor: (batch_size, 1) - this is also correct
            pass
        else:
            raise RuntimeError(f"Value tensor shape mismatch: expected (batch_size,) or (batch_size, 1), got {z.shape}")
    except Exception as e:
        logger.error(f"Tensor validation failed: {e}")
        logger.error(f"States shape: {s.shape}, Policy shape: {pi.shape}, Values shape: {z.shape}")
        raise

    # Data Augmentation: geometric transforms aligned with action space
    if augment:
        r = torch.rand(1).item()
        if r < 0.5:
            # Horizontal flip (mirror files)
            s = torch.flip(s, dims=[3])
            # Ensure policy tensor is contiguous before reshaping
            pi_cont = pi.contiguous()
            pi_sh = pi_cont.reshape(-1, *POLICY_SHAPE)  # (B, 8, 8, 73)
            pi_sh = torch.flip(pi_sh, dims=[2])
            from azchess.encoding import build_horizontal_flip_permutation
            perm = build_horizontal_flip_permutation()
            perm_t = torch.as_tensor(perm, device=pi_sh.device, dtype=torch.long)
            pi_sh = pi_sh.index_select(-1, perm_t)
            # Ensure the final policy tensor is contiguous
            pi = pi_sh.reshape(-1, np.prod(POLICY_SHAPE)).contiguous()
        elif r < 0.75 and augment_rotate180:
            # 180-degree rotation (flip ranks and files)
            s = torch.flip(s, dims=[2, 3])
            # Ensure policy tensor is contiguous before reshaping
            pi_cont = pi.contiguous()
            pi_sh = pi_cont.reshape(-1, *POLICY_SHAPE)
            pi_sh = torch.flip(pi_sh, dims=[1, 2])
            from azchess.encoding import build_rotate180_permutation
            perm = build_rotate180_permutation()
            perm_t = torch.as_tensor(perm, device=pi_sh.device, dtype=torch.long)
            pi_sh = pi_sh.index_select(-1, perm_t)
            # Ensure the final policy tensor is contiguous
            pi = pi_sh.reshape(-1, np.prod(POLICY_SHAPE)).contiguous()
    
    # Validate tensors after augmentation to catch any contiguity issues
    if not s.is_contiguous():
        raise RuntimeError(f"States tensor lost contiguity after augmentation. Shape: {s.shape}, strides: {s.stride()}")
    if not pi.is_contiguous():
        raise RuntimeError(f"Policy tensor lost contiguity after augmentation. Shape: {pi.shape}, strides: {pi.stride()}")
    if not z.is_contiguous():
        raise RuntimeError(f"Value tensor lost contiguity after augmentation. Shape: {z.shape}, strides: {z.stride()}")

    # Generate self-supervised learning targets (class indices for multi-class prediction)
    ssl_targets = None
    if enable_ssl and hasattr(model, 'create_ssl_targets'):
        logger.info("TRAINING: Creating SSL targets...")
        ssl_targets = model.create_ssl_targets(s)
        ssl_targets = ssl_targets.contiguous()
        logger.info(f"TRAINING: SSL targets created with shape {ssl_targets.shape}")

        # DEBUG: Log SSL targets statistics
        logger.info(f"SSL TARGETS DEBUG: shape={ssl_targets.shape}, min={ssl_targets.min().item()}, max={ssl_targets.max().item()}, sum={ssl_targets.sum().item()}, all_zeros={torch.all(ssl_targets == 0).item()}")
    else:
        logger.warning(f"TRAINING: SSL targets not created - enable_ssl={enable_ssl}, has_create_ssl_targets={hasattr(model, 'create_ssl_targets')}")

    # CRITICAL: Setup precision and device type BEFORE data transfer (OUTSIDE SSL block)
    device_type = device.split(':')[0]
    use_autocast = scaler is not None

    if precision == "bf16":
        _amp_dtype = torch.bfloat16
        scaler = None
    elif precision == "fp16":
        _amp_dtype = torch.float16
    else: # fp32
        use_autocast = False
        _amp_dtype = torch.float32

    logger.debug(f"Using precision: {precision}, autocast: {use_autocast}, dtype: {_amp_dtype}")

    # Ensure input data matches model precision
    if s.dtype != next(model.parameters()).dtype:
        s = s.to(next(model.parameters()).dtype)
        pi = pi.to(next(model.parameters()).dtype)
        z = z.to(next(model.parameters()).dtype)
        if ssl_targets is not None:
            ssl_targets = ssl_targets.to(torch.long) # Target for CrossEntropyLoss should be long

    # CRITICAL: Optimize data transfer to device with non_blocking
    s = s.to(device, non_blocking=True)
    pi = pi.to(device, non_blocking=True)
    z = z.to(device, non_blocking=True)
    if ssl_targets is not None:
        ssl_targets = ssl_targets.to(device, non_blocking=True)

    with torch.autocast(device_type=device_type, dtype=_amp_dtype, enabled=use_autocast):
        # Keep standard contiguous format during training to avoid MPS view/stride issues
        s = s.contiguous()
        if use_wdl and hasattr(model, 'forward_with_features'):
            p, v, ssl_out, feats = model.forward_with_features(s, return_ssl=True)
        else:
            feats = None
            p, v, ssl_out = model(s, return_ssl=True)

        # CRITICAL: Validate policy outputs before computing loss
        if torch.isnan(p).any() or torch.isinf(p).any():
            logger.error(f"Policy output contains NaN/Inf: policy={torch.isnan(p).sum()}/{p.numel()}")
            # Skip this batch to prevent training crash
            return None, None, None, None, None
        
        # Validate policy logits are in reasonable range (match model clamping)
        if p.abs().max() > 5.0:
            logger.warning(f"Policy logits too large: max={p.abs().max()}, clamping to match model")
            p = torch.clamp(p, -5.0, 5.0)
        
        # Ensure contiguity to avoid view-related autograd errors on some backends
        p = p.contiguous()
        v = v.contiguous()
        if ssl_out is not None:
            # Force standard contiguous (NCHW) to avoid channels_last view issues in CrossEntropy backward (MPS)
            try:
                ssl_out = ssl_out.contiguous(memory_format=torch.contiguous_format)
            except Exception:
                ssl_out = ssl_out.contiguous()
        
        # Validate model outputs to catch any contiguity issues
        if not p.is_contiguous():
            raise RuntimeError(f"Policy output tensor is not contiguous. Shape: {p.shape}, strides: {p.stride()}")
        if not v.is_contiguous():
            raise RuntimeError(f"Value output tensor is not contiguous. Shape: {v.shape}, strides: {v.stride()}")
        if ssl_out is not None and not ssl_out.is_contiguous():
            raise RuntimeError(f"SSL output tensor is not contiguous. Shape: {ssl_out.shape}, strides: {ssl_out.stride()}")

        # Optional legality masking for stability: mask logits where target is zero
        # Assumes pi provides positive mass only on legal actions
        if policy_masking:
            try:
                with torch.no_grad():
                    legal_mask = (pi > 0)
                    # If any row is all-zero (fallback), treat all as legal (no mask)
                    valid_rows = legal_mask.any(dim=1, keepdim=True)
                    # Build a keep mask: keep original logits where either row invalid or legal; else set to -1e9
                    keep_mask = valid_rows & legal_mask
                p_for_loss = torch.where(keep_mask, p, torch.full_like(p, -1e9)).contiguous()
            except Exception:
                p_for_loss = p.contiguous()
        else:
            p_for_loss = p.contiguous()
        # Policy loss with optional label smoothing
        if label_smoothing and label_smoothing > 0.0:
            num_actions = p.shape[1]
            smooth = label_smoothing / float(num_actions)
            pi_smooth = (1.0 - label_smoothing) * pi + smooth
            log_probs = nn.functional.log_softmax(p_for_loss.contiguous(), dim=1)
            policy_loss = -(pi_smooth * log_probs).sum(dim=1).mean()
        else:
            log_probs = nn.functional.log_softmax(p_for_loss.contiguous(), dim=1)
            policy_loss = -(pi * log_probs).sum(dim=1).mean()
        
        # Ensure policy loss is contiguous
        policy_loss = policy_loss.contiguous()
        
        # Add policy regularization to prevent uniform outputs
        # This encourages the model to produce diverse, meaningful policies
        policy_probs = torch.softmax(p_for_loss.contiguous(), dim=1)
        uniform_probs = torch.ones_like(policy_probs) / policy_probs.shape[1]
        policy_entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(policy_probs.shape[1], dtype=torch.float32, device=p.device))
        
        # Penalize if policy is too close to uniform (entropy too high)
        # But don't penalize if policy is too concentrated (entropy too low)
        entropy_penalty = torch.relu(policy_entropy - 0.8 * max_entropy)
        policy_reg_loss = 0.1 * entropy_penalty
        
        # Ensure policy regularization loss is contiguous
        policy_reg_loss = policy_reg_loss.contiguous()
        
        # Value loss: MSE or Huber
        if value_loss_type == 'huber':
            value_loss = nn.functional.smooth_l1_loss(v, z, beta=huber_delta)
        else:
            value_loss = nn.functional.mse_loss(v, z)
        
        # Ensure value loss is contiguous
        value_loss = value_loss.contiguous()
        
        ssl_loss = 0.0
        if enable_ssl and ssl_targets is not None and ssl_out is not None:
            # DEBUG: Log SSL computation conditions
            if torch.rand(1).item() < 0.05:  # 5% chance to log
                logger.info(f"SSL COMPUTATION: enable_ssl={enable_ssl}, ssl_targets is not None={ssl_targets is not None}, ssl_out is not None={ssl_out is not None}")

            # SSL warmup: linearly ramp SSL weight over first ssl_warmup_steps
            if ssl_warmup_steps and ssl_warmup_steps > 0:
                ramp = min(1.0, float(current_step) / float(ssl_warmup_steps))
            else:
                ramp = 1.0
            # SSL loss computation using the new method
            ssl_loss = model.get_ssl_loss(s, ssl_targets)
            # Ensure all loss components are contiguous before arithmetic operations
            policy_loss = policy_loss.contiguous()
            policy_reg_loss = policy_reg_loss.contiguous()
            value_loss = value_loss.contiguous()
            ssl_loss = ssl_loss.contiguous()
            loss = policy_loss + policy_reg_loss + value_loss + (ssl_weight * ramp * ssl_target_weight) * ssl_loss
        else:
            # Ensure all loss components are contiguous before arithmetic operations
            policy_loss = policy_loss.contiguous()
            policy_reg_loss = policy_reg_loss.contiguous()
            value_loss = value_loss.contiguous()
            loss = policy_loss + policy_reg_loss + value_loss

        # Optional WDL auxiliary head
        wdl_loss = 0.0
        if use_wdl and wdl_weight > 0.0 and feats is not None and hasattr(model, 'compute_wdl_logits'):
            try:
                wdl_logits = model.compute_wdl_logits(feats)
                if wdl_logits is not None:
                    # Build WDL targets from value targets z
                    # 0: loss (z < -margin), 1: draw (|z| <= margin), 2: win (z > margin)
                    with torch.no_grad():
                        cls = torch.full_like(z, 1, dtype=torch.long)
                        cls = torch.where(z > float(wdl_margin), torch.tensor(2, device=z.device, dtype=torch.long), cls)
                        cls = torch.where(z < -float(wdl_margin), torch.tensor(0, device=z.device, dtype=torch.long), cls)
                    wdl_loss = nn.functional.cross_entropy(wdl_logits, cls)
                    # Ensure loss is contiguous before arithmetic operations
                    wdl_loss = wdl_loss.contiguous()
                    loss = loss + (float(wdl_weight) * wdl_loss)
            except Exception:
                pass
    
    # Guard against NaN/Inf loss; skip backward if not finite
    if torch.isfinite(loss):
        # Ensure loss tensor is contiguous before backward pass
        if not loss.is_contiguous():
            loss = loss.contiguous()
        try:
            if scaler is not None:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()
        except RuntimeError as e:
            msg = str(e)
            if 'view size is not compatible' in msg or 'contiguous subspaces' in msg:
                # Emit targeted diagnostics to help trace offending tensors
                try:
                    logger.error(
                        "Backward view/stride error. Diagnostics: p(%s,%s), v(%s,%s), ssl_out(%s,%s)",
                        tuple(p.shape) if 'p' in locals() else None,
                        tuple(p.stride()) if 'p' in locals() and hasattr(p, 'stride') else None,
                        tuple(v.shape) if 'v' in locals() else None,
                        tuple(v.stride()) if 'v' in locals() and hasattr(v, 'stride') else None,
                        tuple(ssl_out.shape) if 'ssl_out' in locals() and ssl_out is not None else None,
                        tuple(ssl_out.stride()) if 'ssl_out' in locals() and ssl_out is not None and hasattr(ssl_out, 'stride') else None,
                    )
                    logger.error(
                        "Inputs: s(%s,%s), pi(%s,%s), z(%s,%s)",
                        tuple(s.shape), tuple(s.stride()),
                        tuple(pi.shape), tuple(pi.stride()),
                        tuple(z.shape), tuple(z.stride()),
                    )
                except Exception:
                    pass
            elif "Expected a gradient" in msg:
                logger.error(f"Gradient error during backward pass: {e}", exc_info=True)
            else:
                logger.error(f"Runtime error during backward pass: {e}", exc_info=True)
            raise
    else:
        logger.warning("Non-finite loss encountered; skipping backward for this batch")
    
    # Optimizer stepping and clipping are handled by the caller to support grad accumulation
    return loss.item(), policy_loss.item(), value_loss.item(), (ssl_loss.item() if ssl_targets is not None else 0.0), (wdl_loss.item() if (use_wdl and wdl_weight > 0.0) else 0.0)

def get_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Creates a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def train_comprehensive(
    config_path: str = "config.yaml",
    total_steps: int = 10000,
    batch_size: int = 192,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    ema_decay: float = 0.999,
    grad_clip_norm: float = 1.0,
    accum_steps: int = 2,
    warmup_steps: int = 500,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    device: str = "auto",
    use_amp: bool = True,
    augment: bool = True,
    precision: str = "fp16"
):
    """Comprehensive training with progress tracking and ETA."""
    
    # Load configuration
    cfg = Config.load(config_path)
    device = select_device(device)
    
    # Memory cleanup at start of training
    logger.info("Performing memory cleanup at start of training")
    try:
        import gc
        gc.collect()
        if device.startswith('mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("MPS memory cache cleared")
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")
    
    # Apply config overrides for performance optimization
    if cfg.training().get('gradient_accumulation_steps'):
        accum_steps = int(cfg.training().get('gradient_accumulation_steps'))
        logger.info(f"Using gradient accumulation: {accum_steps} steps (effective batch size: {batch_size * accum_steps})")
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = PolicyValueNet.from_config(cfg.model()).to(device)
    
    # Enable memory optimizations
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled for memory efficiency")
    
    # Log model memory usage
    if hasattr(model, 'get_memory_usage'):
        memory_stats = model.get_memory_usage()
        logger.info(f"Model memory usage: {memory_stats}")
    
    # Optional compile for speed (PyTorch 2.1+)
    try:
        if bool(cfg.training().get('compile', False)):
            # torch already imported at module scope; avoid local import that would shadow name
            model = torch.compile(model, mode=cfg.training().get('compile_mode', 'default'))
            logger.info("torch.compile enabled")
    except Exception as _e:
        logger.warning(f"torch.compile not enabled: {_e}")

    # Use AMP + channels_last for MPS/CUDA; avoid hard quantization of BatchNorm layers
    if device.startswith('mps'):
        logger.info("Using MPS with autocast; not forcing FP16 parameters to keep BatchNorm stable")
    
    # Load best checkpoint if available
    best_ckpt = Path(checkpoint_dir) / "best.pt"
    start_step = 0
    state = None
    if best_ckpt.exists():
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        try:
            state = torch.load(best_ckpt, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats robustly
            if "model_ema" in state:
                logger.info("Loading EMA model state")
                model.load_state_dict(state["model_ema"])
            elif "model" in state:
                logger.info("Loading model state")
                model.load_state_dict(state["model"])
            elif "model_state_dict" in state:
                logger.info("Loading model_state_dict")
                model.load_state_dict(state["model_state_dict"])
            else:
                # Try to find any key that looks like model weights
                model_keys = [k for k in state.keys() if 'model' in k.lower() and 'state' in k.lower()]
                if model_keys:
                    logger.info(f"Found model key: {model_keys[0]}")
                    model.load_state_dict(state[model_keys[0]])
                else:
                    logger.warning("No recognizable model state found in checkpoint, starting fresh")
                    state = None
            
            if state is not None:
                start_step = state.get("step", state.get("global_step", 0))
                logger.info(f"Resuming from step {start_step}")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch")
            state = None
    else:
        logger.info("Starting training from scratch")
    
    # Initialize optimizer, scheduler, and EMA
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps)
    ema = EMA(model, ema_decay)
    
    # Initialize GradScaler for mixed precision in a device-agnostic way
    # Use precision parameter to determine if AMP should be enabled
    if precision == "fp32":
        use_amp = False
    elif precision in ["fp16", "bf16"]:
        use_amp = True
    # Otherwise use the passed use_amp parameter
    
    if use_amp:
        device_type = device.split(":")[0]
        if device_type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = torch.amp.GradScaler(device="mps")
    else:
        scaler = None
    if scaler is not None:
        logger.info(f"Using Automatic Mixed Precision (AMP) with {precision} precision.")
    else:
        logger.info(f"Using {precision} precision without AMP.")

    # Load optimizer and scheduler state if resuming
    if state is not None:
        # Try different key names for optimizer
        optimizer_key = None
        for key in ["optimizer", "optimizer_state_dict"]:
            if key in state and state[key] is not None:
                optimizer_key = key
                break
        
        if optimizer_key:
            try:
                optimizer.load_state_dict(state[optimizer_key])
                logger.info(f"Loaded optimizer state from {optimizer_key}")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state from {optimizer_key}: {e}, starting fresh")
        else:
            logger.info("No optimizer state found, starting fresh")
            
        # Try different key names for scheduler
        scheduler_key = None
        for key in ["scheduler", "scheduler_state_dict"]:
            if key in state and state[key] is not None:
                scheduler_key = key
                break
                
        if scheduler_key:
            try:
                scheduler.load_state_dict(state[scheduler_key])
                logger.info(f"Loaded scheduler state from {scheduler_key}")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state from {scheduler_key}: {e}, starting fresh")
        else:
            logger.info("No scheduler state found, starting fresh")
    else:
        logger.info("No checkpoint state, starting fresh")

    # Apply warmup learning rate if starting from scratch
    if start_step < warmup_steps:
        warmup_factor = float(start_step) / float(max(1, warmup_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * warmup_factor
        logger.info(f"Applied warmup LR: {learning_rate * warmup_factor:.6f}")

    # Setup data using DataManager
    data_manager = DataManager(base_dir=cfg.get("data_dir", "data"))
    data_stats = data_manager.get_stats()
    external_stats = data_manager.get_external_data_stats()
    
    if data_stats.total_samples == 0 and external_stats['external_total'] == 0:
        raise ValueError("No training data found by DataManager!")
    
    logger.info(f"DataManager found {data_stats.total_shards} shards with {data_stats.total_samples} total samples.")
    logger.info(f"External training data: {external_stats['tactical_samples']} tactical + {external_stats['openings_samples']} openings = {external_stats['external_total']} total samples.")
    
    # Check if curriculum learning is enabled
    use_curriculum = cfg.training().get("use_curriculum", False)
    curriculum_phases = cfg.training().get("curriculum_phases", [])
    
    if use_curriculum and curriculum_phases:
        logger.info(f"Curriculum learning enabled with {len(curriculum_phases)} phases")
        current_phase = curriculum_phases[0]['name']  # Start with first phase
        logger.info(f"Starting with curriculum phase: {current_phase}")
    else:
        logger.info("Using standard training data mixing")
        current_phase = "mixed"
    
    # Use appropriate batch method based on configuration
    if use_curriculum and curriculum_phases:
        # Curriculum learning - will be handled in training loop
        batch_generator = None
    else:
        # Standard training - use external data if available, fallback to replay buffer
        if external_stats['external_total'] > 0:
            logger.info("Using external training data with self-play mixing")
            batch_generator = None  # Will use get_curriculum_batch in training loop
        else:
            logger.info("Using replay buffer only")
            batch_generator = data_manager.get_training_batch(batch_size, device)
    
    logger.info(f"Training for {total_steps - start_step} steps.")
    logger.info(f"Batch size: {batch_size}, Gradient Accumulation: {accum_steps}")
    logger.info(f"Device: {device}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)

    # Get training config (needed for all device types)
    tr_cfg = cfg.training()  # Get training config

    # Set MPS memory limit if specified
    memory_limit_gb = tr_cfg.get('memory_limit_gb', 12)  # Default 12GB if not specified
    if device.startswith('mps'):
        import os
        memory_ratio = min(memory_limit_gb / 18.0, 0.9)  # Cap at 90% to be safe
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(memory_ratio)
        logger.info(f"Set MPS memory limit to {memory_limit_gb}GB (ratio: {memory_ratio:.2f})")

        # Enable model memory optimizations
        if hasattr(model, 'enable_memory_optimization'):
            model.enable_memory_optimization()
            logger.info("Enabled model memory optimizations for MPS")

    # Training loop
    current_step = start_step
    
    pbar = tqdm(total=total_steps, desc="Training Progress", unit="step", initial=start_step)
    
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_ssl_loss = 0.0
    running_wdl_loss = 0.0

    # Watchdog for detecting training hangs
    last_progress_time = time.time()
    watchdog_timeout = 300  # 5 minutes without progress

    start_time = time.time()
    optimizer.zero_grad()

    try:
        while current_step < total_steps:
            # Handle curriculum learning phase transitions
            if use_curriculum and curriculum_phases:
                current_phase_info = None
                for phase in curriculum_phases:
                    if current_step < phase.get('steps', float('inf')):
                        current_phase_info = phase
                        break
                
                if current_phase_info and current_phase_info['name'] != current_phase:
                    current_phase = current_phase_info['name']
                    logger.info(f"Transitioning to curriculum phase: {current_phase}")
            
            # Get training batch based on current configuration
            try:
                if use_curriculum and curriculum_phases:
                    # Curriculum learning - use phase-specific data
                    batch_dict = data_manager.get_curriculum_batch(batch_size, current_phase)
                    if batch_dict is None:
                        logger.warning(f"No data available for phase {current_phase}, falling back to mixed")
                        batch_dict = data_manager.get_curriculum_batch(batch_size, "mixed")
                    
                    if batch_dict is None:
                        logger.error("No training data available, stopping training")
                        break
                    
                    # Convert dict format to tuple format for existing train_step
                    batch = (batch_dict['s'], batch_dict['pi'], batch_dict['z'])
                    
                elif batch_generator is None:
                    # External data training - use curriculum mixing
                    batch_dict = data_manager.get_curriculum_batch(batch_size, "mixed")
                    if batch_dict is None:
                        logger.error("No external training data available, stopping training")
                        break
                    
                    # Convert dict format to tuple format for existing train_step
                    batch = (batch_dict['s'], batch_dict['pi'], batch_dict['z'])
                    
                else:
                    # Standard replay buffer training
                    batch = next(batch_generator)
                    
            except StopIteration:
                if batch_generator:
                    logger.info("Data stream exhausted, restarting batch generator.")
                    batch_generator = data_manager.get_training_batch(batch_size, device)
                    continue
                else:
                    logger.error("External data stream exhausted, stopping training")
                    break
            except Exception as e:
                logger.warning(f"Error getting batch: {e}, skipping...")
                continue

            # Error recovery wrapper for train_step
            try:
                # tr_cfg already assigned at function start
                loss_values = train_step(
                model, optimizer, scaler, batch, device, accum_steps, augment,
                augment_rotate180=bool(tr_cfg.get('augment_rotate180', True)),
                ssl_weight=float(tr_cfg.get('ssl_weight', 0.1)), enable_ssl=bool(tr_cfg.get('self_supervised', False)),
                label_smoothing=float(tr_cfg.get('policy_label_smoothing', 0.0)),
                value_loss_type=str(tr_cfg.get('value_loss', 'mse')),
                huber_delta=float(tr_cfg.get('huber_delta', 1.0)),
                policy_masking=bool(tr_cfg.get('policy_masking', True)),
                ssl_warmup_steps=int(tr_cfg.get('ssl_warmup_steps', 0)),
                current_step=int(current_step),
                ssl_target_weight=float(tr_cfg.get('ssl_target_weight', 1.0)),
                use_wdl=bool(cfg.model().get('wdl', False)),
                wdl_weight=float(tr_cfg.get('wdl_weight', 0.0)),
                wdl_margin=float(tr_cfg.get('wdl_margin', 0.25)),
                precision=tr_cfg.get("precision", "fp16")
                )
                
                # Check if train_step returned None (indicating invalid batch)
                if loss_values is None:
                    logger.warning("train_step returned None, skipping batch")
                    continue
                
                loss, policy_loss, value_loss, ssl_loss, wdl_loss = loss_values
                
                # Validate loss values (they are Python floats from .item() calls)
                if not (np.isfinite(loss) and np.isfinite(policy_loss) and 
                       np.isfinite(value_loss) and np.isfinite(ssl_loss)):
                    logger.warning(f"Non-finite loss detected: loss={loss}, policy={policy_loss}, value={value_loss}, ssl={ssl_loss}")
                    # Skip this batch and continue training
                    continue
                    
            except Exception as e:
                logger.error(f"Error in train_step: {e}", exc_info=True)
                logger.error(f"Batch info: device={device}, batch_size={batch[0].shape[0] if batch else 'unknown'}")
                
                # Try to recover from the error
                try:
                    # Clear gradients and reset optimizer state
                    optimizer.zero_grad()
                    if scaler:
                        scaler.update()
                    logger.info("Recovered from training error, continuing...")
                except Exception as recovery_error:
                    logger.error(f"Failed to recover from training error: {recovery_error}")
                
                # Skip this batch and continue training
                continue
            
            running_loss = 0.98 * running_loss + 0.02 * loss
            running_policy_loss = 0.98 * running_policy_loss + 0.02 * policy_loss
            running_value_loss = 0.98 * running_value_loss + 0.02 * value_loss
            running_ssl_loss = 0.98 * running_ssl_loss + 0.02 * ssl_loss
            running_wdl_loss = 0.98 * running_wdl_loss + 0.02 * wdl_loss
            
            # CRITICAL: Update SSL curriculum difficulty during training
            if hasattr(model, 'update_ssl_curriculum') and cfg.model().get('ssl_curriculum', False):
                model.update_ssl_curriculum(current_step, total_steps)
            
            if (current_step + 1) % accum_steps == 0:
                try:
                    if scaler:
                        scaler.unscale_(optimizer)
                    if grad_clip_norm > 0:
                        # CRITICAL: Ultra-aggressive gradient clipping to prevent NaN/Inf
                        total_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=False)
                        if torch.isnan(total_norm) or torch.isinf(total_norm):
                            logger.warning(f"Gradient norm is NaN/Inf: {total_norm}, applying emergency clipping")
                            # Emergency gradient clipping with very small norm
                            nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=False)
                            total_norm = 0.01

                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # CRITICAL: Step scheduler AFTER optimizer to avoid LR warning
                    scheduler.step()
                    optimizer.zero_grad()
                    ema.update(model)

                    # Update watchdog timer
                    last_progress_time = time.time()

                except Exception as e:
                    logger.error(f"Error in optimization step: {e}")
                    logger.error(f"Current step: {current_step}, Loss: {running_loss:.4f}")
                    
                    # Try to recover from optimization error
                    try:
                        optimizer.zero_grad()
                        if scaler:
                            scaler.update()
                        logger.info("Recovered from optimization error, continuing...")
                    except Exception as recovery_error:
                        logger.error(f"Failed to recover from optimization error: {recovery_error}")
                    
                    # Force a checkpoint and continue
                    try:
                        checkpoint_name = f"emergency_checkpoint_step_{current_step}.pt"
                        checkpoint_path = Path(checkpoint_dir) / checkpoint_name
                        save_checkpoint(model, ema, optimizer, scheduler, current_step, checkpoint_path)
                        logger.info(f"Saved emergency checkpoint: {checkpoint_path}")
                    except Exception as checkpoint_error:
                        logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
                    
                    last_progress_time = time.time()
                
                # Check for training hangs (moved outside the try block)
                if time.time() - last_progress_time > watchdog_timeout:
                    logger.error(f"Training appears to be hanging (no progress for {watchdog_timeout}s)")
                    logger.error(f"Current step: {current_step}, Loss: {running_loss:.4f}")
                    # Force a checkpoint and continue
                    try:
                        checkpoint_name = f"emergency_checkpoint_step_{current_step}.pt"
                        checkpoint_path = Path(checkpoint_dir) / checkpoint_name
                        save_checkpoint(model, ema, optimizer, scheduler, current_step, checkpoint_path)
                        logger.info(f"Saved emergency checkpoint: {checkpoint_path}")
                    except Exception as checkpoint_error:
                        logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
                    last_progress_time = time.time()
            
            pbar.update(1)
            current_step += 1
            
            if current_step % 10 == 0:
                writer.add_scalar('Loss/total', running_loss, current_step)
                writer.add_scalar('Loss/policy', running_policy_loss, current_step)
                writer.add_scalar('Loss/value', running_value_loss, current_step)
                writer.add_scalar('Loss/ssl', running_ssl_loss, current_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], current_step)
                if cfg.model().get('wdl', False) and float(tr_cfg.get('wdl_weight', 0.0)) > 0.0:
                    writer.add_scalar('Loss/wdl', running_wdl_loss, current_step)
            
            if current_step % 100 == 0:
                elapsed_time = time.time() - start_time
                steps_per_second = (current_step - start_step) / elapsed_time
                eta_seconds = (total_steps - current_step) / steps_per_second if steps_per_second > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
                
                # Calculate memory usage
                if hasattr(model, 'get_memory_usage'):
                    memory_usage = model.get_memory_usage()
                else:
                    memory_usage = "N/A"
                
                pbar.set_postfix({
                    'Loss': f'{running_loss:.4f}',
                    'Policy': f'{running_policy_loss:.4f}',
                    'Value': f'{running_value_loss:.4f}',
                    'SSL': f'{running_ssl_loss:.4f}',
                    'LR': f"{optimizer.param_groups[0]['lr']:.1e}",
                    'Memory': memory_usage,
                    'ETA': str(eta),
                })
            
            # Save checkpoints based on configuration
            checkpoint_freq = cfg.get("checkpoint_save_freq", 1000)
            if current_step % checkpoint_freq == 0:
                checkpoint_name = f"{cfg.get('checkpoint_prefix', 'model')}_step_{current_step}.pt"
                checkpoint_path = Path(checkpoint_dir) / checkpoint_name
                save_checkpoint(model, ema, optimizer, scheduler, current_step, checkpoint_path)
                logger.info(f"Saved checkpoint at step {current_step}: {checkpoint_path}")
            
            # Log validation info based on configuration
            validation_freq = cfg.get("validation_freq", 500)
            if current_step % validation_freq == 0:
                logger.info(f"Step {current_step}: Validation checkpoint - Loss: {running_loss:.4f}, Policy: {running_policy_loss:.4f}, Value: {running_value_loss:.4f}, SSL: {running_ssl_loss:.4f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        pbar.close()
        
        # Save final checkpoint with enhanced prefix
        checkpoint_prefix = cfg.get("checkpoint_prefix", "enhanced")
        final_checkpoint = Path(checkpoint_dir) / f"{checkpoint_prefix}_final.pt"
        save_checkpoint(model, ema, optimizer, scheduler, current_step, final_checkpoint)
        logger.info(f"Saved final checkpoint: {final_checkpoint}")
        
        # Save enhanced checkpoint (don't overwrite baseline best.pt)
        enhanced_checkpoint = Path(checkpoint_dir) / f"{checkpoint_prefix}_best.pt"
        save_checkpoint(model, ema, optimizer, scheduler, current_step, enhanced_checkpoint)
        logger.info(f"Saved enhanced checkpoint: {enhanced_checkpoint}")
        
        logger.info(f"Baseline checkpoint 'best.pt' preserved for comparison")
        
        total_time = time.time() - start_time
        logger.info(f"Training finished in {timedelta(seconds=int(total_time))}")
        logger.info(f"Final step: {current_step}")
        # Consistent summary line
        try:
            logger.info(f"TRAIN SUMMARY: steps={current_step} batch={batch_size} accum={accum_steps} time_s={int(total_time)} lr={optimizer.param_groups[0]['lr']:.2e}")
        except Exception:
            pass
        
        # JSONL train summary
        try:
            import json as _json
            logs_dir = Path(cfg.training().get("log_dir", "logs"))
            logs_dir.mkdir(parents=True, exist_ok=True)
            _rec = {
                'type': 'train_summary',
                'steps': int(current_step),
                'batch_size': int(batch_size),
                'accum_steps': int(accum_steps),
                'time_seconds': int(total_time),
                'lr': float(optimizer.param_groups[0]['lr']),
                'timestamp': int(time.time()),
            }
            with (logs_dir / 'train_summary.jsonl').open('a') as _f:
                _f.write(_json.dumps(_rec) + "\n")
        except Exception:
            pass

        writer.close()

def save_checkpoint(model, ema, optimizer, scheduler, step, path):
    """Save a training checkpoint with robust error handling."""
    try:
        # Create consistent checkpoint format
        state = {
            'step': step,
            'global_step': step,  # Alternative key for compatibility
            'model': model.state_dict(),
            'model_state_dict': model.state_dict(),  # Alternative key for compatibility
            'model_ema': ema.shadow if ema.shadow is not None else None,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',  # Version tracking
            'config': {
                'model_type': 'PolicyValueNet',
                'architecture': 'V2_Enhanced'
            }
        }
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with error handling
        torch.save(state, path)
        logger.info(f"Checkpoint saved successfully: {path}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint {path}: {e}")
        # Try to save minimal checkpoint
        try:
            minimal_state = {
                'step': step,
                'model': model.state_dict(),
                'timestamp': datetime.now().isoformat(),
                'error': f"Minimal save due to: {str(e)}"
            }
            torch.save(minimal_state, path)
            logger.info(f"Minimal checkpoint saved: {path}")
        except Exception as e2:
            logger.error(f"Failed to save even minimal checkpoint: {e2}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Matrix0 Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--steps", type=int, default=6500, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Max learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/mps/cuda)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp", help="Disable Automatic Mixed Precision")
    parser.add_argument("--no-augment", action="store_false", dest="augment", help="Disable data augmentation")
    
    args = parser.parse_args()
    
    train_comprehensive(
        config_path=args.config,
        total_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        grad_clip_norm=args.grad_clip,
        accum_steps=args.accum_steps,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        use_amp=args.use_amp,
        augment=args.augment
    )

def train_from_config(config_path: str = "config.yaml"):
    """Training function that works with orchestrator - uses config file only."""
    cfg = Config.load(config_path)
    
    # Extract training parameters from config
    train_cfg = cfg.training()
    
    train_comprehensive(
        config_path=config_path,
        total_steps=train_cfg.get("steps_per_epoch", 10000),
        batch_size=train_cfg.get("batch_size", 256),
        learning_rate=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        ema_decay=train_cfg.get("ema_decay", 0.999),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        accum_steps=train_cfg.get("accum_steps", 1),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
        log_dir=train_cfg.get("log_dir", "logs"),
        device=cfg.get("device", "auto"),
        use_amp=train_cfg.get("precision", "fp16") != "fp32",
        augment=True
    )

if __name__ == "__main__":
    main()
