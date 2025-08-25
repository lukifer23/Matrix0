#!/usr/bin/env python3
"""
Comprehensive Training Script for Matrix0
- Integrated DataManager for robust data handling.
- Advanced training features: mixed precision, LR scheduling, and data augmentation.
- Progress bars for all phases with ETA.
- Proper checkpointing and TensorBoard logging.
"""

import argparse
import math
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from azchess.config import Config, select_device
from azchess.data_manager import DataManager
from azchess.encoding import POLICY_SHAPE, encode_board, move_to_index
from azchess.logging_utils import setup_logging
from azchess.model import PolicyValueNet
from azchess.training.npz_dataset import build_training_dataloader
from azchess.utils import (add_memory_alert_callback, clear_memory_cache,
                           emergency_memory_cleanup, get_memory_usage,
                           log_tensor_stats, safe_config_get,
                           start_memory_monitoring)

# Setup logging
logger = setup_logging(level=logging.INFO)

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


def _ensure_contiguous(t: torch.Tensor, name: str) -> torch.Tensor:
    """Force a tensor into standard contiguous memory format.

    Some backends (e.g., MPS) are sensitive to tensors with channels_last
    layout.  We attempt to set the contiguous memory format and log a warning
    if that fails before falling back to the default ``contiguous`` call.
    """
    if t is None:
        return None
    try:
        return t.contiguous(memory_format=torch.contiguous_format)
    except Exception as e:  # pragma: no cover - backend specific
        logger.warning(f"Failed to set contiguous format for {name}: {e}")
        return t.contiguous()


def _check_contiguous(outputs: dict):
    """Validate that output tensors are contiguous.

    Raises:
        RuntimeError: if any tensor is not contiguous.
    """
    for name, tensor in outputs.items():
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(
                f"{name} output tensor is not contiguous. Shape: {tensor.shape}, strides: {tensor.stride()}"
            )


def apply_policy_mask(p: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Mask policy logits using the provided target distribution.

    Any logits corresponding to zero-probability entries in ``pi`` are set to a
    large negative value.  If masking fails for any reason the original logits
    are returned and the error is logged.
    """
    try:
        with torch.no_grad():
            legal_mask = pi > 0
            valid_rows = legal_mask.any(dim=1, keepdim=True)
            keep_mask = valid_rows & legal_mask
        return torch.where(keep_mask, p, torch.full_like(p, -1e9))
    except Exception as e:
        logger.error(f"Policy masking failed: {e}")
        return p

def train_step(model, optimizer, scaler, batch, device: str, accum_steps: int = 1, augment: bool = True,
               augment_rotate180: bool = True,
               ssl_weight: float = 0.1, enable_ssl: bool = True,
               label_smoothing: float = 0.0, value_loss_type: str = 'mse', huber_delta: float = 1.0,
               policy_masking: bool = True, ssl_warmup_steps: int = 0, current_step: int = 0, ssl_target_weight: float = 1.0,
               use_wdl: bool = False, wdl_weight: float = 0.0, wdl_margin: float = 0.25, precision: str = "fp16"):
    """Single training step with augmentation, mixed precision, and self-supervised learning."""
    import time

    import torch
    start_time = time.time()

    model.train()
    # Support batches as tuple/list (s, pi, z[, legal_mask]) or dict with optional 'legal_mask'
    legal_mask_np = None
    if isinstance(batch, dict):
        s = batch['s']
        pi = batch['pi']
        z = batch['z']
        legal_mask_np = batch.get('legal_mask', None)
    else:
        if len(batch) == 4:
            s, pi, z, legal_mask_np = batch
        else:
            s, pi, z = batch

    # PERFORMANCE PROFILING: Data preparation
    data_prep_start = time.time()

    # Convert numpy arrays to PyTorch tensors on CPU
    s = torch.from_numpy(s)
    pi = torch.from_numpy(pi)
    z = torch.from_numpy(z)
    legal_mask_t = None
    if legal_mask_np is not None:
        try:
            if isinstance(legal_mask_np, np.ndarray):
                legal_mask_t = torch.from_numpy(legal_mask_np)
            elif torch.is_tensor(legal_mask_np):
                legal_mask_t = legal_mask_np
            else:
                # Unknown type; try numpy conversion
                legal_mask_t = torch.from_numpy(np.asarray(legal_mask_np))
            # Normalize to bool mask of shape (N, 4672)
            if legal_mask_t.dtype != torch.bool:
                legal_mask_t = legal_mask_t.to(dtype=torch.bool)
        except Exception:
            legal_mask_t = None
    # Normalize common shape variant for values: (N,) or (N,1) → (N,)
    if z.dim() == 2 and z.size(1) == 1:
        z = z.reshape(z.size(0))
    
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

    # PERFORMANCE PROFILING: Data prep complete
    data_prep_time = time.time() - data_prep_start
    if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):  # Log every 10 steps
        logger.debug(f"PERF: Data preparation: {data_prep_time:.3f}s")

    # Setup precision and device type BEFORE moving tensors
    device_type = device.split(':')[0]
    use_autocast = scaler is not None

    if precision == "bf16":
        _amp_dtype = torch.bfloat16
        use_autocast = False  # Disable autocast for bf16 but keep scaler
    elif precision == "fp16":
        _amp_dtype = torch.float16
    else: # fp32
        use_autocast = False
        _amp_dtype = torch.float32

    logger.debug(f"Using precision: {precision}, autocast: {use_autocast}, dtype: {_amp_dtype}, device: {device_type}")

    # Move tensors to target device once
    s = s.to(device, non_blocking=True)
    pi = pi.to(device, non_blocking=True)
    z = z.to(device, non_blocking=True)
    if legal_mask_t is not None:
        legal_mask_t = legal_mask_t.to(device, non_blocking=True)

    # Data Augmentation: geometric transforms aligned with action space
    if augment:
        r = torch.rand(1, device=s.device).item()
        if r < 0.5:
            # Horizontal flip (mirror files)
            s = torch.flip(s, dims=[3]).contiguous()
            pi_sh = torch.flip(pi.view(-1, *POLICY_SHAPE), dims=[2]).contiguous()
            from azchess.encoding import build_horizontal_flip_permutation
            perm = build_horizontal_flip_permutation()
            perm_t = torch.as_tensor(perm, device=pi_sh.device, dtype=torch.long)
            pi = pi_sh.index_select(-1, perm_t).view(-1, np.prod(POLICY_SHAPE))
            if legal_mask_t is not None:
                lm_sh = torch.flip(legal_mask_t.view(-1, *POLICY_SHAPE), dims=[2]).contiguous()
                legal_mask_t = lm_sh.index_select(-1, perm_t).view(-1, np.prod(POLICY_SHAPE)).to(dtype=torch.bool)
        elif r < 0.75 and augment_rotate180:
            # 180-degree rotation (flip ranks and files)
            s = torch.flip(s, dims=[2, 3]).contiguous()
            pi_sh = torch.flip(pi.view(-1, *POLICY_SHAPE), dims=[1, 2]).contiguous()
            from azchess.encoding import build_rotate180_permutation
            perm = build_rotate180_permutation()
            perm_t = torch.as_tensor(perm, device=pi_sh.device, dtype=torch.long)
            pi = pi_sh.index_select(-1, perm_t).view(-1, np.prod(POLICY_SHAPE))
            if legal_mask_t is not None:
                lm_sh = torch.flip(legal_mask_t.view(-1, *POLICY_SHAPE), dims=[1, 2]).contiguous()
                legal_mask_t = lm_sh.index_select(-1, perm_t).view(-1, np.prod(POLICY_SHAPE)).to(dtype=torch.bool)

    # PERFORMANCE PROFILING: Start SSL target creation
    ssl_target_start = time.time()

    # Generate self-supervised learning targets (enhanced multi-task SSL)
    ssl_targets = None
    if enable_ssl and hasattr(model, 'create_ssl_targets'):
        # Replace signal.alarm with a monotonic time budget using a worker thread
        # This avoids signal usage and remains macOS/MPS friendly
        import threading
        from queue import Queue

        result_q: Queue = Queue(maxsize=1)

        def _ssl_worker():
            try:
                out = model.create_ssl_targets(s)
                try:
                    result_q.put(out, block=False)
                except Exception:
                    pass
            except Exception as e:
                try:
                    result_q.put(e, block=False)
                except Exception:
                    pass

        worker = threading.Thread(target=_ssl_worker, daemon=True)
        worker.start()
        # Configurable budget (seconds); keep conservative default
        ssl_time_budget_s = 10.0
        worker.join(timeout=ssl_time_budget_s)

        if worker.is_alive():
            logger.warning(f"SSL target creation exceeded {ssl_time_budget_s:.1f}s; skipping SSL for this batch")
            enable_ssl = False
            ssl_targets = None
        else:
            try:
                res = result_q.get_nowait()
                if isinstance(res, Exception):
                    logger.warning(f"SSL target creation failed: {res}; skipping SSL for this batch")
                    enable_ssl = False
                    ssl_targets = None
                else:
                    ssl_targets = res
            except Exception:
                # No result available; skip SSL for this batch
                logger.warning("SSL target creation returned no result; skipping SSL for this batch")
                enable_ssl = False
                ssl_targets = None

    # PERFORMANCE PROFILING: SSL target creation complete
    ssl_target_time = time.time() - ssl_target_start
    if current_step % 10 == 0:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"PERF: SSL target creation: {ssl_target_time:.3f}s")
        if ssl_target_time > 5.0:  # Log if it's taking too long
            logger.warning(f"SSL target creation is very slow: {ssl_target_time:.3f}s - this indicates a performance issue")

        # DEBUG: Log SSL targets statistics (only when available)
        if ssl_targets is not None and torch.rand(1).item() < 0.1:  # 10% chance to log
            try:
                if isinstance(ssl_targets, torch.Tensor):
                    logger.info(
                        f"SSL TARGETS DEBUG: shape={tuple(ssl_targets.shape)}, min={ssl_targets.min().item()}, max={ssl_targets.max().item()}, sum={ssl_targets.sum().item()}, all_zeros={torch.all(ssl_targets == 0).item()}"
                    )
                elif isinstance(ssl_targets, dict):
                    keys = ",".join(sorted(list(ssl_targets.keys())))
                    logger.info(f"SSL TARGETS DEBUG: dict keys={keys}")
            except Exception:
                pass

    if ssl_targets is None and enable_ssl:
        logger.warning(f"TRAINING: SSL targets not created - enable_ssl={enable_ssl}, has_create_ssl_targets={hasattr(model, 'create_ssl_targets')}")

    # Do not unconditionally cast SSL targets; handle per-target at loss time

    # PERFORMANCE PROFILING: Start forward pass
    forward_start = time.time()

    # Do not force model parameter dtypes; keep params/buffers in fp32 for stability on MPS

    # Prepare input dtype for forward to match parameter dtype on MPS
    s_forward = s
    if use_autocast and device_type == "mps" and s.dtype != _amp_dtype:
        s_forward = s.to(dtype=_amp_dtype)

    # CRITICAL: Enhanced autocast handling with proper type consistency
    with torch.autocast(device_type=device_type, dtype=_amp_dtype, enabled=use_autocast):
        if use_wdl and hasattr(model, 'forward_with_features'):
            p, v, ssl_out, feats = model.forward_with_features(s_forward, return_ssl=True)
        else:
            feats = None
            p, v, ssl_out = model(s_forward, return_ssl=True)

    # Do not restore dtypes on MPS; keeping consistent param dtype avoids mixed-type ops

    # PERFORMANCE PROFILING: Forward pass complete
    forward_time = time.time() - forward_start
    if True:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"PERF: Forward pass: {forward_time:.3f}s")
            logger.debug(f"DEBUG: Output dtypes - p: {p.dtype}, v: {v.dtype}, ssl_out: {ssl_out.dtype if ssl_out is not None else 'None'}")

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
        if ssl_out is not None:
            ssl_out = _ensure_contiguous(ssl_out, "ssl_out")

        # Validate model outputs to catch any contiguity issues
        _check_contiguous({"policy": p, "value": v, "ssl_out": ssl_out})

        # Optional legality masking: prefer explicit legal_mask; else fall back to pi>0 heuristic
        if legal_mask_t is not None:
            # Ensure mask has correct shape
            if legal_mask_t.dim() == 1:
                legal_mask_t = legal_mask_t.view(-1, p.shape[1])
            p_for_loss = torch.where(legal_mask_t, p, torch.full_like(p, -1e9))
        elif policy_masking:
            p_for_loss = apply_policy_mask(p, pi)
        else:
            p_for_loss = p

        # Diagnostics: log mask coverage and illegal probability mass periodically
        try:
            if legal_mask_t is not None and current_step % 200 == 0:
                with torch.no_grad():
                    legal_counts = legal_mask_t.sum(dim=1).float()
                    probs_unmasked = torch.softmax(p, dim=1)
                    legal_mass = (probs_unmasked * legal_mask_t.float()).sum(dim=1)
                    illegal_mass = (1.0 - legal_mass).mean().item()
                    logger.info(
                        "LEGAL_MASK: count mean=%.1f min=%d max=%d | illegal_prob_mass=%.4f",
                        float(legal_counts.mean().item()), int(legal_counts.min().item()), int(legal_counts.max().item()), illegal_mass
                    )
        except Exception:
            pass

        # DEBUG: Inspect policy targets to ensure they're non-empty distributions
        try:
            if current_step % 100 == 0:
                with torch.no_grad():
                    pi_row_sum = pi.sum(dim=1)
                    pi_pos_counts = (pi > 0).sum(dim=1)
                    logger.info(
                        "POLICY_DEBUG: row_sum mean=%.4f min=%.4f max=%.4f | pos_count mean=%.1f min=%d max=%d",
                        float(pi_row_sum.mean().item()),
                        float(pi_row_sum.min().item()),
                        float(pi_row_sum.max().item()),
                        float(pi_pos_counts.float().mean().item()),
                        int(pi_pos_counts.min().item()),
                        int(pi_pos_counts.max().item()),
                    )
        except Exception:
            pass

        # Policy loss with optional label smoothing
        if label_smoothing and label_smoothing > 0.0:
            num_actions = p.shape[1]
            smooth = label_smoothing / float(num_actions)
            pi_smooth = (1.0 - label_smoothing) * pi + smooth
            log_probs = nn.functional.log_softmax(p_for_loss, dim=1)
            policy_loss = -(pi_smooth * log_probs).sum(dim=1).mean()
        else:
            log_probs = nn.functional.log_softmax(p_for_loss, dim=1)
            policy_loss = -(pi * log_probs).sum(dim=1).mean()

        # DEBUG: Log unusually small policy loss values to catch empty targets early
        try:
            if current_step % 100 == 0 and float(policy_loss.detach().item()) < 1e-3:
                logger.warning("POLICY_DEBUG: very small policy_loss=%.6f; check policy targets above", float(policy_loss.detach().item()))
        except Exception:
            pass
        
        # Add policy regularization to prevent uniform outputs
        # This encourages the model to produce diverse, meaningful policies
        policy_probs = torch.softmax(p_for_loss, dim=1)
        uniform_probs = torch.ones_like(policy_probs) / policy_probs.shape[1]
        policy_entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum(dim=1).mean()
        max_entropy = torch.log(torch.tensor(policy_probs.shape[1], dtype=torch.float32, device=p.device))
        
        # Penalize if policy is too close to uniform (entropy too high)
        # But don't penalize if policy is too concentrated (entropy too low)
        entropy_penalty = torch.relu(policy_entropy - 0.8 * max_entropy)
        policy_reg_loss = 0.1 * entropy_penalty
        
        # PERFORMANCE PROFILING: Start loss computation
        loss_comp_start = time.time()

        # Value loss: ensure dtype alignment on MPS/AMP to avoid graph type errors
        if z.dtype != v.dtype:
            z = z.to(dtype=v.dtype)
        if value_loss_type == 'huber':
            value_loss = nn.functional.smooth_l1_loss(v, z, beta=huber_delta)
        else:
            value_loss = nn.functional.mse_loss(v, z)

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

            # Compute piece SSL loss directly from forward's ssl_out when using piece-only targets
            ssl_loss = torch.tensor(0.0, device=device, dtype=policy_loss.dtype)
            try:
                # If ssl_targets is a Tensor, assume piece-only targets
                if isinstance(ssl_targets, torch.Tensor) and ssl_out is not None:
                    logits = ssl_out.permute(0, 2, 3, 1).reshape(-1, ssl_out.size(1))
                    targets_flat = ssl_targets
                    if targets_flat.dim() == 4 and targets_flat.size(1) == ssl_out.size(1):
                        targets_flat = torch.argmax(targets_flat, dim=1)
                    elif targets_flat.dim() == 3:
                        pass
                    else:
                        # Already flat or unexpected; reshape later
                        pass
                    targets_flat = targets_flat.reshape(-1).long()
                    if targets_flat.numel() == logits.size(0):
                        ssl_loss = nn.functional.cross_entropy(logits, targets_flat, reduction='mean')
                else:
                    # Fallback to enhanced SSL (dict) path if present
                    if hasattr(model, 'get_enhanced_ssl_loss') and isinstance(ssl_targets, dict):
                        ssl_loss = model.get_enhanced_ssl_loss(s, ssl_targets)
            except Exception as e:
                logger.warning(f"SSL-from-forward computation failed: {e}; setting ssl_loss=0 for this batch")
                ssl_loss = torch.tensor(0.0, device=device, dtype=policy_loss.dtype)

        # PERFORMANCE PROFILING: Loss computation complete
        loss_comp_time = time.time() - loss_comp_start
        if current_step % 10 == 0:
            logger.info(f"PERF: Loss computation: {loss_comp_time:.3f}s")

        # CRITICAL FIX: Ensure all loss components have consistent dtypes BEFORE arithmetic
        # This prevents MPS type mismatch errors during loss combination
        # Ensure all tensors have the same dtype (use the model's dtype)
        target_dtype = policy_loss.dtype
        if value_loss.dtype != target_dtype:
            value_loss = value_loss.to(dtype=target_dtype)
        if ssl_loss.dtype != target_dtype:
            ssl_loss = ssl_loss.to(dtype=target_dtype)

        # Combine losses with consistent dtypes
        if enable_ssl and (isinstance(ssl_loss, torch.Tensor) and ssl_loss.detach().item() > 0):
            loss = policy_loss + policy_reg_loss + value_loss + (ssl_weight * ramp * ssl_target_weight) * ssl_loss
        else:
            loss = policy_loss + policy_reg_loss + value_loss

        # Optional WDL auxiliary head
        wdl_loss = torch.tensor(0.0, device=device, dtype=target_dtype)
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
                    # Ensure loss has consistent dtype before arithmetic operations
                    if wdl_loss.dtype != loss.dtype:
                        wdl_loss = wdl_loss.to(dtype=loss.dtype)
                    loss = loss + (float(wdl_weight) * wdl_loss)
            except Exception:
                pass
    
    # PERFORMANCE PROFILING: Start backward pass
    backward_start = time.time()

    # Guard against NaN/Inf loss; skip backward if not finite; also guard if loss wasn't set due to earlier skip
    if 'loss' in locals() and isinstance(loss, torch.Tensor) and torch.isfinite(loss):
        try:
            # Use scaler for mixed precision if available and valid
            if scaler is not None and use_autocast:
                try:
                    scaler.scale(loss / accum_steps).backward()
                except Exception as scaler_err:
                    logger.warning(f"Scaler error during backward pass: {scaler_err}, skipping this batch")
                    optimizer.zero_grad(set_to_none=True)
                    return None, None, None, None, None
            else:
                (loss / accum_steps).backward()

            # PERFORMANCE PROFILING: Backward pass complete
            backward_time = time.time() - backward_start
            if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"PERF: Backward pass: {backward_time:.3f}s")

            # PERFORMANCE PROFILING: Total step time
            total_step_time = time.time() - start_time
            if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"PERF: Total step time: {total_step_time:.3f}s")
                logger.debug(f"PERF: Breakdown - Data: {data_prep_time:.3f}s, SSL: {ssl_target_time:.3f}s, Forward: {forward_time:.3f}s, Loss: {loss_comp_time:.3f}s, Backward: {backward_time:.3f}s")
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
        logger.warning("Skipping backward for this batch (loss not set or non-finite)")
    
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
    warmup_steps: float = 500,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    device: str = "auto",
    use_amp: bool = True,
    augment: bool = True,
    precision: str = "fp16",
):
    """Train the model with comprehensive features and memory optimizations."""

    # Ensure torch is available (fix for import scope issues)
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Load configuration
    cfg = Config.load(config_path)
    device = select_device(device)
    
    # Memory cleanup at start of training
    logger.info("Performing memory cleanup at start of training")
    try:
        clear_memory_cache(device)
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
    
    # Enable memory optimizations with configurable gradient checkpointing
    if hasattr(model, 'enable_gradient_checkpointing'):
        # Get checkpointing strategy from config (default to adaptive)
        checkpoint_strategy = safe_config_get(cfg, 'gradient_checkpointing_strategy', 'adaptive', section='training')
        model.enable_gradient_checkpointing(strategy=checkpoint_strategy)
        logger.info(f"Gradient checkpointing enabled with strategy '{checkpoint_strategy}' for memory efficiency")
    
    # Log memory usage
    try:
        system_memory = get_memory_usage(device)
        logger.info(f"System memory usage: {system_memory['memory_gb']:.2f}GB on {device}")

        if hasattr(model, 'get_memory_usage'):
            model_memory_stats = model.get_memory_usage()
            logger.info(f"Model memory usage: {model_memory_stats}")
    except Exception as e:
        logger.debug(f"Could not log memory usage: {e}")
    
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
            missing = []
            unexpected = []
            if "model_ema" in state and state["model_ema"] is not None:
                logger.info("Loading EMA model state (non-strict)")
                r = model.load_state_dict(state["model_ema"], strict=False)
                missing, unexpected = list(r.missing_keys), list(r.unexpected_keys)
            elif "model" in state and state["model"] is not None:
                logger.info("Loading model state (non-strict)")
                r = model.load_state_dict(state["model"], strict=False)
                missing, unexpected = list(r.missing_keys), list(r.unexpected_keys)
            elif "model_state_dict" in state and state["model_state_dict"] is not None:
                logger.info("Loading model_state_dict (non-strict)")
                r = model.load_state_dict(state["model_state_dict"], strict=False)
                missing, unexpected = list(r.missing_keys), list(r.unexpected_keys)
            else:
                missing, unexpected = [], []

            if missing or unexpected:
                try:
                    logger.warning(f"Checkpoint key diff: missing={len(missing)} unexpected={len(unexpected)}")
                    if missing:
                        logger.warning(f"Missing (first 10): {missing[:10]}")
                    if unexpected:
                        logger.warning(f"Unexpected (first 10): {unexpected[:10]}")
                except Exception:
                    pass
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
    # Scheduler defined on UPDATE steps when using gradient accumulation
    num_updates = int(math.ceil(float(total_steps) / float(max(1, accum_steps))))
    warmup_updates = int(math.ceil(float(warmup_steps) / float(max(1, accum_steps))))
    scheduler = get_lr_scheduler(optimizer, num_updates, warmup_updates)
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
        try:
            if device_type == "cuda":
                scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0)
            else:
                scaler = torch.amp.GradScaler(device="mps", init_scale=65536.0, growth_factor=2.0)
        except Exception as scaler_init_error:
            logger.warning(f"Failed to create scaler: {scaler_init_error}, will use regular precision")
            scaler = None
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
    use_curriculum = safe_config_get(cfg, "use_curriculum", False, section='training')
    curriculum_phases = safe_config_get(cfg, "curriculum_phases", [], section='training')
    
    if use_curriculum and curriculum_phases:
        logger.info(f"Curriculum learning enabled with {len(curriculum_phases)} phases")
        current_phase = curriculum_phases[0]['name']  # Start with first phase
        logger.info(f"Starting with curriculum phase: {current_phase}")
    else:
        logger.info("Using standard training data mixing")
        current_phase = "mixed"
    
    # Use appropriate batch method based on configuration
    dataloader = None
    loader_iter = None
    if use_curriculum and curriculum_phases:
        # Curriculum learning - will be handled in training loop (phase-aware)
        batch_generator = None
        logger.info("DataLoader disabled for curriculum mode (phase-dependent).")
    else:
        # Standard training - prefer DataLoader-backed batches for throughput
        dl_workers = int(safe_config_get(cfg, 'dataloader_workers', 2, section='training'))
        dl_prefetch = int(safe_config_get(cfg, 'prefetch_factor', 2, section='training'))
        if external_stats['external_total'] > 0:
            logger.info("Using DataLoader over external+mixed batches")
            dataloader = build_training_dataloader(
                data_manager,
                batch_size=batch_size,
                device=device,
                mode='mixed',
                num_workers=dl_workers,
                prefetch_factor=dl_prefetch,
                persistent_workers=True,
            )
        else:
            logger.info("Using DataLoader over replay buffer shards")
            dataloader = build_training_dataloader(
                data_manager,
                batch_size=batch_size,
                device=device,
                mode='replay',
                num_workers=dl_workers,
                prefetch_factor=dl_prefetch,
                persistent_workers=True,
            )
        loader_iter = iter(dataloader) if dataloader is not None else None
        batch_generator = None
    
    logger.info(f"Training for {total_steps - start_step} steps.")
    logger.info(f"Batch size: {batch_size}, Gradient Accumulation: {accum_steps}")
    logger.info(f"Device: {device}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)

    # Get training config (needed for all device types)
    tr_cfg = cfg.training()  # Get training config

    # Memory limits are now set at the beginning of the orchestrator main() function
    memory_limit_gb = safe_config_get(cfg, 'memory_limit_gb', 12, section='training')  # Default 12GB if not specified
    if device.startswith('mps'):
        import os

        # Check current memory settings
        current_high_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.8')
        current_low_ratio = os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.6')

        logger.info(f"Current MPS memory settings - High: {current_high_ratio}, Low: {current_low_ratio}")
        logger.info(f"Target memory limit: {memory_limit_gb}GB (configured in config.yaml)")

        # Enable model memory optimizations
        if hasattr(model, 'enable_memory_optimization'):
            model.enable_memory_optimization()
            logger.info("Enabled model memory optimizations for MPS")

    # Initialize training heartbeat monitoring
    last_heartbeat = time.time()
    heartbeat_interval = 30.0  # Heartbeat every 30 seconds (like self-play but less frequent)

    # Initialize advanced memory monitoring system
    memory_warning_threshold = safe_config_get(cfg, 'memory_warning_threshold', 0.85, section='training')  # 85% default
    memory_critical_threshold = safe_config_get(cfg, 'memory_critical_threshold', 0.95, section='training')  # 95% default

    # Start comprehensive memory monitoring
    try:
        start_memory_monitoring(
            device=device,
            warning_threshold=memory_warning_threshold,
            critical_threshold=memory_critical_threshold,
            check_interval=30.0  # Check every 30 seconds
        )
        logger.info("Advanced memory monitoring system started")

        # Add custom alert callback for training-specific actions
        def training_memory_alert_callback(alert):
            if alert.alert_type == 'critical':
                logger.critical(f"CRITICAL MEMORY: Training may become unstable. Memory: {alert.memory_usage_gb:.2f}GB")
            elif alert.alert_type == 'warning':
                logger.warning(f"HIGH MEMORY: Monitor training stability. Memory: {alert.memory_usage_gb:.2f}GB")

        add_memory_alert_callback(training_memory_alert_callback)

    except Exception as e:
        logger.warning(f"Could not start advanced memory monitoring: {e}")
        # Fallback to basic monitoring
        last_memory_warning = 0
        memory_warning_cooldown = 300  # 5 minutes between warnings

    def get_system_memory_usage():
        """Get current system memory usage in GB for heartbeat monitoring."""
        try:
            # Use unified memory management
            usage = get_memory_usage(device)
            return usage.get('memory_gb', 0.0)
        except Exception as e:
            logger.debug(f"Could not get memory usage: {e}")
            return 0.0

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

    # Training heartbeat for monitoring
    last_heartbeat = time.time()
    heartbeat_interval = 60  # Log every 60 seconds

    accum_counter = 0  # micro-steps within an accumulation window
    updates_done = 0   # total optimizer updates completed

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
            
            # PERFORMANCE PROFILING: Start batch preparation
            batch_prep_start = time.time()

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
                    
                    # Convert dict format to tuple format for existing train_step, include legal_mask if present
                    if 'legal_mask' in batch_dict:
                        batch = (batch_dict['s'], batch_dict['pi'], batch_dict['z'], batch_dict['legal_mask'])
                    else:
                        batch = (batch_dict['s'], batch_dict['pi'], batch_dict['z'])
                    
                elif dataloader is not None:
                    # DataLoader-backed training path
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        # Recreate iterator (IterableDataset can be re-used)
                        loader_iter = iter(dataloader)
                        batch = next(loader_iter)
                else:
                    # Fallback: Standard replay buffer training via generator
                    batch = next(batch_generator)
                    
            except StopIteration:
                if dataloader is not None:
                    logger.info("DataLoader exhausted, recreating iterator")
                    loader_iter = iter(dataloader)
                    continue
                if batch_generator:
                    logger.info("Data stream exhausted, restarting batch generator.")
                    batch_generator = data_manager.get_training_batch(batch_size, device)
                    continue
                logger.error("Data stream exhausted, stopping training")
                break
            except Exception as e:
                logger.warning(f"Error getting batch: {e}, skipping...")
                continue

            # Error recovery wrapper for train_step
            try:
                # PERFORMANCE PROFILING: Batch preparation complete
                batch_prep_time = time.time() - batch_prep_start
                if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.info(f"PERF: Batch preparation: {batch_prep_time:.3f}s")

                # PERFORMANCE PROFILING: Start train_step
                train_step_start = time.time()

                # tr_cfg already assigned at function start
                loss_values = train_step(
                model, optimizer, scaler, batch, device, accum_steps, augment,
                augment_rotate180=safe_config_get(cfg, 'augment_rotate180', True, section='training'),
                ssl_weight=float(tr_cfg.get('ssl_weight', 0.1)), enable_ssl=bool(cfg.model().get('self_supervised', False)),
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
                
                # PERFORMANCE PROFILING: train_step complete
                train_step_time = time.time() - train_step_start
                if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.info(f"PERF: train_step call: {train_step_time:.3f}s")

                loss, policy_loss, value_loss, ssl_loss, wdl_loss = loss_values

                # PERFORMANCE PROFILING: Start post-processing
                post_proc_start = time.time()

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

                    # Handle scaler recovery - it may be corrupted after memory errors
                    if scaler:
                        try:
                            scaler.update()
                        except Exception as scaler_error:
                            logger.warning(f"Scaler corrupted after memory error, creating new one: {scaler_error}")
                            # Recreate the scaler with proper initialization
                            if device.startswith('mps'):
                                scaler = torch.amp.GradScaler(device="mps", init_scale=65536.0, growth_factor=2.0)
                            else:
                                scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0)

                            # Force initialization by calling step() and update() with a dummy operation
                            try:
                                # Create a dummy parameter to force scaler initialization
                                dummy_param = torch.nn.Parameter(torch.tensor([1.0], device=device))
                                dummy_optimizer = torch.optim.SGD([dummy_param], lr=0.01)

                                # This will properly initialize the scaler
                                with torch.no_grad():
                                    dummy_loss = torch.tensor(1.0, device=device)
                                    dummy_loss.backward()
                                    scaler.step(dummy_optimizer)
                                    scaler.update()

                                # Clean up
                                dummy_param.grad = None
                                logger.info("Scaler successfully reinitialized")
                            except Exception as init_error:
                                logger.warning(f"Scaler initialization failed: {init_error}, will use regular backward pass")
                                scaler = None  # Disable scaler if we can't initialize it

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
                # PERFORMANCE PROFILING: Start optimizer step
                optimizer_start = time.time()

                try:
                    # Check if scaler is still valid before using it
                    scaler_valid = (scaler is not None and
                                   hasattr(scaler, '_scale') and
                                   scaler._scale is not None and
                                   scaler._scale > 0)

                    # Gradient accumulation: only update optimizer every accum_steps
                    accum_counter += 1
                    do_update = (accum_counter % max(1, accum_steps) == 0)

                    if do_update:
                        if scaler_valid:
                            scaler.unscale_(optimizer)
                        if grad_clip_norm > 0:
                            # CRITICAL: Ultra-aggressive gradient clipping to prevent NaN/Inf
                            total_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=False)
                            if torch.isnan(total_norm) or torch.isinf(total_norm):
                                logger.warning(f"Gradient norm is NaN/Inf: {total_norm}, applying emergency clipping")
                                # Emergency gradient clipping with very small norm
                                nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=False)
                                total_norm = 0.01

                        # CRITICAL: Do optimizer step and scheduler step
                        if scaler_valid:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                            # Recreate scaler if it was corrupted
                            if device.startswith('mps'):
                                scaler = torch.amp.GradScaler(device="mps")
                            else:
                                scaler = torch.cuda.amp.GradScaler()

                        # CRITICAL: Step scheduler AFTER optimizer to avoid LR warning
                        scheduler.step()
                        optimizer.zero_grad()
                        ema.update(model)
                        updates_done += 1

                    # PERFORMANCE PROFILING: Optimizer step complete
                    optimizer_time = time.time() - optimizer_start
                    if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"PERF: Optimizer step: {optimizer_time:.3f}s")

                    # Update watchdog timer
                    last_progress_time = time.time()

                    # PERFORMANCE PROFILING: Post-processing complete
                    post_proc_time = time.time() - post_proc_start
                    if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"PERF: Post-processing: {post_proc_time:.3f}s")

                    # PERFORMANCE PROFILING: Total iteration time
                    total_iter_time = time.time() - batch_prep_start
                    if current_step % 10 == 0 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"PERF: Total iteration: {total_iter_time:.3f}s")
                        logger.debug(f"PERF: Full breakdown - Batch: {batch_prep_time:.3f}s, TrainStep: {train_step_time:.3f}s, Post: {post_proc_time:.3f}s, Optimizer: {optimizer_time:.3f}s")

                    # Training heartbeat monitoring (similar to self-play workers)
                    current_time = time.time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        memory_usage = get_system_memory_usage()
                        lr_current = scheduler.get_last_lr()[0] if scheduler else 0.0

                        # Get detailed memory info for MPS
                        memory_info = ""
                        if device.startswith('mps'):
                            try:
                                import torch.mps
                                memory_info = " | MPS Available" if torch.mps.is_available() else " | MPS Unavailable"
                            except:
                                pass

                        # Memory monitoring and alerting (threshold-driven cache clears only)
                        memory_usage_gb = get_system_memory_usage()
                        if memory_limit_gb > 0 and memory_usage_gb > 0:
                            memory_ratio = memory_usage_gb / memory_limit_gb
                            current_time = time.time()

                            if memory_ratio >= memory_critical_threshold and (current_time - last_memory_warning) > memory_warning_cooldown:
                                logger.warning(f"CRITICAL MEMORY USAGE: {memory_usage_gb:.2f}GB / {memory_limit_gb:.2f}GB ({memory_ratio:.1%})")
                                logger.warning("Consider reducing batch size or enabling gradient checkpointing")
                                emergency_memory_cleanup(device)
                                last_memory_warning = current_time

                            elif memory_ratio >= memory_warning_threshold and (current_time - last_memory_warning) > memory_warning_cooldown:
                                logger.warning(f"HIGH MEMORY USAGE: {memory_usage_gb:.2f}GB / {memory_limit_gb:.2f}GB ({memory_ratio:.1%})")
                                # Threshold-based cleanup only
                                clear_memory_cache(device)
                                last_memory_warning = current_time

                        logger.info(f"TRAINING_HB: Step {current_step}/{total_steps} (updates={updates_done}) | "
                                  f"Loss: {running_loss:.4f} | "
                                  f"Policy: {running_policy_loss:.4f} | "
                                  f"Value: {running_value_loss:.4f} | "
                                  f"SSL: {running_ssl_loss:.4f} | "
                                  f"LR: {lr_current:.6f} | "
                                  f"Memory: {memory_usage}GB{memory_info} | "
                                  f"Device: {device}")

                        last_heartbeat = current_time

                        # Skip unconditional cleanup; rely on threshold-driven cleanup above

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
                try:
                    system_memory = get_memory_usage(device)
                    memory_usage = f"{system_memory['memory_gb']:.2f}GB"
                except Exception:
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
        interrupted = True
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        pbar.close()

        # Save final checkpoint with enhanced prefix, but handle KeyboardInterrupt gracefully
        if not locals().get('interrupted', False):
            try:
                checkpoint_prefix = cfg.get("checkpoint_prefix", "enhanced")
                final_checkpoint = Path(checkpoint_dir) / f"{checkpoint_prefix}_final.pt"
                save_checkpoint(model, ema, optimizer, scheduler, current_step, final_checkpoint)
                logger.info(f"Saved final checkpoint: {final_checkpoint}")
            except KeyboardInterrupt:
                logger.info("Checkpoint saving interrupted by user - training session complete")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save final checkpoint: {checkpoint_error}")
        else:
            logger.info("Skipping final checkpoint save due to user interruption")
        
        # Save enhanced checkpoint (don't overwrite baseline best.pt)
        if not locals().get('interrupted', False):
            try:
                enhanced_checkpoint = Path(checkpoint_dir) / f"{checkpoint_prefix}_best.pt"
                save_checkpoint(model, ema, optimizer, scheduler, current_step, enhanced_checkpoint)
                logger.info(f"Saved enhanced checkpoint: {enhanced_checkpoint}")
            except KeyboardInterrupt:
                logger.info("Enhanced checkpoint saving interrupted by user")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save enhanced checkpoint: {checkpoint_error}")
        else:
            logger.info("Skipping enhanced checkpoint save due to user interruption")
        
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
        # Handle optional EMA safely
        ema_shadow = None
        try:
            if ema is not None and getattr(ema, 'shadow', None) is not None:
                ema_shadow = ema.shadow
        except Exception:
            ema_shadow = None
        state = {
            'step': step,
            'global_step': step,  # Alternative key for compatibility
            'model': model.state_dict(),
            'model_state_dict': model.state_dict(),  # Alternative key for compatibility
            'model_ema': ema_shadow,
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
        precision="fp16",
        device=args.device,
        use_amp=args.use_amp,
        augment=args.augment
    )

def train_from_config(config_path: str = "config.yaml"):
    """Training function that works with orchestrator - uses config file only."""
    cfg = Config.load(config_path)
    
    # Extract training parameters from config
    train_cfg = cfg.training()
    
    # Support both 'lr' and 'learning_rate' in config for compatibility
    lr_cfg = train_cfg.get("lr", train_cfg.get("learning_rate", 0.001))
    train_comprehensive(
        config_path=config_path,
        total_steps=train_cfg.get("steps_per_epoch", 10000),
        batch_size=train_cfg.get("batch_size", 256),
        learning_rate=lr_cfg,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        ema_decay=train_cfg.get("ema_decay", 0.999),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        accum_steps=train_cfg.get("accum_steps", 1),
        precision=train_cfg.get("precision", "fp16"),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
        log_dir=train_cfg.get("log_dir", "logs"),
        device=cfg.get("device", "auto"),
        use_amp=train_cfg.get("use_amp", True),
        augment=True
    )

if __name__ == "__main__":
    # Allows running with: python -m azchess.training.train
    main()
