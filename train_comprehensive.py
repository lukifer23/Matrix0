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
               ssl_weight: float = 0.1, enable_ssl: bool = True):
    """Single training step with augmentation, mixed precision, and self-supervised learning."""
    model.train()
    s, pi, z = batch
    s = torch.from_numpy(s).to(device)
    pi = torch.from_numpy(pi).to(device)
    z = torch.from_numpy(z).to(device)

    # Data Augmentation: Random horizontal flip
    if augment and torch.rand(1).item() > 0.5:
        s = torch.flip(s, dims=[3])  # Flip along the width dimension (files)
        # Use reshape instead of view to handle potential non-contiguous tensors after flip
        pi = torch.flip(pi.reshape(-1, *POLICY_SHAPE), dims=[2]).reshape(-1, np.prod(POLICY_SHAPE))

    # Generate self-supervised learning targets (piece presence mask)
    ssl_target = None
    if enable_ssl and hasattr(model, 'create_ssl_targets'):
        ssl_target = model.create_ssl_targets(s)

    # Ensure input data matches model precision
    if s.dtype != next(model.parameters()).dtype:
        s = s.to(next(model.parameters()).dtype)
        pi = pi.to(next(model.parameters()).dtype)
        z = z.to(next(model.parameters()).dtype)
        if ssl_target is not None:
            ssl_target = ssl_target.to(torch.long) # Target for CrossEntropyLoss should be long

    # Mixed Precision Forward Pass
    device_type = device.split(':')[0]
    use_autocast = (scaler is not None) or (device_type == 'mps')
    with torch.autocast(device_type=device_type, enabled=use_autocast):
        p, v, ssl_out = model(s, return_ssl=True)
        log_probs = nn.functional.log_softmax(p, dim=1)
        policy_loss = -(pi * log_probs).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(v, z)
        
        ssl_loss = 0.0
        if enable_ssl and ssl_target is not None and ssl_out is not None:
            ssl_loss = nn.functional.cross_entropy(ssl_out, ssl_target)
            loss = policy_loss + value_loss + ssl_weight * ssl_loss
        else:
            loss = policy_loss + value_loss
    
    # Scale loss and backward pass
    if scaler is not None:
        scaler.scale(loss / accum_steps).backward()
    else:
        (loss / accum_steps).backward()
    
    # Optimizer stepping and clipping are handled by the caller to support grad accumulation
    return loss.item(), policy_loss.item(), value_loss.item(), ssl_loss if ssl_target is not None else 0.0

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
    total_steps: int = 6500,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    ema_decay: float = 0.999,
    grad_clip_norm: float = 1.0,
    accum_steps: int = 1,
    warmup_steps: int = 500,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    device: str = "auto",
    use_amp: bool = True,
    augment: bool = True
):
    """Comprehensive training with progress tracking and ETA."""
    
    # Load configuration
    cfg = Config.load(config_path)
    device = select_device(device)
    
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
    
    # Use AMP + channels_last for MPS/CUDA; avoid hard quantization of BatchNorm layers
    if device.startswith('mps'):
        logger.info("Using MPS with autocast; not forcing FP16 parameters to keep BatchNorm stable")
    
    # Load best checkpoint if available
    best_ckpt = Path(checkpoint_dir) / "best.pt"
    start_step = 0
    state = None
    if best_ckpt.exists():
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        state = torch.load(best_ckpt, map_location=device)
        if "model_ema" in state:
            model.load_state_dict(state["model_ema"])
        else:
            model.load_state_dict(state["model"])
        start_step = state.get("step", 0)
        logger.info(f"Resuming from step {start_step}")
    else:
        logger.info("Starting training from scratch")
    
    # Initialize optimizer, scheduler, and EMA
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps)
    ema = EMA(model, ema_decay)
    
    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.startswith("cuda") else None
    if scaler:
        logger.info("Using Automatic Mixed Precision (AMP).")

    # Load optimizer and scheduler state if resuming
    if state is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        logger.info("Loaded optimizer state")
    if state is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
        logger.info("Loaded scheduler state")
    
    # Apply warmup learning rate if starting from scratch
    if start_step < warmup_steps:
        warmup_factor = float(start_step) / float(max(1, warmup_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * warmup_factor
        logger.info(f"Applied warmup LR: {learning_rate * warmup_factor:.6f}")

    # Setup data using DataManager
    data_manager = DataManager(base_dir=cfg.get("data_dir", "data"))
    data_stats = data_manager.get_stats()
    if data_stats.total_samples == 0:
        raise ValueError("No training data found by DataManager!")
    logger.info(f"DataManager found {data_stats.total_shards} shards with {data_stats.total_samples} total samples.")
    
    batch_generator = data_manager.get_training_batch(batch_size, device)
    
    logger.info(f"Training for {total_steps - start_step} steps.")
    logger.info(f"Batch size: {batch_size}, Gradient Accumulation: {accum_steps}")
    logger.info(f"Device: {device}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    current_step = start_step
    
    pbar = tqdm(total=total_steps, desc="Training Progress", unit="step", initial=start_step)
    
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_ssl_loss = 0.0
    
    start_time = time.time()
    optimizer.zero_grad()

    try:
        while current_step < total_steps:
            try:
                batch = next(batch_generator)
            except StopIteration:
                logger.info("Data stream exhausted, restarting batch generator.")
                batch_generator = data_manager.get_training_batch(batch_size, device)
                continue
            except Exception as e:
                logger.warning(f"Error getting batch: {e}, skipping...")
                continue

            loss, policy_loss, value_loss, ssl_loss = train_step(
                model, optimizer, scaler, batch, device, accum_steps, augment, 
                ssl_weight=0.1, enable_ssl=True
            )
            
            running_loss = 0.98 * running_loss + 0.02 * loss
            running_policy_loss = 0.98 * running_policy_loss + 0.02 * policy_loss
            running_value_loss = 0.98 * running_value_loss + 0.02 * value_loss
            running_ssl_loss = 0.98 * running_ssl_loss + 0.02 * ssl_loss
            
            if (current_step + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                ema.update(model)
            
            pbar.update(1)
            current_step += 1
            
            if current_step % 10 == 0:
                writer.add_scalar('Loss/total', running_loss, current_step)
                writer.add_scalar('Loss/policy', running_policy_loss, current_step)
                writer.add_scalar('Loss/value', running_value_loss, current_step)
                writer.add_scalar('Loss/ssl', running_ssl_loss, current_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], current_step)
            
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
    """Save a training checkpoint."""
    state = {
        'step': step,
        'model': model.state_dict(),
        'model_ema': ema.shadow,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    torch.save(state, path)

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
        use_amp=cfg.get("precision", "fp16") != "fp32",
        augment=True
    )

if __name__ == "__main__":
    main()
