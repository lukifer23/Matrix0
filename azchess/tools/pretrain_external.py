#!/usr/bin/env python3
"""
Pretrain the Matrix0 model on external curated data (tactical + openings),
evaluate against the untrained baseline, and optionally produce a checkpoint
that can be used as the orchestrator's starting point.

Usage (example):
  python -m azchess.tools.pretrain_external \
    --config config.yaml \
    --steps 2000 \
    --batch-size 128 \
    --device auto \
    --checkpoint-in checkpoints/v2_base.pt \
    --checkpoint-out checkpoints/pretrained_external.pt

This script does NOT run automatically; the orchestrator can be pointed at the
produced checkpoint via config or CLI once validated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim

from azchess.config import Config, select_device
from azchess.model import PolicyValueNet
from azchess.data_manager import DataManager
from azchess.training.train import get_lr_scheduler, save_checkpoint
from azchess.utils import setup_logging, clear_memory_cache, get_memory_usage

logger = setup_logging(level=logging.INFO)


def evaluate_quick(model: PolicyValueNet, dm: DataManager, device: str, batches: int = 10, batch_size: int = 128, source: str = "mixed") -> dict:
    """Quick metric proxy evaluation on external data (policy/value losses)."""
    model.eval()
    total_pol, total_val, n = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(batches):
            batch = dm.get_external_training_batch(batch_size, source=source)
            if not batch:
                break
            s = torch.from_numpy(batch['s']).to(device)
            pi = torch.from_numpy(batch['pi']).to(device)
            z = torch.from_numpy(batch['z']).to(device)
            if z.dim() == 2 and z.size(1) == 1:
                z = z.reshape(z.size(0))
            p, v = model(s)[0:2]
            log_probs = nn.functional.log_softmax(p, dim=1)
            pol = (-(pi * log_probs).sum(dim=1).mean()).item()
            if v.dtype != z.dtype:
                z = z.to(v.dtype)
            val = nn.functional.mse_loss(v, z).item()
            total_pol += pol
            total_val += val
            n += 1
    return {"policy_loss": (total_pol / max(1, n)), "value_loss": (total_val / max(1, n)), "batches": n}


def main():
    ap = argparse.ArgumentParser(description="Matrix0 External Pretraining")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--checkpoint-in", type=str, default="checkpoints/v2_base.pt")
    ap.add_argument("--checkpoint-out", type=str, default="checkpoints/pretrain_external.pt")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--source", type=str, default="mixed", choices=["mixed", "tactical", "openings"], help="External data source to use")
    ap.add_argument("--save-every", type=int, default=500, help="Save intermediate checkpoints every N steps (0 to disable)")
    ap.add_argument("--checkpoint-prefix", type=str, default="pretrain_external", help="Prefix for saved checkpoints")
    ap.add_argument("--resume-from", type=str, default="", help="Resume from a specific checkpoint path")
    ap.add_argument("--auto-resume", action="store_true", help="Auto resume from latest step checkpoint in output dir")
    ap.add_argument("--use-amp", action="store_true", help="Enable autocast mixed precision during forward")
    ap.add_argument("--progress-interval", type=int, default=20, help="Update progress/memory every N steps")
    ap.add_argument("--clip-every", type=int, default=5, help="Apply gradient clipping every N steps (1 = every step)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = select_device(args.device)
    dm = DataManager(base_dir=cfg.get("data_dir", "data"))

    # Verify curated external data availability
    ext_stats = dm.get_external_data_stats()
    logger.info(f"External data stats: tactical={ext_stats['tactical_samples']} openings={ext_stats['openings_samples']} total={ext_stats['external_total']}")
    if ext_stats['external_total'] == 0:
        logger.warning("No external curated data found. Ensure tactical/openings NPZ files exist under data/training/.")

    # Model
    model = PolicyValueNet.from_config(cfg.model()).to(device)
    ckpt_in = Path(args.checkpoint_in)
    if ckpt_in.exists():
        logger.info(f"Loading baseline checkpoint: {ckpt_in}")
        try:
            state = torch.load(ckpt_in, map_location=device, weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            elif "model" in state:
                model.load_state_dict(state["model"], strict=False)
        except Exception as e:
            logger.warning(f"Failed to load {ckpt_in}: {e}")

    # Quick baseline eval
    base_metrics = evaluate_quick(model, dm, device, batches=args.eval_batches, batch_size=args.batch_size, source=args.source)
    logger.info(f"Baseline (external) metrics: {base_metrics}")

    # Optimizer / scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.steps, warmup_steps=min(200, args.steps // 10))

    # Resume support
    start_step = 0
    resume_path: Path | None = None
    if args.resume_from:
        rp = Path(args.resume_from)
        if rp.exists():
            resume_path = rp
        else:
            logger.warning(f"--resume-from path not found: {rp}")
    elif args.auto_resume:
        out_dir = Path(args.checkpoint_out).parent
        pattern = re.compile(rf"^{re.escape(args.checkpoint_prefix)}_step_(\d+)\\.pt$")
        latest_step = -1
        latest_path = None
        for p in out_dir.glob(f"{args.checkpoint_prefix}_step_*.pt"):
            m = pattern.match(p.name)
            if m:
                st = int(m.group(1))
                if st > latest_step:
                    latest_step = st
                    latest_path = p
        if latest_path is not None:
            resume_path = latest_path
            logger.info(f"Auto-resume: found latest checkpoint {latest_path}")

    if resume_path is not None:
        try:
            state = torch.load(resume_path, map_location=device, weights_only=False)
            # Load model
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            elif "model" in state:
                model.load_state_dict(state["model"], strict=False)
            # Load optimizer/scheduler if present
            if "optimizer" in state and state["optimizer"] is not None:
                try:
                    optimizer.load_state_dict(state["optimizer"])
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
            if "scheduler" in state and state["scheduler"] is not None:
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
            start_step = int(state.get("step", state.get("global_step", 0)))
            logger.info(f"Resuming from step {start_step} ({resume_path})")
        except Exception as e:
            logger.warning(f"Failed to resume from {resume_path}: {e}")

    # Train only on external data (mixed tactics + openings)
    model.train()
    start = time.time()
    from tqdm import tqdm
    pbar = tqdm(total=args.steps, desc="Pretraining (external)", unit="step", initial=start_step)
    running_pol, running_val = 0.0, 0.0
    start_global = time.time()
    for step in range(start_step, args.steps):
        batch = dm.get_external_training_batch(args.batch_size, source=args.source)
        if not batch:
            logger.warning("No external batch available; stopping early")
            break

        s = torch.from_numpy(batch['s']).to(device)
        pi = torch.from_numpy(batch['pi']).to(device)
        z = torch.from_numpy(batch['z']).to(device)
        if z.dim() == 2 and z.size(1) == 1:
            z = z.reshape(z.size(0))

        optimizer.zero_grad(set_to_none=True)

        # Forward (optionally with AMP)
        device_type = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")
        if args.use_amp and device_type in ("cuda", "mps"):
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=True):
                p, v = model(s)[0:2]
        else:
            p, v = model(s)[0:2]

        log_probs = nn.functional.log_softmax(p, dim=1)
        policy_loss = -(pi * log_probs).sum(dim=1).mean()
        if v.dtype != z.dtype:
            z = z.to(v.dtype)
        value_loss = nn.functional.mse_loss(v, z)

        loss = policy_loss + value_loss
        loss.backward()
        # Clip gradients at a lower frequency to reduce overhead
        if args.clip_every > 0 and (step + 1) % args.clip_every == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        optimizer.step()
        scheduler.step()

        # Update progress and running metrics
        running_pol = 0.98 * running_pol + 0.02 * float(policy_loss.item())
        running_val = 0.98 * running_val + 0.02 * float(value_loss.item())
        if (step + 1) % max(1, args.progress_interval) == 0:
            mem = get_memory_usage(device).get("memory_gb", 0.0)
            eta_s = (args.steps - (step + 1)) * max(1e-6, (time.time() - start_global) / (step + 1))
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pol": f"{policy_loss.item():.4f}",
                "val": f"{value_loss.item():.4f}",
                "r_pol": f"{running_pol:.4f}",
                "r_val": f"{running_val:.4f}",
                "mem": f"{mem:.2f}GB",
                "ETA": f"{int(eta_s)}s"
            })
        pbar.update(1)

        # Optional periodic save
        if args.save_every and (step + 1) % args.save_every == 0:
            tag = f"{args.checkpoint_prefix}_step_{step+1}.pt"
            save_checkpoint(model, None, optimizer, scheduler, step + 1, Path(args.checkpoint_out).parent / tag)
            logger.info(f"Saved intermediate checkpoint: {tag}")

    # Save checkpoint
    pbar.close()
    out_path = Path(args.checkpoint_out)
    save_checkpoint(model, None, optimizer, scheduler, step+1 if 'step' in locals() else 0, out_path)
    logger.info(f"Saved pretrained checkpoint: {out_path}")

    # Quick post-train eval
    post_metrics = evaluate_quick(model, dm, device, batches=args.eval_batches, batch_size=args.batch_size, source=args.source)
    logger.info(f"Post-train (external) metrics: {post_metrics}")

    # Optional: compare via enhanced_eval tool if desired
    logger.info("Compare models with:\n  python -m azchess.tools.enhanced_eval --baseline checkpoints/v2_base.pt --candidate checkpoints/pretrain_external.pt")


if __name__ == "__main__":
    main()


