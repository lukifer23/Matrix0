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
import logging
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from azchess.config import Config, select_device
from azchess.data_manager import DataManager
from azchess.model import PolicyValueNet
from azchess.training.train import get_lr_scheduler, save_checkpoint, train_step as core_train_step, EMA
from azchess.utils import clear_memory_cache, get_memory_usage, setup_logging

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


def _scan_stockfish_sources(dm: DataManager) -> dict:
    """Scan DB for Stockfish-tagged shards and group by domain/subcategory prefixes.

    Returns a dict mapping prefix -> count of shards.
    """
    sources: dict[str, int] = {}
    try:
        for shard in dm._get_all_shards():  # internal enumeration ok within tool
            src = (shard.source or "")
            if not src.startswith("stockfish:"):
                continue
            # Normalize to prefix 'stockfish:domain/subcategory'
            sources[src] = sources.get(src, 0) + shard.sample_count
    except Exception:
        pass
    return sources


def _build_stockfish_phases(dm: DataManager, total_steps: int) -> list[tuple[str, list[str], int]]:
    """Create phase plan: (name, prefixes, steps). Skips phases with no data."""
    src_counts = _scan_stockfish_sources(dm)

    def has(pref: str) -> bool:
        return any(k.startswith(pref) for k in src_counts.keys())

    plan: list[tuple[str, list[str], float]] = []
    # Desired weights
    desired = [
        ("openings", ["stockfish:openings/"], 0.22),
        ("king_safety", ["stockfish:king_safety/"], 0.12),
        ("weakness_backrank", ["stockfish:weaknesses/back_rank_weakness"], 0.12),
        ("positional", ["stockfish:positional/"], 0.12),
        ("endgames", ["stockfish:endgames/"], 0.22),
        ("tactical", ["stockfish:tactical/"], 0.20),
    ]
    # Filter available and normalize weights
    available: list[tuple[str, list[str], float]] = []
    total_w = 0.0
    for name, prefs, w in desired:
        if any(has(p) for p in prefs):
            available.append((name, prefs, w))
            total_w += w
    if not available:
        # Fallback to all stockfish if unknown
        available = [("all", ["stockfish:"], 1.0)]
        total_w = 1.0
    # Allocate integer steps respecting total_steps
    phases: list[tuple[str, list[str], int]] = []
    remaining = total_steps
    for i, (name, prefs, w) in enumerate(available):
        steps_i = int(round((w / total_w) * total_steps)) if i < len(available) - 1 else remaining
        steps_i = max(0, steps_i)
        phases.append((name, prefs, steps_i))
        remaining -= steps_i
    # Fix rounding drift
    if phases and remaining != 0:
        name, prefs, steps0 = phases[-1]
        phases[-1] = (name, prefs, max(0, steps0 + remaining))
    # Remove 0-step phases
    phases = [(n, p, s) for (n, p, s) in phases if s > 0]
    return phases


def _evaluate_quick_stockfish(model: PolicyValueNet, dm: DataManager, device: str, batches: int = 5, batch_size: int = 128) -> dict:
    """Lightweight proxy eval over mixed Stockfish prefixes (opens+tactics+endgames)."""
    model.eval()
    total_pol, total_val, n = 0.0, 0.0, 0
    prefixes = [
        "stockfish:openings/",
        "stockfish:tactical/",
        "stockfish:endgames/",
        "stockfish:king_safety/",
        "stockfish:weaknesses/",
        "stockfish:positional/",
    ]
    try:
        batch_iter = dm.get_training_batch_by_source_prefixes(batch_size, prefixes)
    except Exception:
        return {"policy_loss": 0.0, "value_loss": 0.0, "batches": 0}

    with torch.no_grad():
        for _ in range(batches):
            try:
                s_np, pi_np, z_np, _lm = next(batch_iter)
            except StopIteration:
                break
            s = torch.from_numpy(s_np).to(device)
            pi = torch.from_numpy(pi_np).to(device)
            z = torch.from_numpy(z_np).to(device)
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
    ap.add_argument("--source", type=str, default="mixed", choices=["mixed", "tactical", "openings"], help="Deprecated: legacy external data selector")
    ap.add_argument("--stockfish-root", type=str, default="data/stockfish_games", help="Root directory of Stockfish-generated datasets")
    ap.add_argument("--curriculum", type=str, default="stockfish", choices=["stockfish", "legacy"], help="Training data curriculum: stockfish (from stockfish_root) or legacy (tactical/openings)")
    ap.add_argument("--phase-workers", type=int, default=0, help="Internal dataloader workers for stockfish sampling (0 = none)")
    ap.add_argument("--phase-prefetch", type=int, default=2, help="Prefetch factor when using workers > 0")
    ap.add_argument("--save-every", type=int, default=500, help="Save intermediate checkpoints every N steps (0 to disable)")
    ap.add_argument("--checkpoint-prefix", type=str, default="pretrain_external", help="Prefix for saved checkpoints")
    ap.add_argument("--resume-from", type=str, default="", help="Resume from a specific checkpoint path")
    ap.add_argument("--auto-resume", action="store_true", help="Auto resume from latest step checkpoint in output dir")
    ap.add_argument("--use-amp", action="store_true", help="Enable autocast mixed precision during forward")
    ap.add_argument("--progress-interval", type=int, default=20, help="Update progress/memory every N steps")
    ap.add_argument("--clip-every", type=int, default=5, help="Apply gradient clipping every N steps (1 = every step)")
    # Fine-grained loss/SSL controls
    ap.add_argument("--ssl-weight", type=float, default=0.2, help="Weight for SSL loss component")
    ap.add_argument("--ssl-warmup-steps", type=int, default=1000, help="Linear SSL weight warmup steps")
    ap.add_argument("--label-smoothing", type=float, default=0.03, help="Policy label smoothing for external data")
    ap.add_argument("--value-loss", type=str, choices=["mse", "huber"], default="huber", help="Value loss type")
    ap.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta (beta)")
    ap.add_argument("--lr-warmup-steps", type=int, default=400, help="Learning rate warmup steps for scheduler")
    # EMA controls
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for shadow weights (higher = smoother)")
    ap.add_argument("--no-ema", action="store_true", help="Disable EMA maintenance and saving")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = select_device(args.device)
    dm = DataManager(base_dir=cfg.get("data_dir", "data"))

    # Always attempt to import stockfish tree so shards are registered with source tags
    try:
        imported = dm.import_stockfish_tree(args.stockfish_root, move_files=False)
        if imported:
            logger.info(f"Imported {imported} Stockfish shards from {args.stockfish_root}")
    except Exception as e:
        logger.warning(f"Stockfish import skipped/failed: {e}")

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

    # Quick baseline eval (legacy). If curriculum is stockfish, do a light pass via stockfish prefixes
    if args.curriculum == "legacy":
        base_metrics = evaluate_quick(model, dm, device, batches=args.eval_batches, batch_size=args.batch_size, source=args.source)
        logger.info(f"Baseline (external, legacy) metrics: {base_metrics}")
    else:
        try:
            base_metrics = _evaluate_quick_stockfish(model, dm, device, batches=min(10, args.eval_batches), batch_size=args.batch_size)
            logger.info(f"Baseline (stockfish) metrics: {base_metrics}")
        except Exception as e:
            logger.warning(f"Baseline stockfish eval failed: {e}")

    # Optimizer / scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Allow CLI override for LR warmup steps
    scheduler = get_lr_scheduler(optimizer, args.steps, warmup_steps=int(args.lr_warmup_steps))
    # Initialize EMA (optional)
    ema = None if args.no_ema else EMA(model, decay=float(args.ema_decay))

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

    # Train on stockfish curriculum or legacy external data
    model.train()
    start = time.time()
    from tqdm import tqdm
    pbar = tqdm(total=args.steps, desc="Pretraining (external)", unit="step", initial=start_step)
    running_pol, running_val = 0.0, 0.0
    start_global = time.time()

    # Mixed precision scaler (CUDA only; disable on MPS/CPU)
    scaler = None
    try:
        if args.use_amp and device.startswith("cuda"):
            scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)
    except Exception as e:
        logger.warning(f"AMP scaler unavailable: {e}")
        scaler = None

    def _phase_iter(prefixes):
        while True:
            try:
                it = dm.get_training_batch_by_source_prefixes(args.batch_size, prefixes)
                for b in it:
                    yield b
            except RuntimeError:
                # No data; stop
                return

    if args.curriculum == "stockfish":
        phases = _build_stockfish_phases(dm, total_steps=args.steps)
        logger.info(f"Stockfish curriculum phases: {[(nm, st) for nm, _, st in phases]}")
        current_step = start_step
        for phase_idx, (phase_name, prefixes, phase_steps) in enumerate(phases):
            if phase_steps <= 0:
                continue
            logger.info(f"Phase {phase_idx+1}/{len(phases)}: {phase_name} for {phase_steps} steps")
            it = _phase_iter(prefixes)
            for _ in range(phase_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = _phase_iter(prefixes)
                    try:
                        batch = next(it)
                    except StopIteration:
                        logger.warning(f"No data for phase {phase_name}, skipping remaining steps")
                        break

                # Train step using core_train_step (includes SSL, masking, precision guards)
                loss_values = core_train_step(
                    model, optimizer, scaler, batch, device,
                    accum_steps=1, augment=True,
                    ssl_weight=float(args.ssl_weight), enable_ssl=True,
                    label_smoothing=float(args.label_smoothing), value_loss_type=str(args.value_loss), huber_delta=float(args.huber_delta),
                    policy_masking=True, ssl_warmup_steps=int(args.ssl_warmup_steps), current_step=current_step,
                    ssl_target_weight=1.0, use_wdl=False, wdl_weight=0.0, wdl_margin=0.25,
                    precision=("fp16" if scaler is not None else "fp32")
                )
                if loss_values is None:
                    continue
                loss, policy_loss, value_loss, ssl_loss, _ = loss_values

                # Optimizer step + scheduler
                if scaler is not None:
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                    if args.clip_every > 0 and (current_step + 1) % args.clip_every == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.clip_every > 0 and (current_step + 1) % args.clip_every == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # Update EMA after optimizer step
                if ema is not None:
                    ema.update(model)

                # Metrics
                running_pol = 0.98 * running_pol + 0.02 * float(policy_loss)
                running_val = 0.98 * running_val + 0.02 * float(value_loss)
                if (current_step + 1) % max(1, args.progress_interval) == 0:
                    mem = get_memory_usage(device).get("memory_gb", 0.0)
                    eta_s = (args.steps - (current_step + 1)) * max(1e-6, (time.time() - start_global) / max(1, current_step - start_step + 1))
                    pbar.set_postfix({
                        "loss": f"{float(loss):.4f}",
                        "pol": f"{float(policy_loss):.4f}",
                        "val": f"{float(value_loss):.4f}",
                        "ssl": f"{float(ssl_loss):.4f}",
                        "r_pol": f"{running_pol:.4f}",
                        "r_val": f"{running_val:.4f}",
                        "mem": f"{mem:.2f}GB",
                        "ETA": f"{int(eta_s)}s"
                    })
                pbar.update(1)
                current_step += 1

                # Optional periodic save
                if args.save_every and (current_step % args.save_every == 0):
                    tag = f"{args.checkpoint_prefix}_step_{current_step}.pt"
                    save_checkpoint(model, None, optimizer, scheduler, current_step, Path(args.checkpoint_out).parent / tag)
                    logger.info(f"Saved intermediate checkpoint: {tag}")

                if current_step >= args.steps:
                    break
            if current_step >= args.steps:
                break
    else:
        # Legacy path (tactical/openings)
        for step in range(start_step, args.steps):
            batch = dm.get_external_training_batch(args.batch_size, source=args.source)
            if not batch:
                logger.warning("No external batch available; stopping early")
                break

            loss_values = core_train_step(
                model, optimizer, scaler, batch, device,
                accum_steps=1, augment=True,
                ssl_weight=float(args.ssl_weight), enable_ssl=True,
                label_smoothing=float(args.label_smoothing), value_loss_type=str(args.value_loss), huber_delta=float(args.huber_delta),
                policy_masking=True, ssl_warmup_steps=int(args.ssl_warmup_steps), current_step=step,
                ssl_target_weight=1.0, use_wdl=False, wdl_weight=0.0, wdl_margin=0.25,
                precision=("fp16" if scaler is not None else "fp32")
            )
            if loss_values is None:
                continue
            loss, policy_loss, value_loss, ssl_loss, _ = loss_values

            if scaler is not None:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                if args.clip_every > 0 and (step + 1) % args.clip_every == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_every > 0 and (step + 1) % args.clip_every == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

            running_pol = 0.98 * running_pol + 0.02 * float(policy_loss)
            running_val = 0.98 * running_val + 0.02 * float(value_loss)
            if (step + 1) % max(1, args.progress_interval) == 0:
                mem = get_memory_usage(device).get("memory_gb", 0.0)
                eta_s = (args.steps - (step + 1)) * max(1e-6, (time.time() - start_global) / (step + 1))
                pbar.set_postfix({
                    "loss": f"{float(loss):.4f}",
                    "pol": f"{float(policy_loss):.4f}",
                    "val": f"{float(value_loss):.4f}",
                    "ssl": f"{float(ssl_loss):.4f}",
                    "r_pol": f"{running_pol:.4f}",
                    "r_val": f"{running_val:.4f}",
                    "mem": f"{mem:.2f}GB",
                    "ETA": f"{int(eta_s)}s"
                })
            pbar.update(1)

            if args.save_every and (step + 1) % args.save_every == 0:
                tag = f"{args.checkpoint_prefix}_step_{step+1}.pt"
                save_checkpoint(model, None, optimizer, scheduler, step + 1, Path(args.checkpoint_out).parent / tag)
                logger.info(f"Saved intermediate checkpoint: {tag}")

    # Save checkpoint
    pbar.close()
    out_path = Path(args.checkpoint_out)
    save_checkpoint(model, ema, optimizer, scheduler, step+1 if 'step' in locals() else 0, out_path)
    logger.info(f"Saved pretrained checkpoint: {out_path}")

    # Quick post-train eval
    if args.curriculum == "legacy":
        post_metrics = evaluate_quick(model, dm, device, batches=args.eval_batches, batch_size=args.batch_size, source=args.source)
        logger.info(f"Post-train (external, legacy) metrics: {post_metrics}")
    else:
        try:
            post_metrics = _evaluate_quick_stockfish(model, dm, device, batches=min(10, args.eval_batches), batch_size=args.batch_size)
            logger.info(f"Post-train (stockfish) metrics: {post_metrics}")
        except Exception as e:
            logger.warning(f"Post-train stockfish eval failed: {e}")

    # Optional: compare via enhanced_eval tool if desired
    logger.info("Compare models with:\n  python -m azchess.tools.enhanced_eval --model-a checkpoints/pretrain_external.pt --model-b checkpoints/best.pt")


if __name__ == "__main__":
    main()


