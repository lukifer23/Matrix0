from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Process, Queue
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List

from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from .config import Config
from .logging_utils import setup_logging
from .selfplay.internal import selfplay_worker, math_div_ceil
from .train import main as train_main
from .arena import play_match
import gc
import torch
from .data_manager import DataManager
from .monitor import dir_stats, disk_free, memory_usage_bytes


def orchestrate(cfg_path: str, games_override: int | None = None, eval_games_override: int | None = None, 
                workers_override: int | None = None, sims_override: int | None = None, cpuct_override: float | None = None,
                dirichlet_alpha_override: float | None = None, selection_jitter_override: float | None = None,
                opening_plies_override: int | None = None, resign_threshold_override: float | None = None,
                max_game_length_override: int | None = None, lr_override: float | None = None,
                batch_size_override: int | None = None, epochs_override: int | None = None,
                steps_per_epoch_override: int | None = None, accum_steps_override: int | None = None,
                weight_decay_override: float | None = None, ema_decay_override: float | None = None,
                grad_clip_override: float | None = None, promotion_threshold_override: float | None = None,
                device_override: str | None = None, max_retries_override: int | None = None,
                backoff_seconds_override: int | None = None, doctor_fix: bool | None = None, seed: int | None = None):
    cfg = Config.load(cfg_path)
    logger = setup_logging(cfg.training().get("log_dir", "logs"))

    orch = cfg.raw.get("orchestrator", {})
    games_target = int(orch.get("games_per_cycle", 64))
    eval_games = int(orch.get("eval_games_per_cycle", 20))
    if games_override is not None:
        games_target = int(games_override)
    if eval_games_override is not None:
        eval_games = int(eval_games_override)
    promote_thr = float(orch.get("promotion_threshold", 0.55))
    if promotion_threshold_override is not None:
        promote_thr = float(promotion_threshold_override)
    keep_top_k = int(orch.get("keep_top_k", 3))
    run_doctor_fix = bool(orch.get("doctor_fix", False)) if doctor_fix is None else bool(doctor_fix)
    max_retries = int(orch.get("max_retries", 0))
    if max_retries_override is not None:
        max_retries = int(max_retries_override)
    backoff = int(orch.get("backoff_seconds", 5))
    if backoff_seconds_override is not None:
        backoff = int(backoff_seconds_override)
    external_engine_integration = bool(orch.get("external_engine_integration", False))

    sp_cfg = cfg.to_dict()
    
    # Apply command-line overrides to self-play config
    if workers_override is not None:
        sp_cfg["selfplay"]["num_workers"] = int(workers_override)
    if sims_override is not None:
        sp_cfg["selfplay"]["num_simulations"] = int(sims_override)
    if cpuct_override is not None:
        sp_cfg["selfplay"]["cpuct"] = float(cpuct_override)
    if dirichlet_alpha_override is not None:
        sp_cfg["selfplay"]["dirichlet_alpha"] = float(dirichlet_alpha_override)
    if selection_jitter_override is not None:
        sp_cfg["selfplay"]["selection_jitter"] = float(selection_jitter_override)
    if opening_plies_override is not None:
        sp_cfg["selfplay"]["opening_random_plies"] = int(opening_plies_override)
    if resign_threshold_override is not None:
        sp_cfg["selfplay"]["resign_threshold"] = float(resign_threshold_override)
    if max_game_length_override is not None:
        sp_cfg["selfplay"]["max_game_len"] = int(max_game_length_override)
    
    # Apply command-line overrides to training config
    if lr_override is not None:
        sp_cfg["training"]["lr"] = float(lr_override)
    if batch_size_override is not None:
        sp_cfg["training"]["batch_size"] = int(batch_size_override)
    if epochs_override is not None:
        sp_cfg["training"]["epochs"] = int(epochs_override)
    if steps_per_epoch_override is not None:
        sp_cfg["training"]["steps_per_epoch"] = int(steps_per_epoch_override)
    if accum_steps_override is not None:
        sp_cfg["training"]["accum_steps"] = int(accum_steps_override)
    if weight_decay_override is not None:
        sp_cfg["training"]["weight_decay"] = float(weight_decay_override)
    if ema_decay_override is not None:
        sp_cfg["training"]["ema_decay"] = float(ema_decay_override)
    if grad_clip_override is not None:
        sp_cfg["training"]["grad_clip_norm"] = float(grad_clip_override)
    
    # Apply device override
    if device_override is not None:
        sp_cfg["device"] = str(device_override)
    
    # Apply reproducible seed if provided
    if seed is not None:
        try:
            import random, numpy as np, torch as _torch
            random.seed(seed)
            np.random.seed(seed)
            try:
                _torch.manual_seed(seed)
            except Exception:
                pass
            sp_cfg["seed"] = int(seed)
        except Exception:
            pass
    workers = sp_cfg["selfplay"].get("num_workers", 2)
    games_per_worker = math_div_ceil(games_target, workers)

    Path(cfg.training().get("checkpoint_dir", "checkpoints")).mkdir(parents=True, exist_ok=True)
    best_ckpt = Path(cfg.training().get("checkpoint_dir", "checkpoints")) / "best.pt"
    
    # Auto-create best.pt from model.pt if it doesn't exist
    if not best_ckpt.exists():
        model_ckpt = Path(cfg.training().get("checkpoint_dir", "checkpoints")) / "model.pt"
        if model_ckpt.exists():
            logger.info(f"Creating best.pt from existing model.pt")
            try:
                import shutil
                shutil.copy2(model_ckpt, best_ckpt)
                logger.info(f"Successfully created {best_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to copy model.pt to best.pt: {e}")
                logger.info("Will create new model with random weights")
        else:
            logger.info("No existing checkpoints found. Will create new model with random weights")

    with Progress(
        "{task.description}",
        BarColumn(),
        "{task.completed}/{task.total}",
        "•", TimeElapsedColumn(), "•", TimeRemainingColumn(),
    ) as progress:
        # Self-play
        def run_selfplay_once():
            logger.info("Starting self-play")
            q: Queue = Queue()
            procs: List[Process] = []
            sp_task = progress.add_task("Self-Play games", total=workers * games_per_worker)
            ckpt_for_sp = str(best_ckpt) if best_ckpt.exists() else None
            for i in range(workers):
                p = Process(target=selfplay_worker, args=(i, sp_cfg, ckpt_for_sp, games_per_worker, q))
                p.start()
                procs.append(p)
            stats = {"moves": 0, "time": 0.0, "win": 0, "loss": 0, "draw": 0}
            done = 0
            try:
                while done < workers * games_per_worker:
                    msg = q.get()
                    if isinstance(msg, dict) and msg.get("type") == "game":
                        done += 1
                        progress.update(sp_task, advance=1)
                        stats["moves"] += int(msg["moves"]) if "moves" in msg else 0
                        stats["time"] += float(msg["secs"]) if "secs" in msg else 0.0
                        res = float(msg.get("result", 0.0))
                        if res > 0: stats["win"] += 1
                        elif res < 0: stats["loss"] += 1
                        else: stats["draw"] += 1
            finally:
                for p in procs:
                    p.join()
            return stats, done

        # External engine self-play (if enabled)
        def run_external_engine_selfplay():
            if not external_engine_integration:
                logger.info("External engine integration disabled, skipping external engine self-play")
                return {"moves": 0, "time": 0.0, "win": 0, "loss": 0, "draw": 0}, 0
            
            try:
                from .selfplay.external_engine_worker import ExternalEngineSelfPlay
                from .engines import EngineManager
                import asyncio
                
                logger.info("Starting external engine self-play")
                
                async def run_external_selfplay():
                    engine_manager = EngineManager(cfg.to_dict())
                    await engine_manager.start_all_engines()
                    
                    try:
                        selfplay = ExternalEngineSelfPlay(cfg, engine_manager)
                        external_games = int(games_target * cfg.selfplay().get("external_engine_ratio", 0.3))
                        
                        if external_games > 0:
                            games = await selfplay.generate_games(
                                external_games, 
                                cfg.selfplay().get("buffer_dir", "data/selfplay")
                            )
                            logger.info(f"Generated {len(games)} external engine games")
                            return len(games)
                        else:
                            logger.info("No external engine games configured")
                            return 0
                    finally:
                        await engine_manager.cleanup()
                
                external_games_count = asyncio.run(run_external_selfplay())
                return {"moves": 0, "time": 0.0, "win": 0, "loss": 0, "draw": 0}, external_games_count
                
            except ImportError as e:
                logger.warning(f"External engine support not available: {e}")
                return {"moves": 0, "time": 0.0, "win": 0, "loss": 0, "draw": 0}, 0
            except Exception as e:
                logger.error(f"External engine self-play failed: {e}")
                return {"moves": 0, "time": 0.0, "win": 0, "loss": 0, "draw": 0}, 0

        # Retry loop for self-play
        attempt = 0
        while True:
            try:
                stats, done = run_selfplay_once()
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise
                logger.warning(f"Self-play failed: {e}. Retrying in {backoff}s...")
                sleep(backoff)
                attempt += 1

        # Run external engine self-play
        external_stats, external_done = run_external_engine_selfplay()
        
        # Combine stats
        total_done = done + external_done
        combined_stats = {
            "moves": stats["moves"] + external_stats["moves"],
            "time": stats["time"] + external_stats["time"],
            "win": stats["win"] + external_stats["win"],
            "loss": stats["loss"] + external_stats["loss"],
            "draw": stats["draw"] + external_stats["draw"]
        }
        
        avg_moves = combined_stats["moves"] / max(1, total_done)
        avg_time = combined_stats["time"] / max(1, total_done)
        logger.info(f"Self-play done: games={total_done} (internal={done}, external={external_done}), avg_moves={avg_moves:.1f}, avg_time={avg_time:.1f}s, W/L/D={combined_stats['win']}/{combined_stats['loss']}/{combined_stats['draw']}")

        # Report directory and resource stats
        sp_stats = dir_stats(sp_cfg["selfplay"]["buffer_dir"])
        logger.info(f"Self-play buffer: files={sp_stats.files} size={sp_stats.bytes/1e6:.1f}MB")
        df = disk_free(".")
        if df is not None:
            logger.info(f"Disk free: {df/1e9:.1f}GB")
        mu = memory_usage_bytes()
        if mu is not None:
            logger.info(f"Memory usage: {mu/1024/1024:.1f}MB")

        # Compact self-play to replay
        logger.info("Compacting self-play to replay (DataManager)")
        dm = DataManager(base_dir=cfg.get("data_dir", "data"))
        dm.compact_selfplay_to_replay()
        
        replay_stats = dir_stats(cfg.training().get("replay_dir", "data/replays"))
        logger.info(f"Replay buffer after compaction: {replay_stats.files} shards, {replay_stats.bytes / 1e6:.1f}MB")

        # Data integrity check
        if run_doctor_fix:
            logger.info("Running data integrity check")
            dm = DataManager(base_dir=cfg.get("data_dir", "data"))
            valid, corrupted = dm.validate_data_integrity()
            logger.info(f"Doctor: valid={valid} corrupted={corrupted}")
            if corrupted > 0:
                quarantined_count = dm.quarantine_corrupted_shards()
                logger.info(f"Quarantined {quarantined_count} corrupted shards.")

        # Training
        # Retry loop for training
        attempt = 0
        while True:
            try:
                logger.info("Starting training")
                t0 = perf_counter()
                # Optional epochs-per-cycle override via env
                try:
                    os.environ["MATRIX0_TRAIN_EPOCHS"] = str(int(orch.get("train_epochs_per_cycle", 1)))
                except Exception:
                    pass
                train_main()
                train_secs = perf_counter() - t0
                logger.info(f"Training finished in {train_secs:.1f}s")
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise
                logger.warning(f"Training failed: {e}. Retrying in {backoff}s...")
                sleep(backoff)
                attempt += 1

        # Memory cleanup
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        # Promote new checkpoint if better
        ckpt_dir = Path(cfg.training().get("checkpoint_dir", "checkpoints"))
        latest = ckpt_dir / "model.pt"
        if not best_ckpt.exists():
            logger.info("No best checkpoint found; promoting latest as best.")
            best_ckpt.write_bytes(latest.read_bytes())
            return
        logger.info("Evaluating latest vs best")
        # Retry loop for evaluation
        attempt = 0
        while True:
            try:
                score = play_match(str(latest), str(best_ckpt), eval_games, cfg, seed=seed)
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise
                logger.warning(f"Evaluation failed: {e}. Retrying in {backoff}s...")
                sleep(backoff)
                attempt += 1
        win_rate = score / float(eval_games)
        logger.info(f"Eval result: win_rate={win_rate:.3f} over {eval_games} games")
        if win_rate >= promote_thr:
            logger.info("Promoting latest to best")
            best_ckpt.write_bytes(latest.read_bytes())

        # Per-cycle JSONL summary
        try:
            import json
            summary = {
                "type": "cycle_summary",
                "games_target": games_target,
                "games_internal": int(done),
                "games_external": int(external_done),
                "replay_files": int(dir_stats(cfg.training().get("replay_dir", "data/replays")).files),
                "train_secs": float(train_secs),
                "eval_games": int(eval_games),
                "win_rate": float(win_rate),
                "timestamp": int(__import__("time").time()),
            }
            logs_dir = Path(cfg.training().get("log_dir", "logs"))
            logs_dir.mkdir(parents=True, exist_ok=True)
            with (logs_dir / "cycle_summary.jsonl").open("a") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception:
            pass

        # Prune old checkpoints if desired
        ckpts = sorted(glob.glob(str(ckpt_dir / "model_*.pt")))
        if len(ckpts) > keep_top_k:
            for p in ckpts[:-keep_top_k]:
                try:
                    os.remove(p)
                except OSError:
                    pass


def main():
    ap = argparse.ArgumentParser(description="Matrix0 Training Orchestrator - Flexible command-line configuration")
    
    # Core configuration
    ap.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    
    # Self-play parameters
    ap.add_argument("--games", type=int, default=None, help="Override total games per cycle")
    ap.add_argument("--workers", type=int, default=None, help="Override number of self-play workers")
    ap.add_argument("--sims", type=int, default=None, help="Override MCTS simulations per move")
    ap.add_argument("--cpuct", type=float, default=None, help="Override MCTS cpuct parameter")
    ap.add_argument("--dirichlet-alpha", type=float, default=None, help="Override MCTS dirichlet alpha")
    ap.add_argument("--selection-jitter", type=float, default=None, help="Override MCTS selection jitter")
    ap.add_argument("--opening-plies", type=int, default=None, help="Override random opening plies")
    ap.add_argument("--resign-threshold", type=float, default=None, help="Override resignation threshold")
    ap.add_argument("--max-game-length", type=int, default=None, help="Override maximum game length")
    
    # Training parameters
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ap.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    ap.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="Override training steps per epoch")
    ap.add_argument("--accum-steps", type=int, default=None, help="Override gradient accumulation steps")
    ap.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")
    ap.add_argument("--ema-decay", type=float, default=None, help="Override EMA decay")
    ap.add_argument("--grad-clip", type=float, default=None, help="Override gradient clipping norm")
    
    # Evaluation parameters
    ap.add_argument("--eval-games", type=int, default=None, help="Override evaluation games per cycle")
    ap.add_argument("--promotion-threshold", type=float, default=None, help="Override promotion threshold")
    
    # System parameters
    ap.add_argument("--device", type=str, default=None, help="Override device (auto/cpu/mps/cuda)")
    ap.add_argument("--doctor-fix", action="store_true", help="Quarantine corrupted data files before training")
    ap.add_argument("--external-engines", action="store_true", help="Enable external engine integration")
    ap.add_argument("--seed", type=int, default=None, help="Deterministic seed for self-play/eval")
    ap.add_argument("--max-retries", type=int, default=None, help="Override max retries for failed operations")
    ap.add_argument("--backoff-seconds", type=int, default=None, help="Override backoff delay between retries")
    
    args = ap.parse_args()
    
    # Enforce strict encoding via env if configured
    try:
        import os as _os
        cfg = Config.load(args.config)
        if bool(cfg.get("strict_encoding", False)):
            _os.environ["MATRIX0_STRICT_ENCODING"] = "1"
    except Exception:
        pass
    
    orchestrate(
        args.config, 
        games_override=args.games, 
        eval_games_override=args.eval_games,
        workers_override=args.workers,
        sims_override=args.sims,
        cpuct_override=args.cpuct,
        dirichlet_alpha_override=args.dirichlet_alpha,
        selection_jitter_override=args.selection_jitter,
        opening_plies_override=args.opening_plies,
        resign_threshold_override=args.resign_threshold,
        max_game_length_override=args.max_game_length,
        lr_override=args.lr,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
        steps_per_epoch_override=args.steps_per_epoch,
        accum_steps_override=args.accum_steps,
        weight_decay_override=args.weight_decay,
        ema_decay_override=args.ema_decay,
        grad_clip_override=args.grad_clip,
        promotion_threshold_override=args.promotion_threshold,
        device_override=args.device,
        max_retries_override=args.max_retries,
        backoff_seconds_override=args.backoff_seconds,
        doctor_fix=args.doctor_fix, 
        seed=args.seed
    )


if __name__ == "__main__":
    main()
