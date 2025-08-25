from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import queue as pyqueue
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List

import numpy as np
import torch
from rich.progress import (BarColumn, Progress, TimeElapsedColumn,
                           TimeRemainingColumn)
from torch.multiprocessing import Event as MPEvent
from torch.multiprocessing import Process, Queue

from azchess.training.train import train_from_config as train_main

from .arena import play_match
from .config import Config, select_device
from .data_manager import DataManager
from .elo import EloBook, update_elo
from .logging_utils import setup_logging
from .monitor import dir_stats, disk_free, memory_usage_bytes
from .selfplay.inference import (run_inference_server,
                                 setup_shared_memory_for_worker)
from .selfplay.internal import math_div_ceil, selfplay_worker
from .utils import (clear_memory_cache, get_memory_usage, log_config_summary,
                    safe_config_get)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")


def cleanup_temp_files(data_dir: Path) -> None:
    """Clean up temporary files from previous runs."""
    logger = logging.getLogger(__name__)

    # Clean up any leftover temp files
    temp_patterns = ["*.tmp", "*.temp", "temp_*", "*_temp"]
    for pattern in temp_patterns:
        for temp_file in data_dir.glob(pattern):
            try:
                temp_file.unlink()
            except FileNotFoundError:
                logger.debug("Temp file not found during cleanup: %s", temp_file)
            except OSError as e:
                logger.warning("Failed to delete temp file %s: %s", temp_file, e)


# Top-level helper for external engine process (macOS spawn-safe)
def _run_external_proc(proc_id: int, cfg_path: str, out_dir: str, n_games: int):
    import asyncio as _asyncio

    from .config import Config as _Cfg
    from .selfplay.external_engine_worker import \
        external_engine_worker as _worker
    _cfg = _Cfg.load(cfg_path)
    _asyncio.run(_worker(proc_id=proc_id, config=_cfg, output_dir=out_dir, num_games=n_games))


@dataclass
class OrchestratorOverrides:
    games: int | None = None
    eval_games: int | None = None
    workers: int | None = None
    sims: int | None = None
    cpuct: float | None = None
    dirichlet_alpha: float | None = None
    selection_jitter: float | None = None
    opening_plies: int | None = None
    resign_threshold: float | None = None
    max_game_length: int | None = None
    lr: float | None = None
    batch_size: int | None = None
    epochs: int | None = None
    steps_per_epoch: int | None = None
    accum_steps: int | None = None
    weight_decay: float | None = None
    ema_decay: float | None = None
    grad_clip: float | None = None
    promotion_threshold: float | None = None
    device: str | None = None
    max_retries: int | None = None
    backoff_seconds: int | None = None
    doctor_fix: bool | None = None
    seed: int | None = None
    no_shared_infer: bool | None = None


def orchestrate(
    cfg_path: str,
    overrides: OrchestratorOverrides | None = None,
    tui_mode: str = "bars",
    quick_start: bool = False,
):
    cfg = Config.load(cfg_path)
    log_dir = safe_config_get(cfg, "log_dir", "logs", section="training")
    logger = setup_logging(log_dir)

    # Clear structured log file at start of each run
    structured_log_path = Path(log_dir) / "structured.jsonl"
    if structured_log_path.exists():
        try:
            structured_log_path.unlink()
            logger.info(f"Cleared previous structured log: {structured_log_path}")
        except Exception as e:
            logger.warning(f"Could not clear structured log: {e}")

    # Validate MCTS config
    from .mcts import MCTSConfig
    mcts_cfg_dict = cfg.mcts()
    if not mcts_cfg_dict:
        raise ValueError("MCTS config section is missing in config.yaml")
    
    _ = MCTSConfig.from_dict(mcts_cfg_dict)
    required_keys = set(MCTSConfig.__dataclass_fields__.keys())
    missing_keys = required_keys - set(mcts_cfg_dict.keys())
    if missing_keys:
        raise ValueError(f"Missing required MCTS config keys: {sorted(missing_keys)}")

    try:
        import os as _os
        dev_req = safe_config_get(cfg, "device", "auto")
        from .config import select_device as _sel
        dev_sel = _sel(dev_req)
        # Set MPS-friendly env if we are likely to use MPS
        if dev_sel == 'mps':
            _os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # Memory limits are now set at the beginning of main() before torch imports
        mps_built = getattr(torch.backends.mps, 'is_built', lambda: False)()
        mps_avail = getattr(torch.backends.mps, 'is_available', lambda: False)()
        logger.info(f"Device requested={dev_req} selected={dev_sel} | MPS built={mps_built} available={mps_avail}")
    except Exception:
        pass

    overrides = overrides or OrchestratorOverrides()
    orch = cfg.raw.get("orchestrator", {})

    # Dynamic game count logic: check if this is first run
    initial_games = int(orch.get("initial_games", 64))
    subsequent_games = int(orch.get("subsequent_games", 64))

    # Check if we have existing checkpoints to determine if this is first run
    checkpoint_dir = Path("checkpoints")
    existing_checkpoints = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []

    # Determine game count based on context and overrides
    if quick_start:
        # Force quick start mode regardless of checkpoints
        games_target = initial_games
        logger.info(f"Quick start mode: using {initial_games} games for rapid iteration")
    elif overrides.games is not None:
        # Command-line override takes precedence
        games_target = int(overrides.games)
        logger.info(f"Command-line override: using {games_target} games")
    elif len(existing_checkpoints) <= 1:  # Only base model or no models
        games_target = initial_games
        logger.info(f"First run detected: using {initial_games} games for quick start")
    else:
        games_target = subsequent_games
        logger.info(f"Subsequent run: using {subsequent_games} games for full cycle")

    eval_games = int(orch.get("eval_games_per_cycle", 20))
    if overrides.eval_games is not None:
        eval_games = int(overrides.eval_games)
    promote_thr = float(orch.get("promotion_threshold", 0.55))
    if overrides.promotion_threshold is not None:
        promote_thr = float(overrides.promotion_threshold)
    keep_top_k = int(orch.get("keep_top_k", 3))
    run_doctor_fix = bool(orch.get("doctor_fix", False)) if overrides.doctor_fix is None else bool(overrides.doctor_fix)
    max_retries = int(orch.get("max_retries", 0))
    if overrides.max_retries is not None:
        max_retries = int(overrides.max_retries)
    backoff = int(orch.get("backoff_seconds", 5))
    if overrides.backoff_seconds is not None:
        backoff = int(overrides.backoff_seconds)
    external_engine_integration = bool(orch.get("external_engine_integration", False))

    sp_cfg = cfg.to_dict()

    # Apply device-specific presets if configured
    try:
        from copy import deepcopy
        if bool(sp_cfg.get("use_presets", True)) and isinstance(sp_cfg.get("presets", None), dict):
            dev = sp_cfg.get("device", cfg.get("device", "auto"))
            from .config import select_device as _sel_dev
            dev_sel = _sel_dev(dev)
            preset = sp_cfg["presets"].get(dev_sel)
            if preset:
                # Merge selfplay and training presets without clobbering existing keys unless absent
                for section in ("selfplay", "training", "mcts", "eval"):
                    if section in preset:
                        sp_cfg.setdefault(section, {})
                        for k, v in preset[section].items():
                            if k not in sp_cfg[section]:
                                sp_cfg[section][k] = v
                logger.info(f"Applied {dev_sel} presets to config")
    except Exception as e:
        logger.warning(f"Failed to apply presets: {e}")
    
    # Apply command-line overrides to self-play config
    if overrides.workers is not None:
        sp_cfg["selfplay"]["num_workers"] = int(overrides.workers)
    if overrides.sims is not None:
        sp_cfg["selfplay"]["num_simulations"] = int(overrides.sims)
    if overrides.cpuct is not None:
        sp_cfg["selfplay"]["cpuct"] = float(overrides.cpuct)
    if overrides.dirichlet_alpha is not None:
        sp_cfg["selfplay"]["dirichlet_alpha"] = float(overrides.dirichlet_alpha)
    if overrides.selection_jitter is not None:
        sp_cfg["selfplay"]["selection_jitter"] = float(overrides.selection_jitter)
    if overrides.opening_plies is not None:
        sp_cfg["selfplay"]["opening_random_plies"] = int(overrides.opening_plies)
    if overrides.resign_threshold is not None:
        sp_cfg["selfplay"]["resign_threshold"] = float(overrides.resign_threshold)
    if overrides.max_game_length is not None:
        sp_cfg["selfplay"]["max_game_len"] = int(overrides.max_game_length)
    if overrides.no_shared_infer is True:
        sp_cfg["selfplay"]["shared_inference"] = False

    # Apply command-line overrides to training config
    if overrides.lr is not None:
        sp_cfg["training"]["lr"] = float(overrides.lr)
    if overrides.batch_size is not None:
        sp_cfg["training"]["batch_size"] = int(overrides.batch_size)
    if overrides.epochs is not None:
        sp_cfg["training"]["epochs"] = int(overrides.epochs)
    if overrides.steps_per_epoch is not None:
        sp_cfg["training"]["steps_per_epoch"] = int(overrides.steps_per_epoch)
    if overrides.accum_steps is not None:
        sp_cfg["training"]["accum_steps"] = int(overrides.accum_steps)
    if overrides.weight_decay is not None:
        sp_cfg["training"]["weight_decay"] = float(overrides.weight_decay)
    if overrides.ema_decay is not None:
        sp_cfg["training"]["ema_decay"] = float(overrides.ema_decay)
    if overrides.grad_clip is not None:
        sp_cfg["training"]["grad_clip_norm"] = float(overrides.grad_clip)

    # Apply device override
    if overrides.device is not None:
        sp_cfg["device"] = str(overrides.device)

    # Apply reproducible seed if provided
    if overrides.seed is not None:
        try:
            import random

            import numpy as np
            import torch as _torch
            random.seed(overrides.seed)
            np.random.seed(overrides.seed)
            try:
                torch.manual_seed(overrides.seed)
            except Exception:
                pass
            sp_cfg["seed"] = int(overrides.seed)
        except Exception:
            pass
    workers = sp_cfg["selfplay"].get("num_workers", 2)
    games_per_worker = math_div_ceil(games_target, workers)

    Path(cfg.training().get("checkpoint_dir", "checkpoints")).mkdir(parents=True, exist_ok=True)
    
    # Check for checkpoint override in orchestrator config
    checkpoint_override = cfg.orchestrator().get("checkpoint_override")
    if checkpoint_override:
        best_ckpt = Path(cfg.training().get("checkpoint_dir", "checkpoints")) / checkpoint_override
        logger.info(f"Using checkpoint override: {best_ckpt}")
    else:
        best_ckpt = Path(cfg.training().get("checkpoint_dir", "checkpoints")) / "best.pt"
        logger.info(f"Using default checkpoint: {best_ckpt}")
    
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
            logger.info("DEBUG: About to start training phase")

    with Progress(
        "{task.description}",
        BarColumn(),
        "{task.completed}/{task.total}",
        "•", TimeElapsedColumn(), "•", TimeRemainingColumn(),
        transient=False,
    ) as progress:
        # Self-play
        def run_selfplay_once():
            # Clean up any leftover temp files from previous runs
            cleanup_temp_files(Path(cfg.get("data_dir", "data")))

            logger.info("Starting self-play")
            # Reduce MCTS verbosity during self-play to keep TUI readable
            try:
                import logging as _logging
                _mcts_logger = __import__('logging').getLogger('azchess.mcts')
                _mcts_logger.setLevel(_logging.ERROR)
            except Exception:
                pass
            
            # Ensure torch is available in this scope
            import torch
            q: Queue = Queue()
            procs: List[Process] = []
            use_table = (tui_mode == "table")
            if not use_table:
                sp_task = progress.add_task("Self-Play (total)", total=workers * games_per_worker)
                worker_tasks: Dict[int, int] = {}
                for i in range(workers):
                    worker_tasks[i] = progress.add_task(f"W{i}", total=games_per_worker)
                stats_task = progress.add_task("W/L/D: 0/0/0", total=1)
                hb_task = progress.add_task("HB", total=1)
            ckpt_for_sp = str(best_ckpt) if best_ckpt.exists() else None
            logger.info(f"Checkpoint path for self-play: {ckpt_for_sp}")
            
            # Pre-load the model state dict in the main process to avoid issues in spawned processes
            model_state_dict = None
            if ckpt_for_sp:
                try:
                    logger.info(f"Loading checkpoint from: {ckpt_for_sp}")
                    state = torch.load(ckpt_for_sp, map_location='cpu', weights_only=False)
                    logger.info(f"Checkpoint keys: {list(state.keys())}")
                    # Extract the actual model state dict from the checkpoint
                    if "model_state_dict" in state:
                        model_state_dict = state["model_state_dict"]
                        logger.info("Using model_state_dict from checkpoint")
                    elif "model_ema" in state:
                        model_state_dict = state["model_ema"]
                        logger.info("Using model_ema from checkpoint")
                    elif "model" in state:
                        model_state_dict = state["model"]
                        logger.info("Using model from checkpoint")
                    else:
                        # Fallback: assume the checkpoint itself is the model state dict
                        model_state_dict = state
                        logger.info("Using checkpoint directly as model state dict")

                    logger.info(f"Model state dict loaded successfully with {len(model_state_dict)} layers")
                    
                    # Log actual parameter count for clarity
                    try:
                        from azchess.model import PolicyValueNet
                        temp_model = PolicyValueNet.from_config(sp_cfg["model"])
                        actual_params = temp_model.count_parameters()
                        logger.info(f"Model architecture has {actual_params:,} total parameters")
                        del temp_model  # Clean up temporary model
                    except Exception as e:
                        logger.debug(f"Could not determine parameter count: {e}")
                    
                    # If using checkpoint override, copy to best.pt for future runs
                    if checkpoint_override and checkpoint_override != "best.pt":
                        best_pt_path = Path(cfg.training().get("checkpoint_dir", "checkpoints")) / "best.pt"
                        try:
                            import shutil
                            shutil.copy2(ckpt_for_sp, best_pt_path)
                            logger.info(f"Copied {checkpoint_override} to {best_pt_path} for future runs")
                        except Exception as e:
                            logger.warning(f"Failed to copy checkpoint override to best.pt: {e}")
                    
                except Exception as e:
                    logger.error(f"Failed to pre-load checkpoint state_dict: {e}")
                    raise

            # Start shared inference server if beneficial
            infer_proc: Process | None = None
            stop_event = None
            shared_memory_resources = []
            try:
                dev = select_device(sp_cfg.get("device", cfg.get("device", "auto")))
            except Exception:
                dev = "cpu"
            shared_infer_enabled = bool(sp_cfg.get("selfplay", {}).get("shared_inference", True))
            if shared_infer_enabled and dev != "cpu":
                logger.info(f"Setting up shared memory and launching inference server on device: {dev}")
                stop_event = MPEvent()
                server_ready_event = MPEvent()
                
                # Create shared memory resources for each worker with optimized batch sizes
                model_params = sp_cfg["model"]
                sp_params = sp_cfg["selfplay"]
                # Use larger batch sizes for better GPU utilization
                optimized_batch_size = max(32, sp_params.get('batch_size', 32))
                for i in range(workers):
                    res = setup_shared_memory_for_worker(
                        worker_id=i,
                        planes=model_params['planes'],
                        policy_size=model_params['policy_size'],
                        max_batch_size=optimized_batch_size
                    )
                    shared_memory_resources.append(res)

                # In compact mode, ask child to minimize stdout noise
                compact = bool(cfg.get("orchestrator", {}).get("ui", {}).get("compact", False))
                if compact:
                    os.environ["MATRIX0_COMPACT_LOG"] = "1"

                infer_proc = Process(target=run_inference_server, args=(dev, sp_cfg["model"], model_state_dict, stop_event, server_ready_event, shared_memory_resources))
                infer_proc.start()
                
                logger.info("Waiting for inference server to initialize...")
                if not server_ready_event.wait(timeout=60):
                    logger.error("Inference server failed to start in time. Aborting.")
                    if infer_proc.is_alive(): infer_proc.terminate()
                    raise RuntimeError("Inference server startup timeout")
                logger.info("Inference server is ready.")

            for i in range(workers):
                os.environ.setdefault('MATRIX0_WORKER_LOG_LEVEL', 'WARNING')
                # Pass the specific shared memory resource for this worker
                sm_res = shared_memory_resources[i] if shared_memory_resources else None
                p = Process(target=selfplay_worker, args=(i, sp_cfg, ckpt_for_sp, games_per_worker, q, sm_res))
                p.start()
                procs.append(p)

            stats = {"moves": 0, "time": 0.0, "win": 0, "loss": 0, "draw": 0, "res_w": 0, "res_b": 0, "avg_ms_per_move_sum": 0.0, "avg_sims_sum": 0.0, "games_with_metrics": 0}
            done = 0
            # For table mode, track per-worker stats
            per_worker = {i: {"done": 0, "avg_ms": 0.0, "avg_sims": 0.0, "moves": 0, "hb_ts": 0.0} for i in range(workers)}

            # Helper: monitor, respawn or adjust targets if workers die
            def _check_and_respawn_workers(last_progress_ts: float,
                                          per_worker_done: Dict[int, int]) -> None:
                nonlocal infer_proc, stop_event, shared_memory_resources, model_params, optimized_batch_size
                now = time.time()
                for i, p in enumerate(list(procs)):
                    if p.is_alive():
                        continue
                    # Worker died. Determine remaining games and respawn if needed
                    remaining = max(0, games_per_worker - per_worker_done.get(i, 0))
                    if remaining <= 0:
                        continue
                    logger.warning(f"Worker {i} died early (done={per_worker_done.get(i,0)}/{games_per_worker}). Respawning for remaining {remaining} games.")
                    # Reuse the same shared memory resource if present
                    sm_res = shared_memory_resources[i] if shared_memory_resources else None
                    new_p = Process(target=selfplay_worker, args=(i, sp_cfg, ckpt_for_sp, remaining, q, sm_res))
                    new_p.start()
                    procs[i] = new_p
                # If using shared inference, ensure server is still alive; restart if needed
                if shared_infer_enabled and infer_proc is not None and not infer_proc.is_alive():
                    logger.warning("Inference server died; restarting.")
                    try:
                        # Attempt clean stop signal (in case it is in zombie state)
                        if stop_event is not None:
                            stop_event.set()
                    except Exception:
                        pass
                    # Start a new server process
                    stop_event = MPEvent()
                    server_ready_event = MPEvent()

                    # CRITICAL: Recreate shared memory resources with fresh Event objects
                    # This ensures workers and the new server use the same Event objects
                    from .selfplay.inference import \
                        setup_shared_memory_for_worker
                    new_shared_memory_resources = []
                    for i in range(len(shared_memory_resources)):
                        try:
                            new_res = setup_shared_memory_for_worker(
                                worker_id=i,
                                planes=model_params['planes'],
                                policy_size=model_params['policy_size'],
                                max_batch_size=optimized_batch_size
                            )
                            new_shared_memory_resources.append(new_res)
                            logger.debug(f"Recreated shared memory resource {i} for inference server restart")
                        except Exception as e:
                            logger.error(f"Failed to recreate shared memory resource {i}: {e}")
                            # Try to reuse old resource if recreation fails
                            new_shared_memory_resources.append(shared_memory_resources[i])

                    # Replace the old resources with new ones
                    shared_memory_resources = new_shared_memory_resources
                    logger.info("Updated shared memory resources with fresh Event objects for server restart")

                    infer_proc = Process(
                        target=run_inference_server,
                        args=(dev, sp_cfg["model"], model_state_dict, stop_event, server_ready_event, shared_memory_resources)
                    )
                    infer_proc.start()
                    if not server_ready_event.wait(timeout=60):
                        logger.error("Restarted inference server failed to start in time.")
                        logger.error("This may indicate a critical issue with shared memory or device initialization.")
                        logger.error("Check logs for detailed error messages from the inference server process.")
                        # Try to terminate the failed process
                        try:
                            if infer_proc.is_alive():
                                infer_proc.terminate()
                                infer_proc.join(timeout=5)
                                logger.info("Terminated failed inference server process")
                        except Exception as term_error:
                            logger.error(f"Failed to terminate failed inference server: {term_error}")
                        infer_proc = None  # Reset to None so it won't be checked again

            try:
                if use_table:
                    from rich.live import Live
                    from rich.table import Table
                    
                    def mem_title() -> str:
                        try:
                            import psutil
                            vm = psutil.virtual_memory()
                            total_gb = vm.total / (1024**3)
                            used_gb = (vm.total - vm.available) / (1024**3)
                            return f"Self-Play | Mem {used_gb:.1f}/{total_gb:.1f} GB ({vm.percent:.0f}%)"
                        except Exception:
                            return "Self-Play"

                    table = Table(title=mem_title())
                    table.add_column("Worker", justify="left")
                    table.add_column("Done/Total", justify="right")
                    table.add_column("Avg ms/move", justify="right")
                    table.add_column("Avg sims", justify="right")
                    table.add_column("Moves", justify="right")
                    table.add_column("HB(s)", justify="right")
                    table.add_column("W/L/D", justify="right")
                    table.add_column("Res W/B", justify="right")
                    
                    with Live(table, refresh_per_second=4, transient=False) as live:
                        for i in range(workers):
                            w = per_worker[i]
                            hb_age = 0.0
                            if w["hb_ts"] > 0:
                                hb_age = max(0.0, time.perf_counter() - w["hb_ts"])
                            table.add_row(
                                f"W{i}",
                                f"{w['done']}/{games_per_worker}",
                                f"{w['avg_ms']:.1f}",
                                f"{w['avg_sims']:.1f}",
                                f"{w['moves']}",
                                f"{hb_age:.0f}",
                                f"{stats['win']}/{stats['loss']}/{stats['draw']}",
                                f"{stats['res_w']}/{stats['res_b']}"
                            )
                        # Ensure initial rows render before first message
                        try:
                            live.refresh()
                        except Exception:
                            pass
                        
                        last_msg_time = time.time()
                        total_target = workers * games_per_worker
                        while done < total_target:
                            try:
                                msg = q.get(timeout=2.0)
                            except pyqueue.Empty:
                                # Periodically check for dead workers and respawn if needed
                                _check_and_respawn_workers(last_msg_time, {k: v["done"] for k, v in per_worker.items()})
                                # Detect total stall (no progress) and break to allow retry loop
                                if time.time() - last_msg_time > 300:
                                    raise RuntimeError("Self-play appears stalled (no progress for 300s)")
                                continue
                            # Any message counts as progress for stall detection
                            last_msg_time = time.time()
                            # Heartbeat updates (update in-place; don't print new lines)
                            if isinstance(msg, dict) and msg.get("type") == "heartbeat":
                                wid = int(msg.get("proc", -1))
                                if wid in per_worker:
                                    per_worker[wid]["moves"] = int(msg.get("moves", 0))
                                    per_worker[wid]["hb_ts"] = time.perf_counter()
                                try:
                                    # bump a dummy heartbeat task to redraw if present
                                    progress.update(hb_task, completed=0)
                                except Exception:
                                    pass
                                # Heartbeat processed - no table refresh needed
                                continue
                            if isinstance(msg, dict) and msg.get("type") == "game":
                                done += 1
                                wid = int(msg.get("proc", -1))
                                if wid in per_worker:
                                    per_worker[wid]["done"] += 1
                                    ms = float(msg.get("avg_ms_per_move", 0.0))
                                    sims = float(msg.get("avg_sims", 0.0))
                                    c = per_worker[wid]["done"]
                                    per_worker[wid]["avg_ms"] = (per_worker[wid]["avg_ms"] * (c - 1) + ms) / max(1, c)
                                    per_worker[wid]["avg_sims"] = (per_worker[wid]["avg_sims"] * (c - 1) + sims) / max(1, c)
                                stats["moves"] += int(msg.get("moves", 0))
                                stats["time"] += float(msg.get("secs", 0.0))
                                if "avg_ms_per_move" in msg and "avg_sims" in msg:
                                    stats["avg_ms_per_move_sum"] += float(msg["avg_ms_per_move"])
                                    stats["avg_sims_sum"] += float(msg["avg_sims"])
                                    stats["games_with_metrics"] += 1
                                res = float(msg.get("result", 0.0))
                                if res > 0: stats["win"] += 1
                                elif res < 0: stats["loss"] += 1
                                else: stats["draw"] += 1
                                if bool(msg.get("resigned", False)):
                                    rc = msg.get("resigner", None)
                                    if rc == 'W':
                                        stats['res_w'] += 1
                                    elif rc == 'B':
                                        stats['res_b'] += 1
                                # Rebuild table with memory and heartbeat info
                                new_table = Table(title=mem_title())
                                new_table.add_column("Worker", justify="left")
                                new_table.add_column("Done/Total", justify="right")
                                new_table.add_column("Avg ms/move", justify="right")
                                new_table.add_column("Avg sims", justify="right")
                                new_table.add_column("Moves", justify="right")
                                new_table.add_column("HB(s)", justify="right")
                                new_table.add_column("W/L/D", justify="right")
                                new_table.add_column("Res W/B", justify="right")
                                
                                for i in range(workers):
                                    w = per_worker[i]
                                    hb_age = 0.0
                                    if w["hb_ts"] > 0:
                                        hb_age = max(0.0, time.perf_counter() - w["hb_ts"])
                                    new_table.add_row(
                                        f"W{i}",
                                        f"{w['done']}/{games_per_worker}",
                                        f"{w['avg_ms']:.1f}",
                                        f"{w['avg_sims']:.1f}",
                                        f"{w['moves']}",
                                        f"{hb_age:.0f}",
                                        f"{stats['win']}/{stats['loss']}/{stats['draw']}",
                                        f"{stats['res_w']}/{stats['res_b']}"
                                    )
                                
                                live.update(new_table)
                else:
                    last_msg_time = time.time()
                    total_target = workers * games_per_worker
                    while done < total_target:
                        try:
                            msg = q.get(timeout=2.0)
                        except pyqueue.Empty:
                            # Periodically check for dead workers and respawn if needed
                            # Build minimal per-worker-done map from progress tasks when in bars mode
                            # We don't track per-worker here tightly; maintain a simple done counter only.
                            # Use zeros so we only respawn if a worker died very early.
                            per_worker_done = {i: 0 for i in range(workers)}
                            _check_and_respawn_workers(last_msg_time, per_worker_done)
                            if time.time() - last_msg_time > 300:
                                raise RuntimeError("Self-play appears stalled (no progress for 300s)")
                            continue
                        # Any message counts as progress for stall detection
                        last_msg_time = time.time()
                        if isinstance(msg, dict) and msg.get("type") == "game":
                            done += 1
                            progress.update(sp_task, advance=1)
                            wid = int(msg.get("proc", -1))
                            if wid in worker_tasks:
                                progress.update(worker_tasks[wid], advance=1)
                            stats["moves"] += int(msg.get("moves", 0))
                            stats["time"] += float(msg.get("secs", 0.0))
                            if "avg_ms_per_move" in msg and "avg_sims" in msg:
                                stats["avg_ms_per_move_sum"] += float(msg["avg_ms_per_move"])
                                stats["avg_sims_sum"] += float(msg["avg_sims"])
                                stats["games_with_metrics"] += 1
                            res = float(msg.get("result", 0.0))
                            if res > 0: stats["win"] += 1
                            elif res < 0: stats["loss"] += 1
                            else: stats["draw"] += 1
                            progress.update(stats_task, description=f"W/L/D: {stats['win']}/{stats['loss']}/{stats['draw']}")
            finally:
                for p in procs:
                    p.join()
                if infer_proc is not None:
                    try:
                        if stop_event: stop_event.set()
                        infer_proc.join(timeout=5)
                    except Exception:
                        pass
            return stats, done

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
        # Post self-play: compact data, validate/quarantine if requested
        logger.info("Compacting self-play data into replay buffer")
        dm = DataManager(base_dir=cfg.get("data_dir", "data"))
        try:
            dm.compact_selfplay_to_replay()
        except Exception as e:
            logger.warning(f"Data compaction failed: {e}")

        # Re-initialize DataManager to ensure it sees the new data
        dm = DataManager(base_dir=cfg.get("data_dir", "data"))

        if run_doctor_fix:
            try:
                valid, corrupted = dm.validate_data_integrity()
                logger.info(f"Data integrity: valid={valid} corrupted={corrupted}")
                if corrupted > 0:
                    qn = dm.quarantine_corrupted_shards()
                    logger.info(f"Quarantined {qn} corrupted shards")
            except Exception as e:
                logger.warning(f"Doctor fix failed: {e}")

        # Import extra replay dirs (e.g., Lichess) into the DB for training
        try:
            extra_dirs = cfg.training().get("extra_replay_dirs", []) or []
            if isinstance(extra_dirs, list) and extra_dirs:
                imported_total = 0
                for d in extra_dirs:
                    try:
                        n = dm.import_replay_dir(d, source="external", move_files=False)
                        imported_total += int(n)
                    except Exception as e:
                        logger.warning(f"Import of extra replay dir {d} failed: {e}")
                if imported_total:
                    logger.info(f"Imported {imported_total} external shards from extra_replay_dirs for training")
        except Exception as e:
            logger.warning(f"Loading extra replay dirs failed: {e}")

        # External CSV ingestion disabled per request (sufficient NPZ shards available)

        # Ensure all common NPZ directories are registered with the DB
        try:
            ndirs = _register_external_npz_dirs(cfg, dm)
            if ndirs:
                logger.info(f"Registered NPZ shards from {ndirs} external directories")
        except Exception as e:
            logger.warning(f"Registering NPZ directories failed: {e}")

        # Ensure we have usable data before training
        try:
            stats = DataManager(base_dir=cfg.get("data_dir", "data")).get_stats()
            if int(stats.total_samples) <= 0:
                logger.warning("No training data available after ingestion; skipping training and evaluation.")
                return
        except Exception:
            logger.warning("Could not obtain data stats; attempting training anyway.")

        # Memory cleanup between phases (especially important for MPS)
        logger.info("Performing memory cleanup between self-play and training phases")
        try:
            clear_memory_cache('auto')
            logger.info("Memory cleanup completed")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

        # Train phase
        logger.info("Starting training phase")
        try:
            train_main(cfg_path)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        # Determine candidate checkpoint produced by training
        ckpt_dir = Path(cfg.training().get("checkpoint_dir", "checkpoints"))
        candidate = ckpt_dir / "enhanced_best.pt"
        if not candidate.exists():
            # Fallback to any *_best.pt excluding best.pt
            cand_list = [p for p in ckpt_dir.glob("*_best.pt") if p.name != "best.pt"]
            cand_list.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if cand_list:
                candidate = cand_list[0]
            else:
                # Last resort: use enhanced_final.pt
                alt = ckpt_dir / "enhanced_final.pt"
                candidate = alt if alt.exists() else None

        if candidate is None or not Path(candidate).exists():
            logger.warning("No candidate checkpoint produced by training; skipping evaluation.")
            return

        # If there is no best checkpoint yet, promote immediately
        if not best_ckpt.exists():
            import shutil
            try:
                shutil.copy2(candidate, best_ckpt)
                logger.info(f"Promoted initial best checkpoint: {best_ckpt}")
            except Exception as e:
                logger.error(f"Failed to set initial best checkpoint: {e}")
            return

        # Evaluate candidate vs current best
        if eval_games > 0:
            logger.info(f"Evaluating candidate {candidate} vs best {best_ckpt}")
            # Parallelize eval games if configured
            eval_workers = int(cfg.eval().get("workers", 1))
            eval_num_sims = int(cfg.eval().get("num_simulations", cfg.mcts().get("num_simulations", 500)))
            score = play_match(
                ckpt_a=str(candidate),
                ckpt_b=str(best_ckpt),
                games=eval_games,
                cfg=cfg,
                seed=seed or int(time.time()),
                workers=eval_workers,
                num_sims=eval_num_sims
            )
            win_rate = score / float(eval_games)
            logger.info(f"Evaluation complete: win_rate={win_rate:.3f} threshold={promote_thr:.3f}")
        else:
            logger.info(f"Skipping evaluation (eval_games={eval_games})")
            score = 0.0
            win_rate = 0.0

        # Update Elo ratings for bookkeeping
        try:
            elopath = Path(cfg.training().get("checkpoint_dir", "checkpoints")) / "elo.json"
            book = EloBook(elopath)
            state = book.load()
            r_best = float(state.get("best", 1500.0))
            r_cand = float(state.get("candidate", 1500.0))
            r_cand_new, r_best_new = update_elo(r_cand, r_best, win_rate)
            state["best"] = r_best_new
            state["candidate"] = r_cand_new
            state.setdefault("history", []).append({
                "ts": int(time.time()),
                "candidate": r_cand_new,
                "best": r_best_new,
                "score": float(score),
                "games": int(eval_games),
            })
            book.save(state)
            logger.info(f"Elo updated: candidate={r_cand_new:.1f}, best={r_best_new:.1f}")
        except Exception as e:
            logger.warning(f"Elo update failed: {e}")

        # Promotion and top-k management
        if win_rate >= promote_thr:
            import shutil
            try:
                # Archive current best
                ts = time.strftime("%Y%m%d_%H%M%S")
                archive = ckpt_dir / f"best_archive_{ts}.pt"
                shutil.copy2(best_ckpt, archive)
            except Exception:
                pass
            try:
                shutil.copy2(candidate, best_ckpt)
                logger.info(f"Promoted candidate to best: {best_ckpt}")
            except Exception as e:
                logger.error(f"Failed to promote candidate: {e}")
            # Enforce keep_top_k archives
            try:
                arch = sorted(ckpt_dir.glob("best_archive_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                for p in arch[keep_top_k:]:
                    try:
                        p.unlink()
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            logger.info("Candidate did not meet promotion threshold; keeping current best.")

def main():
    # Parse arguments first to get config
    ap = argparse.ArgumentParser(description="Matrix0 Training Orchestrator - Flexible command-line configuration")

    # Core configuration
    ap.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")

    # Parse just the config argument first
    temp_args, _ = ap.parse_known_args()

    # Set MPS memory limits BEFORE any torch imports
    try:
        cfg = Config.load(temp_args.config)
        import os

        # Get memory limit from config and set appropriate ratios
        memory_limit_gb = cfg.training().get('memory_limit_gb', 12)
        memory_ratio = min(memory_limit_gb / 18.0, 0.9)  # Cap at 90% to be safe

        # Set environment variables before PyTorch initializes
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(memory_ratio)
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = str(memory_ratio * 0.8)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        print(f"✅ Set MPS memory limit to {memory_limit_gb}GB (ratio: {memory_ratio:.2f})")
    except Exception as e:
        print(f"⚠️  Could not set memory limits: {e}")

    # Force spawn start method for torch/multiprocessing compatibility on macOS
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Can only be set once
    
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
    ap.add_argument("--tui", type=str, default=None, choices=["bars", "table"], help="Progress UI: bars or table")
    ap.add_argument("--no-shared-infer", action="store_true", help="Disable shared inference server (debug/compat)")
    ap.add_argument("--quick-start", action="store_true", help="Force quick start mode (uses initial_games regardless of checkpoints)")
    
    args = ap.parse_args()
    
    # Enforce strict encoding via env if configured
    try:
        import os as _os
        cfg = Config.load(args.config)
        if bool(cfg.get("strict_encoding", False)):
            _os.environ["MATRIX0_STRICT_ENCODING"] = "1"
    except Exception:
        pass
    
    # Determine TUI mode: CLI overrides config; if CLI not provided, use config default
    tui_cfg = Config.load(args.config).orchestrator().get("tui", "bars")
    chosen_tui = args.tui or tui_cfg
    overrides = OrchestratorOverrides(
        games=args.games,
        eval_games=args.eval_games,
        workers=args.workers,
        sims=args.sims,
        cpuct=args.cpuct,
        dirichlet_alpha=args.dirichlet_alpha,
        selection_jitter=args.selection_jitter,
        opening_plies=args.opening_plies,
        resign_threshold=args.resign_threshold,
        max_game_length=args.max_game_length,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        accum_steps=args.accum_steps,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        promotion_threshold=args.promotion_threshold,
        device=args.device,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        doctor_fix=args.doctor_fix or None,
        seed=args.seed,
        no_shared_infer=True if args.no_shared_infer else None,
    )

    orchestrate(
        args.config,
        overrides=overrides,
        tui_mode=chosen_tui,
        quick_start=bool(args.quick_start),
    )

def _register_external_npz_dirs(cfg: Config, dm: DataManager) -> int:
    """Register NPZ shards in the DB from common replay directories.

    Returns number of directories successfully imported.
    """
    count_dirs = 0
    data_root = Path(cfg.get('data_dir', 'data'))
    dirs = [
        Path(cfg.training().get('replay_dir', 'data/replays')),
        data_root / 'replays',
        data_root / 'openings',
        data_root / 'training',
        data_root / 'tactical',
    ]
    # Deduplicate while preserving order
    seen = set()
    uniq_dirs = []
    for d in dirs + [Path(p) for p in (cfg.training().get('extra_replay_dirs', []) or [])]:
        try:
            rp = d.resolve()
        except Exception:
            rp = d
        if rp in seen:
            continue
        seen.add(rp)
        uniq_dirs.append(d)
    
    # Log external training data availability
    try:
        external_stats = dm.get_external_data_stats()
        if external_stats['external_total'] > 0:
            logger = setup_logging(cfg.training().get("log_dir", "logs"))
            logger.info(f"External training data available: {external_stats['tactical_samples']} tactical + {external_stats['openings_samples']} openings = {external_stats['external_total']} total samples")
            
            # Check curriculum configuration
            if cfg.training().get('use_curriculum', False):
                curriculum_phases = cfg.training().get('curriculum_phases', [])
                logger.info(f"Curriculum learning enabled with {len(curriculum_phases)} phases:")
                for phase in curriculum_phases:
                    logger.info(f"  - {phase['name']}: steps 0-{phase['steps']} ({phase.get('description', '')})")
    except Exception as e:
        # Non-critical, continue with directory registration
        pass
    
    # Filter out external training data directories to avoid import errors
    # These are handled separately by the enhanced DataManager
    external_training_dirs = [
        data_root / 'training',  # Contains our processed external data
        data_root / 'tactical',  # Contains our processed tactical data
        data_root / 'openings',  # Contains our processed openings data
    ]
    
    # Remove external training dirs from the import list
    uniq_dirs = [d for d in uniq_dirs if d not in external_training_dirs]
    
    for d in uniq_dirs:
        try:
            n = dm.import_replay_dir(str(d), source='external', move_files=False)
            if n:
                count_dirs += 1
        except Exception:
            continue
    return count_dirs


if __name__ == "__main__":
    main()
