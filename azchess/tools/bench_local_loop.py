from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

from azchess.config import Config, select_device
from azchess.data_manager import DataManager
from azchess.model import PolicyValueNet
from azchess.utils.env_info import collect_environment_info


def _sync_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _run_stage(name: str, cmd: List[str], cwd: Path, env: Dict[str, str]) -> Dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    duration = time.perf_counter() - start
    rec = {
        "name": name,
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "seconds": duration,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with code {proc.returncode}\n{proc.stderr[-4000:]}")
    return rec


def _npz_files(data_dir: Path) -> List[Path]:
    roots = [data_dir / "selfplay", data_dir / "replays"]
    files: List[Path] = []
    for root in roots:
        files.extend(sorted(root.glob("*.npz")))
    files.extend(sorted((data_dir / "teacher_games").rglob("*.npz")))
    files.extend(sorted((data_dir / "stockfish_games").rglob("*.npz")))
    return files


def copy_anchor_shards(anchor_dirs: Iterable[str], run_data_dir: Path, max_files_per_dir: int = 0) -> Dict[str, Any]:
    """Copy stable training shards into the run replay buffer before training.

    Self-play generation still writes only fresh data. Anchors are copied after
    generation so the training stage sees both fresh and prior validated shards.
    """
    copied: List[str] = []
    replay_dir = run_data_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for idx, raw_dir in enumerate(anchor_dirs):
        source_dir = Path(raw_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Anchor data directory not found: {source_dir}")
        files = _npz_files(source_dir)
        if max_files_per_dir > 0:
            files = files[:max_files_per_dir]
        if not files:
            raise FileNotFoundError(f"No NPZ anchor shards found in: {source_dir}")
        for shard_idx, src in enumerate(files):
            dest = replay_dir / f"anchor{idx:02d}_{shard_idx:04d}_{src.name}"
            if dest.exists():
                raise FileExistsError(f"Anchor destination already exists: {dest}")
            shutil.copy2(src, dest)
            copied.append(str(dest))
    return {
        "dirs": [str(Path(d)) for d in anchor_dirs],
        "max_files_per_dir": int(max_files_per_dir),
        "copied_files": len(copied),
        "destination": str(replay_dir),
    }


def _entropy(policy: np.ndarray) -> np.ndarray:
    clipped = np.clip(policy, 1e-12, 1.0)
    return -np.sum(np.where(policy > 0, policy * np.log(clipped), 0.0), axis=1)


def summarize_npz_shards(data_dir: Path, max_files: int = 128) -> Dict[str, Any]:
    files = _npz_files(data_dir)[:max_files]
    total_samples = 0
    policy_sums: List[np.ndarray] = []
    policy_entropy: List[np.ndarray] = []
    policy_top_prob: List[np.ndarray] = []
    policy_support: List[np.ndarray] = []
    values: List[np.ndarray] = []
    legal_counts: List[np.ndarray] = []
    legal_policy_mass: List[np.ndarray] = []
    ssl_ranges: Dict[str, List[tuple[float, float]]] = {}
    meta_moves: List[float] = []
    meta_avg_sims: List[float] = []
    meta_results: List[float] = []
    meta_capped: List[float] = []
    meta_resigned: List[float] = []
    meta_terminal: List[float] = []
    meta_tablebase: List[float] = []
    meta_adjudicated_draw: List[float] = []
    meta_value_bootstrap: List[float] = []
    meta_value_weight: List[float] = []
    value_weights: List[np.ndarray] = []
    result_sources: Dict[str, int] = {}
    source_metrics: Dict[str, Dict[str, Any]] = {}

    for path in files:
        with np.load(path, mmap_mode="r") as data:
            if "s" not in data or "pi" not in data or "z" not in data:
                raise ValueError(f"Shard missing required s/pi/z arrays: {path}")
            pi = np.asarray(data["pi"], dtype=np.float32)
            z = np.asarray(data["z"], dtype=np.float32).reshape(-1)
            if pi.ndim != 2:
                raise ValueError(f"Policy array must be rank 2 in {path}, got {pi.shape}")
            total_samples += int(pi.shape[0])
            row_sum = pi.sum(axis=1)
            policy_sums.append(row_sum)
            normalized = pi / np.clip(row_sum[:, None], 1e-12, None)
            entropy = _entropy(normalized)
            policy_entropy.append(entropy)
            top_prob = np.max(normalized, axis=1)
            support = np.count_nonzero(normalized > 1e-6, axis=1).astype(np.float32)
            policy_top_prob.append(top_prob)
            policy_support.append(support)
            values.append(z)
            source = "unknown"
            if "meta_result_source" in data:
                raw_sources = np.asarray(data["meta_result_source"]).reshape(-1)
                if raw_sources.size:
                    source = str(raw_sources[0])
            source_rec = source_metrics.setdefault(
                source,
                {
                    "shards": 0,
                    "samples": 0,
                    "policy_entropy": [],
                    "policy_top_prob": [],
                    "policy_support": [],
                    "value": [],
                    "value_weight": [],
                    "legal_count": [],
                    "legal_policy_mass": [],
                    "moves": [],
                    "avg_sims": [],
                },
            )
            source_rec["shards"] += 1
            source_rec["samples"] += int(pi.shape[0])
            source_rec["policy_entropy"].append(entropy)
            source_rec["policy_top_prob"].append(top_prob)
            source_rec["policy_support"].append(support)
            source_rec["value"].append(z)
            if "legal_mask" in data:
                legal = np.asarray(data["legal_mask"], dtype=np.float32)
                if legal.shape != pi.shape:
                    raise ValueError(f"legal_mask shape {legal.shape} does not match pi {pi.shape} in {path}")
                legal_count = legal.sum(axis=1)
                legal_mass = (normalized * legal).sum(axis=1)
                legal_counts.append(legal_count)
                legal_policy_mass.append(legal_mass)
                source_rec["legal_count"].append(legal_count)
                source_rec["legal_policy_mass"].append(legal_mass)
            for key in data.files:
                if key.startswith("ssl_"):
                    arr = np.asarray(data[key])
                    ssl_ranges.setdefault(key, []).append((float(np.min(arr)), float(np.max(arr))))
            if "meta_moves" in data:
                moves_arr = np.asarray(data["meta_moves"]).reshape(-1)
                meta_moves.extend(float(v) for v in moves_arr)
                source_rec["moves"].extend(float(v) for v in moves_arr)
            if "meta_avg_sims" in data:
                sims_arr = np.asarray(data["meta_avg_sims"]).reshape(-1)
                meta_avg_sims.extend(float(v) for v in sims_arr)
                source_rec["avg_sims"].extend(float(v) for v in sims_arr)
            if "meta_result" in data:
                meta_results.extend(float(v) for v in np.asarray(data["meta_result"]).reshape(-1))
            if "meta_capped" in data:
                meta_capped.extend(float(v) for v in np.asarray(data["meta_capped"]).reshape(-1))
            if "meta_resigned" in data:
                meta_resigned.extend(float(v) for v in np.asarray(data["meta_resigned"]).reshape(-1))
            if "meta_terminal" in data:
                meta_terminal.extend(float(v) for v in np.asarray(data["meta_terminal"]).reshape(-1))
            if "meta_tablebase" in data:
                meta_tablebase.extend(float(v) for v in np.asarray(data["meta_tablebase"]).reshape(-1))
            if "meta_adjudicated_draw" in data:
                meta_adjudicated_draw.extend(float(v) for v in np.asarray(data["meta_adjudicated_draw"]).reshape(-1))
            if "meta_value_bootstrap" in data:
                meta_value_bootstrap.extend(float(v) for v in np.asarray(data["meta_value_bootstrap"]).reshape(-1))
            if "meta_value_weight" in data:
                meta_value_weight.extend(float(v) for v in np.asarray(data["meta_value_weight"]).reshape(-1))
            if "meta_result_source" in data:
                for raw in np.asarray(data["meta_result_source"]).reshape(-1):
                    source_name = str(raw)
                    result_sources[source_name] = result_sources.get(source_name, 0) + 1
            if "value_weight" in data:
                vw_arr = np.asarray(data["value_weight"], dtype=np.float32)
                value_weights.append(vw_arr)
                source_rec["value_weight"].append(vw_arr)

    def stats(arrays: Iterable[np.ndarray]) -> Optional[Dict[str, float]]:
        joined = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
        joined = [a for a in joined if a.size]
        if not joined:
            return None
        arr = np.concatenate(joined)
        return {
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    return {
        "shards": len(files),
        "samples": total_samples,
        "policy_row_sum": stats(policy_sums),
        "policy_entropy": stats(policy_entropy),
        "policy_top_prob": stats(policy_top_prob),
        "policy_support": stats(policy_support),
        "value": stats(values),
        "legal_count": stats(legal_counts),
        "legal_policy_mass": stats(legal_policy_mass),
        "ssl_ranges": {
            key: {
                "min": min(v[0] for v in ranges),
                "max": max(v[1] for v in ranges),
            }
            for key, ranges in sorted(ssl_ranges.items())
        },
        "game_moves": stats([np.asarray(meta_moves, dtype=np.float32)]) if meta_moves else None,
        "avg_sims": stats([np.asarray(meta_avg_sims, dtype=np.float32)]) if meta_avg_sims else None,
        "game_results": stats([np.asarray(meta_results, dtype=np.float32)]) if meta_results else None,
        "value_weight": stats(value_weights),
        "game_value_weight": stats([np.asarray(meta_value_weight, dtype=np.float32)]) if meta_value_weight else None,
        "game_outcomes": {
            "capped": int(np.sum(meta_capped)) if meta_capped else 0,
            "resigned": int(np.sum(meta_resigned)) if meta_resigned else 0,
            "terminal": int(np.sum(meta_terminal)) if meta_terminal else 0,
            "tablebase": int(np.sum(meta_tablebase)) if meta_tablebase else 0,
            "adjudicated_draw": int(np.sum(meta_adjudicated_draw)) if meta_adjudicated_draw else 0,
            "value_bootstrap": int(np.sum(meta_value_bootstrap)) if meta_value_bootstrap else 0,
            "result_sources": dict(sorted(result_sources.items())),
        },
        "source_metrics": {
            source: {
                "shards": int(rec["shards"]),
                "samples": int(rec["samples"]),
                "policy_entropy": stats(rec["policy_entropy"]),
                "policy_top_prob": stats(rec["policy_top_prob"]),
                "policy_support": stats(rec["policy_support"]),
                "value": stats(rec["value"]),
                "value_weight": stats(rec["value_weight"]),
                "legal_count": stats(rec["legal_count"]),
                "legal_policy_mass": stats(rec["legal_policy_mass"]),
                "moves": stats([np.asarray(rec["moves"], dtype=np.float32)]) if rec["moves"] else None,
                "avg_sims": stats([np.asarray(rec["avg_sims"], dtype=np.float32)]) if rec["avg_sims"] else None,
            }
            for source, rec in sorted(source_metrics.items())
        },
    }


def _save_initial_checkpoint(cfg: Config, path: Path, device: str) -> None:
    model = PolicyValueNet.from_config(cfg.model()).to(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 0,
            "global_step": 0,
            "model": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "version": "local-loop-init",
        },
        path,
    )


def _prepare_initial_checkpoint(cfg: Config, path: Path, device: str, init_checkpoint: Optional[str]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if init_checkpoint:
        source = Path(init_checkpoint)
        if not source.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {source}")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        _state_dict_from_checkpoint(checkpoint, source)
        shutil.copy2(source, path)
        return {"mode": "provided", "source": str(source), "path": str(path)}
    _save_initial_checkpoint(cfg, path, device)
    return {"mode": "fresh", "source": None, "path": str(path)}


def _state_dict_from_checkpoint(checkpoint: Dict[str, Any], path: Path) -> Dict[str, torch.Tensor]:
    for key in ("model_ema", "model", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError(f"No model state dict found in checkpoint: {path}")


def _sample_training_batch(data_dir: Path, batch_size: int) -> Dict[str, np.ndarray]:
    dm = DataManager(base_dir=str(data_dir))
    try:
        batch = next(dm.get_training_batch(batch_size=batch_size, device="cpu"))
    except RuntimeError:
        batch = next(
            dm.get_training_batch_by_source_prefixes(
                batch_size=batch_size,
                prefixes=["teacher:", "stockfish:", "external"],
            )
        )
    if isinstance(batch, dict):
        return batch
    if len(batch) >= 5:
        s, pi, z, legal, value_weight = batch[:5]
        out = {"s": s, "pi": pi, "z": z, "value_weight": value_weight}
        if legal is not None:
            out["legal_mask"] = legal
        return out
    if len(batch) == 4:
        s, pi, z, legal = batch
        return {"s": s, "pi": pi, "z": z, "legal_mask": legal}
    s, pi, z = batch
    return {"s": s, "pi": pi, "z": z}


def _sample_eval_batch(data_dir: Path, batch_size: int, source_prefixes: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    dm = DataManager(base_dir=str(data_dir))
    if source_prefixes:
        batch = next(dm.get_training_batch_by_source_prefixes(batch_size=batch_size, prefixes=source_prefixes))
        if isinstance(batch, dict):
            return batch
    return _sample_training_batch(data_dir, batch_size=batch_size)


def _summarize_metric_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("No metric records to summarize")
    summary: Dict[str, Any] = {"batches": len(records)}
    keys = sorted(set().union(*(record.keys() for record in records)))
    for key in keys:
        values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
        if len(values) == len(records):
            arr = np.asarray(values, dtype=np.float64)
            summary[key] = float(np.mean(arr))
            summary[f"{key}_std"] = float(np.std(arr))
    for key in ("checkpoint",):
        if key in records[0]:
            summary[key] = records[0][key]
    return summary


@torch.no_grad()
def _evaluate_loaded_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str,
    batch: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    x = torch.from_numpy(np.asarray(batch["s"], dtype=np.float32)).to(device)
    pi_target = np.asarray(batch["pi"], dtype=np.float32)
    row_sum = pi_target.sum(axis=1, keepdims=True)
    pi_target = pi_target / np.clip(row_sum, 1e-12, None)
    z_target = np.asarray(batch["z"], dtype=np.float32).reshape(-1)

    start = time.perf_counter()
    policy_logits, value_pred = model(x, return_ssl=False)
    _sync_device(device)
    seconds = time.perf_counter() - start

    log_probs = torch.log_softmax(policy_logits, dim=1).detach().cpu().numpy()
    probs = np.exp(log_probs)
    value = value_pred.detach().cpu().numpy().reshape(-1)
    ce = -np.sum(pi_target * log_probs, axis=1)
    target_entropy = _entropy(pi_target)
    top1_match = np.argmax(probs, axis=1) == np.argmax(pi_target, axis=1)
    rec: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "batch_size": int(x.shape[0]),
        "seconds": seconds,
        "positions_per_second": float(x.shape[0] / max(seconds, 1e-9)),
        "policy_ce": float(np.mean(ce)),
        "policy_kl": float(np.mean(ce - target_entropy)),
        "policy_top1_match": float(np.mean(top1_match)),
        "target_entropy": float(np.mean(target_entropy)),
        "pred_entropy": float(np.mean(_entropy(probs))),
        "value_mse": float(np.mean((value - z_target) ** 2)),
        "value_pred_mean": float(np.mean(value)),
        "value_target_mean": float(np.mean(z_target)),
    }
    if "legal_mask" in batch:
        legal = np.asarray(batch["legal_mask"], dtype=np.float32)
        if legal.shape == probs.shape:
            rec["legal_policy_mass"] = float(np.mean(np.sum(probs * legal, axis=1)))
            masked_logits = policy_logits.masked_fill(torch.from_numpy(legal <= 0).to(device), -1e9)
            masked_log_probs = torch.log_softmax(masked_logits, dim=1).detach().cpu().numpy()
            masked_probs = np.exp(masked_log_probs)
            legal_ce = -np.sum(pi_target * masked_log_probs, axis=1)
            rec["policy_legal_ce"] = float(np.mean(legal_ce))
            rec["policy_legal_kl"] = float(np.mean(legal_ce - target_entropy))
            rec["policy_legal_top1_match"] = float(np.mean(np.argmax(masked_probs, axis=1) == np.argmax(pi_target, axis=1)))
            rec["legal_pred_entropy"] = float(np.mean(_entropy(masked_probs)))
    return rec


def _load_eval_model(checkpoint_path: Path, cfg: Config, device: str) -> torch.nn.Module:
    model = PolicyValueNet.from_config(cfg.model()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = model.load_state_dict(_state_dict_from_checkpoint(checkpoint, checkpoint_path), strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if (missing or unexpected) and os.environ.get("MATRIX0_STRICT_CHECKPOINT") == "1":
        raise RuntimeError(
            f"Checkpoint did not match model exactly for eval: {checkpoint_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    model.eval()
    return model


@torch.no_grad()
def evaluate_checkpoint_batches(
    checkpoint_path: Path,
    cfg: Config,
    data_dir: Path,
    device: str,
    batch_size: int,
    batches: int,
    source_prefixes: Optional[List[str]] = None,
    fixed_batches: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    count = max(1, int(batches))
    model = _load_eval_model(checkpoint_path, cfg, device)
    records: List[Dict[str, Any]] = []
    for idx in range(count):
        if fixed_batches is not None:
            batch = fixed_batches[idx]
        else:
            batch = _sample_eval_batch(data_dir, batch_size=batch_size, source_prefixes=source_prefixes)
        records.append(_evaluate_loaded_model(model, checkpoint_path, device, batch))
    return _summarize_metric_records(records)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: Config,
    data_dir: Path,
    device: str,
    batch_size: int,
    batch: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    if batch is None:
        batch = _sample_training_batch(data_dir, batch_size=batch_size)
    model = _load_eval_model(checkpoint_path, cfg, device)
    return _evaluate_loaded_model(model, checkpoint_path, device, batch)


def write_loop_config(base_cfg: Config, run_dir: Path, args: argparse.Namespace) -> Path:
    raw = deepcopy(base_cfg.to_dict())
    data_dir = run_dir / "data"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    raw["data_dir"] = str(data_dir)
    raw.setdefault("selfplay", {})
    raw["selfplay"].update(
        {
            "num_workers": int(args.workers),
            "max_games": int(args.games),
            "max_game_len": int(args.max_game_len),
            "num_simulations": int(args.sims),
            "capped_value_weight": float(args.capped_value_weight),
            "min_resign_plies": int(args.min_resign_plies)
            if args.min_resign_plies is not None
            else int(args.max_game_len + 1),
            "resign_threshold": float(args.resign_threshold) if args.resign_threshold is not None else -1.0,
        }
    )
    for arg_name, cfg_name, caster in (
        ("temperature_start", "temperature_start", float),
        ("temperature_end", "temperature_end", float),
        ("temperature_moves", "temperature_moves", int),
        ("opening_random_plies", "opening_random_plies", int),
        ("low_visit_threshold", "low_visit_threshold", int),
        ("selection_jitter", "selection_jitter", float),
        ("fpu", "fpu", float),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            raw["selfplay"][cfg_name] = caster(value)
    if args.resign_consecutive_bad is not None:
        raw["selfplay"]["resign_consecutive_bad"] = int(args.resign_consecutive_bad)
    if args.resign_window is not None:
        raw["selfplay"]["resign_window"] = int(args.resign_window)
    if args.resign_value_margin is not None:
        raw["selfplay"]["resign_value_margin"] = float(args.resign_value_margin)
    if args.resign_min_entropy is not None:
        raw["selfplay"]["resign_min_entropy"] = float(args.resign_min_entropy)
    if (
        args.draw_halfmove_cap is not None
        or args.draw_material_threshold is not None
        or args.draw_min_plies is not None
        or args.draw_window is not None
        or args.draw_min_unique is not None
    ):
        draw_cfg = dict(raw["selfplay"].get("draw", {}) or {})
        draw_cfg["enabled"] = True
        if args.draw_halfmove_cap is not None:
            draw_cfg["halfmove_cap"] = int(args.draw_halfmove_cap)
        if args.draw_material_threshold is not None:
            draw_cfg["material_draw_threshold"] = int(args.draw_material_threshold)
        if args.draw_min_plies is not None:
            draw_cfg["min_plies"] = int(args.draw_min_plies)
        if args.draw_window is not None:
            draw_cfg["window"] = int(args.draw_window)
        if args.draw_min_unique is not None:
            draw_cfg["min_unique"] = int(args.draw_min_unique)
        raw["selfplay"]["draw"] = draw_cfg
    raw.setdefault("mcts", {})
    raw["mcts"].update(
        {
            "num_simulations": int(args.sims),
            "inference_batch_size": int(args.inference_batch_size),
            "simulation_batch_size": int(args.inference_batch_size),
            "parallel_simulations": True,
        }
    )
    for arg_name, cfg_name, caster in (
        ("cpuct", "cpuct", float),
        ("cpuct_start", "cpuct_start", float),
        ("cpuct_end", "cpuct_end", float),
        ("cpuct_plies", "cpuct_plies", int),
        ("dirichlet_alpha", "dirichlet_alpha", float),
        ("dirichlet_frac", "dirichlet_frac", float),
        ("dirichlet_plies", "dirichlet_plies", int),
        ("selection_jitter", "selection_jitter", float),
        ("fpu", "fpu", float),
        ("fpu_reduction", "fpu_reduction", float),
        ("draw_penalty", "draw_penalty", float),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            raw["mcts"][cfg_name] = caster(value)
    if getattr(args, "disable_entropy_noise", False):
        raw["mcts"]["enable_entropy_noise"] = False
    raw.setdefault("training", {})
    raw["training"].update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "warmup_steps": int(args.warmup_steps),
            "steps_per_epoch": int(args.train_steps),
            "epochs": 1,
            "checkpoint_prefix": "local_loop",
            "checkpoint_save_freq": max(int(args.train_steps) + 1, 2),
            "use_curriculum": False,
            "dataloader_workers": int(args.dataloader_workers),
            "legal_mass_weight": float(args.legal_mass_weight),
        }
    )
    if args.ssl_weight is not None:
        raw["training"]["ssl_weight"] = float(args.ssl_weight)
    if args.policy_label_smoothing is not None:
        raw["training"]["policy_label_smoothing"] = float(args.policy_label_smoothing)
    raw["checkpoint_prefix"] = "local_loop"
    raw["checkpoint_save_freq"] = max(int(args.train_steps) + 1, 2)
    path = run_dir / "local_loop_config.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return path


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = [path for path in checkpoint_dir.glob("*.pt") if path.name != "local_loop_init.pt"]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No trained checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def run_local_loop(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path.cwd()
    base_cfg = Config.load(args.config)
    device = select_device(args.device if args.device else base_cfg.get("device", "auto"))
    run_dir = Path(args.run_dir) if args.run_dir else Path("logs/local_loop") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_loop_config(base_cfg, run_dir, args)
    loop_cfg = Config.load(str(config_path))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["MATRIX0_STRICT_DATA"] = "1"
    env["MATRIX0_STRICT_CHECKPOINT"] = "1"
    os.environ["MATRIX0_STRICT_DATA"] = "1"
    os.environ["MATRIX0_STRICT_CHECKPOINT"] = "1"
    if args.mps_target_batch:
        env["MATRIX0_MPS_TARGET_BATCH"] = str(args.mps_target_batch)

    initial_ckpt = run_dir / "checkpoints" / "local_loop_init.pt"
    initial_checkpoint_info = _prepare_initial_checkpoint(loop_cfg, initial_ckpt, device, args.init_checkpoint)

    stages: List[Dict[str, Any]] = []
    selfplay_cmd = [
        sys.executable,
        "-m",
        "azchess.selfplay",
        "--config",
        str(config_path),
        "--ckpt",
        str(initial_ckpt),
        "--workers",
        str(args.workers),
        "--games",
        str(args.games),
    ]
    stages.append(_run_stage("selfplay", selfplay_cmd, repo, env))

    data_after_selfplay = summarize_npz_shards(run_dir / "data")

    anchor_info: Optional[Dict[str, Any]] = None
    if args.train_anchor_data_dir:
        anchor_info = copy_anchor_shards(
            args.train_anchor_data_dir,
            run_dir / "data",
            max_files_per_dir=int(args.train_anchor_max_files),
        )

    data_metrics_before = summarize_npz_shards(run_dir / "data")
    eval_data_dir = Path(args.eval_data_dir) if args.eval_data_dir else run_dir / "data"
    eval_prefixes = args.eval_source_prefix if args.eval_source_prefix else None
    eval_batch_size = min(args.eval_batch_size, args.batch_size)
    eval_batches = [
        _sample_eval_batch(eval_data_dir, batch_size=eval_batch_size, source_prefixes=eval_prefixes)
        for _ in range(max(1, int(args.eval_batches)))
    ]
    eval_before = evaluate_checkpoint_batches(
        initial_ckpt,
        loop_cfg,
        eval_data_dir,
        device,
        eval_batch_size,
        batches=len(eval_batches),
        fixed_batches=eval_batches,
    )

    if args.skip_train:
        final_ckpt = initial_ckpt
        data_metrics_after = data_metrics_before
        eval_after = eval_before
        train_steps_per_second: Optional[float] = None
    else:
        train_cmd = [
            sys.executable,
            "-m",
            "azchess.training.train",
            "--config",
            str(config_path),
            "--steps",
            str(args.train_steps),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--warmup-steps",
            str(args.warmup_steps),
            "--checkpoint-dir",
            str(run_dir / "checkpoints"),
            "--log-dir",
            str(run_dir / "logs"),
            "--device",
            device,
            "--dataloader-workers",
            str(args.dataloader_workers),
        ]
        if args.no_amp:
            train_cmd.append("--no-amp")
        stages.append(_run_stage("train", train_cmd, repo, env))

        final_ckpt = latest_checkpoint(run_dir / "checkpoints")
        data_metrics_after = summarize_npz_shards(run_dir / "data")
        eval_after = evaluate_checkpoint_batches(
            final_ckpt,
            loop_cfg,
            eval_data_dir,
            device,
            eval_batch_size,
            batches=len(eval_batches),
            fixed_batches=eval_batches,
        )
        train_seconds = stages[-1]["seconds"]
        train_steps_per_second = float(args.train_steps / max(train_seconds, 1e-9))

    report = {
        "type": "matrix0_local_loop_benchmark",
        "timestamp": datetime.now().isoformat(),
        "repo": str(repo),
        "run_dir": str(run_dir),
        "config": str(config_path),
        "platform": platform.platform(),
        "device": device,
        "env": collect_environment_info(device).to_dict(),
        "args": vars(args),
        "eval_data_dir": str(eval_data_dir),
        "eval_source_prefix": eval_prefixes,
        "checkpoints": {
            "initial": str(initial_ckpt),
            "initial_info": initial_checkpoint_info,
            "final": str(final_ckpt),
        },
        "stages": stages,
        "anchor_data": anchor_info,
        "throughput": {
            "selfplay_games_per_hour": float(args.games / max(stages[0]["seconds"], 1e-9) * 3600.0),
            "train_steps_per_second": train_steps_per_second,
        },
        "data_after_selfplay": data_after_selfplay,
        "data_before_train": data_metrics_before,
        "data_after_train": data_metrics_after,
        "eval": {
            "before": eval_before,
            "after": eval_after,
            "delta": {
                key: float(eval_after[key] - eval_before[key])
                for key in (
                    "policy_ce",
                    "policy_kl",
                    "policy_legal_ce",
                    "policy_legal_kl",
                    "value_mse",
                    "legal_policy_mass",
                )
                if key in eval_before and key in eval_after
            },
        },
    }
    out_path = Path(args.output) if args.output else run_dir / "local_loop_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    report["output"] = str(out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Matrix0 self-play -> train -> eval loop and write JSON metrics.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to seed self-play/training instead of fresh init.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sims", type=int, default=1)
    parser.add_argument("--max-game-len", type=int, default=16)
    parser.add_argument("--capped-value-weight", type=float, default=0.25)
    parser.add_argument("--min-resign-plies", type=int, default=None)
    parser.add_argument("--resign-threshold", type=float, default=None)
    parser.add_argument("--resign-consecutive-bad", type=int, default=None)
    parser.add_argument("--resign-window", type=int, default=None)
    parser.add_argument("--resign-value-margin", type=float, default=None)
    parser.add_argument("--resign-min-entropy", type=float, default=None)
    parser.add_argument("--draw-halfmove-cap", type=int, default=None)
    parser.add_argument("--draw-material-threshold", type=int, default=None)
    parser.add_argument("--draw-min-plies", type=int, default=None)
    parser.add_argument("--draw-window", type=int, default=None)
    parser.add_argument("--draw-min-unique", type=int, default=None)
    parser.add_argument("--cpuct", type=float, default=None)
    parser.add_argument("--cpuct-start", type=float, default=None)
    parser.add_argument("--cpuct-end", type=float, default=None)
    parser.add_argument("--cpuct-plies", type=int, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--dirichlet-frac", type=float, default=None)
    parser.add_argument("--dirichlet-plies", type=int, default=None)
    parser.add_argument("--selection-jitter", type=float, default=None)
    parser.add_argument("--disable-entropy-noise", action="store_true")
    parser.add_argument("--fpu", type=float, default=None)
    parser.add_argument("--fpu-reduction", type=float, default=None)
    parser.add_argument("--draw-penalty", type=float, default=None)
    parser.add_argument("--temperature-start", type=float, default=None)
    parser.add_argument("--temperature-end", type=float, default=None)
    parser.add_argument("--temperature-moves", type=int, default=None)
    parser.add_argument("--opening-random-plies", type=int, default=None)
    parser.add_argument("--low-visit-threshold", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--mps-target-batch", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--eval-data-dir", default=None, help="Optional stable data directory for before/after checkpoint eval.")
    parser.add_argument("--eval-source-prefix", action="append", default=[], help="Optional source prefix filter for eval data. Repeatable.")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--legal-mass-weight", type=float, default=0.05)
    parser.add_argument("--ssl-weight", type=float, default=None)
    parser.add_argument("--policy-label-smoothing", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only generate self-play data and report shard/generator metrics.")
    parser.add_argument(
        "--train-anchor-data-dir",
        action="append",
        default=[],
        help="Copy NPZ shards from this data directory into the run replay buffer after self-play and before training. Repeatable.",
    )
    parser.add_argument("--train-anchor-max-files", type=int, default=0, help="Optional max anchor shards to copy from each anchor directory; 0 means all.")
    args = parser.parse_args()
    report = run_local_loop(args)
    data = report["data_after_train"]
    fresh_data = report["data_after_selfplay"]
    def quality_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "avg_sims": metrics.get("avg_sims"),
            "policy_entropy": metrics.get("policy_entropy"),
            "policy_top_prob": metrics.get("policy_top_prob"),
            "policy_support": metrics.get("policy_support"),
            "legal_count": metrics.get("legal_count"),
            "legal_policy_mass": metrics.get("legal_policy_mass"),
            "source_metrics": metrics.get("source_metrics"),
        }

    print(
        json.dumps(
            {
                "output": report["output"],
                "throughput": report["throughput"],
                "eval_delta": report["eval"]["delta"],
                "fresh_data": {
                    "samples": fresh_data["samples"],
                    "shards": fresh_data["shards"],
                    "game_outcomes": fresh_data["game_outcomes"],
                    "value_weight": fresh_data["value_weight"],
                    "game_value_weight": fresh_data["game_value_weight"],
                    "quality": quality_summary(fresh_data),
                },
                "data": {
                    "samples": data["samples"],
                    "shards": data["shards"],
                    "game_outcomes": data["game_outcomes"],
                    "value_weight": data["value_weight"],
                    "game_value_weight": data["game_value_weight"],
                    "quality": quality_summary(data),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
