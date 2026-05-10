from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from azchess.config import Config, select_device
from azchess.tools.bench_local_loop import _load_eval_model, _sample_eval_batch, summarize_npz_shards


def _entropy(policy: np.ndarray) -> np.ndarray:
    clipped = np.clip(policy, 1e-12, 1.0)
    return -np.sum(np.where(policy > 0, policy * np.log(clipped), 0.0), axis=1)


def _bucket_label(value: float, bins: List[float]) -> str:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            return f"[{lo:g},{hi:g})"
    return f">={bins[-1]:g}"


def _rank_of_target(probs: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    ranks = np.empty(probs.shape[0], dtype=np.int32)
    for row, idx in enumerate(target_idx):
        ranks[row] = int(np.where(order[row] == idx)[0][0]) + 1
    return ranks


@torch.no_grad()
def _model_position_metrics(model: torch.nn.Module, device: str, batch: Dict[str, np.ndarray], pi: np.ndarray) -> Dict[str, np.ndarray]:
    x = torch.from_numpy(np.asarray(batch["s"], dtype=np.float32)).to(device)
    logits, value_pred = model(x, return_ssl=False)
    log_probs = torch.log_softmax(logits, dim=1).detach().cpu().numpy()
    probs = np.exp(log_probs)

    target_idx = np.argmax(pi, axis=1)
    ce = -np.sum(pi * log_probs, axis=1)
    top_idx = np.argmax(probs, axis=1)

    out: Dict[str, np.ndarray] = {
        "ce": ce,
        "top_idx": top_idx,
        "top1": top_idx == target_idx,
        "pred_entropy": _entropy(probs),
        "target_top_prob_pred": probs[np.arange(probs.shape[0]), target_idx],
        "target_rank": _rank_of_target(probs, target_idx),
        "pred_top_prob": np.max(probs, axis=1),
        "value": value_pred.detach().cpu().numpy().reshape(-1),
    }

    if "legal_mask" in batch:
        legal = np.asarray(batch["legal_mask"], dtype=np.float32)
        if legal.shape == probs.shape:
            mask = torch.from_numpy(legal <= 0).to(device)
            masked_logits = logits.masked_fill(mask, -1e9)
            masked_log_probs = torch.log_softmax(masked_logits, dim=1).detach().cpu().numpy()
            masked_probs = np.exp(masked_log_probs)
            legal_ce = -np.sum(pi * masked_log_probs, axis=1)
            legal_top_idx = np.argmax(masked_probs, axis=1)
            out.update(
                {
                    "legal_ce": legal_ce,
                    "legal_top_idx": legal_top_idx,
                    "legal_top1": legal_top_idx == target_idx,
                    "legal_pred_entropy": _entropy(masked_probs),
                    "legal_mass": np.sum(probs * legal, axis=1),
                    "legal_target_top_prob_pred": masked_probs[np.arange(masked_probs.shape[0]), target_idx],
                    "legal_target_rank": _rank_of_target(masked_probs, target_idx),
                    "legal_pred_top_prob": np.max(masked_probs, axis=1),
                }
            )
    return out


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _summarize_subset(name: str, mask: np.ndarray, target: Dict[str, np.ndarray], a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = int(np.sum(mask))
    rec: Dict[str, Any] = {"bucket": name, "positions": n}
    if n == 0:
        return rec

    key_ce = "legal_ce" if "legal_ce" in a and "legal_ce" in b else "ce"
    key_top1 = "legal_top1" if "legal_top1" in a and "legal_top1" in b else "top1"
    key_rank = "legal_target_rank" if "legal_target_rank" in a and "legal_target_rank" in b else "target_rank"
    key_target_prob = "legal_target_top_prob_pred" if "legal_target_top_prob_pred" in a and "legal_target_top_prob_pred" in b else "target_top_prob_pred"
    key_pred_entropy = "legal_pred_entropy" if "legal_pred_entropy" in a and "legal_pred_entropy" in b else "pred_entropy"
    key_top_idx = "legal_top_idx" if "legal_top_idx" in a and "legal_top_idx" in b else "top_idx"

    a_ce = a[key_ce][mask]
    b_ce = b[key_ce][mask]
    a_top = a[key_top1][mask]
    b_top = b[key_top1][mask]
    ce_better = a_ce < b_ce
    top1_worse = (~a_top) & b_top
    top1_better = a_top & (~b_top)

    rec.update(
        {
            "target_entropy": _mean(target["entropy"][mask]),
            "target_top_prob": _mean(target["top_prob"][mask]),
            "legal_count": _mean(target["legal_count"][mask]),
            "a_ce": _mean(a_ce),
            "b_ce": _mean(b_ce),
            "delta_ce_a_minus_b": _mean(a_ce - b_ce),
            "a_top1": _mean(a_top.astype(np.float32)),
            "b_top1": _mean(b_top.astype(np.float32)),
            "delta_top1_a_minus_b": _mean(a_top.astype(np.float32) - b_top.astype(np.float32)),
            "a_target_prob": _mean(a[key_target_prob][mask]),
            "b_target_prob": _mean(b[key_target_prob][mask]),
            "a_target_rank": _mean(a[key_rank][mask].astype(np.float32)),
            "b_target_rank": _mean(b[key_rank][mask].astype(np.float32)),
            "a_pred_entropy": _mean(a[key_pred_entropy][mask]),
            "b_pred_entropy": _mean(b[key_pred_entropy][mask]),
            "ce_better_rate": _mean(ce_better.astype(np.float32)),
            "top1_worse_rate": _mean(top1_worse.astype(np.float32)),
            "top1_better_rate": _mean(top1_better.astype(np.float32)),
            "ce_better_top1_worse_rate": _mean((ce_better & top1_worse).astype(np.float32)),
            "model_top_agreement": _mean((a[key_top_idx][mask] == b[key_top_idx][mask]).astype(np.float32)),
        }
    )
    if "legal_mass" in a and "legal_mass" in b:
        rec["a_legal_mass"] = _mean(a["legal_mass"][mask])
        rec["b_legal_mass"] = _mean(b["legal_mass"][mask])
    return rec


def _bucket_summaries(target: Dict[str, np.ndarray], a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = target["entropy"].shape[0]
    all_mask = np.ones(n, dtype=bool)
    report: Dict[str, Any] = {"overall": _summarize_subset("all", all_mask, target, a, b)}

    top_bins = [0.0, 0.15, 0.25, 0.4, 0.6, 0.8, 1.000001]
    entropy_bins = [0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 10.0]
    legal_bins = [0.0, 8.0, 16.0, 32.0, 48.0, 80.0]

    for name, values, bins in (
        ("by_target_top_prob", target["top_prob"], top_bins),
        ("by_target_entropy", target["entropy"], entropy_bins),
        ("by_legal_count", target["legal_count"], legal_bins),
    ):
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            rows.append(_summarize_subset(_bucket_label((lo + hi) / 2.0, bins), (values >= lo) & (values < hi), target, a, b))
        report[name] = rows
    return report


def _concat(parts: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[np.ndarray]] = {}
    for part in parts:
        for key, value in part.items():
            grouped.setdefault(key, []).append(np.asarray(value))
    return {key: np.concatenate(values, axis=0) for key, values in grouped.items()}


def run_diagnostics(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = Config.load(args.config)
    device = select_device(args.device if args.device else cfg.get("device", "auto"))
    data_dir = Path(args.data_dir)
    source_prefixes = args.source_prefix if args.source_prefix else None

    model_a = _load_eval_model(Path(args.model_a), cfg, device)
    model_b = _load_eval_model(Path(args.model_b), cfg, device)

    target_parts: List[Dict[str, np.ndarray]] = []
    a_parts: List[Dict[str, np.ndarray]] = []
    b_parts: List[Dict[str, np.ndarray]] = []
    z_parts: List[np.ndarray] = []
    for _ in range(max(1, int(args.batches))):
        batch = _sample_eval_batch(data_dir, batch_size=args.batch_size, source_prefixes=source_prefixes)
        pi = np.asarray(batch["pi"], dtype=np.float32)
        pi = pi / np.clip(pi.sum(axis=1, keepdims=True), 1e-12, None)
        target_idx = np.argmax(pi, axis=1)
        legal_count = (
            np.asarray(batch["legal_mask"], dtype=np.float32).sum(axis=1)
            if "legal_mask" in batch
            else np.full((pi.shape[0],), np.nan, dtype=np.float32)
        )
        target_parts.append(
            {
                "entropy": _entropy(pi),
                "top_prob": pi[np.arange(pi.shape[0]), target_idx],
                "legal_count": legal_count.astype(np.float32),
                "top_idx": target_idx.astype(np.int32),
            }
        )
        a_parts.append(_model_position_metrics(model_a, device, batch, pi))
        b_parts.append(_model_position_metrics(model_b, device, batch, pi))
        z_parts.append(np.asarray(batch["z"], dtype=np.float32).reshape(-1))

    target = _concat(target_parts)
    a = _concat(a_parts)
    b = _concat(b_parts)
    z = np.concatenate(z_parts, axis=0)
    report = {
        "type": "matrix0_policy_target_diagnostics",
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "data_dir": str(data_dir),
        "source_prefix": source_prefixes,
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "batch_size": int(args.batch_size),
        "batches": int(args.batches),
        "positions": int(target["entropy"].shape[0]),
        "data": summarize_npz_shards(data_dir),
        "buckets": _bucket_summaries(target, a, b),
        "value": {
            "a_mse": float(np.mean((a["value"] - z) ** 2)),
            "b_mse": float(np.mean((b["value"] - z) ** 2)),
            "delta_mse_a_minus_b": float(np.mean((a["value"] - z) ** 2) - np.mean((b["value"] - z) ** 2)),
            "target_mean": float(np.mean(z)),
            "a_mean": float(np.mean(a["value"])),
            "b_mean": float(np.mean(b["value"])),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose per-position policy CE/top-1 tradeoffs between two checkpoints.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--source-prefix", action="append", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_diagnostics(args)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
