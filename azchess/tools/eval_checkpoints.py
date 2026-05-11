from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from azchess.config import Config, select_device
from azchess.tools.bench_local_loop import (
    _delta_source_metrics,
    _sample_eval_batch,
    evaluate_checkpoint_batches,
    summarize_npz_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Matrix0 checkpoints on local replay/self-play data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--source-prefix", action="append", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = select_device(args.device if args.device else cfg.get("device", "auto"))
    data_dir = Path(args.data_dir)
    source_prefixes = args.source_prefix if args.source_prefix else None
    fixed_batches = [
        _sample_eval_batch(data_dir, batch_size=args.batch_size, source_prefixes=source_prefixes)
        for _ in range(max(1, int(args.batches)))
    ]
    metrics_a = evaluate_checkpoint_batches(
        Path(args.model_a),
        cfg,
        data_dir,
        device,
        args.batch_size,
        batches=len(fixed_batches),
        fixed_batches=fixed_batches,
    )
    metrics_b = evaluate_checkpoint_batches(
        Path(args.model_b),
        cfg,
        data_dir,
        device,
        args.batch_size,
        batches=len(fixed_batches),
        fixed_batches=fixed_batches,
    )
    numeric_keys = sorted(set(metrics_a) & set(metrics_b))
    delta = {
        key: float(metrics_a[key] - metrics_b[key])
        for key in numeric_keys
        if isinstance(metrics_a[key], (int, float)) and isinstance(metrics_b[key], (int, float))
    }
    report = {
        "type": "matrix0_checkpoint_eval",
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "data_dir": str(data_dir),
        "device": device,
        "batch_size": int(args.batch_size),
        "batches": int(len(fixed_batches)),
        "source_prefix": source_prefixes,
        "data": summarize_npz_shards(data_dir),
        "model_a": metrics_a,
        "model_b": metrics_b,
        "delta_a_minus_b": delta,
        "source_delta_a_minus_b": _delta_source_metrics(metrics_b, metrics_a),
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
