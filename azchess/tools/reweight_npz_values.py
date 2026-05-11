from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOURCES = ("capped", "unfinished")


def _npz_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.npz") if not p.name.endswith(".tmp.npz"))


def _source_name(data: np.lib.npyio.NpzFile) -> str:
    if "meta_result_source" not in data:
        return "unknown"
    values = np.asarray(data["meta_result_source"]).reshape(-1)
    if values.size == 0:
        return "unknown"
    return str(values[0])


def _rewrite_shard(src: Path, dst: Path, sources: set[str], value_weight: float) -> dict[str, Any]:
    with np.load(src, allow_pickle=True) as data:
        source = _source_name(data)
        payload = {key: data[key] for key in data.files}
        changed = source in sources
        positions = int(payload["s"].shape[0]) if "s" in payload else 0

        old_value_weight_mean = None
        if "value_weight" in payload:
            old_value_weight_mean = float(np.asarray(payload["value_weight"], dtype=np.float32).mean())

        old_meta_value_weight = None
        if "meta_value_weight" in payload:
            meta_arr = np.asarray(payload["meta_value_weight"], dtype=np.float32).reshape(-1)
            if meta_arr.size:
                old_meta_value_weight = float(meta_arr[0])

        if changed:
            if "value_weight" in payload:
                payload["value_weight"] = np.full_like(
                    np.asarray(payload["value_weight"], dtype=np.float32),
                    float(value_weight),
                    dtype=np.float32,
                )
            if "meta_value_weight" in payload:
                payload["meta_value_weight"] = np.full_like(
                    np.asarray(payload["meta_value_weight"], dtype=np.float32),
                    float(value_weight),
                    dtype=np.float32,
                )
            if value_weight == 0.0 and "meta_value_bootstrap" in payload:
                payload["meta_value_bootstrap"] = np.zeros_like(np.asarray(payload["meta_value_bootstrap"]))

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(dst)

    return {
        "source": source,
        "positions": positions,
        "changed": changed,
        "old_value_weight_mean": old_value_weight_mean,
        "old_meta_value_weight": old_meta_value_weight,
        "new_value_weight": float(value_weight) if changed else old_value_weight_mean,
    }


def reweight_npz_tree(src_dir: Path, dst_dir: Path, sources: set[str], value_weight: float) -> dict[str, Any]:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    if src_dir == dst_dir:
        raise ValueError("Destination must be different from source")
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise FileExistsError(f"Destination exists and is not empty: {dst_dir}")

    files = _npz_files(src_dir)
    if not files:
        raise ValueError(f"No NPZ shards found under {src_dir}")

    report: dict[str, Any] = {
        "source_dir": str(src_dir),
        "output_dir": str(dst_dir),
        "sources": sorted(sources),
        "value_weight": float(value_weight),
        "files": 0,
        "changed_files": 0,
        "positions": 0,
        "changed_positions": 0,
        "by_source": {},
    }

    for src in files:
        rel = src.relative_to(src_dir)
        item = _rewrite_shard(src, dst_dir / rel, sources, value_weight)
        report["files"] += 1
        report["positions"] += int(item["positions"])
        if item["changed"]:
            report["changed_files"] += 1
            report["changed_positions"] += int(item["positions"])
        source_rec = report["by_source"].setdefault(
            item["source"],
            {"files": 0, "changed_files": 0, "positions": 0, "changed_positions": 0},
        )
        source_rec["files"] += 1
        source_rec["positions"] += int(item["positions"])
        if item["changed"]:
            source_rec["changed_files"] += 1
            source_rec["changed_positions"] += int(item["positions"])

    for src in src_dir.rglob("*"):
        if src.is_dir() or src.suffix == ".npz":
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    report_path = dst_dir / "reweight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy NPZ data while overriding value weights for selected result sources.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Source data directory containing NPZ shards")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination data directory to create")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Result source to reweight. May be repeated. Defaults to capped and unfinished.",
    )
    parser.add_argument("--value-weight", type=float, default=0.0, help="New value_weight for matching shards")
    args = parser.parse_args()

    sources = set(args.source or DEFAULT_SOURCES)
    report = reweight_npz_tree(args.input_dir, args.output_dir, sources, args.value_weight)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
