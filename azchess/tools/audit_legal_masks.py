from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def audit_file(path: Path) -> dict:
    try:
        with np.load(path, mmap_mode='r') as data:
            n = int(len(data['s']))
            has_mask = 'legal_mask' in data
            if not has_mask:
                return {'file': str(path), 'samples': n, 'has_mask': False}
            lm = data['legal_mask']
            # Normalize shapes: (N,4672)
            if lm.ndim == 3 and lm.shape[1:] == (8, 8, 73):
                lm = lm.reshape(lm.shape[0], -1)
            legal_counts = lm.sum(axis=1)
            return {
                'file': str(path),
                'samples': n,
                'has_mask': True,
                'legal_mean': float(legal_counts.mean()),
                'legal_min': int(legal_counts.min()),
                'legal_max': int(legal_counts.max()),
                'legal_ratio': float(lm.sum() / max(1, lm.size)),
            }
    except Exception as e:
        return {'file': str(path), 'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description="Audit legal_mask presence and stats in NPZ shards")
    ap.add_argument('--dir', type=str, default='data', help='Base data directory')
    ap.add_argument('--subset', type=str, choices=['selfplay', 'replays'], default='selfplay', help='Subset to audit')
    ap.add_argument('--limit', type=int, default=10, help='Max files to audit')
    args = ap.parse_args()

    base = Path(args.dir)
    target = base / args.subset
    files = sorted([p for p in target.glob('*.npz')])[:args.limit]
    if not files:
        print(f"No files found in {target}")
        return

    rows = [audit_file(p) for p in files]
    # Print concise table
    headers = ["file", "samples", "has_mask", "legal_mean", "legal_min", "legal_max", "legal_ratio", "error"]
    print(",".join(headers))
    for r in rows:
        print(",".join(str(r.get(h, '')) for h in headers))


if __name__ == '__main__':
    main()


