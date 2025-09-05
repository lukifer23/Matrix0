#!/usr/bin/env python3
"""
Add EMA weights to a checkpoint if missing by copying current model weights.

Usage:
  python tools/add_ema_to_checkpoint.py --in checkpoints/pretrained_40k_step_35000.pt \
                                        --out checkpoints/pretrained_35k_ema.pt

Notes:
- If the input already has 'model_ema', it is preserved unless --force is set.
- This does not perform training; EMA will equal current model weights.
  For a smoothed EMA, consider a short training run first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Add or seed EMA to a Matrix0 checkpoint")
    ap.add_argument("--in", dest="in_path", required=True, help="Input checkpoint path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output checkpoint path")
    ap.add_argument("--force", action="store_true", help="Overwrite existing 'model_ema' in input")
    args = ap.parse_args()

    src = Path(args.in_path)
    dst = Path(args.out_path)
    if not src.exists():
        raise SystemExit(f"Input checkpoint not found: {src}")

    chk = torch.load(src, map_location="cpu", weights_only=False)
    # Determine base state dict
    model_state = chk.get('model') or chk.get('model_state_dict')
    if model_state is None:
        # Some minimal checkpoints may just contain the raw state dict
        if isinstance(chk, dict) and all(hasattr(v, 'shape') for v in chk.values() if hasattr(v, 'shape') or True):
            model_state = chk
        else:
            raise SystemExit("Could not locate model state_dict in checkpoint")

    if ('model_ema' in chk) and (chk['model_ema'] is not None) and (not args.force):
        print("EMA already present; use --force to overwrite. Saving a copy anyway.")
    else:
        chk['model_ema'] = {k: v.clone() for k, v in model_state.items()}
        print("Seeded 'model_ema' from current model weights.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(chk, dst)
    print(f"Wrote checkpoint with EMA: {dst}")


if __name__ == "__main__":
    main()

