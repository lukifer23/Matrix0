from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config, select_device
from ..model import PolicyValueNet


def run_benchmark(cfg_path: str = "config.yaml", batches=(1, 8, 32), steps: int = 200):
    cfg = Config.load(cfg_path)
    device = select_device(cfg.get("device", "auto"))
    model = PolicyValueNet.from_config(cfg.model()).to(device)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    model.eval()

    device_type = device.split(':')[0]
    use_amp = device_type in ("cuda", "mps")

    print(f"Device: {device}")
    for bsz in batches:
        x = torch.randn(bsz, cfg.model().get('planes', 19), 8, 8, device=device)
        try:
            x = x.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass
        # Warmup
        with torch.autocast(device_type=device_type, enabled=use_amp):
            for _ in range(10):
                _ = model(x, return_ssl=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast(device_type=device_type, enabled=use_amp):
            for _ in range(steps):
                p, v = model(x, return_ssl=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
        iters = steps
        items = steps * bsz
        print(f"batch={bsz:>3} | {items/dt:.1f} states/s | {iters/dt:.1f} it/s | time={dt:.3f}s")


if __name__ == "__main__":
    run_benchmark()

