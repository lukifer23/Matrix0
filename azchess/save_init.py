from __future__ import annotations

from pathlib import Path
import torch

from .config import Config, select_device
from .model import PolicyValueNet


def main():
    cfg = Config.load()
    device = select_device(cfg.get("device", "auto"))
    model = PolicyValueNet.from_config(cfg.model()).to(device)
    out_dir = Path(cfg.training().get("checkpoint_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model.pt"
    torch.save({"model": model.state_dict()}, out)
    print(f"Saved untrained model to {out}")


if __name__ == "__main__":
    main()

