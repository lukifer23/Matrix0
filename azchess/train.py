from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import Config, select_device
from .data_manager import DataManager
from .model import PolicyValueNet


class ReplayDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.samples: List[Tuple[str, int]] = []
        for p in paths:
            try:
                with np.load(p) as data:
                    n = data["s"].shape[0]
                for i in range(n):
                    self.samples.append((p, i))
            except Exception:
                # Skip corrupted files
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, i = self.samples[idx]
        with np.load(p) as data:
            s = data["s"][i]
            pi = data["pi"][i]
            z = data["z"][i]
        return s, pi, z


def collate(batch):
    s = torch.from_numpy(np.stack([b[0] for b in batch], axis=0))
    pi = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))
    z = torch.from_numpy(np.stack([b[2] for b in batch], axis=0))
    return s, pi, z


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow:
                p.copy_(self.shadow[k])


def train_step(model, optimizer, batch, device: str, accum_steps: int = 1):
    model.train()
    s, pi, z = batch
    s = s.to(device)
    pi = pi.to(device)
    z = z.to(device)

    with torch.autocast(device if device != "cpu" else "cpu", enabled=(device != "cpu")):
        p, v = model(s)
        log_probs = nn.functional.log_softmax(p, dim=1)
        policy_loss = -(pi * log_probs).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(v, z)
        loss = policy_loss + value_loss

    (loss / max(1, accum_steps)).backward()

    return loss.item(), policy_loss.item(), value_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--replay", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    # Strict encoding enforcement via env
    try:
        import os as _os
        if bool(cfg.get("strict_encoding", False)):
            _os.environ["MATRIX0_STRICT_ENCODING"] = "1"
    except Exception:
        pass
    
    device = select_device(cfg.get("device", "auto"))

    model = PolicyValueNet.from_config(cfg.model()).to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.training().get("lr", 1e-3), 
        weight_decay=cfg.training().get("weight_decay", 1e-4)
    )
    steps_per_epoch = cfg.training().get("steps_per_epoch", 1000)
    epochs = cfg.training().get("epochs", 1)
    
    # Allow optional override from env for orchestrator cycles
    try:
        import os as _os
        if "MATRIX0_TRAIN_EPOCHS" in _os.environ:
            epochs = int(_os.environ["MATRIX0_TRAIN_EPOCHS"])
    except Exception:
        pass
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps_per_epoch * epochs))
    ema_decay = float(cfg.training().get("ema_decay", 0.0))
    ema = EMA(model, ema_decay) if ema_decay > 0 else None

    # Load dataset using DataManager
    data_manager = DataManager(base_dir=cfg.get("data_dir", "data"))
    batch_size = cfg.training().get("batch_size", 256)
    
    log_dir = cfg.training().get("log_dir", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    steps = 0
    accum_steps = max(1, int(cfg.training().get("accum_steps", 1)))
    optimizer.zero_grad(set_to_none=True)
    
    # Resume support
    ckpt_dir = Path(cfg.training().get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "model.pt"
    if args.resume and out.exists():
        state = torch.load(out, map_location=device)
        model.load_state_dict(state.get("model_ema", state["model"]))
        if "opt" in state:
            optimizer.load_state_dict(state["opt"])
        if "sched" in state:
            scheduler.load_state_dict(state["sched"])
        if ema and "model_ema" in state:
            ema.shadow = {k: v.to(device) for k, v in state["model_ema"].items()}

    for epoch in range(epochs):
        batch_generator = data_manager.get_training_batch(batch_size, device)
        
        for step in range(steps_per_epoch):
            try:
                batch = next(batch_generator)
            except StopIteration:
                break
            
            loss, policy_loss, value_loss = train_step(model, optimizer, batch, device, accum_steps)
            
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
            
            # Logging
            if step % 100 == 0:
                writer.add_scalar("Loss/total", loss, steps)
                writer.add_scalar("Loss/policy", policy_loss, steps)
                writer.add_scalar("Loss/value", value_loss, steps)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], steps)
            
            steps += 1
        
        scheduler.step()
        
        # Save checkpoint
        state = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "epoch": epoch,
            "steps": steps,
        }
        if ema:
            state["model_ema"] = ema.shadow
        torch.save(state, out)
        
        print(f"Epoch {epoch+1}/{epochs} completed. Steps: {steps}")
    
    writer.close()
    print(f"Training completed. Total steps: {steps}")


if __name__ == "__main__":
    main()
