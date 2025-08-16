from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.9)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se: bool = False, se_ratio: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = _norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = _norm(channels)
        self.use_se = se
        if se:
            hidden = max(8, int(channels * se_ratio))
            self.se_fc1 = nn.Linear(channels, hidden)
            self.se_fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            # Squeeze
            w = F.adaptive_avg_pool2d(out, 1).flatten(1)
            w = F.relu(self.se_fc1(w), inplace=True)
            w = torch.sigmoid(self.se_fc2(w)).unsqueeze(-1).unsqueeze(-1)
            out = out * w
        out = out + x
        out = F.relu(out, inplace=True)
        return out


@dataclass
class NetConfig:
    planes: int = 19
    channels: int = 160
    blocks: int = 14
    policy_size: int = 4672
    se: bool = False
    se_ratio: float = 0.25


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        C = cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.planes, C, kernel_size=3, padding=1, bias=False),
            _norm(C),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(C, se=cfg.se, se_ratio=cfg.se_ratio) for _ in range(cfg.blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=1, bias=False),
            _norm(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, cfg.policy_size)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=1, bias=False),
            _norm(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(32 * 8 * 8, C)
        self.value_fc2 = nn.Linear(C, 1)

        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_head(x)
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)

        v = self.value_head(x)
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = torch.tanh(self.value_fc2(v))
        return p, v.squeeze(-1)

    @staticmethod
    def from_config(d: dict) -> "PolicyValueNet":
        return PolicyValueNet(NetConfig(**d))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def to_coreml(self, example_input: torch.Tensor, out_path: str) -> None:
        try:
            import coremltools as ct
        except Exception as e:
            raise RuntimeError("coremltools not available") from e

        self.eval()
        traced = torch.jit.trace(self, example_input)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=example_input.shape)],
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.save(out_path)
