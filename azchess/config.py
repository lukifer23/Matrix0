from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str = "config.yaml") -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config(data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.raw

    # Convenience nested getters
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    def selfplay(self) -> Dict[str, Any]:
        return self.raw.get("selfplay", {})

    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    def eval(self) -> Dict[str, Any]:
        return self.raw.get("eval", {})

    def engines(self) -> Dict[str, Any]:
        return self.raw.get("engines", {})

    def orchestrator(self) -> Dict[str, Any]:
        return self.raw.get("orchestrator", {})


def select_device(cfg_device: str = "auto") -> str:
    try:
        import torch
        if cfg_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        if cfg_device == "auto" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

