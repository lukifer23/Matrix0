from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch

from azchess.config import select_device


@dataclass(frozen=True)
class EnvironmentInfo:
    python: str
    platform: str
    machine: str
    processor: str
    cpu_count: int
    torch: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    selected_device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_environment_info(device: str = "auto") -> EnvironmentInfo:
    return EnvironmentInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 0,
        torch=torch.__version__,
        mps_built=bool(getattr(torch.backends.mps, "is_built", lambda: False)()),
        mps_available=bool(getattr(torch.backends.mps, "is_available", lambda: False)()),
        cuda_available=bool(torch.cuda.is_available()),
        selected_device=select_device(device),
    )


def log_environment_info(logger, device: str = "auto") -> EnvironmentInfo:
    info = collect_environment_info(device)
    logger.info("Environment:")
    for key, value in info.to_dict().items():
        logger.info("  %s: %s", key, value)
    return info
