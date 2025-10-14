"""
Unified Device Management for Matrix0
Centralizes device selection, validation, and management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DeviceManager:
    """Unified device management for consistent device handling across modules."""

    def __init__(self):
        self._device_cache: Dict[str, str] = {}

    def select_device(self, requested_device: str = "auto") -> str:
        """
        Unified device selection logic.

        Args:
            requested_device: Device request ("auto", "cpu", "mps", "cuda", "cuda:0", etc.)

        Returns:
            Selected device string
        """
        # Check cache first
        if requested_device in self._device_cache:
            return self._device_cache[requested_device]

        selected = requested_device

        if requested_device == "auto":
            # Auto-select best available device
            try:
                import torch

                if torch.cuda.is_available():
                    selected = "cuda"
                    logger.info("Auto-selected CUDA device")
                elif torch.backends.mps.is_available():
                    selected = "mps"
                    logger.info("Auto-selected MPS device")
                else:
                    selected = "cpu"
                    logger.info("Auto-selected CPU device")

            except ImportError:
                selected = "cpu"
                logger.info("PyTorch not available, using CPU")

        elif requested_device.startswith("cuda"):
            # Validate CUDA device
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    selected = "cpu"
                elif ":" in requested_device:
                    # Specific CUDA device
                    try:
                        device_idx = int(requested_device.split(":")[1])
                    except ValueError:
                        logger.warning(f"Invalid CUDA device spec '{requested_device}', using cuda:0")
                        device_idx = 0
                    if device_idx >= torch.cuda.device_count():
                        logger.warning(f"CUDA device {device_idx} not available, using cuda:0")
                        selected = "cuda:0"
            except ImportError:
                logger.warning("PyTorch CUDA not available, falling back to CPU")
                selected = "cpu"

        elif requested_device == "mps":
            # Validate MPS device
            try:
                import torch
                if not torch.backends.mps.is_available():
                    logger.warning("MPS requested but not available, falling back to CPU")
                    selected = "cpu"
            except ImportError:
                logger.warning("PyTorch MPS not available, falling back to CPU")
                selected = "cpu"

        # Cache the result
        self._device_cache[requested_device] = selected

        return selected

    def validate_device(self, device: str) -> bool:
        """Validate that a device is available and working."""
        try:
            import torch

            if device == "cpu":
                return True
            elif device.startswith("cuda"):
                if not torch.cuda.is_available():
                    return False
                idx = 0
                if ":" in device:
                    try:
                        idx = int(device.split(":")[1])
                    except ValueError:
                        return False
                return 0 <= idx < torch.cuda.device_count()
            elif device == "mps":
                return torch.backends.mps.is_available()
            else:
                return False

        except ImportError:
            return device == "cpu"

    def get_device_info(self, device: str) -> Dict[str, Any]:
        """Get detailed information about a device."""
        info = {
            "device": device,
            "available": self.validate_device(device),
            "type": device.split(":")[0]
        }

        try:
            import torch

            if device.startswith("cuda") and info["available"]:
                idx = 0
                if ":" in device:
                    try:
                        idx = int(device.split(":")[1])
                    except ValueError:
                        logger.debug(f"Invalid CUDA device spec '{device}', defaulting to index 0")
                        idx = 0
                if not (0 <= idx < torch.cuda.device_count()):
                    raise ValueError(f"CUDA device index out of range: {idx}")
                props = torch.cuda.get_device_properties(idx)
                info.update({
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}"
                })

            elif device == "mps" and info["available"]:
                info.update({
                    "name": "Apple Silicon MPS",
                    "memory_gb": None  # MPS doesn't expose this
                })

        except Exception as e:
            logger.debug(f"Could not get device info: {e}")

        return info

    def setup_device(self, device: str) -> None:
        """Setup device-specific configurations."""
        device_type = device.split(":")[0]

        if device_type == "mps":
            # MPS-specific setup
            import os
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            logger.info("MPS device configured")

        elif device_type == "cuda":
            # CUDA-specific setup
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("CUDA device configured")

        else:
            logger.info("CPU device configured")


# Global instance for easy access
device_manager = DeviceManager()


def select_device(requested_device: str = "auto") -> str:
    """Convenience function to select device."""
    return device_manager.select_device(requested_device)


def validate_device(device: str) -> bool:
    """Convenience function to validate device."""
    return device_manager.validate_device(device)


def _check_mps_performance() -> bool:
    """Check for known MPS performance issues (module-level function)."""
    manager = DeviceManager()
    return manager._check_mps_performance()


def get_device_info(device: str) -> Dict[str, Any]:
    """Convenience function to get device info."""
    return device_manager.get_device_info(device)


def setup_device(device: str) -> None:
    """Convenience function to setup device."""
    device_manager.setup_device(device)
