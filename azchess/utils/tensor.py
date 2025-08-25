"""
Unified Tensor Utilities for Matrix0
Centralizes tensor operations, validation, and utilities.
"""

from __future__ import annotations

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class TensorUtils:
    """Unified tensor utilities for consistent tensor handling."""

    @staticmethod
    def ensure_contiguous(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Ensure tensor is contiguous with logging."""
        if not tensor.is_contiguous():
            logger.debug(f"Making {name} contiguous (shape: {tensor.shape}, device: {tensor.device})")
            return tensor.contiguous()
        return tensor

    @staticmethod
    def ensure_contiguous_array(array: np.ndarray, name: str = "array") -> np.ndarray:
        """Ensure numpy array is contiguous."""
        if not array.flags.c_contiguous:
            logger.debug(f"Making {name} contiguous (shape: {array.shape})")
            return np.ascontiguousarray(array)
        return array

    @staticmethod
    def validate_shapes(*tensors: torch.Tensor, names: Optional[list] = None) -> bool:
        """Validate tensor shapes are compatible."""
        if not tensors:
            return True

        base_shape = tensors[0].shape
        names = names or [f"tensor_{i}" for i in range(len(tensors))]

        for i, (tensor, name) in enumerate(zip(tensors, names)):
            if tensor.shape != base_shape:
                logger.error(f"Shape mismatch for {name}: expected {base_shape}, got {tensor.shape}")
                return False

        return True

    @staticmethod
    def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Check tensor for NaN/Inf values and other issues."""
        health = {
            "name": name,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "device": tensor.device,
            "contiguous": tensor.is_contiguous(),
            "has_nan": False,
            "has_inf": False,
            "nan_count": 0,
            "inf_count": 0
        }

        if tensor.dtype.is_floating_point:
            health["has_nan"] = torch.isnan(tensor).any().item()
            health["has_inf"] = torch.isinf(tensor).any().item()
            health["nan_count"] = torch.isnan(tensor).sum().item()
            health["inf_count"] = torch.isinf(tensor).sum().item()

        return health

    @staticmethod
    def safe_tensor_to_device(tensor: torch.Tensor, device: str, name: str = "tensor") -> torch.Tensor:
        """Safely move tensor to device with error handling."""
        try:
            return tensor.to(device)
        except Exception as e:
            logger.warning(f"Failed to move {name} to {device}: {e}")
            return tensor

    @staticmethod
    def create_tensor(*args, device: str = "cpu", dtype: Optional[torch.dtype] = None,
                     **kwargs) -> torch.Tensor:
        """Create tensor with unified device and dtype handling."""
        try:
            tensor = torch.tensor(*args, **kwargs)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            if device != "cpu":
                tensor = tensor.to(device)
            return tensor
        except Exception as e:
            logger.error(f"Failed to create tensor: {e}")
            # Fallback to CPU tensor
            return torch.tensor(*args, **kwargs)

    @staticmethod
    def log_tensor_stats(tensor: torch.Tensor, name: str = "tensor", level: int = logging.DEBUG) -> None:
        """Log comprehensive tensor statistics."""
        if not logger.isEnabledFor(level):
            return

        stats = {
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "contiguous": tensor.is_contiguous(),
            "memory_mb": tensor.numel() * tensor.element_size() / (1024 * 1024)
        }

        if tensor.dtype.is_floating_point:
            stats.update({
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "has_nan": torch.isnan(tensor).any().item(),
                "has_inf": torch.isinf(tensor).any().item()
            })

        logger.log(level, f"{name} stats: {stats}")


# Global instance for easy access
tensor_utils = TensorUtils()


def ensure_contiguous(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Convenience function."""
    return tensor_utils.ensure_contiguous(tensor, name)


def ensure_contiguous_array(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Convenience function."""
    return tensor_utils.ensure_contiguous_array(array, name)


def validate_tensor_shapes(*tensors: torch.Tensor, names: Optional[list] = None) -> bool:
    """Convenience function."""
    return tensor_utils.validate_shapes(*tensors, names=names)


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
    """Convenience function."""
    return tensor_utils.check_tensor_health(tensor, name)


def safe_to_device(tensor: torch.Tensor, device: str, name: str = "tensor") -> torch.Tensor:
    """Convenience function."""
    return tensor_utils.safe_tensor_to_device(tensor, device, name)


def create_tensor(*args, device: str = "cpu", dtype: Optional[torch.dtype] = None, **kwargs) -> torch.Tensor:
    """Convenience function."""
    return tensor_utils.create_tensor(*args, device=device, dtype=dtype, **kwargs)


def log_tensor_stats(tensor: torch.Tensor, name: str = "tensor", level: int = logging.DEBUG) -> None:
    """Convenience function."""
    tensor_utils.log_tensor_stats(tensor, name, level)
