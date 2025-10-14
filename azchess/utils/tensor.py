"""
Unified tensor utilities for Matrix0 chess AI.

This module provides consistent tensor handling utilities across the entire codebase,
eliminating duplication and ensuring MPS compatibility.
"""

import logging
import torch

logger = logging.getLogger(__name__)


class TensorUtils:
    """Unified tensor utility class."""

    @staticmethod
    def ensure_contiguous_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Ensure tensor is contiguous, with debug logging for MPS compatibility.

        Args:
            tensor: Input tensor that may need to be made contiguous
            name: Name for logging purposes

        Returns:
            Contiguous tensor
        """
        if not tensor.is_contiguous():
            logger.debug(f"Making tensor {name} contiguous")
            return tensor.contiguous()
        return tensor


    @staticmethod
    def ensure_contiguous_array(array, name: str = "array"):
        """Ensure numpy array is contiguous for MPS compatibility.

        Args:
            array: Input numpy array that may need to be made contiguous
            name: Name for logging purposes

        Returns:
            Contiguous numpy array
        """
        if hasattr(array, 'flags') and not array.flags.c_contiguous:
            logger.debug(f"Making array {name} contiguous")
            import numpy as np
            return np.ascontiguousarray(array)
        return array

    @staticmethod
    def check_contiguous(outputs: dict) -> None:
        """Validate that output tensors (or dicts of tensors) are contiguous.

        Args:
            outputs: Dictionary of tensors to check

        Raises:
            RuntimeError: if any tensor is not contiguous.
        """
        for name, tensor in outputs.items():
            if tensor is None:
                continue
            if torch.is_tensor(tensor):
                if not tensor.is_contiguous():
                    raise RuntimeError(
                        f"{name} output tensor is not contiguous. Shape: {tensor.shape}, strides: {tensor.stride()}"
                    )
            elif isinstance(tensor, dict):
                for sub_name, sub_tensor in tensor.items():
                    if torch.is_tensor(sub_tensor) and not sub_tensor.is_contiguous():
                        raise RuntimeError(
                            f"{name}[{sub_name}] tensor is not contiguous. Shape: {sub_tensor.shape}, strides: {sub_tensor.stride()}"
                        )


    # Additional utility methods
    @staticmethod
    def check_tensor_health(tensor: torch.Tensor) -> dict:
        """Check tensor health and return diagnostic information.

        Args:
            tensor: Tensor to check

        Returns:
            Dictionary with health metrics
        """
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "contiguous": tensor.is_contiguous(),
            "requires_grad": tensor.requires_grad,
            "nan_count": torch.isnan(tensor).sum().item(),
            "inf_count": torch.isinf(tensor).sum().item(),
        }

    @staticmethod
    def create_tensor(*args, **kwargs) -> torch.Tensor:
        """Create a tensor with proper device placement.

        Args:
            *args: Positional arguments for torch.tensor
            **kwargs: Keyword arguments for torch.tensor

        Returns:
            Created tensor
        """
        return torch.tensor(*args, **kwargs)

    @staticmethod
    def log_tensor_stats(tensor: torch.Tensor, name: str) -> None:
        """Log tensor statistics for debugging.

        Args:
            tensor: Tensor to analyze
            name: Name for logging
        """
        if logger.isEnabledFor(logging.DEBUG):
            stats = TensorUtils.check_tensor_health(tensor)
            logger.debug(f"Tensor {name} stats: {stats}")

    @staticmethod
    def safe_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
        """Safely move tensor to device.

        Args:
            tensor: Tensor to move
            device: Target device

        Returns:
            Tensor on target device
        """
        try:
            return tensor.to(device)
        except Exception as e:
            logger.warning(f"Failed to move tensor to device {device}: {e}")
            return tensor

    @staticmethod
    def validate_tensor_shapes(tensors: dict) -> bool:
        """Validate that tensors have expected shapes.

        Args:
            tensors: Dictionary of tensors to validate

        Returns:
            True if all tensors have valid shapes
        """
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            if torch.is_tensor(tensor):
                if tensor.numel() == 0:
                    logger.warning(f"Tensor {name} is empty")
                    return False
        return True


# Backward compatibility - export functions directly
def ensure_contiguous_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Ensure tensor is contiguous, with debug logging for MPS compatibility."""
    return TensorUtils.ensure_contiguous_tensor(tensor, name)


def ensure_contiguous_array(array, name: str = "array"):
    """Ensure numpy array is contiguous for MPS compatibility."""
    return TensorUtils.ensure_contiguous_array(array, name)


def check_contiguous(outputs: dict) -> None:
    """Validate that output tensors (or dicts of tensors) are contiguous."""
    return TensorUtils.check_contiguous(outputs)


def ensure_contiguous(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Ensure tensor is contiguous (alias for ensure_contiguous_tensor)."""
    return TensorUtils.ensure_contiguous_tensor(tensor, name)


def check_tensor_health(tensor: torch.Tensor) -> dict:
    """Check tensor health and return diagnostic information."""
    return TensorUtils.check_tensor_health(tensor)


def create_tensor(*args, **kwargs) -> torch.Tensor:
    """Create a tensor with proper device placement."""
    return TensorUtils.create_tensor(*args, **kwargs)


def log_tensor_stats(tensor: torch.Tensor, name: str) -> None:
    """Log tensor statistics for debugging."""
    return TensorUtils.log_tensor_stats(tensor, name)


def safe_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Safely move tensor to device."""
    return TensorUtils.safe_to_device(tensor, device)


def validate_tensor_shapes(tensors: dict) -> bool:
    """Validate that tensors have expected shapes."""
    return TensorUtils.validate_tensor_shapes(tensors)