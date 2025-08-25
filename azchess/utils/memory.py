"""
Unified Memory Management for Matrix0
Centralizes all memory management operations for consistent behavior across modules.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified memory management for MPS and CUDA devices."""

    @staticmethod
    def clear_cache(device: str = "auto") -> None:
        """Clear GPU memory cache for specified device."""
        try:
            if device in ["mps", "auto"]:
                try:
                    import torch.mps
                    torch.mps.empty_cache()
                    logger.debug("MPS cache cleared")
                except Exception as e:
                    logger.debug(f"MPS cache clear failed: {e}")

            if device in ["cuda", "auto"]:
                try:
                    import torch.cuda
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
                except Exception as e:
                    logger.debug(f"CUDA cache clear failed: {e}")

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.debug(f"Memory cache clear failed: {e}")

    @staticmethod
    def get_memory_usage(device: str = "auto") -> dict:
        """Get memory usage statistics."""
        usage = {"device": device, "memory_gb": 0.0}

        try:
            if device == "mps":
                import torch.mps
                if torch.mps.is_available():
                    usage["memory_gb"] = 0.0  # MPS doesn't expose memory usage easily
                    usage["available"] = True
                else:
                    usage["available"] = False

            elif device.startswith("cuda"):
                import torch.cuda
                if torch.cuda.is_available():
                    usage["memory_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
                    usage["available"] = True
                else:
                    usage["available"] = False

            else:
                # CPU memory
                import psutil
                vm = psutil.virtual_memory()
                usage["memory_gb"] = vm.used / (1024**3)
                usage["available"] = True

        except Exception as e:
            logger.debug(f"Memory usage check failed: {e}")
            usage["available"] = False

        return usage

    @staticmethod
    def optimize_for_device(device: str) -> None:
        """Apply device-specific memory optimizations."""
        if device == "mps":
            # MPS-specific optimizations
            import os
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

        elif device.startswith("cuda"):
            # CUDA-specific optimizations
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def emergency_cleanup(device: str = "auto") -> None:
        """Emergency memory cleanup when facing OOM errors."""
        logger.warning("Performing emergency memory cleanup")

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear all caches
        MemoryManager.clear_cache(device)

        # Try to free any remaining GPU memory
        try:
            if device == "mps":
                import torch.mps
                torch.mps.empty_cache()
            elif device.startswith("cuda"):
                import torch.cuda
                torch.cuda.empty_cache()
        except Exception:
            pass


# Global instance for easy access
memory_manager = MemoryManager()


def clear_memory_cache(device: str = "auto") -> None:
    """Convenience function to clear memory cache."""
    memory_manager.clear_cache(device)


def get_memory_usage(device: str = "auto") -> dict:
    """Convenience function to get memory usage."""
    return memory_manager.get_memory_usage(device)


def emergency_memory_cleanup(device: str = "auto") -> None:
    """Convenience function for emergency cleanup."""
    memory_manager.emergency_cleanup(device)
