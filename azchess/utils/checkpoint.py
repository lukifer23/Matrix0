"""
Unified Checkpoint Management for Matrix0
Centralizes checkpoint saving, loading, and validation operations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Unified checkpoint management for consistent checkpoint operations."""

    @staticmethod
    def save_checkpoint(model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None,
                       step: int = 0,
                       loss: Optional[float] = None,
                       metrics: Optional[Dict[str, Any]] = None,
                       path: Union[str, Path] = "checkpoint.pt",
                       ema_model: Optional[torch.nn.Module] = None,
                       **kwargs) -> bool:
        """Save a comprehensive checkpoint."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create checkpoint state
            checkpoint = {
                'step': step,
                'global_step': step,  # Alternative key for compatibility
                'model_state_dict': model.state_dict(),
                'timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'config': {
                    'model_type': model.__class__.__name__,
                    'architecture': 'V2_Enhanced'
                }
            }

            # Add optional components
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()

            if ema_model is not None:
                checkpoint['ema_state_dict'] = ema_model.state_dict()

            if loss is not None:
                checkpoint['loss'] = loss

            if metrics is not None:
                checkpoint['metrics'] = metrics

            # Add any additional kwargs
            checkpoint.update(kwargs)

            # Save checkpoint
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved successfully: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint {path}: {e}")
            return False

    @staticmethod
    def load_checkpoint(path: Union[str, Path],
                       model: Optional[torch.nn.Module] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None,
                       device: str = "cpu",
                       strict: bool = False,
                       ema_model: Optional[torch.nn.Module] = None) -> Optional[Dict[str, Any]]:
        """Load a checkpoint with comprehensive error handling."""
        try:
            path = Path(path)
            if not path.exists():
                logger.error(f"Checkpoint not found: {path}")
                return None

            # Load checkpoint
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            logger.info(f"Checkpoint loaded: {path}")

            # Load model state
            if model is not None:
                model_key = None
                for key in ['model_state_dict', 'model', 'state_dict']:
                    if key in checkpoint:
                        model_key = key
                        break

                if model_key:
                    try:
                        model.load_state_dict(checkpoint[model_key], strict=strict)
                        logger.info(f"Model loaded from '{model_key}'")
                    except Exception as e:
                        logger.warning(f"Failed to load model: {e}")
                else:
                    logger.error("No model state found in checkpoint")

            # Load optimizer state
            if optimizer is not None:
                optimizer_key = None
                for key in ['optimizer_state_dict', 'optimizer']:
                    if key in checkpoint:
                        optimizer_key = key
                        break

                if optimizer_key:
                    try:
                        optimizer.load_state_dict(checkpoint[optimizer_key])
                        logger.info(f"Optimizer loaded from '{optimizer_key}'")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer: {e}")

            # Load scheduler state
            if scheduler is not None:
                scheduler_key = None
                for key in ['scheduler_state_dict', 'scheduler']:
                    if key in checkpoint:
                        scheduler_key = key
                        break

                if scheduler_key:
                    try:
                        scheduler.load_state_dict(checkpoint[scheduler_key])
                        logger.info(f"Scheduler loaded from '{scheduler_key}'")
                    except Exception as e:
                        logger.warning(f"Failed to load scheduler: {e}")

            # Load scaler state
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                try:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logger.info("Scaler loaded from 'scaler_state_dict'")
                except Exception as e:
                    logger.warning(f"Failed to load scaler: {e}")

            # Load EMA model state
            if ema_model is not None and 'ema_state_dict' in checkpoint:
                try:
                    ema_model.load_state_dict(checkpoint['ema_state_dict'])
                    logger.info("EMA model loaded from 'ema_state_dict'")
                except Exception as e:
                    logger.warning(f"Failed to load EMA model: {e}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")
            return None

    @staticmethod
    def get_checkpoint_info(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get information about a checkpoint without loading it."""
        try:
            path = Path(path)
            if not path.exists():
                return None

            # Load metadata only
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

            info = {
                'path': str(path),
                'step': checkpoint.get('step', 0),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'version': checkpoint.get('version', 'unknown'),
                'loss': checkpoint.get('loss'),
                'metrics': checkpoint.get('metrics'),
                'has_optimizer': 'optimizer_state_dict' in checkpoint,
                'has_scheduler': 'scheduler_state_dict' in checkpoint,
                'has_scaler': 'scaler_state_dict' in checkpoint,
                'has_ema': 'ema_state_dict' in checkpoint
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get checkpoint info for {path}: {e}")
            return None

    @staticmethod
    def validate_checkpoint(path: Union[str, Path]) -> bool:
        """Validate that a checkpoint file is readable and contains expected keys."""
        try:
            path = Path(path)
            if not path.exists():
                return False

            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

            # Check for essential keys
            required_keys = ['model_state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    logger.warning(f"Checkpoint missing required key: {key}")
                    return False

            # Check step information
            if 'step' not in checkpoint and 'global_step' not in checkpoint:
                logger.warning("Checkpoint missing step information")
                return False

            return True

        except Exception as e:
            logger.error(f"Checkpoint validation failed for {path}: {e}")
            return False


# Global instance for easy access
checkpoint_manager = CheckpointManager()


def save_checkpoint(*args, **kwargs) -> bool:
    """Convenience function."""
    return checkpoint_manager.save_checkpoint(*args, **kwargs)


def load_checkpoint(*args, **kwargs) -> Optional[Dict[str, Any]]:
    """Convenience function."""
    return checkpoint_manager.load_checkpoint(*args, **kwargs)


def get_checkpoint_info(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Convenience function."""
    return checkpoint_manager.get_checkpoint_info(path)


def validate_checkpoint(path: Union[str, Path]) -> bool:
    """Convenience function."""
    return checkpoint_manager.validate_checkpoint(path)
