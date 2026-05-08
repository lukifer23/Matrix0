from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..data_manager import DataManager

logger = logging.getLogger(__name__)


class NPZBatchIterableDataset(IterableDataset):
    """Iterable dataset that yields ready-to-train batches from DataManager.

    This design preserves existing train_step expectations (NumPy arrays),
    while enabling DataLoader workers to prefetch batches.

    Modes:
    - 'replay': yields from DataManager.get_training_batch()
    - 'mixed':  yields from DataManager.get_curriculum_batch(..., phase='mixed')
    - 'phase:<name>': yields curriculum batches for a specific phase name
    """

    def __init__(
        self,
        data_manager: DataManager,
        batch_size: int,
        device: str = "cpu",
        mode: str = "mixed",
    ) -> None:
        super().__init__()
        self.dm = data_manager
        self.batch_size = int(batch_size)
        self.device = device
        self.mode = str(mode)
        self._batches_seen = 0

    def _log_batch_sanity(self, batch: Dict[str, np.ndarray]) -> None:
        self._batches_seen += 1
        if self._batches_seen != 1 and self._batches_seen % 200 != 0:
            return
        try:
            pi = batch["pi"]
            z = batch["z"]
            row_sum = pi.sum(axis=1)
            positive = pi > 0
            entropy = -np.sum(np.where(positive, pi * np.log(np.clip(pi, 1e-12, 1.0)), 0.0), axis=1)
            msg = (
                "Batch sanity mode=%s n=%d policy_sum=%.4f/%.4f/%.4f "
                "entropy=%.3f value=%.3f/%.3f"
            )
            args = (
                self.mode,
                int(pi.shape[0]),
                float(row_sum.min()),
                float(row_sum.mean()),
                float(row_sum.max()),
                float(entropy.mean()),
                float(np.min(z)),
                float(np.max(z)),
            )
            if "legal_mask" in batch:
                legal = batch["legal_mask"]
                legal_counts = legal.reshape(legal.shape[0], -1).sum(axis=1)
                msg += " legal=%.1f/%.1f/%.1f"
                args += (float(legal_counts.min()), float(legal_counts.mean()), float(legal_counts.max()))
            ssl_keys = sorted(k for k in batch if k.startswith("ssl_"))
            if ssl_keys:
                msg += " ssl=%s"
                args += (",".join(ssl_keys),)
            logger.info(msg, *args)
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Batch sanity metrics failed for mode={self.mode}: {exc}") from exc

    def __iter__(self) -> Iterator[Union[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], Dict[str, np.ndarray]]]:
        # Replay mode: stream shards via DataManager iterator
        if self.mode == "replay":
            for batch in self.dm.get_training_batch(self.batch_size, self.device):
                # Ensure we return a tuple matching train_step expectations
                if isinstance(batch, tuple) and len(batch) in (3, 4, 5):
                    yield batch
                else:
                    try:
                        s = batch.get("s")
                        pi = batch.get("pi")
                        z = batch.get("z")
                        lm = batch.get("legal_mask", None)
                        vw = batch.get("value_weight", None)
                        if lm is not None or vw is not None:
                            yield (s, pi, z, lm, vw) if vw is not None else (s, pi, z, lm)
                        else:
                            yield (s, pi, z)
                    except Exception:
                        continue
            return

        # Mixed or specific curriculum phase: query DM each time
        phase = "mixed"
        if self.mode.startswith("phase:"):
            phase = self.mode.split(":", 1)[1] or "mixed"

        while True:
            batch_dict: Optional[Dict[str, np.ndarray]] = self.dm.get_curriculum_batch(
                self.batch_size, phase
            )
            if batch_dict is None:
                # If a specific phase had no data, try mixed as fallback
                if phase != "mixed":
                    batch_dict = self.dm.get_curriculum_batch(self.batch_size, "mixed")
                if batch_dict is None:
                    # No data available; stop gracefully
                    logger.warning("NPZ dataset: no data available (phase=%s)", phase)
                    return

            # Preserve every key, including precomputed ssl_* targets. train_step
            # accepts dict batches and will consume those targets directly.
            self._log_batch_sanity(batch_dict)
            yield batch_dict


def build_training_dataloader(
    data_manager: DataManager,
    batch_size: int,
    device: str,
    mode: str,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    """Construct a DataLoader for NPZ batches with MPS-friendly defaults."""
    from torch.utils.data import DataLoader

    ds = NPZBatchIterableDataset(data_manager, batch_size, device=device, mode=mode)

    # pin_memory has negligible benefit on MPS; keep False.
    dl = DataLoader(
        ds,
        batch_size=None,  # dataset yields full batches already
        num_workers=max(0, int(num_workers)),
        persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
        pin_memory=False,
    )
    return dl
