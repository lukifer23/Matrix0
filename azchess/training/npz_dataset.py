from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional, Tuple

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

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        # Replay mode: stream shards via DataManager iterator
        if self.mode == "replay":
            for batch in self.dm.get_training_batch(self.batch_size, self.device):
                # Ensure we return a 3- or 4-tuple matching train_step expectations
                if isinstance(batch, tuple) and (len(batch) == 3 or len(batch) == 4):
                    yield batch
                else:
                    try:
                        s = batch.get("s")
                        pi = batch.get("pi")
                        z = batch.get("z")
                        lm = batch.get("legal_mask", None)
                        yield (s, pi, z, lm) if lm is not None else (s, pi, z)
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

            s = batch_dict.get("s")
            pi = batch_dict.get("pi")
            z = batch_dict.get("z")
            lm = batch_dict.get("legal_mask", None)

            # Yield tuple to match train_step inputs
            yield (s, pi, z, lm) if lm is not None else (s, pi, z)


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

