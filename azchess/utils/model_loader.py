from __future__ import annotations

from typing import Tuple

import logging

import torch

from ..config import Config, select_device
from ..mcts import MCTS, MCTSConfig
from ..model import PolicyValueNet


def load_model_and_mcts(cfg: Config, checkpoint: str) -> Tuple[PolicyValueNet, MCTS]:
    """Load a model checkpoint and construct its MCTS helper.

    This function centralizes device selection, handling of EMA weights and
    MCTS configuration so that callers can rely on a consistent behaviour.
    """
    # Use explicit device if set, otherwise auto-select
    requested_device = cfg.get("device", "auto")
    if requested_device == "cpu":
        # Force CPU usage when explicitly requested
        device = "cpu"
    else:
        device = select_device(requested_device)
    mcfg_dict = dict(cfg.mcts())
    eval_section = cfg.eval()
    if isinstance(eval_section, dict) and eval_section:
        allowed = set(MCTSConfig.__dataclass_fields__.keys())
        overrides = {k: v for k, v in eval_section.items() if k in allowed}
        if overrides:
            logger = logging.getLogger(__name__)
            logger.debug(f"Applying MCTS overrides from eval config: {sorted(overrides.keys())}")
            mcfg_dict.update(overrides)
    mcfg = MCTSConfig.from_dict(mcfg_dict)

    model = PolicyValueNet.from_config(cfg.model()).to(device)

    # Allow loading checkpoints with NumPy scalar types.
    try:
        import numpy  # noqa: F401
        from torch.serialization import add_safe_globals
        try:
            add_safe_globals([numpy.core.multiarray.scalar])
        except Exception:
            pass
        try:
            add_safe_globals([numpy._core.multiarray.scalar])
        except Exception:
            pass
    except Exception:
        pass

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_ema", state.get("model", state)), strict=False)

    mcts = MCTS(mcfg, model, device)
    return model, mcts
