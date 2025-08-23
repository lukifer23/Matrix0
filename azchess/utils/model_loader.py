from __future__ import annotations

from typing import Tuple

import torch

from ..config import Config, select_device
from ..model import PolicyValueNet
from ..mcts import MCTS, MCTSConfig


def load_model_and_mcts(cfg: Config, checkpoint: str) -> Tuple[PolicyValueNet, MCTS]:
    """Load a model checkpoint and construct its MCTS helper.

    This function centralizes device selection, handling of EMA weights and
    MCTS configuration so that callers can rely on a consistent behaviour.
    """
    device = select_device(cfg.get("device", "auto"))
    mcfg_dict = cfg.eval() or cfg.mcts()
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
