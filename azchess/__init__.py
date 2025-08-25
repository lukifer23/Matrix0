"""Convenience imports for the :mod:`azchess` package.

Importing :mod:`azchess` now exposes commonly used submodules directly::

    from azchess import config, data_manager, mcts, model, orchestrator, arena, encoding, elo

This mirrors the layout of the package and keeps ``from azchess import *``
functional without creating circular imports.
"""

from . import arena, config, data_manager, elo, encoding, mcts, model, orchestrator

__all__ = ["config", "data_manager", "mcts", "model", "orchestrator", "arena", "encoding", "elo"]

