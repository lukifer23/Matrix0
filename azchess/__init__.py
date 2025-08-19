"""Convenience imports for the :mod:`azchess` package.

Importing :mod:`azchess` now exposes commonly used submodules directly::

    from azchess import config, data_manager, mcts, model

This mirrors the layout of the package and keeps ``from azchess import *``
functional without creating circular imports.
"""

from . import config, data_manager, mcts, model

__all__ = ["config", "data_manager", "mcts", "model"]

