"""External engine integration for Matrix0."""

from .engine_manager import EngineManager
from .uci_bridge import UCIClient

__all__ = ["UCIClient", "EngineManager"]
