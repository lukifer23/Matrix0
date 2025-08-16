"""External engine integration for Matrix0."""

from .uci_bridge import UCIClient
from .engine_manager import EngineManager

__all__ = ["UCIClient", "EngineManager"]
