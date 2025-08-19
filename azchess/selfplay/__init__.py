"""Self-play modules for Matrix0.

Note: External engine worker import is optional to avoid hard dependency
when external engine integration is disabled.
"""

__all__ = []

try:
    from .external_engine_worker import ExternalEngineSelfPlay, external_engine_worker, GameResult
    __all__ += ["ExternalEngineSelfPlay", "external_engine_worker", "GameResult"]
except Exception:
    # External engines not required for core self-play
    pass
