from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str = "config.yaml") -> "Config":
        """Load configuration data from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configuration value or the provided default."""
        return self.raw.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying configuration dictionary."""
        return self.raw

    # Convenience nested getters
    def model(self) -> Dict[str, Any]:
        """Model configuration section."""
        return self.raw.get("model", {})

    def selfplay(self) -> Dict[str, Any]:
        """Self-play configuration section."""
        return self.raw.get("selfplay", {})

    def draw(self) -> Dict[str, Any]:
        """Draw adjudication configuration section.

        Merge top-level `draw` with `selfplay.draw`, with selfplay-specific
        values taking precedence when present.
        """
        try:
            top = self.raw.get("draw", {}) or {}
            sp = (self.raw.get("selfplay", {}) or {}).get("draw", {}) or {}
            # selfplay overrides top-level
            merged = dict(top)
            merged.update(sp)
            return merged
        except Exception:
            return self.raw.get("draw", {})

    def training(self) -> Dict[str, Any]:
        """Training configuration section."""
        return self.raw.get("training", {})

    def eval(self) -> Dict[str, Any]:
        """Evaluation configuration section."""
        return self.raw.get("eval", {})

    def mcts(self) -> Dict[str, Any]:
        """MCTS configuration section."""
        return self.raw.get("mcts", {})

    def engines(self) -> Dict[str, Any]:
        """Engines configuration section."""
        return self.raw.get("engines", {})

    def openings(self) -> Dict[str, Any]:
        """Opening book configuration section."""
        return self.raw.get("openings", {})

    def external_data(self) -> Dict[str, Any]:
        """External data configuration section."""
        return self.raw.get("external_data", {})

    def orchestrator(self) -> Dict[str, Any]:
        """Orchestrator configuration section."""
        return self.raw.get("orchestrator", {})
    
    def model_v2(self) -> Dict[str, Any]:
        """V2 model configuration section."""
        return self.raw.get("model", {})
    
    def is_v2_enabled(self) -> bool:
        """Check if V2 features are enabled."""
        model_cfg = self.model_v2()
        return (
            model_cfg.get("channels", 160) == 192 or
            model_cfg.get("blocks", 14) == 16 or
            model_cfg.get("norm") == "group" or
            model_cfg.get("activation") == "silu" or
            model_cfg.get("preact", False) or
            model_cfg.get("policy_factor_rank", 0) > 0
        )


def select_device(cfg_device: str = "auto") -> str:
    """Select best available device string via unified DeviceManager.

    Delegates to ``azchess.utils.device.select_device`` to keep a single source
    of truth for device policy across the codebase.
    """
    try:
        from azchess.utils.device import select_device as _select_device  # local import to avoid cycles
        return _select_device(cfg_device)
    except Exception:
        # Conservative fallback
        return "cpu"
