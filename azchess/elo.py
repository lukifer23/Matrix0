from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

K_DEFAULT = 20.0


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, sa: float, k: float = K_DEFAULT) -> tuple[float, float]:
    """Update Elo ratings for A and B.

    ra, rb: pre-match ratings
    sa: average score for A over games (1 win, 0.5 draw, 0 loss)
    """
    ea = expected_score(ra, rb)
    delta = k * (sa - ea)
    return ra + delta, rb - delta


@dataclass
class EloBook:
    path: Path

    def load(self) -> dict:
        """Load Elo ratings; provide sensible defaults across tools if file missing/corrupt."""
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        # Provide defaults for multiple tools that may track different keys
        return {
            "best": 1500.0,
            "enhanced_best": 1500.0,
            "candidate": 1500.0,
            "baseline": 1500.0,
            "history": [],
        }

    def save(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
