from __future__ import annotations

"""
Rating utilities for Matrix0
- Elo helpers (already in azchess.elo)
- Glicko‑2 implementation for more robust rating with uncertainty

API focuses on minimal, dependency‑free update for one player vs repeated
opponent(s). Formulas follow Glickman (2001/2012) with standard tau.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

# Conversion constant between Elo scale and Glicko-2 scale
_SCALE = 173.7178


@dataclass
class Glicko2Rating:
    rating: float = 1500.0
    rd: float = 350.0
    sigma: float = 0.06


def _to_glicko(mu_elo: float, rd_elo: float) -> Tuple[float, float]:
    return (mu_elo - 1500.0) / _SCALE, rd_elo / _SCALE


def _from_glicko(mu: float, phi: float) -> Tuple[float, float]:
    return mu * _SCALE + 1500.0, phi * _SCALE


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + (3.0 * phi * phi) / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def update_glicko2_player(player: Glicko2Rating,
                          opp_list: Iterable[Tuple[float, float]],
                          outcomes: Iterable[float],
                          tau: float = 0.5) -> Glicko2Rating:
    """Update a single player's Glicko‑2 rating given opponents and outcomes.

    Args:
        player: current Glicko2Rating (rating, rd, sigma) on Elo scale
        opp_list: iterable of (opponent_rating, opponent_rd) in Elo scale
        outcomes: iterable of results for this player per opponent: 1.0 win, 0.5 draw, 0.0 loss
        tau: volatility constraint (standard ~0.5)

    Returns updated Glicko2Rating on Elo scale.
    """
    opp = list(opp_list)
    res = list(outcomes)
    if not opp or not res or len(opp) != len(res):
        return player  # nothing to update

    mu, phi = _to_glicko(player.rating, player.rd)
    sigma = float(player.sigma)

    opp_mu_phi = [(_to_glicko(r, rd)) for (r, rd) in opp]

    # v: estimated variance of the rating based on opponents
    v_inv = 0.0
    for (mu_j, phi_j), s_j in zip(opp_mu_phi, res):
        E_j = _E(mu, mu_j, phi_j)
        v_inv += (_g(phi_j) ** 2) * E_j * (1.0 - E_j)
    if v_inv <= 0.0:
        return player
    v = 1.0 / v_inv

    # Delta: the estimated rating improvement
    delta_sum = 0.0
    for (mu_j, phi_j), s_j in zip(opp_mu_phi, res):
        E_j = _E(mu, mu_j, phi_j)
        delta_sum += _g(phi_j) * (s_j - E_j)
    delta = v * delta_sum

    # Volatility update via iterative method
    a = math.log(sigma * sigma)
    A = a
    B = None
    # Helper f(x)
    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return (num / den) - ((x - a) / (tau * tau))

    # Find B as per Glicko-2 paper
    if delta * delta > (phi * phi + v):
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while True:
            B = a - k * tau
            if f(B) < 0:
                break
            k += 1

    # Newton iteration on [A, B]
    fa = f(A)
    fb = f(B)
    # Ensure opposite signs to use a robust iteration (fallback on bisection Newton hybrid)
    for _ in range(100):
        C = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if abs(C - A) < 1e-12:
            break
        if fc * fb < 0:
            A, fa = B, fb
        else:
            fa = fa / 2.0
        B, fb = C, fc
    new_a = A
    new_sigma = math.exp(new_a / 2.0)

    # Update phi to phi_star then phi'
    phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
    phi_prime = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / v))
    mu_prime = mu + (phi_prime * phi_prime) * delta_sum

    new_rating, new_rd = _from_glicko(mu_prime, phi_prime)
    return Glicko2Rating(rating=new_rating, rd=new_rd, sigma=new_sigma)


def update_glicko2_batch(player: Glicko2Rating,
                         opponent: Glicko2Rating,
                         wins: int,
                         draws: int,
                         losses: int,
                         tau: float = 0.5) -> Glicko2Rating:
    """Update player rating against a single repeated opponent using counts."""
    opp_list = [(opponent.rating, opponent.rd)] * max(0, int(wins + draws + losses))
    outcomes: List[float] = [1.0] * max(0, int(wins)) + [0.5] * max(0, int(draws)) + [0.0] * max(0, int(losses))
    return update_glicko2_player(player, opp_list, outcomes, tau=tau)

