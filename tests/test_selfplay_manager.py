import random
import numpy as np
import torch
import chess

from experiments.grpo.mcts.mcts_integration import MCTS, MCTSConfig, SelfPlayManager


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        policy = torch.zeros((batch, 4672), dtype=torch.float32)
        value = torch.zeros((batch, 1), dtype=torch.float32)
        return policy, value


class DeterministicMCTS(MCTS):
    def _sample_move_from_policy(self, policy, legal_moves):
        return legal_moves[0], 0.0


def mcts_factory():
    model = DummyModel()
    cfg = MCTSConfig(num_simulations=1, dirichlet_frac=0.0)
    return DeterministicMCTS(model, cfg)


def _set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _generate_games():
    spm = SelfPlayManager(mcts_factory, num_workers=4)
    return spm.generate_games(num_games=8, max_moves=2)


def test_selfplay_manager_deterministic_concurrent():
    _set_seeds()
    games1 = _generate_games()
    _set_seeds()
    games2 = _generate_games()
    assert games1 == games2
