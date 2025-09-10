import pytest
import torch
import torch.nn as nn

from experiments.grpo.training.grpo_trainer import (
    GRPOEvaluator,
    Trajectory,
    TrajectoryStep,
)


class DummyModel(nn.Module):
    """Simple model producing deterministic policy and value"""

    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(4, 4, bias=False)
        self.value = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.policy.weight.zero_()
            self.value.weight.zero_()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.policy(x), self.value(x).squeeze(-1)


def _create_trajectory():
    state1 = torch.zeros(1, 4)
    state2 = torch.ones(1, 4)
    mask = torch.ones(4)

    step1 = TrajectoryStep(
        state=state1,
        action=0,
        log_prob=0.0,
        value=0.0,
        reward=0.0,
        done=False,
        legal_mask=mask,
    )
    step2 = TrajectoryStep(
        state=state2,
        action=1,
        log_prob=0.0,
        value=0.0,
        reward=1.0,
        done=True,
        legal_mask=mask,
    )
    return Trajectory(
        steps=[step1, step2], total_reward=1.0, length=2, game_result=1.0
    )


def test_grpo_evaluator_basic():
    model = DummyModel()
    evaluator = GRPOEvaluator(model)
    traj = _create_trajectory()

    result = evaluator.evaluate_game(traj)
    assert result["game_length"] == 2
    assert pytest.approx(0.5) == result["move_accuracy"]
    assert pytest.approx(1.0) == result["avg_value_error"]

    agg = evaluator.evaluate_games([traj])
    assert pytest.approx(0.5) == agg["avg_move_accuracy"]
    assert pytest.approx(1.0) == agg["avg_value_error"]

