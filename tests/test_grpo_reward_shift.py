import copy
import torch
import torch.nn as nn

from experiments.grpo.training.grpo_trainer import (
    GRPOTrainer,
    GRPOConfig,
    Trajectory,
    TrajectoryStep,
)


class DummyModel(nn.Module):
    """Minimal policy/value model for testing."""

    def __init__(self):
        super().__init__()
        # Two-action policy with constant logits and a scalar value head
        self.logits = nn.Parameter(torch.zeros(2))
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        batch = states.shape[0]
        return self.logits.expand(batch, -1), self.value.expand(batch, 1)


def make_traj(reward: float) -> Trajectory:
    step = TrajectoryStep(
        state=torch.zeros(1, 4),
        action=0,
        log_prob=float(torch.log(torch.tensor(0.5))),
        value=0.0,
        reward=reward,
        done=True,
    )
    return Trajectory(steps=[step], total_reward=reward, length=1, game_result=0.0)


def test_policy_update_changes_with_reward_shift():
    torch.manual_seed(0)
    model = DummyModel()
    config = GRPOConfig(group_size=3, ppo_epochs=1, batch_size=8, learning_rate=0.1)
    trainer = GRPOTrainer(model, config)

    base_group = [make_traj(r) for r in [0.0, 1.0, 2.0]]
    init_state = copy.deepcopy(model.state_dict())
    trainer._train_on_group(base_group)
    base_logits = model.logits.clone().detach()

    # Reset model and optimizer
    model.load_state_dict(init_state)
    trainer = GRPOTrainer(model, config)

    shifted_group = [make_traj(r) for r in [2.0, 1.0, 2.0]]
    trainer._train_on_group(shifted_group)
    shifted_logits = model.logits.clone().detach()

    assert not torch.allclose(base_logits, shifted_logits)
