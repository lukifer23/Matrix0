import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from azchess.training.train import POLICY_SHAPE, apply_policy_mask, train_step


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal parameter to satisfy optimizer
        self.lin = nn.Linear(1, 1)

    def forward(self, x, return_ssl=True):
        batch = x.size(0)
        p = torch.zeros(batch, int(np.prod(POLICY_SHAPE)), dtype=torch.float32)
        v = torch.zeros(batch, dtype=torch.float32)
        ssl = torch.zeros(batch, 1, dtype=torch.float32)
        return p, v, ssl


def test_train_step_illegal_policy_shape(caplog):
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    s = np.zeros((1, 8, 8, 8), dtype=np.float32)
    bad_pi = np.zeros((1, int(np.prod(POLICY_SHAPE)) + 1), dtype=np.float32)
    z = np.zeros((1,), dtype=np.float32)

    batch = (s, bad_pi, z)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            train_step(model, optimizer, None, batch, "cpu", augment=False, enable_ssl=False, policy_masking=False, precision="fp32")
    assert "Policy tensor shape mismatch" in caplog.text


def test_apply_policy_mask():
    p = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pi = torch.tensor([[0.1, 0.9, 0.0], [0.0, 0.0, 0.0]])
    masked = apply_policy_mask(p, pi)
    assert masked[0, 2] < -1e8  # illegal move masked
    assert (masked[1] < -1e8).all()  # all-zero targets -> fully masked
