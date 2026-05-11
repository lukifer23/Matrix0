import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from azchess.training.train import (
    POLICY_SHAPE,
    apply_policy_mask,
    apply_trainable_scope,
    select_checkpoint_model_state,
    legal_policy_ce_loss,
    legal_policy_mass_loss,
    train_step,
)


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


class SourceFilterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, return_ssl=True):
        batch = x.size(0)
        p = torch.zeros(batch, int(np.prod(POLICY_SHAPE)), dtype=torch.float32, device=x.device)
        p[:, 0] = self.bias + 4.0
        v = self.bias.expand(batch).clone()
        ssl = torch.zeros(batch, 1, dtype=torch.float32, device=x.device)
        return p, v, ssl


class ScopedTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk_bn = nn.BatchNorm2d(19)
        self.policy_head = nn.Linear(1, int(np.prod(POLICY_SHAPE)))
        self.value_fc1 = nn.Linear(1, 1)

    def forward(self, x, return_ssl=True):
        batch = x.size(0)
        _ = self.trunk_bn(x)
        seed = torch.ones((batch, 1), dtype=torch.float32, device=x.device)
        p = torch.zeros(batch, int(np.prod(POLICY_SHAPE)), dtype=torch.float32, device=x.device)
        v = self.value_fc1(seed).reshape(batch)
        ssl = torch.zeros(batch, 1, dtype=torch.float32, device=x.device)
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
            train_step(
                model,
                optimizer,
                None,
                batch,
                "cpu",
                augment=False,
                enable_ssl=False,
                ssrl_weight=0.0,
                enable_ssrl=False,
                policy_masking=False,
                precision="fp32",
            )
    assert "Policy tensor shape mismatch" in caplog.text


def test_apply_policy_mask():
    p = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pi = torch.tensor([[0.1, 0.9, 0.0], [0.0, 0.0, 0.0]])
    masked = apply_policy_mask(p, pi)
    assert masked[0, 2] < -1e8  # illegal move masked
    assert (masked[1] < -1e8).all()  # all-zero targets -> fully masked


def test_legal_policy_mass_loss_penalizes_illegal_probability():
    p = torch.tensor([[0.0, 0.0, 0.0], [4.0, -4.0, -4.0]], dtype=torch.float32)
    legal = torch.tensor([[True, False, False], [True, False, False]])

    loss = legal_policy_mass_loss(p, legal)

    assert loss.item() > 0.0
    assert loss.item() < 1.0


def test_legal_policy_mass_loss_backprops_to_illegal_logits():
    p = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
    legal = torch.tensor([[True, False, False, False]])

    loss = legal_policy_mass_loss(p, legal)
    loss.backward()

    assert p.grad is not None
    assert p.grad[0, 0].item() < 0.0
    assert p.grad[0, 1:].sum().item() > 0.0


def test_legal_policy_ce_loss_renormalizes_to_legal_moves():
    p = torch.tensor([[0.0, 2.0, 8.0, -3.0]], dtype=torch.float32)
    pi = torch.tensor([[0.25, 0.75, 0.0, 0.0]], dtype=torch.float32)
    legal = torch.tensor([[True, True, False, False]])

    loss = legal_policy_ce_loss(p, pi, legal)
    expected = -(pi[:, :2] * torch.log_softmax(p[:, :2], dim=1)).sum(dim=1).mean()

    assert torch.allclose(loss, expected)


def test_legal_policy_ce_loss_backprops_only_through_legal_logits():
    p = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
    pi = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    legal = torch.tensor([[True, True, False, False]])

    loss = legal_policy_ce_loss(p, pi, legal)
    loss.backward()

    assert p.grad is not None
    assert p.grad[0, 0].item() > 0.0
    assert p.grad[0, 1].item() < 0.0
    assert torch.equal(p.grad[0, 2:], torch.zeros(2))


def test_train_step_can_exclude_result_sources_from_value_loss():
    model = SourceFilterModel()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    s = np.zeros((2, 19, 8, 8), dtype=np.float32)
    pi = np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32)
    z = np.array([10.0, 0.0], dtype=np.float32)
    batch = {
        "s": s,
        "pi": pi,
        "z": z,
        "value_weight": np.ones((2,), dtype=np.float32),
        "result_source": np.array(["capped", "terminal"]),
    }

    _, _, unfiltered_value_loss, *_ = train_step(
        model,
        optimizer,
        None,
        batch,
        "cpu",
        augment=False,
        enable_ssl=False,
        ssrl_weight=0.0,
        enable_ssrl=False,
        policy_masking=False,
        precision="fp32",
    )
    optimizer.zero_grad()
    _, _, filtered_value_loss, *_ = train_step(
        model,
        optimizer,
        None,
        batch,
        "cpu",
        augment=False,
        enable_ssl=False,
        ssrl_weight=0.0,
        enable_ssrl=False,
        policy_masking=False,
        precision="fp32",
        value_exclude_sources=["capped"],
    )

    assert unfiltered_value_loss > 40.0
    assert filtered_value_loss == 0.0


def test_train_step_source_filters_match_prefixes():
    model = SourceFilterModel()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.array([0.0, 10.0], dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
        "result_source": np.array(["teacher:bootstrap_007", "capped"]),
    }

    _, _, value_loss, *_ = train_step(
        model,
        optimizer,
        None,
        batch,
        "cpu",
        augment=False,
        enable_ssl=False,
        ssrl_weight=0.0,
        enable_ssrl=False,
        policy_masking=False,
        precision="fp32",
        value_include_sources=["teacher:"],
    )

    assert value_loss == 0.0


def test_train_step_can_exclude_teacher_from_policy_loss_but_keep_value():
    model = SourceFilterModel()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    pi = np.zeros((2, policy_size), dtype=np.float32)
    pi[0, 0] = 1.0
    pi[1, 1] = 1.0
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": pi,
        "z": np.array([10.0, 0.0], dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
        "result_source": np.array(["teacher:bootstrap_007", "terminal"]),
    }

    _, unfiltered_policy_loss, unfiltered_value_loss, *_ = train_step(
        model,
        optimizer,
        None,
        batch,
        "cpu",
        augment=False,
        enable_ssl=False,
        ssrl_weight=0.0,
        enable_ssrl=False,
        policy_masking=False,
        precision="fp32",
    )
    optimizer.zero_grad()
    _, filtered_policy_loss, filtered_value_loss, *_ = train_step(
        model,
        optimizer,
        None,
        batch,
        "cpu",
        augment=False,
        enable_ssl=False,
        ssrl_weight=0.0,
        enable_ssrl=False,
        policy_masking=False,
        precision="fp32",
        policy_exclude_sources=["teacher:"],
    )

    assert filtered_policy_loss > unfiltered_policy_loss
    assert unfiltered_value_loss > 40.0
    assert filtered_value_loss == unfiltered_value_loss


def test_value_head_scope_freezes_non_value_params_and_norm_stats():
    model = ScopedTrainModel()
    stats = apply_trainable_scope(model, "value_head")

    assert stats["trainable_params"] == sum(p.numel() for p in model.value_fc1.parameters())
    assert all(param.requires_grad for param in model.value_fc1.parameters())
    assert not any(param.requires_grad for param in model.policy_head.parameters())
    assert not any(param.requires_grad for param in model.trunk_bn.parameters())

    optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.1)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.ones((2,), dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
    }

    train_step(
        model,
        optimizer,
        None,
        batch,
        "cpu",
        augment=False,
        enable_ssl=False,
        ssrl_weight=0.0,
        enable_ssrl=False,
        policy_masking=False,
        precision="fp32",
    )

    assert model.trunk_bn.training is False
    assert model.value_fc1.weight.grad is not None
    assert model.policy_head.weight.grad is None


def test_train_step_skips_backward_for_noop_source_filtered_batch():
    model = ScopedTrainModel()
    apply_trainable_scope(model, "value_head")
    optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.1)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.ones((2,), dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
        "result_source": np.array(["capped", "capped"]),
    }

    loss, policy_loss, value_loss, *_ = train_step(
        model,
        optimizer,
        None,
        batch,
        "cpu",
        augment=False,
        enable_ssl=False,
        ssrl_weight=0.0,
        enable_ssrl=False,
        policy_masking=False,
        precision="fp32",
        value_include_sources=["terminal"],
        policy_include_sources=["__none__"],
    )

    assert loss == 0.0
    assert policy_loss == 0.0
    assert value_loss == 0.0
    assert model.value_fc1.weight.grad is None


def test_checkpoint_state_selection_prefers_ema_once():
    state = {
        "model": {"weight": torch.tensor([1.0])},
        "model_ema": {"weight": torch.tensor([2.0])},
        "model_state_dict": {"weight": torch.tensor([3.0])},
    }

    key, selected = select_checkpoint_model_state(state)

    assert key == "model_ema"
    assert selected["weight"].item() == 2.0
