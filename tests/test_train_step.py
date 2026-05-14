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
    parse_source_weight_specs,
    policy_distillation_loss,
    train_step,
    value_distillation_loss,
    value_mean_distillation_loss,
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


class FixedValueTeacher(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = float(value)

    def forward(self, x, return_ssl=True):
        batch = x.size(0)
        p = torch.zeros(batch, int(np.prod(POLICY_SHAPE)), dtype=torch.float32, device=x.device)
        v = torch.full((batch,), self.value, dtype=torch.float32, device=x.device)
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


class MovesLeftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moves_left_head = nn.Linear(1, 1)

    def forward_with_features(self, x, return_ssl=True):
        batch = x.size(0)
        feats = torch.ones((batch, 1), dtype=torch.float32, device=x.device)
        p = torch.zeros(batch, int(np.prod(POLICY_SHAPE)), dtype=torch.float32, device=x.device)
        v = torch.zeros(batch, dtype=torch.float32, device=x.device)
        ssl = torch.zeros(batch, 1, dtype=torch.float32, device=x.device)
        return p, v, ssl, feats

    def compute_moves_left(self, feats):
        return torch.sigmoid(self.moves_left_head(feats)).reshape(-1)


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


def test_train_step_applies_value_source_weights():
    model = SourceFilterModel()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.array([1.0, 3.0], dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
        "result_source": np.array(["capped", "terminal"]),
    }

    _, _, unweighted_value_loss, *_ = train_step(
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
        policy_include_sources=["__none__"],
    )
    optimizer.zero_grad()
    _, _, weighted_value_loss, *_ = train_step(
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
        policy_include_sources=["__none__"],
        value_source_weights={"terminal": 3.0},
    )

    assert unweighted_value_loss == 5.0
    assert weighted_value_loss == 7.0


def test_parse_source_weight_specs_validates_input():
    assert parse_source_weight_specs(["terminal=2", "capped=0.5"]) == {
        "terminal": 2.0,
        "capped": 0.5,
    }
    with pytest.raises(ValueError):
        parse_source_weight_specs(["terminal"])
    with pytest.raises(ValueError):
        parse_source_weight_specs(["terminal=-1"])


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


def test_train_step_uses_moves_left_auxiliary_loss():
    model = MovesLeftModel()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.zeros((2,), dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
        "moves_left": np.array([10.0, 1.0], dtype=np.float32),
    }

    *_, moves_left_loss = train_step(
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
        policy_include_sources=["__none__"],
        moves_left_weight=1.0,
        moves_left_scale=32.0,
    )

    assert moves_left_loss > 0.0
    assert model.moves_left_head.weight.grad is not None


def test_checkpoint_state_selection_prefers_ema_once():
    state = {
        "model": {"weight": torch.tensor([1.0])},
        "model_ema": {"weight": torch.tensor([2.0])},
        "model_state_dict": {"weight": torch.tensor([3.0])},
    }

    key, selected = select_checkpoint_model_state(state)

    assert key == "model_ema"
    assert selected["weight"].item() == 2.0


def test_policy_distillation_loss_backprops_when_policy_targets_filtered_out():
    student = SourceFilterModel()
    teacher = SourceFilterModel()
    with torch.no_grad():
        student.bias.fill_(0.0)
        teacher.bias.fill_(2.0)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    optimizer = optim.SGD(student.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.zeros((2,), dtype=np.float32),
        "value_weight": np.zeros((2,), dtype=np.float32),
        "result_source": np.array(["terminal", "terminal"]),
    }

    loss, policy_loss, value_loss, *_ = train_step(
        student,
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
        policy_include_sources=["__none__"],
        policy_distill_model=teacher,
        policy_distill_weight=1.0,
    )

    assert policy_distillation_loss(
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([[2.0, 0.0]], dtype=torch.float32),
    ).item() > 0.0
    assert loss > 0.0
    assert policy_loss == 0.0
    assert value_loss == 0.0
    assert student.bias.grad is not None


def test_value_mean_distillation_loss_penalizes_source_mean_drift():
    student_value = torch.tensor([0.1, 0.3, -0.2, -0.4], dtype=torch.float32)
    teacher_value = torch.zeros_like(student_value)
    weights = torch.ones_like(student_value)
    sources = np.array(["capped", "capped", "terminal", "terminal"])

    loss = value_mean_distillation_loss(student_value, teacher_value, weights, sources)

    assert torch.allclose(loss, torch.tensor((0.2**2 + 0.3**2) / 2, dtype=torch.float32))


def test_value_distillation_loss_penalizes_per_position_drift():
    student_value = torch.tensor([0.1, 0.3, -0.2], dtype=torch.float32)
    teacher_value = torch.zeros_like(student_value)
    weights = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)

    loss = value_distillation_loss(student_value, teacher_value, weights)

    assert loss.item() == pytest.approx(((0.1**2) + 2.0 * (0.2**2)) / 3.0)


def test_train_step_value_mean_distillation_uses_teacher_value():
    student = SourceFilterModel()
    with torch.no_grad():
        student.bias.fill_(0.1)
    teacher = FixedValueTeacher(value=0.0)
    teacher.eval()
    optimizer = optim.SGD(student.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((4, 19, 8, 8), dtype=np.float32),
        "pi": np.full((4, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.full((4,), 0.1, dtype=np.float32),
        "value_weight": np.ones((4,), dtype=np.float32),
        "result_source": np.array(["capped", "capped", "terminal", "terminal"]),
    }

    loss, policy_loss, value_loss, *_ = train_step(
        student,
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
        policy_include_sources=["__none__"],
        policy_distill_model=teacher,
        policy_distill_weight=0.0,
        value_mean_distill_weight=10.0,
    )

    assert policy_loss == 0.0
    assert value_loss == pytest.approx(0.0, abs=1e-8)
    assert loss == pytest.approx(0.1, rel=1e-5)
    assert student.bias.grad is not None


def test_train_step_value_distillation_uses_teacher_value():
    student = SourceFilterModel()
    with torch.no_grad():
        student.bias.fill_(0.2)
    teacher = FixedValueTeacher(value=0.0)
    optimizer = optim.SGD(student.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.full((2,), 0.2, dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
        "result_source": np.array(["capped", "terminal"]),
    }

    loss, policy_loss, value_loss, *_ = train_step(
        student,
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
        policy_include_sources=["__none__"],
        policy_distill_model=teacher,
        policy_distill_weight=0.0,
        value_distill_weight=5.0,
    )

    assert policy_loss == 0.0
    assert value_loss == pytest.approx(0.0, abs=1e-8)
    assert loss == pytest.approx(0.2, rel=1e-5)
    assert student.bias.grad is not None


def test_strict_data_rejects_all_zero_value_weight_after_source_filter(monkeypatch):
    monkeypatch.setenv("MATRIX0_STRICT_DATA", "1")
    model = SourceFilterModel()
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    policy_size = int(np.prod(POLICY_SHAPE))
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.full((2, policy_size), 1.0 / policy_size, dtype=np.float32),
        "z": np.zeros((2,), dtype=np.float32),
        "value_weight": np.ones((2,), dtype=np.float32),
        "result_source": np.array(["capped", "capped"]),
    }

    with pytest.raises(ValueError, match="All value samples"):
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
            value_include_sources=["terminal"],
        )
