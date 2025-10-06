import pytest
import torch
import torch.nn as nn

from azchess.model.resnet import NetConfig, PolicyValueNet, ChessSpecificFeatures


@pytest.mark.parametrize("norm", ["batch", "group"])
@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_chess_specific_features_respects_config(norm: str, activation: str) -> None:
    cfg = NetConfig(
        planes=19,
        channels=64,
        blocks=1,
        policy_size=4672,
        se=False,
        attention=False,
        chess_features=True,
        piece_square_tables=True,
        self_supervised=False,
        norm=norm,
        activation=activation,
    )

    model = PolicyValueNet(cfg)

    assert isinstance(model.chess_features, ChessSpecificFeatures)

    if norm == "batch":
        expected_norm = nn.BatchNorm2d
    else:
        expected_norm = nn.GroupNorm

    assert isinstance(model.chess_features.pst_norm, expected_norm)
    assert isinstance(model.chess_features.interaction_norm, expected_norm)

    expected_activation = nn.SiLU if activation == "silu" else nn.ReLU
    assert isinstance(model.chess_features.pst_activation, expected_activation)
    assert isinstance(model.chess_features.interaction_activation, expected_activation)

    x = torch.randn(2, cfg.planes, 8, 8)
    policy, value = model(x, return_ssl=False)
    assert policy.shape == (2, cfg.policy_size)
    assert value.shape == (2,)
