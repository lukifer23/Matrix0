from types import SimpleNamespace

from azchess.selfplay.internal import _allows_missing_moves_left_head


def test_allows_missing_moves_left_head_for_enabled_model():
    model = SimpleNamespace(moves_left_head=object())

    assert _allows_missing_moves_left_head(
        model,
        [
            "moves_left_head.2.weight",
            "moves_left_head.2.bias",
            "moves_left_head.4.weight",
            "moves_left_head.4.bias",
        ],
        [],
    )


def test_rejects_moves_left_migration_when_head_disabled():
    model = SimpleNamespace(moves_left_head=None)

    assert not _allows_missing_moves_left_head(model, ["moves_left_head.2.weight"], [])


def test_rejects_non_moves_left_missing_or_unexpected_keys():
    model = SimpleNamespace(moves_left_head=object())

    assert not _allows_missing_moves_left_head(model, ["value_head.weight"], [])
    assert not _allows_missing_moves_left_head(model, ["moves_left_head.2.weight"], ["extra.weight"])
