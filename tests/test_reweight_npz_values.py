from __future__ import annotations

import numpy as np

from azchess.tools.reweight_npz_values import reweight_npz_tree


def _write_shard(path, source: str, value_weight: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        s=np.zeros((2, 19, 8, 8), dtype=np.float32),
        pi=np.full((2, 4), 0.25, dtype=np.float32),
        z=np.array([0.5, -0.5], dtype=np.float32),
        value_weight=np.full((2,), value_weight, dtype=np.float32),
        meta_result_source=np.array([source]),
        meta_value_weight=np.array([value_weight], dtype=np.float32),
        meta_value_bootstrap=np.array([1 if source == "capped" else 0], dtype=np.int8),
    )


def test_reweight_npz_tree_zeroes_capped_values_and_preserves_terminal(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_shard(src / "selfplay" / "capped.npz", "capped", 0.25)
    _write_shard(src / "selfplay" / "terminal.npz", "terminal", 1.0)
    (src / "data_metadata.db").write_text("metadata")

    report = reweight_npz_tree(src, dst, {"capped"}, 0.0)

    assert report["files"] == 2
    assert report["changed_files"] == 1
    assert report["changed_positions"] == 2
    assert (dst / "data_metadata.db").read_text() == "metadata"

    with np.load(dst / "selfplay" / "capped.npz") as capped:
        np.testing.assert_allclose(capped["pi"], 0.25)
        np.testing.assert_allclose(capped["value_weight"], 0.0)
        np.testing.assert_allclose(capped["meta_value_weight"], 0.0)
        np.testing.assert_array_equal(capped["meta_value_bootstrap"], np.array([0], dtype=np.int8))

    with np.load(dst / "selfplay" / "terminal.npz") as terminal:
        np.testing.assert_allclose(terminal["value_weight"], 1.0)
        np.testing.assert_allclose(terminal["meta_value_weight"], 1.0)
        np.testing.assert_array_equal(terminal["meta_value_bootstrap"], np.array([0], dtype=np.int8))
