import numpy as np
import chess

from azchess.data_manager import DataManager
from azchess.encoding import encode_board


POLICY_SIZE = 4672


def _write_mock_npz(base_dir, subdir, filename, sample_count):
    path = base_dir / subdir
    path.mkdir(parents=True, exist_ok=True)
    board = chess.Board()
    positions = np.stack([encode_board(board) for _ in range(sample_count)], axis=0)
    policy_targets = np.zeros((sample_count, POLICY_SIZE), dtype=np.float32)
    value_targets = np.zeros((sample_count,), dtype=np.float32)
    np.savez(
        path / filename,
        positions=positions,
        policy_targets=policy_targets,
        value_targets=value_targets,
    )


def test_tactical_batch_handles_small_file(tmp_path):
    _write_mock_npz(tmp_path, "tactical", "tactical_positions.npz", sample_count=2)
    manager = DataManager(base_dir=str(tmp_path))

    batch = manager._get_tactical_batch(batch_size=4)

    assert batch is not None
    assert batch['s'].shape == (4, 19, 8, 8)
    assert batch['pi'].shape == (4, POLICY_SIZE)
    assert batch['z'].shape == (4,)
    assert batch['legal_mask'].shape == (4, POLICY_SIZE)


def test_openings_batch_handles_small_file(tmp_path):
    _write_mock_npz(tmp_path, "openings", "openings_positions.npz", sample_count=1)
    manager = DataManager(base_dir=str(tmp_path))

    batch = manager._get_openings_batch(batch_size=3)

    assert batch is not None
    assert batch['s'].shape == (3, 19, 8, 8)
    assert batch['pi'].shape == (3, POLICY_SIZE)
    assert batch['z'].shape == (3,)
    assert batch['legal_mask'].shape == (3, POLICY_SIZE)


def test_replay_batch_derives_moves_left_from_meta_moves(tmp_path):
    manager = DataManager(base_dir=str(tmp_path))
    board = chess.Board()
    positions = np.stack([encode_board(board) for _ in range(3)], axis=0)
    policy_targets = np.full((3, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32)
    value_targets = np.zeros((3,), dtype=np.float32)
    manager.add_selfplay_data(
        {
            "s": positions,
            "pi": policy_targets,
            "z": value_targets,
            "legal_mask": np.ones((3, POLICY_SIZE), dtype=np.uint8),
            "meta_moves": np.array([3], dtype=np.int32),
            "meta_result_source": np.array(["terminal"]),
        },
        worker_id=0,
        game_id=0,
    )

    batch = next(manager.get_training_batch(batch_size=3))

    assert isinstance(batch, dict)
    assert sorted(batch["moves_left"].tolist()) == [1.0, 2.0, 3.0]
