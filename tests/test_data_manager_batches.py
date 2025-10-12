import numpy as np

from azchess.data_manager import DataManager


def test_replay_batch_exposes_ssl_targets(tmp_path):
    dm = DataManager(base_dir=str(tmp_path))

    sample_count = 4
    states = np.zeros((sample_count, dm.expected_planes, 8, 8), dtype=np.float32)
    policies = np.zeros((sample_count, 4672), dtype=np.float32)
    values = np.zeros((sample_count,), dtype=np.float32)
    legal_mask = np.ones((sample_count, 4672), dtype=np.uint8)
    ssl_piece = np.arange(sample_count * 7 * 8 * 8, dtype=np.float32).reshape(sample_count, 7, 8, 8)

    shard_data = {
        "s": states,
        "pi": policies,
        "z": values,
        "legal_mask": legal_mask,
        "ssl_piece": ssl_piece,
    }

    dm.add_training_data(shard_data, shard_id=0, source="selfplay")

    batch_iter = dm.get_training_batch(batch_size=2)
    batch = next(batch_iter)

    assert isinstance(batch, dict)
    assert batch["s"].shape == (2, dm.expected_planes, 8, 8)
    assert batch["pi"].shape == (2, 4672)
    assert batch["z"].shape == (2,)
    assert batch["legal_mask"].shape == (2, 4672)
    assert "ssl_piece" in batch
    assert batch["ssl_piece"].shape == (2, 7, 8, 8)
    assert batch["ssl_piece"].dtype == np.float32
