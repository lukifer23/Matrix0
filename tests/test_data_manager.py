import numpy as np

from azchess.data_manager import DataManager


def _make_shard_data(num_samples: int, fill_value: float) -> dict[str, np.ndarray]:
    states = np.full((num_samples, 19, 8, 8), fill_value, dtype=np.float32)
    policies = np.full((num_samples, 4672), fill_value, dtype=np.float32)
    values = np.full((num_samples,), fill_value, dtype=np.float32)
    return {'s': states, 'pi': policies, 'z': values}


def test_training_batch_balances_external_and_selfplay(tmp_path):
    np.random.seed(0)

    manager = DataManager(base_dir=str(tmp_path))

    manager.add_training_data(_make_shard_data(16, 1.0), shard_id=0, source="selfplay")
    manager.add_training_data(_make_shard_data(8, 2.0), shard_id=1, source="stockfish:mixed")

    batch_size = 10
    generator = manager.get_training_batch(batch_size)

    total_samples = 0
    external_samples = 0
    num_batches = 12

    for _ in range(num_batches):
        batch = next(generator)
        states = batch[0]
        total_samples += states.shape[0]
        external_samples += int(np.sum(np.isclose(states[:, 0, 0, 0], 2.0)))

    observed_ratio = external_samples / total_samples

    assert np.isclose(observed_ratio, 0.3, atol=0.05)
