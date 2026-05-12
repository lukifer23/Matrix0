import numpy as np

from azchess.data_manager import DataManager


def _make_shard_data(num_samples: int, fill_value: float) -> dict[str, np.ndarray]:
    states = np.full((num_samples, 19, 8, 8), fill_value, dtype=np.float32)
    policies = np.full((num_samples, 4672), fill_value, dtype=np.float32)
    values = np.full((num_samples,), fill_value, dtype=np.float32)
    return {'s': states, 'pi': policies, 'z': values}


def _make_sourced_shard_data(num_samples: int, source: str, value_weight: float, fill_value: float = 1.0) -> dict[str, np.ndarray]:
    data = _make_shard_data(num_samples, fill_value)
    data["value_weight"] = np.full((num_samples,), value_weight, dtype=np.float32)
    data["meta_result_source"] = np.array([source])
    return data


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
        states = batch["s"] if isinstance(batch, dict) else batch[0]
        total_samples += states.shape[0]
        external_samples += int(np.sum(np.isclose(states[:, 0, 0, 0], 2.0)))

    observed_ratio = external_samples / total_samples

    assert np.isclose(observed_ratio, 0.3, atol=0.05)


def test_training_batch_includes_result_source_metadata(tmp_path):
    manager = DataManager(base_dir=str(tmp_path))
    manager.add_training_data(_make_sourced_shard_data(4, "capped", 0.25), shard_id=0, source="selfplay")

    batch = next(manager.get_training_batch(4))

    assert isinstance(batch, dict)
    assert batch["result_source"].tolist() == ["capped"] * 4
    np.testing.assert_allclose(batch["value_weight"], 0.25)


def test_training_batch_by_result_source_mix_balances_prefixes(tmp_path):
    manager = DataManager(base_dir=str(tmp_path))
    manager.add_training_data(_make_sourced_shard_data(16, "capped", 0.25, fill_value=1.0), shard_id=0, source="replay")
    manager.add_training_data(_make_sourced_shard_data(16, "tablebase", 1.0, fill_value=2.0), shard_id=1, source="replay")
    manager.add_training_data(_make_sourced_shard_data(16, "terminal", 1.0, fill_value=3.0), shard_id=2, source="replay")

    batch = next(
        manager.get_training_batch_by_result_source_mix(
            12,
            {
                "capped": 0.25,
                "tablebase": 0.50,
                "terminal": 0.25,
            },
        )
    )

    sources, counts = np.unique(batch["result_source"], return_counts=True)
    assert dict(zip(sources.tolist(), counts.tolist())) == {"capped": 3, "tablebase": 6, "terminal": 3}
    assert batch["s"].shape[0] == 12
    assert batch["value_weight"].shape[0] == 12
