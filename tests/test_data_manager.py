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


def test_result_source_mix_preserves_value_weights_when_one_source_omits_them(tmp_path):
    manager = DataManager(base_dir=str(tmp_path))
    capped = _make_sourced_shard_data(8, "capped", 0.25, fill_value=1.0)
    terminal = _make_sourced_shard_data(8, "terminal", 1.0, fill_value=2.0)
    terminal.pop("value_weight")
    manager.add_training_data(capped, shard_id=0, source="replay")
    manager.add_training_data(terminal, shard_id=1, source="replay")

    batch = next(manager.get_training_batch_by_result_source_mix(8, {"capped": 0.5, "terminal": 0.5}))

    assert "value_weight" in batch
    capped_weights = batch["value_weight"][batch["result_source"] == "capped"]
    terminal_weights = batch["value_weight"][batch["result_source"] == "terminal"]
    np.testing.assert_allclose(capped_weights, 0.25)
    np.testing.assert_allclose(terminal_weights, 1.0)


def test_result_source_mix_uses_sample_count_shard_weights(tmp_path, monkeypatch):
    manager = DataManager(base_dir=str(tmp_path))
    manager.add_training_data(_make_sourced_shard_data(4, "terminal", 1.0, fill_value=0.25), shard_id=0, source="replay")
    manager.add_training_data(_make_sourced_shard_data(40, "terminal", 1.0, fill_value=0.75), shard_id=1, source="replay")

    original_choice = np.random.choice
    observed_probabilities = []

    def recording_choice(a, size=None, replace=True, p=None):
        if p is not None and np.asarray(a).dtype.kind in {"U", "S", "O"}:
            observed_probabilities.append(np.asarray(p, dtype=np.float64))
        return original_choice(a, size=size, replace=replace, p=p)

    monkeypatch.setattr(np.random, "choice", recording_choice)

    batch = next(manager.get_training_batch_by_result_source_mix(8, {"terminal": 1.0}))

    assert batch["s"].shape[0] == 8
    assert observed_probabilities
    np.testing.assert_allclose(observed_probabilities[0], np.array([4.0 / 44.0, 40.0 / 44.0]))


def test_stockfish_mixed_batch_accepts_source_prefix_dict_batches(tmp_path):
    manager = DataManager(base_dir=str(tmp_path))
    manager.add_training_data(
        _make_sourced_shard_data(8, "stockfish:openings/main", 1.0, fill_value=1.0),
        shard_id=0,
        source="stockfish:openings/main",
    )
    manager.add_training_data(
        _make_sourced_shard_data(8, "stockfish:tactical/main", 0.5, fill_value=0.5),
        shard_id=1,
        source="stockfish:tactical/main",
    )

    batch = manager._get_stockfish_mixed_batch(6)

    assert batch is not None
    assert batch["s"].shape[0] == batch["value_weight"].shape[0] == batch["result_source"].shape[0]
    assert set(np.unique(batch["result_source"])) == {
        "stockfish:openings/main",
        "stockfish:tactical/main",
    }
    np.testing.assert_allclose(
        batch["value_weight"][batch["result_source"] == "stockfish:tactical/main"],
        0.5,
    )


def test_mixed_external_batch_preserves_source_metadata(tmp_path, monkeypatch):
    manager = DataManager(base_dir=str(tmp_path))

    tactical = _make_sourced_shard_data(3, "tactical", 1.0, fill_value=1.0)
    tactical["result_source"] = np.asarray(["tactical"] * 3)
    tactical.pop("meta_result_source")
    openings = _make_sourced_shard_data(3, "openings", 1.0, fill_value=2.0)
    openings["result_source"] = np.asarray(["openings"] * 3)
    openings.pop("meta_result_source")

    monkeypatch.setattr(manager, "_get_tactical_batch", lambda _batch_size: tactical)
    monkeypatch.setattr(manager, "_get_openings_batch", lambda _batch_size: openings)

    batch = manager._get_mixed_external_batch(6)

    assert batch is not None
    assert batch["value_weight"].shape[0] == 6
    assert batch["result_source"].shape[0] == 6
    assert set(np.unique(batch["result_source"])) == {"openings", "tactical"}


def test_curriculum_mixed_batch_preserves_external_and_replay_metadata(tmp_path):
    manager = DataManager(base_dir=str(tmp_path))
    manager.add_training_data(
        _make_sourced_shard_data(16, "selfplay", 0.75, fill_value=1.0),
        shard_id=0,
        source="selfplay",
    )
    manager.add_training_data(
        _make_sourced_shard_data(16, "stockfish:openings/main", 1.0, fill_value=0.5),
        shard_id=1,
        source="stockfish:openings/main",
    )

    batch = manager._get_curriculum_mixed_batch(8)

    assert batch is not None
    assert batch["s"].shape[0] == batch["value_weight"].shape[0] == batch["result_source"].shape[0]
    assert "selfplay" in set(np.unique(batch["result_source"]))
    assert "stockfish:openings/main" in set(np.unique(batch["result_source"]))
