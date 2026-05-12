import numpy as np

from azchess.training.npz_dataset import NPZBatchIterableDataset


class DummyDataManager:
    def get_curriculum_batch(self, batch_size, phase):
        return {
            "s": np.zeros((batch_size, 19, 8, 8), dtype=np.float32),
            "pi": np.full((batch_size, 4672), 1.0 / 4672.0, dtype=np.float32),
            "z": np.zeros((batch_size,), dtype=np.float32),
            "legal_mask": np.ones((batch_size, 4672), dtype=np.uint8),
            "moves_left": np.arange(batch_size, 0, -1, dtype=np.float32),
            "ssl_piece": np.zeros((batch_size, 13, 8, 8), dtype=np.float32),
            "ssl_threat": np.zeros((batch_size, 8, 8), dtype=np.float32),
        }

    def get_training_batch_by_result_source_mix(self, batch_size, source_mix):
        yield {
            "s": np.zeros((batch_size, 19, 8, 8), dtype=np.float32),
            "pi": np.full((batch_size, 4672), 1.0 / 4672.0, dtype=np.float32),
            "z": np.zeros((batch_size,), dtype=np.float32),
            "result_source": np.asarray(["terminal"] * batch_size),
        }


def test_curriculum_dataloader_preserves_ssl_targets():
    dataset = NPZBatchIterableDataset(DummyDataManager(), batch_size=2, mode="phase:openings")
    batch = next(iter(dataset))

    assert isinstance(batch, dict)
    assert batch["s"].shape == (2, 19, 8, 8)
    assert batch["legal_mask"].shape == (2, 4672)
    assert batch["moves_left"].shape == (2,)
    assert batch["ssl_piece"].shape == (2, 13, 8, 8)
    assert batch["ssl_threat"].shape == (2, 8, 8)


def test_dataset_uses_result_source_mix_when_configured():
    dataset = NPZBatchIterableDataset(
        DummyDataManager(),
        batch_size=3,
        mode="replay",
        result_source_mix={"terminal": 1.0},
    )
    batch = next(iter(dataset))

    assert batch["s"].shape == (3, 19, 8, 8)
    assert batch["result_source"].tolist() == ["terminal", "terminal", "terminal"]
