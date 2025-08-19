import numpy as np
from pathlib import Path

from azchess.data_manager import DataManager


# numpy saves .npz files by appending the extension automatically when given a
# filename with a different suffix. The DataManager expects the temporary file
# to retain the provided ``.npz.tmp`` suffix, so we patch ``np.savez_compressed``
# to write to the exact path when the first argument is a string/Path.
import numpy as _np


def _safe_savez(file, *args, **kwargs):
    if isinstance(file, (str, Path)):
        with open(file, 'wb') as f:
            return _orig_savez(f, *args, **kwargs)
    return _orig_savez(file, *args, **kwargs)


_orig_savez = _np.savez_compressed


def _dummy_data(samples: int = 1):
    return {
        's': np.zeros((samples, 1), dtype=np.float32),
        'pi': np.zeros((samples, 1), dtype=np.float32),
        'z': np.zeros((samples,), dtype=np.float32),
    }


def test_shard_record_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(_np, "savez_compressed", _safe_savez)
    dm = DataManager(base_dir=str(tmp_path))
    shard_path = dm.add_training_data(_dummy_data(), shard_id=0)

    shards = dm._get_all_shards()
    assert len(shards) == 1
    assert shards[0].path == shard_path

    valid, corrupted = dm.validate_data_integrity()
    assert valid == 1 and corrupted == 0

    # Corrupt the file and ensure validation catches it
    Path(shard_path).write_bytes(b'corrupt')
    valid, corrupted = dm.validate_data_integrity()
    assert corrupted == 1


def test_cleanup_old_shards(tmp_path):
    dm = DataManager(base_dir=str(tmp_path))

    for i in range(3):
        fp = dm.replays_dir / f"shard_{i}.npz"
        np.savez(fp, **_dummy_data())
        checksum = dm._calculate_checksum(fp)
        dm._record_shard(str(fp), fp.stat().st_size, 1, f"20210101_00000{i}", checksum, source="selfplay")

    dm.cleanup_old_shards(keep_recent=1)

    shards = dm._get_all_shards()
    assert len(shards) == 1
    assert Path(shards[0].path).name == "shard_2.npz"
    assert Path(shards[0].path).exists()
