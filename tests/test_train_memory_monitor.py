from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from azchess.utils.memory_monitor import MemoryAlert, memory_monitor


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):  # pragma: no cover - not used in this test
        return self.linear(x)

    def enable_gradient_checkpointing(self, strategy: str = "adaptive") -> None:
        self._checkpoint_strategy = strategy

    def get_memory_usage(self):
        return {"parameters_gb": 0.0}

    def enable_memory_optimization(self) -> None:
        self._memory_optimization_enabled = True


class DummyPolicyValueNet:
    @classmethod
    def from_config(cls, _cfg):
        return DummyModel()


class DummyDataManager:
    def __init__(self, *_, **__):
        self._stats = SimpleNamespace(total_samples=1, total_shards=1)

    def get_stats(self):
        return self._stats

    def get_external_data_stats(self):
        return {"external_total": 0, "tactical_samples": 0, "openings_samples": 0}

    def get_curriculum_batch(self, *_args, **_kwargs):
        return None

    def get_training_batch(self, *_args, **_kwargs):
        while False:
            yield None


class DummySummaryWriter:
    def __init__(self, *_args, **_kwargs) -> None:
        self.scalars = []

    def add_scalar(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class DummyConfig:
    def __init__(self, log_dir: str) -> None:
        self._root = {}
        self._training = {
            "gradient_accumulation_steps": None,
            "compile": False,
            "compile_mode": "default",
            "steps_per_epoch": 1,
            "use_curriculum": False,
            "curriculum_phases": [],
            "dataloader_workers": 0,
            "prefetch_factor": 0,
            "memory_limit_gb": 16,
            "memory_warning_threshold": 0.85,
            "memory_critical_threshold": 0.95,
            "ssl_weight": 0.0,
            "policy_label_smoothing": 0.0,
            "value_loss": "mse",
            "huber_delta": 1.0,
            "policy_masking": False,
            "ssl_warmup_steps": 0,
            "ssl_target_weight": 1.0,
            "ssl_targets_provider": "auto",
            "wdl_weight": 0.0,
            "wdl_margin": 0.25,
            "precision": "fp32",
            "ssl_every_n": 1,
            "ssl_chunk_size": 0,
            "log_dir": log_dir,
        }
        self._model = {"self_supervised": False, "wdl": False, "ssl_curriculum": False}

    def training(self):
        return self._training

    def model(self):
        return self._model

    def get(self, key, default=None):
        return self._root.get(key, default)


def test_train_comprehensive_memory_monitor_cleanup(monkeypatch, tmp_path):
    from azchess.training import train

    # Ensure monitor is not running before the test
    memory_monitor.stop_monitoring()
    memory_monitor.alert_callbacks.clear()

    dummy_log_dir = tmp_path / "logs"
    dummy_ckpt_dir = tmp_path / "ckpts"

    # Patch heavy dependencies with lightweight stand-ins
    monkeypatch.setattr(train, "PolicyValueNet", DummyPolicyValueNet)
    monkeypatch.setattr(train, "DataManager", DummyDataManager)
    monkeypatch.setattr(train, "build_training_dataloader", lambda *a, **k: None)
    monkeypatch.setattr(train, "SummaryWriter", DummySummaryWriter)
    monkeypatch.setattr(train, "clear_memory_cache", lambda *a, **k: None)
    monkeypatch.setattr(train, "get_memory_usage", lambda *a, **k: {"memory_gb": 1.0})
    monkeypatch.setattr(train, "emergency_memory_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(train, "save_checkpoint", lambda *a, **k: None)

    def fake_config_load(_path):
        return DummyConfig(str(dummy_log_dir))

    monkeypatch.setattr(train.Config, "load", staticmethod(fake_config_load))

    warning_messages: list[str] = []
    critical_messages: list[str] = []

    def capture_warning(msg, *args, **kwargs):
        text = msg % args if args else msg
        warning_messages.append(text)

    def capture_critical(msg, *args, **kwargs):
        text = msg % args if args else msg
        critical_messages.append(text)

    monkeypatch.setattr(train.logger, "warning", capture_warning)
    monkeypatch.setattr(train.logger, "critical", capture_critical)

    callback_lengths: list[int] = []
    alert_fire_counts: list[int] = []

    def tracked_add_callback(callback):
        pre_len = len(memory_monitor.alert_callbacks)
        callback_lengths.append(pre_len)
        memory_monitor.add_alert_callback(callback)
        post_len = len(memory_monitor.alert_callbacks)
        callback_lengths.append(post_len)

        before_warning = len(warning_messages)
        alert = MemoryAlert(
            alert_type="warning",
            message="unit-test alert",
            memory_usage_gb=1.0,
            memory_limit_gb=2.0,
            timestamp=time.time(),
            device="cpu",
        )
        memory_monitor._send_alert(alert)
        high_memory_logs = [
            msg for msg in warning_messages[before_warning:] if msg.startswith("HIGH MEMORY:")
        ]
        alert_fire_counts.append(len(high_memory_logs))

    monkeypatch.setattr(train, "add_memory_alert_callback", tracked_add_callback)

    run_kwargs = dict(
        config_path="dummy",
        total_steps=0,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.0,
        ema_decay=0.0,
        grad_clip_norm=1.0,
        accum_steps=1,
        warmup_steps=0,
        checkpoint_dir=str(dummy_ckpt_dir),
        log_dir=str(dummy_log_dir),
        device="cpu",
        use_amp=False,
        augment=False,
        precision="fp32",
        epochs=0,
        steps_per_epoch=0,
        init_checkpoint=None,
        resume=False,
        data_mode=None,
        dataloader_workers=0,
        prefetch_factor=0,
    )

    # Run twice to ensure cleanup between runs
    train.train_comprehensive(**run_kwargs)
    assert not memory_monitor.is_monitoring
    assert not (memory_monitor.monitor_thread and memory_monitor.monitor_thread.is_alive())
    assert len(memory_monitor.alert_callbacks) == 0

    train.train_comprehensive(**run_kwargs)
    assert not memory_monitor.is_monitoring
    assert not (memory_monitor.monitor_thread and memory_monitor.monitor_thread.is_alive())
    assert len(memory_monitor.alert_callbacks) == 0

    # Verify callbacks were not accumulated and only fired once per registration
    assert callback_lengths == [0, 1, 0, 1]
    assert alert_fire_counts == [1, 1]
    assert all(msg.startswith("HIGH MEMORY:") for msg in warning_messages if "HIGH MEMORY:" in msg)
    assert critical_messages == []
