import logging

from azchess.training import train


def test_memory_warning_log_emitted_without_exception(monkeypatch, caplog):
    actions = []

    def fake_clear(device: str) -> None:
        actions.append(("clear", device))

    def fake_emergency(device: str) -> None:
        actions.append(("emergency", device))

    monkeypatch.setattr(train, "clear_memory_cache", fake_clear)
    monkeypatch.setattr(train, "emergency_memory_cleanup", fake_emergency)

    with caplog.at_level(logging.WARNING, logger=train.logger.name):
        updated_timestamp = train._handle_memory_thresholds(
            memory_usage_gb=9.0,
            memory_limit_gb=10.0,
            warning_threshold=0.8,
            critical_threshold=0.95,
            last_memory_warning=0.0,
            memory_warning_cooldown=300.0,
            device="cpu",
            now_fn=lambda: 1234.0,
        )

    assert updated_timestamp == 1234.0
    assert ("clear", "cpu") in actions
    assert all(action[0] != "emergency" for action in actions)
    assert any("HIGH MEMORY USAGE" in message for message in caplog.messages)
