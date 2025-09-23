import logging

from azchess.utils import memory_monitor as memory_monitor_module
from azchess.utils.memory_monitor import MemoryMonitor


def test_memory_alert_messages_include_usage(monkeypatch, caplog):
    monitor = MemoryMonitor(warning_threshold=0.5, critical_threshold=0.8, alert_cooldown=0)
    device = "cuda"
    monitor.device_limits[device] = 10.0

    monkeypatch.setattr(memory_monitor_module, "get_memory_usage", lambda d: {"memory_gb": 9.0})
    monkeypatch.setattr(memory_monitor_module, "emergency_memory_cleanup", lambda d: None)
    monkeypatch.setattr(memory_monitor_module, "clear_memory_cache", lambda d: None)

    recorded_alerts = []
    monitor.alert_callbacks.append(recorded_alerts.append)

    with caplog.at_level(logging.CRITICAL):
        monitor._check_memory_and_alert(device)

    assert recorded_alerts, "Expected a critical alert to be recorded"
    critical_alert = recorded_alerts[-1]
    assert critical_alert.alert_type == "critical"
    assert critical_alert.message == "memory 9.0/10.0 GB (90%)"
    assert any(
        message == "MEMORY_ALERT: memory 9.0/10.0 GB (90%)"
        for message in caplog.messages
    ), "Critical MEMORY_ALERT log should include usage details"

    caplog.clear()
    monkeypatch.setattr(memory_monitor_module, "get_memory_usage", lambda d: {"memory_gb": 7.0})
    monitor.last_alert_time -= 1  # allow another alert immediately

    with caplog.at_level(logging.WARNING):
        monitor._check_memory_and_alert(device)

    warning_alert = recorded_alerts[-1]
    assert warning_alert.alert_type == "warning"
    assert warning_alert.message == "memory 7.0/10.0 GB (70%)"
    assert any(
        message == "MEMORY_ALERT: memory 7.0/10.0 GB (70%)"
        for message in caplog.messages
    ), "Warning MEMORY_ALERT log should include usage details"
