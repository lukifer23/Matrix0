from azchess.selfplay.__main__ import _join_or_terminate_workers


class FakeProcess:
    def __init__(self, alive=True):
        self._alive = alive
        self.exitcode = None
        self.join_calls = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.exitcode = -9


def test_join_or_terminate_workers_stops_completed_shutdown_hang():
    proc = FakeProcess(alive=True)

    _join_or_terminate_workers([proc], done=1, total=1, join_timeout=0.0, terminate_timeout=0.0)

    assert proc.terminated
    assert not proc.is_alive()
    assert proc.join_calls == [0.0, 0.0]


def test_join_or_terminate_workers_stops_stalled_incomplete_worker():
    proc = FakeProcess(alive=True)

    _join_or_terminate_workers([proc], done=0, total=1, join_timeout=0.0, terminate_timeout=0.0)

    assert proc.terminated
    assert proc.exitcode == -15


def test_join_or_terminate_workers_leaves_exited_worker_alone():
    proc = FakeProcess(alive=False)

    _join_or_terminate_workers([proc], done=1, total=1, join_timeout=0.0, terminate_timeout=0.0)

    assert not proc.terminated
    assert not proc.killed
    assert proc.join_calls == [0.0]
