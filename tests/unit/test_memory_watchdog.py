"""Unit tests for the CI memory watchdog (``tests/mem_watchdog.py`` + the
``pytest_runtest_teardown`` hook in ``tests/conftest.py``).

The watchdog converts a silent OOM runner-reclaim into a named test failure
(issue #337). These pin the pure decision/measurement helpers and the hook's
fire / no-fire / fire-once behaviour, driving memory readings by monkeypatch so
the tests are deterministic and platform-independent (the real reading is
Linux-only via ``/proc/meminfo``).
"""

import math

import pytest

from tests import conftest, mem_watchdog


class _FakeItem:
    """Minimal stand-in for a pytest item (only ``nodeid`` is read)."""

    def __init__(self, nodeid):
        self.nodeid = nodeid


@pytest.fixture(autouse=True)
def _reset_watchdog_state():
    """Each test starts with the once-per-worker fire latch cleared."""
    conftest._watchdog_fired = False
    yield
    conftest._watchdog_fired = False


# --- pure helpers -----------------------------------------------------------


def test_should_fire_below_floor():
    assert mem_watchdog.watchdog_should_fire(1000.0, 2048.0) is True


def test_should_not_fire_at_or_above_floor():
    assert mem_watchdog.watchdog_should_fire(2048.0, 2048.0) is False
    assert mem_watchdog.watchdog_should_fire(4096.0, 2048.0) is False


def test_should_not_fire_on_unavailable_reading():
    # None reading (non-Linux / unreadable) is inert, whatever the floor.
    assert mem_watchdog.watchdog_should_fire(None, 2048.0) is False


def test_mem_available_is_none_or_positive():
    # Linux: a positive MB reading; elsewhere: None. Never negative/zero-garbage.
    val = mem_watchdog.mem_available_mb()
    assert val is None or val > 0.0


def test_peak_rss_is_positive_or_nan():
    rss = mem_watchdog.peak_rss_mb()
    assert math.isnan(rss) or rss > 0.0


def test_floor_default_and_override(monkeypatch):
    monkeypatch.delenv("AQUAKIN_TEST_MEM_FLOOR_MB", raising=False)
    assert mem_watchdog.mem_floor_mb() == 2048.0
    monkeypatch.setenv("AQUAKIN_TEST_MEM_FLOOR_MB", "512")
    assert mem_watchdog.mem_floor_mb() == 512.0
    # Garbage falls back to the default rather than raising.
    monkeypatch.setenv("AQUAKIN_TEST_MEM_FLOOR_MB", "not-a-number")
    assert mem_watchdog.mem_floor_mb() == 2048.0


def test_enabled_flag_parsing(monkeypatch):
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("AQUAKIN_TEST_MEM_WATCHDOG", truthy)
        assert mem_watchdog.watchdog_enabled() is True
    for falsy in ("0", "false", "", "no"):
        monkeypatch.setenv("AQUAKIN_TEST_MEM_WATCHDOG", falsy)
        assert mem_watchdog.watchdog_enabled() is False


def test_message_names_test_and_floor():
    msg = mem_watchdog.watchdog_message("tests/x.py::test_y", 900.0, 2048.0)
    assert "tests/x.py::test_y" in msg
    assert "900" in msg and "2048" in msg
    assert "memory watchdog" in msg


# --- the hook (fire / no-fire / fire-once) ----------------------------------


def test_hook_fires_when_low(monkeypatch):
    monkeypatch.setenv("AQUAKIN_TEST_MEM_WATCHDOG", "1")
    monkeypatch.setattr(mem_watchdog, "mem_available_mb", lambda: 100.0)
    with pytest.raises(pytest.fail.Exception, match="memory watchdog"):
        conftest.pytest_runtest_teardown(_FakeItem("tests/x.py::test_low"), None)


def test_hook_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("AQUAKIN_TEST_MEM_WATCHDOG", "0")
    monkeypatch.setattr(mem_watchdog, "mem_available_mb", lambda: 1.0)
    # Disabled: no raise even though memory is (fake-)critically low.
    conftest.pytest_runtest_teardown(_FakeItem("tests/x.py::test_disabled"), None)


def test_hook_noop_when_memory_healthy(monkeypatch):
    monkeypatch.setenv("AQUAKIN_TEST_MEM_WATCHDOG", "1")
    monkeypatch.setattr(mem_watchdog, "mem_available_mb", lambda: 8192.0)
    conftest.pytest_runtest_teardown(_FakeItem("tests/x.py::test_healthy"), None)


def test_hook_noop_when_reading_unavailable(monkeypatch):
    monkeypatch.setenv("AQUAKIN_TEST_MEM_WATCHDOG", "1")
    monkeypatch.setattr(mem_watchdog, "mem_available_mb", lambda: None)
    conftest.pytest_runtest_teardown(_FakeItem("tests/x.py::test_no_reading"), None)


def test_hook_fires_only_once_per_worker(monkeypatch):
    monkeypatch.setenv("AQUAKIN_TEST_MEM_WATCHDOG", "1")
    monkeypatch.setattr(mem_watchdog, "mem_available_mb", lambda: 100.0)
    with pytest.raises(pytest.fail.Exception):
        conftest.pytest_runtest_teardown(_FakeItem("tests/x.py::test_first"), None)
    # Latch set: a second low reading does not raise again (no per-test flood).
    conftest.pytest_runtest_teardown(_FakeItem("tests/x.py::test_second"), None)
