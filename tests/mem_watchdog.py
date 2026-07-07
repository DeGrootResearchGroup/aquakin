"""Memory watchdog: turn a silent OOM runner-reclaim into a named test failure.

Every sharded CI job (the fast gate's ``-n auto`` workers, and the ``-n 1``
slow / validation / heavy / xheavy / smoke / refresh shards) accumulates XLA
compilation cache + live JAX buffers as it runs. When a shard's footprint grows
past the hosted runner's RAM -- which happens *silently* as new heavy tests are
added and a shard tips over -- the runner is OOM-reclaimed mid-suite: no test
fails, the job surfaces only as a generic ``"the runner has received a shutdown
signal" / "the operation was canceled"`` cancellation with nothing to point at.
Sharding bounds each process to ~1/N but does not *prevent* this -- more shards
only relocate an overweight one (see ``docs/ci.md``).

This watchdog samples the **system-wide available memory** after every test and,
the moment it falls to a floor, fails the currently-finishing test *loudly* --
naming the test, the worker, and the process peak RSS -- before the OS reclaim.
It is a symptom guard (it does not care *which* solve is heavy), so it never
misfires on the ~80 legitimately-large-but-cheap plant solves the fast gate
already runs (state sizes up to BSM1-scale); it fires only when a shard is
genuinely about to exhaust the runner. That converts the confusing silent
cancellation into an actionable failure: mark the heaviest whole-plant / adjoint
solve in the shard ``@pytest.mark.slow`` (or ``heavy``), or raise the shard
count.

System ``MemAvailable`` (not per-process RSS) is the signal because the OOM
condition is aggregate: under ``-n auto`` one worker can balloon while its
siblings stay small, and it is the *sum* that reclaims the runner. Reading the
shared ``MemAvailable`` catches that regardless of which worker grew, with no
per-job / per-worker-count cap tuning.

Enabled by ``AQUAKIN_TEST_MEM_WATCHDOG=1`` (set at the workflow level in
``.github/workflows/ci.yml`` so every test job inherits it); off by default so a
local run is never surprised. Linux-only in effect -- ``/proc/meminfo`` is the
source, so on macOS / Windows the watchdog reads ``None`` and no-ops. Tunable:
``AQUAKIN_TEST_MEM_FLOOR_MB`` (default 2048) is the available-memory floor.
"""

import os
import sys

# --- configuration (read once at import) ------------------------------------

_TRUE = {"1", "true", "yes", "on"}


def watchdog_enabled() -> bool:
    """True when ``AQUAKIN_TEST_MEM_WATCHDOG`` is set to a truthy value."""
    return os.environ.get("AQUAKIN_TEST_MEM_WATCHDOG", "").strip().lower() in _TRUE


def mem_floor_mb() -> float:
    """Available-memory floor in MB (``AQUAKIN_TEST_MEM_FLOOR_MB``, default 2048).

    When system ``MemAvailable`` drops below this, the watchdog fires. The
    default leaves ~2 GB of headroom on the 16 GB hosted runner -- enough to
    fail cleanly in a test's teardown before the *next* (multi-GB) solve tips
    the runner into an OS reclaim.
    """
    raw = os.environ.get("AQUAKIN_TEST_MEM_FLOOR_MB", "").strip()
    if not raw:
        return 2048.0
    try:
        return float(raw)
    except ValueError:
        return 2048.0


# --- measurement (pure, testable) -------------------------------------------


def mem_available_mb():
    """System available memory in MB from ``/proc/meminfo``, or ``None``.

    Returns ``None`` when ``/proc/meminfo`` is absent or has no
    ``MemAvailable`` line (non-Linux, where the watchdog then no-ops).
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    # "MemAvailable:   12345678 kB" -> MB
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def peak_rss_mb() -> float:
    """This process's peak resident set size in MB (``getrusage``).

    ``ru_maxrss`` is bytes on macOS/BSD and kilobytes on Linux; normalize both.
    Returns ``nan`` if :mod:`resource` is unavailable (e.g. Windows).
    """
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # resource missing / unusable (e.g. Windows)
        return float("nan")
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def watchdog_should_fire(available_mb, floor_mb) -> bool:
    """Fire iff a reading was obtained and it is below the floor.

    A ``None`` reading (non-Linux, or ``/proc/meminfo`` unreadable) never
    fires, so the watchdog is inert off-CI.
    """
    return available_mb is not None and available_mb < floor_mb


def watchdog_message(nodeid: str, available_mb: float, floor_mb: float) -> str:
    """The failure message: what tripped, on which test/worker, and the fix."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    rss = peak_rss_mb()
    rss_note = "" if rss != rss else f" (this process peak RSS {rss:.0f} MB)"  # nan check
    return (
        f"[memory watchdog] system MemAvailable fell to {available_mb:.0f} MB "
        f"(floor {floor_mb:.0f} MB) after {nodeid} on worker {worker}{rss_note}. "
        f"This shard is accumulating memory (XLA compile cache + live JAX buffers) "
        f"toward the runner's OOM limit; the reclaim that would follow is silent -- "
        f"a generic runner cancellation with no failing test. Mark the heaviest "
        f"whole-plant / adjoint solve in this shard @pytest.mark.slow (or heavy), "
        f"or raise the shard count. See docs/ci.md and issue #337."
    )
