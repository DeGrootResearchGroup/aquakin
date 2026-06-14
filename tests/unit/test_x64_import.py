"""``import aquakin`` enables JAX x64 mode, and warns when it overrides an
explicit float32 choice (issue #191).

x64 is GLOBAL, process-wide JAX state, so each scenario runs in a fresh
subprocess: importing aquakin into a process that has already imported JAX at
the default float32 (or with ``JAX_ENABLE_X64`` set off) emits a one-time
warning so the side effect is not silent; a plain fresh import is silent.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import aquakin

# The package root to put on the subprocess's path so it imports *this* aquakin.
_ROOT = str(Path(aquakin.__file__).resolve().parents[1])

# Strip any editable-install finder so the inserted path wins, set up the
# scenario, then record warnings emitted while importing aquakin.
_SCRIPT = """
import sys
sys.meta_path = [m for m in sys.meta_path
                 if "aquakin" not in getattr(type(m), "__module__", "")
                 and "editable" not in getattr(type(m), "__module__", "").lower()]
sys.path.insert(0, {root!r})
{preamble}
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import aquakin
import jax
warned = any("x64" in str(w.message).lower() for w in caught)
print("X64", bool(jax.config.jax_enable_x64))
print("WARNED", warned)
"""


def _run(preamble: str = "", env: dict | None = None):
    import os
    e = dict(os.environ)
    e.pop("JAX_ENABLE_X64", None)         # start from a clean default
    if env:
        e.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(root=_ROOT, preamble=preamble)],
        capture_output=True, text=True, env=e, timeout=180)
    assert proc.returncode == 0, proc.stderr
    out = dict(line.split(" ", 1) for line in proc.stdout.splitlines() if " " in line)
    return out["X64"].strip() == "True", out["WARNED"].strip() == "True"


def test_fresh_import_enables_x64_silently():
    x64, warned = _run("")
    assert x64 and not warned


def test_warns_when_jax_already_imported_at_float32():
    x64, warned = _run("import jax")
    assert x64 and warned


def test_no_warning_when_jax_already_enabled_x64():
    x64, warned = _run("import jax; jax.config.update('jax_enable_x64', True)")
    assert x64 and not warned


def test_warns_when_env_var_forces_x64_off():
    x64, warned = _run("", env={"JAX_ENABLE_X64": "0"})
    assert x64 and warned
