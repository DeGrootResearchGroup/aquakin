"""``scripts/check_constraints.py`` resolves the pins once per supported Python.

The script runs ``uv pip compile`` for each ``Programming Language :: Python ::
3.X`` classifier and fails when any version does not resolve. These tests run
the real script in a scratch directory against a stub ``uv`` that records the
versions it is asked for and fails on the ones named in ``FAIL_ON``, so they
need neither uv nor the network.
"""

import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_constraints.py"

PYPROJECT = """\
[project]
name = "demo"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
]
"""

STUB_UV = """\
#!/bin/sh
while [ $# -gt 0 ]; do
  [ "$1" = "--python-version" ] && version=$2
  shift
done
echo "$version" >> "$(dirname "$0")/calls"
case " $FAIL_ON " in
  *" $version "*) echo "no solution on $version" >&2; exit 1 ;;
esac
"""


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("check_constraints", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supported_pythons_reads_the_versioned_classifiers(script, tmp_path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    assert script.supported_pythons(tmp_path / "pyproject.toml") == ["3.11", "3.12"]


def test_supported_pythons_includes_the_requires_python_floor(script):
    # The floor is the version a too-new pin breaks, so the repository's own
    # classifiers must list it or the check would never resolve for it.
    pyproject = SCRIPT.parents[1] / "pyproject.toml"
    floor = pyproject.read_text().split('requires-python = ">=')[1].split('"')[0]
    assert script.supported_pythons(pyproject)[0] == floor


def test_supported_pythons_without_a_version_raises(script, tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nclassifiers = ["Typing :: Typed"]\n')
    with pytest.raises(ValueError, match=re.escape("no 'Programming Language :: Python :: 3.X'")):
        script.supported_pythons(tmp_path / "pyproject.toml")


def _run(tmp_path, *, uv=True, fail_on=""):
    """Run the script in a scratch project, with the stub uv on PATH if asked."""
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    (tmp_path / "constraints.txt").write_text("jax==0.12.1\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if uv:
        stub = bin_dir / "uv"
        stub.write_text(STUB_UV)
        stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "FAIL_ON": fail_on}
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True, env=env
    )
    calls = bin_dir / "calls"
    asked = calls.read_text().split() if calls.exists() else []
    return result, asked


def test_every_supported_version_resolving_passes(tmp_path):
    result, asked = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert asked == ["3.11", "3.12"]
    assert "Python 3.11: resolves" in result.stderr


def test_a_version_that_does_not_resolve_fails_and_is_named(tmp_path):
    result, asked = _run(tmp_path, fail_on="3.11")
    assert result.returncode == 1
    assert asked == ["3.11", "3.12"]  # every version is still checked
    assert "Python 3.11: does not resolve" in result.stderr
    assert "no solution on 3.11" in result.stderr  # uv's explanation is shown
    assert "Python 3.12: resolves" in result.stderr
    assert "cannot be installed on Python 3.11," in result.stderr


def test_the_lint_extra_provides_uv():
    # The skip message tells the reader to install the `lint` extra; that only
    # works while the extra lists uv.
    pyproject = tomllib.loads((SCRIPT.parents[1] / "pyproject.toml").read_text())
    lint = pyproject["project"]["optional-dependencies"]["lint"]
    assert any(re.match(r"uv\b", requirement) for requirement in lint), lint


def test_without_uv_it_skips_and_passes(tmp_path):
    if os.path.exists("/usr/bin/uv") or os.path.exists("/bin/uv"):
        pytest.skip("uv is installed system-wide, so it cannot be hidden from PATH")
    result, asked = _run(tmp_path, uv=False)
    assert result.returncode == 0
    assert asked == []
    assert "uv not found on PATH -- skipping" in result.stderr
    assert 'pip install -e ".[lint]"' in result.stderr
