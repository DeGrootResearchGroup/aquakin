"""Check that ``constraints.txt`` resolves on every supported Python version.

CI installs with ``pip install -e ".[...]" -c constraints.txt`` on each Python
version in its test matrix, the oldest included. A pin that the oldest version
cannot install (a release that dropped it) fails those jobs before any test
runs, and a resolve on the local interpreter does not show it: pip evaluates the
constraints' environment markers against the Python it runs under, even with
``--python-version``. ``uv pip compile --python-version`` evaluates them for the
named version, so this resolves the project with every extra against the
constraints once per version listed in ``pyproject.toml``'s
``Programming Language :: Python :: 3.X`` classifiers.

Usage::

    python scripts/check_constraints.py

Run it from the repository root. It needs ``uv`` on ``PATH`` and network access
to the package index; without ``uv`` it says so and exits 0, so a contributor
without it is not blocked. The pre-push hook runs it when the commits being
pushed change ``constraints.txt`` or ``pyproject.toml``. Exits 1 when a version
does not resolve, printing uv's explanation.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")


def supported_pythons(pyproject: Path) -> list[str]:
    """Return the Python versions ``pyproject.toml``'s classifiers list.

    Parameters
    ----------
    pyproject : Path
        The project's ``pyproject.toml``.

    Returns
    -------
    list of str
        The ``3.X`` versions, in the order listed.

    Raises
    ------
    ValueError
        If no ``Programming Language :: Python :: 3.X`` classifier is listed,
        since there would then be nothing to check.
    """
    classifiers = tomllib.loads(pyproject.read_text())["project"].get("classifiers", [])
    versions = [m.group(1) for c in classifiers if (m := _CLASSIFIER.match(c))]
    if not versions:
        raise ValueError(
            f"{pyproject} lists no 'Programming Language :: Python :: 3.X' classifier, "
            "so there is no Python version to resolve constraints.txt for."
        )
    return versions


def resolve(uv: str, version: str) -> subprocess.CompletedProcess:
    """Resolve the project, with every extra, against ``constraints.txt``.

    Parameters
    ----------
    uv : str
        Path to the ``uv`` executable.
    version : str
        The Python version to resolve for, such as ``"3.11"``.

    Returns
    -------
    subprocess.CompletedProcess
        The finished ``uv pip compile`` run; a non-zero ``returncode`` means
        the constraints admit no installable set on ``version``.
    """
    return subprocess.run(
        [uv, "pip", "compile", "pyproject.toml", "--all-extras", "-c", "constraints.txt"]
        + ["--python-version", version, "--no-header", "--quiet"],
        capture_output=True,
        text=True,
    )


def main() -> int:
    """Resolve ``constraints.txt`` for each supported version and report.

    Returns
    -------
    int
        0 when every version resolves or ``uv`` is not installed, 1 otherwise.
    """
    uv = shutil.which("uv")
    if uv is None:
        print(
            "check_constraints: uv not found on PATH -- skipping the check that "
            "constraints.txt resolves on every supported Python.\n"
            "                   install it from https://docs.astral.sh/uv/",
            file=sys.stderr,
        )
        return 0
    failed = []
    for version in supported_pythons(Path("pyproject.toml")):
        result = resolve(uv, version)
        if result.returncode == 0:
            print(f"check_constraints: Python {version}: resolves", file=sys.stderr)
        else:
            failed.append(version)
            print(f"check_constraints: Python {version}: does not resolve", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
    if failed:
        print(
            f"check_constraints: constraints.txt cannot be installed on Python "
            f"{', '.join(failed)}, which CI tests. Pin a version that Python can "
            'install, with a marker such as `; python_version < "3.12"` if newer '
            "versions need a different pin.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
