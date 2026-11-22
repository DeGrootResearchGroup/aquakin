"""Set up a Claude Code cloud session: install the project and enable the pre-push hook.

Run by the SessionStart hook in ``.claude/settings.json``. A cloud session starts
from a fresh clone with none of the project's dependencies, so without this
every session begins with a manual install and the committed pre-push hook
(``.githooks/pre-push``) stays off. In a cloud session this script

* installs the project editable with the extras that run the CI gates (``test``,
  which pulls in ``ui``; ``lint``; ``typecheck``), pinned by ``constraints.txt``
  so ruff and pyright match CI, unless the same install already succeeded in
  this Python environment (with ``PIP_NO_CACHE_DIR``, inherited by the build
  subprocess too, so pip keeps no download cache outside the environment); and
* points ``core.hooksPath`` at ``.githooks`` unless it already points somewhere;
  and
* puts the interpreter's scripts directory first on ``PATH`` for the session's
  commands (through ``CLAUDE_ENV_FILE``), because the cloud image ships its own
  ``ruff`` and ``pyright`` ahead of it on ``PATH``, and a bare ``ruff`` would
  otherwise not be the version CI pins.

It does nothing outside a cloud session (``CLAUDE_CODE_REMOTE`` is not
``true``): on a workstation the developer chose their own environment, and
installing into whichever ``python3`` is first on ``PATH`` or rewriting their git
configuration would be a surprise.

"Already installed" is a SHA-256 of the files that decide what gets installed
(``pyproject.toml``, ``constraints.txt``, this script) and the checkout's path,
kept in a marker file in the interpreter's ``purelib`` directory. The marker
therefore disappears with the packages (a new container), and a change to a
pin, an extra, or the checkout's location triggers a fresh install. The marker
is removed before an install and written only after one succeeds, so a failed
or interrupted install is retried at the next start.

The hook never blocks the session: the script exits 0 whatever the install or
git did (an unexpected error is a traceback, which Claude Code reports as a
non-blocking hook error and the session carries on). When there is
nothing to do it prints nothing. Otherwise it prints one JSON object whose
``systemMessage`` tells the user what happened and whose ``additionalContext``
tells Claude, including the command to rerun after a failure. The install has
its own time limit, shorter than the hook's ``timeout`` in the settings file,
so a slow index is reported here rather than the hook being cancelled silently.

The decision (:func:`plan`) is separate from the side effects (:func:`execute`)
so it can be tested without installing anything.

Usage (what the hook runs)::

    python3 scripts/session_start.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: The extras a contributor needs to run the CI gates (``test`` includes ``ui``).
#: ``browser`` is left out: it needs a separate Chromium download.
EXTRAS = ("test", "lint", "typecheck")
#: The repository files that decide what the install puts in the environment.
INPUTS = ("pyproject.toml", "constraints.txt", "scripts/session_start.py")
#: The requirement pip installs (the checkout, editable, with :data:`EXTRAS`).
SPEC = f".[{','.join(EXTRAS)}]"
#: The marker's name in the interpreter's ``purelib`` directory.
MARKER_NAME = ".aquakin-session-start.sha256"
#: What ``core.hooksPath`` is set to.
HOOKS_PATH = ".githooks"
#: Seconds the install may take. Below the hook's ``timeout`` (360 s) in
#: ``.claude/settings.json``, so the script reports a slow install itself.
INSTALL_TIMEOUT_S = 300.0
#: Environment added to the install's: no pip cache under ``~/.cache``. A
#: variable rather than ``--no-cache-dir`` so pip's build-isolation subprocess
#: (which installs setuptools for the editable build) obeys it too.
INSTALL_ENV = {"PIP_NO_CACHE_DIR": "1"}
#: Lines of pip's output passed to Claude when the install fails.
TAIL_LINES = 20


@dataclass(frozen=True)
class Plan:
    """What a session start would do, decided from the environment alone.

    Attributes
    ----------
    install : list of str or None
        The pip command to run, or None when the recorded install is current.
    set_hooks_path : list of str or None
        The git command that enables the pre-push hook, or None when
        ``core.hooksPath`` is already set.
    hooks_path_elsewhere : str or None
        The existing ``core.hooksPath`` when it points somewhere other than
        ``.githooks``; it is left alone and reported.
    env_export : tuple of (str, str) or None
        ``(CLAUDE_ENV_FILE, line)``: the shell line to append so the session's
        commands find the installed tools first, or None when Claude Code gave
        no environment file.
    """

    install: list[str] | None
    set_hooks_path: list[str] | None
    hooks_path_elsewhere: str | None
    env_export: tuple[str, str] | None


def fingerprint(root: Path) -> str:
    """
    Hash the files that decide the install, and where the checkout is.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    str
        A hex SHA-256. It changes when any file in :data:`INPUTS` changes or the
        checkout moves (the editable install points at the checkout's path).
    """
    digest = hashlib.sha256()
    digest.update(str(root.resolve()).encode())
    for name in INPUTS:
        digest.update(b"\0" + name.encode() + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def in_cloud(env: dict[str, str]) -> bool:
    """Whether ``env`` is a Claude Code cloud session's (``CLAUDE_CODE_REMOTE=true``)."""
    return env.get("CLAUDE_CODE_REMOTE") == "true"


def plan(
    python: str,
    scripts_dir: str,
    digest: str,
    recorded: str | None,
    hooks_path: str | None,
    env_file: str | None,
) -> Plan:
    """
    Decide what a session start in a cloud session does.

    Parameters
    ----------
    python : str
        The interpreter to install into.
    scripts_dir : str
        Where that interpreter's packages put their commands (``pytest``,
        ``ruff``, ``pyright``).
    digest : str
        The current :func:`fingerprint`.
    recorded : str or None
        The fingerprint of the last successful install into ``python``, or None
        when there is no marker.
    hooks_path : str or None
        The current ``core.hooksPath``, or None when it is unset.
    env_file : str or None
        ``CLAUDE_ENV_FILE``, the file whose lines run before each of the
        session's commands, or None when it is not set.

    Returns
    -------
    Plan
        What to do. Only ``env_export`` is set when everything is current: the
        environment file is new in every session.
    """
    install = None
    if recorded != digest:
        install = [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--progress-bar",
            "off",
            "--quiet",
            "--editable",
            SPEC,
            "--constraint",
            "constraints.txt",
        ]
    set_hooks_path = None
    if hooks_path is None:
        set_hooks_path = ["git", "config", "core.hooksPath", HOOKS_PATH]
    elsewhere = hooks_path if hooks_path not in (None, HOOKS_PATH) else None
    env_export = None
    if env_file:
        env_export = (env_file, f'export PATH={shlex.quote(scripts_dir)}:"$PATH"')
    return Plan(
        install=install,
        set_hooks_path=set_hooks_path,
        hooks_path_elsewhere=elsewhere,
        env_export=env_export,
    )


#: Runs a command in the repository root with extra environment variables;
#: returns (exit status, combined output).
Runner = Callable[[Sequence[str], float | None, dict[str, str]], tuple[int, str]]


def run_command(
    argv: Sequence[str], timeout: float | None, extra_env: dict[str, str]
) -> tuple[int, str]:
    """
    Run ``argv`` in the current directory and capture its output.

    Parameters
    ----------
    argv : sequence of str
        The command.
    timeout : float or None
        Seconds before the command is killed.
    extra_env : dict of str to str
        Variables added to (or replacing those in) this process's environment.

    Returns
    -------
    tuple of (int, str)
        The exit status and the combined stdout and stderr.

    Raises
    ------
    subprocess.TimeoutExpired
        If the command runs past ``timeout``.
    OSError
        If the command cannot be started.
    """
    done = subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        env={**os.environ, **extra_env},
        check=False,
    )
    return done.returncode, done.stdout


def _tail(text: str) -> str:
    """The last :data:`TAIL_LINES` non-blank lines of ``text``."""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-TAIL_LINES:])


def execute(
    todo: Plan,
    digest: str,
    marker: Path,
    run: Runner = run_command,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object] | None:
    """
    Carry out a :class:`Plan` and describe the outcome.

    Parameters
    ----------
    todo : Plan
        What to do.
    digest : str
        The fingerprint to record after a successful install.
    marker : Path
        The marker file. Removed before the install, written after it succeeds.
    run : Runner, optional
        Runs a command; replaced in tests.
    clock : callable, optional
        A monotonic clock in seconds; replaced in tests.

    Returns
    -------
    dict or None
        The hook's JSON output, or None when there is nothing to report.
    """
    user: list[str] = []
    claude: list[str] = []

    if todo.install is not None:
        retry = shlex.join(
            [todo.install[0], "-m", "pip", "install", "-e", SPEC, "-c", "constraints.txt"]
        )
        marker_error = _remove(marker)
        start = clock()
        try:
            status, output = run(todo.install, INSTALL_TIMEOUT_S, INSTALL_ENV)
        except subprocess.TimeoutExpired:
            status, output = None, f"pip did not finish within {INSTALL_TIMEOUT_S:.0f} s"
        except OSError as exc:
            status, output = None, f"pip could not be started: {exc}"
        if status == 0:
            marker_error = marker_error or _write(marker, digest + "\n")
            if marker_error:
                claude.append(
                    f"The install succeeded but its marker {marker} could not be updated "
                    f"({marker_error}), so the next session start installs again."
                )
            user.append(
                f"Installed aquakin[{','.join(EXTRAS)}] pinned by constraints.txt "
                f"in {clock() - start:.0f} s."
            )
        else:
            user.append(
                "Installing aquakin failed; the session works, but tests, ruff and "
                "pyright may be missing. Claude has the error."
            )
            claude.append(
                "The SessionStart hook could not install the project, so JAX, pytest, "
                f"ruff or pyright may be missing. Rerun it with: {retry}\n"
                f"Last lines of the output:\n{_tail(output)}"
            )

    if todo.set_hooks_path is not None:
        try:
            status, output = run(todo.set_hooks_path, None, {})
        except OSError as exc:
            status, output = None, str(exc)
        if status == 0:
            user.append("Enabled the pre-push hook (core.hooksPath = .githooks).")
        else:
            claude.append(
                "The SessionStart hook could not enable the pre-push hook; run "
                f"`git config core.hooksPath {HOOKS_PATH}`. Git said: {_tail(output)}"
            )
    if todo.hooks_path_elsewhere is not None:
        claude.append(
            f"core.hooksPath is {todo.hooks_path_elsewhere!r}, not {HOOKS_PATH!r}, so the "
            "repository's pre-push gate (.githooks/pre-push) is not active; it was left as is."
        )

    if todo.env_export is not None:
        env_file, line = todo.env_export
        try:
            with open(env_file, "a") as f:
                f.write(line + "\n")
        except OSError as exc:
            claude.append(
                f"The SessionStart hook could not write {env_file} ({exc}), so a bare "
                "`ruff` or `pyright` may be the image's copy rather than the pinned one; "
                f"prefix commands with `{line} &&`."
            )

    if not user and not claude:
        return None
    out: dict[str, object] = {}
    if user:
        out["systemMessage"] = "session start: " + " ".join(user)
    if claude:
        out["hookSpecificOutput"] = {
            "hookEventName": "SessionStart",
            "additionalContext": "\n\n".join(claude),
        }
    return out


def _remove(path: Path) -> str | None:
    """Delete ``path`` if it exists; the error text, or None on success."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        return str(exc)
    return None


def _write(path: Path, text: str) -> str | None:
    """Write ``text`` to ``path``; the error text, or None on success."""
    try:
        path.write_text(text)
    except OSError as exc:
        return str(exc)
    return None


def _git_hooks_path(run: Runner) -> str | None:
    """The repository's ``core.hooksPath``, or None when unset (or git is missing)."""
    try:
        status, output = run(["git", "config", "--get", "core.hooksPath"], 10.0, {})
    except OSError:
        return None
    return output.strip() if status == 0 and output.strip() else None


def main(env: dict[str, str] | None = None, root: Path | None = None) -> int:
    """
    Run the session-start step and print the hook's output.

    Parameters
    ----------
    env : dict of str to str, optional
        The environment; ``os.environ`` by default.
    root : Path, optional
        The repository root; this script's parent's parent by default.

    Returns
    -------
    int
        0, whatever the install or git did, so the hook never blocks the
        session. Only an unexpected error (an input file missing, say) ends the
        script with a traceback instead, which Claude Code shows as a
        non-blocking hook error.
    """
    env = dict(os.environ) if env is None else env
    if not in_cloud(env):
        return 0
    root = Path(__file__).resolve().parents[1] if root is None else root
    os.chdir(root)
    marker = Path(sysconfig.get_path("purelib")) / MARKER_NAME
    recorded = marker.read_text().strip() if marker.is_file() else None
    digest = fingerprint(root)
    todo = plan(
        sys.executable,
        sysconfig.get_path("scripts"),
        digest,
        recorded,
        _git_hooks_path(run_command),
        env.get("CLAUDE_ENV_FILE"),
    )
    out = execute(todo, digest, marker)
    if out is not None:
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
