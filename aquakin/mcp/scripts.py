"""The opt-in script tool: run a Python script the assistant wrote, in a project.

Most of aquakin's API has no tool of its own (a reactor fit, a sensitivity
analysis, a plot). ``run_python`` lets an assistant reach it by writing a
script, as Claude Code does with its own shell. It runs any code, so it is
offered only when the server is started with ``aquakin mcp --allow-python``,
and its annotations mark it as able to change files, so the user's app asks
before each run.

The script runs in a process of its own, in the project folder, with the
Python that runs the server (so ``import aquakin`` works) and matplotlib set
to write files rather than open windows.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: Seconds a script may run when the call does not say, and at most.
DEFAULT_TIMEOUT, MAX_TIMEOUT = 300.0, 3600.0
#: Characters of standard output and of standard error returned (the end of each).
OUTPUT_LIMIT = 20_000


def _tail(text: str) -> tuple[str, bool]:
    """The last :data:`OUTPUT_LIMIT` characters of ``text``, and whether it was cut."""
    if len(text) <= OUTPUT_LIMIT:
        return text, False
    return text[-OUTPUT_LIMIT:], True


def _written_since(project: Path, start_ns: int) -> list[str]:
    """Files under ``project`` modified at or after ``start_ns``, relative, sorted.

    Folders whose name starts with a dot (``.git``) are left out.
    """
    out = []
    for root, dirs, names in os.walk(project):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            path = Path(root) / name
            try:
                modified = path.stat().st_mtime_ns
            except OSError:  # removed while the folder was read
                continue
            if modified >= start_ns:
                out.append(path.relative_to(project).as_posix())
    return sorted(out)


def run_python(
    project: str, code: str, timeout: float = DEFAULT_TIMEOUT, save_as: str | None = None
) -> dict:
    """Run a Python script in a project folder and return its output and the
    files it wrote. Use it for what no other tool does (a reactor fit, a
    sensitivity analysis, a plot saved to a file); look the functions up with
    describe_api first. Prefer a study for an analysis a study kind covers:
    it is saved, and the editor shows it.

    Parameters
    ----------
    project : str
        The folder the script runs in (its working directory).
    code : str
        The script. It runs with the Python that runs aquakin, so ``import
        aquakin`` works; matplotlib writes figures to files (``savefig``).
    timeout : float, optional
        Seconds before the script is stopped (default 300, at most 3600).
    save_as : str, optional
        Also keep the script in the project, at this path relative to it
        (for example ``"scripts/fit_ozone.py"``), so the user can run it again.
        An existing file is not written over.

    Returns
    -------
    dict
        ``exit_code`` (``None`` when stopped at the timeout), ``timed_out``,
        ``seconds``, ``stdout`` and ``stderr`` (their last 20,000 characters,
        with ``truncated`` saying whether either was cut), ``files``: the
        project's files the run wrote, and ``saved_as`` when kept.
    """
    folder = Path(project)
    if not folder.is_dir():
        raise FileNotFoundError(f"{project} is not a folder")
    if not 0 < timeout <= MAX_TIMEOUT:
        raise ValueError(f"timeout must be more than 0 and at most {MAX_TIMEOUT:g} seconds")
    saved = None
    if save_as is not None:
        saved = folder / save_as
        if not saved.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"save_as must be a path inside the project, not {save_as!r}")
        if saved.exists():
            raise FileExistsError(f"{saved} already exists; choose another save_as")
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(code, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="aquakin-script-") as scratch:
        script = saved if saved is not None else Path(scratch) / "script.py"
        if saved is None:
            script.write_text(code, encoding="utf-8")
        start_ns, start = time.time_ns(), time.monotonic()
        try:
            done = subprocess.run(
                [sys.executable, str(script)],
                cwd=folder,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            exit_code, stdout, stderr = done.returncode, done.stdout, done.stderr
        except subprocess.TimeoutExpired as stopped:  # killed by subprocess.run
            exit_code = None
            stdout = _text(stopped.stdout)
            stderr = _text(stopped.stderr)
    out, cut_out = _tail(stdout)
    err, cut_err = _tail(stderr)
    kept = None if saved is None else saved.relative_to(folder).as_posix()
    files = [f for f in _written_since(folder, start_ns) if f != kept]
    result = {
        "exit_code": exit_code,
        "timed_out": exit_code is None,
        "seconds": round(time.monotonic() - start, 3),
        "stdout": out,
        "stderr": err,
        "truncated": cut_out or cut_err,
        "files": files,
    }
    if kept is not None:
        result["saved_as"] = kept
    return result


def _text(data: bytes | None) -> str:
    """The output a stopped process left, as text. ``TimeoutExpired`` holds it
    as bytes even when the run asked for text, or ``None`` when there was none."""
    return "" if data is None else data.decode(errors="replace")
