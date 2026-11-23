"""The cloud-session SessionStart hook (``scripts/session_start.py``).

The decision is tested from hand-built inputs, and the side effects through a
fake command runner, so no test installs anything or depends on the network.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "scripts" / "session_start.py"


@pytest.fixture(scope="module")
def ss():
    spec = importlib.util.spec_from_file_location("session_start", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered while it loads: its dataclass resolves annotations by module.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture
def checkout(tmp_path, ss):
    """A directory holding copies of the files the fingerprint hashes."""
    root = tmp_path / "repo"
    for name in ss.INPUTS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, root / name)
    return root


class FakeRunner:
    """Records each command and answers from a script of results."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, timeout, extra_env):
        self.calls.append((list(argv), timeout, extra_env))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def clock(*times):
    ticks = iter(times)
    return lambda: next(ticks)


# --- where it runs ------------------------------------------------------------


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"CLAUDE_CODE_REMOTE": "true"}, True),
        ({"CLAUDE_CODE_REMOTE": "false"}, False),
        ({"CLAUDE_CODE_REMOTE": "1"}, False),
        ({}, False),
    ],
)
def test_in_cloud_only_when_remote_is_true(ss, env, expected):
    assert ss.in_cloud(env) is expected


def test_main_outside_the_cloud_touches_nothing(ss, tmp_path, capsys):
    """On a workstation it returns 0 before reading, writing or running anything.

    The root does not exist, so any file access or ``chdir`` would raise.
    """
    assert ss.main(env={"PATH": "/usr/bin"}, root=tmp_path / "missing") == 0
    assert capsys.readouterr().out == ""


# --- the fingerprint ----------------------------------------------------------


def test_fingerprint_inputs_include_what_the_install_reads(ss):
    """Every repository file the pip command names is hashed, and the script itself."""
    assert {"pyproject.toml", "constraints.txt", "scripts/session_start.py"} <= set(ss.INPUTS)


@pytest.mark.parametrize("name", ["pyproject.toml", "constraints.txt", "scripts/session_start.py"])
def test_fingerprint_changes_with_each_input(ss, checkout, name):
    before = ss.fingerprint(checkout)
    assert ss.fingerprint(checkout) == before
    with (checkout / name).open("a") as f:
        f.write("\n# changed\n")
    assert ss.fingerprint(checkout) != before


def test_fingerprint_changes_when_the_checkout_moves(ss, checkout, tmp_path):
    """The editable install points at the checkout's path, so a move reinstalls."""
    moved = tmp_path / "elsewhere"
    shutil.copytree(checkout, moved)
    assert ss.fingerprint(moved) != ss.fingerprint(checkout)


# --- the decision -------------------------------------------------------------


def test_plan_installs_the_pinned_extras_when_there_is_no_marker(ss):
    todo = ss.plan("/py/bin/python3", "/py/bin", "abc", None, ".githooks", None)
    assert todo.install == [
        "/py/bin/python3",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--progress-bar",
        "off",
        "--quiet",
        "--editable",
        ".[test,lint,typecheck,docs]",
        "--constraint",
        "constraints.txt",
    ]
    assert todo.set_hooks_path is None
    assert todo.hooks_path_elsewhere is None


def test_plan_reinstalls_when_the_fingerprint_changed(ss):
    assert ss.plan("python3", "/py/bin", "new", "old", ".githooks", None).install is not None


def test_plan_does_nothing_when_everything_is_current(ss):
    todo = ss.plan("python3", "/py/bin", "abc", "abc", ".githooks", None)
    assert todo == ss.Plan(install=None, set_hooks_path=None, hooks_path_elsewhere=None, env_export=None)


def test_plan_enables_the_pre_push_hook_when_unset(ss):
    todo = ss.plan("python3", "/py/bin", "abc", "abc", None, None)
    assert todo.set_hooks_path == ["git", "config", "core.hooksPath", ".githooks"]
    assert todo.hooks_path_elsewhere is None


def test_plan_leaves_a_hooks_path_set_elsewhere_alone(ss):
    todo = ss.plan("python3", "/py/bin", "abc", "abc", "/home/dev/hooks", None)
    assert todo.set_hooks_path is None
    assert todo.hooks_path_elsewhere == "/home/dev/hooks"


def test_plan_puts_the_interpreters_commands_first_on_path(ss):
    """Even with nothing to install, every session gets the PATH line: the
    environment file is new each session, and the image's own ruff would win."""
    todo = ss.plan("python3", "/opt/my env/bin", "abc", "abc", ".githooks", "/tmp/env.sh")
    assert todo.install is None and todo.set_hooks_path is None
    assert todo.env_export == ("/tmp/env.sh", "export PATH='/opt/my env/bin':\"$PATH\"")


# --- carrying it out ----------------------------------------------------------


def test_run_command_passes_extra_env_and_merges_output(ss):
    """The child sees the added variable (how PIP_NO_CACHE_DIR reaches pip's
    build subprocess) and stderr is folded into the captured output."""
    status, output = ss.run_command(
        ["sh", "-c", 'echo "$SESSION_START_X"; echo err >&2; exit 3'],
        10.0,
        {"SESSION_START_X": "seen"},
    )
    assert status == 3
    assert output.split() == ["seen", "err"]


def test_execute_writes_a_path_line_a_shell_resolves_first(ss, tmp_path):
    """Sourcing the line makes a command in the scripts directory shadow one of
    the same name later on PATH, which is the point of writing it."""
    first, later = tmp_path / "pinned bin", tmp_path / "image bin"
    for folder, version in ((first, "pinned"), (later, "image")):
        folder.mkdir()
        tool = folder / "ruff"
        tool.write_text(f"#!/bin/sh\necho {version}\n")
        tool.chmod(0o755)
    env_file = tmp_path / "env.sh"
    env_file.write_text("export EARLIER=1\n")  # another hook's line is kept
    todo = ss.plan("python3", str(first), "abc", "abc", ".githooks", str(env_file))
    assert ss.execute(todo, "abc", tmp_path / "m", run=FakeRunner()) is None  # silent
    assert env_file.read_text().startswith("export EARLIER=1\n")
    shell = subprocess.run(
        ["bash", "-c", f'. "{env_file}" && ruff'],
        env={"PATH": f"{later}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert shell.stdout.strip() == "pinned"


def test_execute_reports_an_unwritable_env_file(ss, tmp_path):
    env_file = tmp_path / "missing-dir" / "env.sh"
    todo = ss.plan("python3", "/py/bin", "abc", "abc", ".githooks", str(env_file))
    out = ss.execute(todo, "abc", tmp_path / "m", run=FakeRunner())
    context = out["hookSpecificOutput"]["additionalContext"]
    assert f"could not write {env_file}" in context
    assert "export PATH=/py/bin:\"$PATH\" &&" in context


def test_execute_nothing_to_do_is_silent_and_runs_nothing(ss, tmp_path):
    run = FakeRunner()
    todo = ss.Plan(install=None, set_hooks_path=None, hooks_path_elsewhere=None, env_export=None)
    assert ss.execute(todo, "abc", tmp_path / "marker", run=run) is None
    assert run.calls == []


def test_execute_successful_install_records_the_fingerprint(ss, tmp_path):
    marker = tmp_path / "marker"
    marker.write_text("stale\n")
    todo = ss.plan("python3", "/py/bin", "abc", "stale", ".githooks", None)
    run = FakeRunner((0, ""))
    out = ss.execute(todo, "abc", marker, run=run, clock=clock(100.0, 112.4))
    assert run.calls == [(todo.install, ss.INSTALL_TIMEOUT_S, {"PIP_NO_CACHE_DIR": "1"})]
    assert marker.read_text() == "abc\n"
    assert out == {
        "systemMessage": "session start: Installed aquakin[test,lint,typecheck,docs] "
        "pinned by constraints.txt in 12 s."
    }


@pytest.mark.parametrize(
    "result, detail",
    [
        ((1, "Collecting jax\nERROR: No matching distribution found for jax==0.10.2\n"),
         "ERROR: No matching distribution found for jax==0.10.2"),
        (subprocess.TimeoutExpired("pip", 300), "pip did not finish within 300 s"),
        (FileNotFoundError(2, "No such file", "python3"), "pip could not be started"),
    ],
)
def test_execute_failed_install_reports_and_retries_next_time(ss, tmp_path, result, detail):
    """A failure removes the stale marker, tells both audiences, and gives the retry command."""
    marker = tmp_path / "marker"
    marker.write_text("stale\n")
    todo = ss.plan("/py/python3", "/py/bin", "abc", "stale", ".githooks", None)
    out = ss.execute(todo, "abc", marker, run=FakeRunner(result), clock=clock(0.0, 1.0))
    assert not marker.exists()
    assert "Installing aquakin failed" in out["systemMessage"]
    context = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "/py/python3 -m pip install -e '.[test,lint,typecheck,docs]' -c constraints.txt" in context
    assert detail in context


def test_execute_reports_a_marker_it_cannot_update(ss, tmp_path):
    """A marker path that is a directory can be neither removed nor written: the
    install still counts as done, and Claude is told the next start reinstalls."""
    marker = tmp_path / "marker"
    marker.mkdir()
    todo = ss.plan("python3", "/py/bin", "abc", None, ".githooks", None)
    out = ss.execute(todo, "abc", marker, run=FakeRunner((0, "")), clock=clock(0.0, 3.0))
    assert out["systemMessage"].endswith("in 3 s.")
    context = out["hookSpecificOutput"]["additionalContext"]
    assert f"marker {marker} could not be updated" in context
    assert "next session start installs again" in context


def test_execute_reports_a_marker_it_cannot_write_after_removing(ss, tmp_path):
    """Removal succeeds (no marker yet) but the write fails."""
    marker = tmp_path / "missing-dir" / "marker"
    todo = ss.plan("python3", "/py/bin", "abc", None, ".githooks", None)
    out = ss.execute(todo, "abc", marker, run=FakeRunner((0, "")), clock=clock(0.0, 1.0))
    assert not marker.exists()
    assert "could not be updated" in out["hookSpecificOutput"]["additionalContext"]


def test_execute_failure_output_is_trimmed_to_the_tail(ss, tmp_path):
    noise = "\n".join(f"line {i}" for i in range(100))
    todo = ss.plan("python3", "/py/bin", "abc", None, ".githooks", None)
    out = ss.execute(todo, "abc", tmp_path / "m", run=FakeRunner((1, noise)), clock=clock(0, 1))
    context = out["hookSpecificOutput"]["additionalContext"]
    assert "line 99" in context and f"line {99 - ss.TAIL_LINES + 1}" in context
    assert f"line {99 - ss.TAIL_LINES}\n" not in context


def test_execute_enables_the_pre_push_hook(ss, tmp_path):
    todo = ss.plan("python3", "/py/bin", "abc", "abc", None, None)
    run = FakeRunner((0, ""))
    out = ss.execute(todo, "abc", tmp_path / "m", run=run)
    assert run.calls == [(["git", "config", "core.hooksPath", ".githooks"], None, {})]
    assert out == {
        "systemMessage": "session start: Enabled the pre-push hook (core.hooksPath = .githooks)."
    }


@pytest.mark.parametrize("result", [(128, "fatal: not in a git directory"), FileNotFoundError("git")])
def test_execute_reports_a_failure_to_enable_the_hook(ss, tmp_path, result):
    todo = ss.plan("python3", "/py/bin", "abc", "abc", None, None)
    out = ss.execute(todo, "abc", tmp_path / "m", run=FakeRunner(result))
    assert "systemMessage" not in out
    assert "git config core.hooksPath .githooks" in out["hookSpecificOutput"]["additionalContext"]


def test_execute_reports_a_hooks_path_set_elsewhere(ss, tmp_path):
    todo = ss.plan("python3", "/py/bin", "abc", "abc", "/home/dev/hooks", None)
    out = ss.execute(todo, "abc", tmp_path / "m", run=FakeRunner())
    context = out["hookSpecificOutput"]["additionalContext"]
    assert "'/home/dev/hooks'" in context and "left as is" in context


@pytest.mark.parametrize(
    "result, expected",
    [
        ((0, ".githooks\n"), ".githooks"),
        ((1, ""), None),  # unset: git config --get exits 1
        (FileNotFoundError("git"), None),  # no git: planned as unset, reported at execute
    ],
)
def test_git_hooks_path_reads_the_setting(ss, result, expected):
    run = FakeRunner(result)
    assert ss._git_hooks_path(run) == expected
    assert run.calls[0][0] == ["git", "config", "--get", "core.hooksPath"]


# --- main in the cloud, with nothing to do -------------------------------------


def test_main_in_the_cloud_is_silent_when_current(ss, checkout, tmp_path, monkeypatch, capsys):
    """A second start: the marker matches and the hook is enabled, so nothing runs or prints."""
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "core.hooksPath", ".githooks"], check=True
    )
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    (purelib / ss.MARKER_NAME).write_text(ss.fingerprint(checkout) + "\n")
    monkeypatch.setattr(ss.sysconfig, "get_path", lambda name: str(purelib))
    monkeypatch.chdir(tmp_path)  # restored afterwards; main() changes into the root
    assert ss.main(env={"CLAUDE_CODE_REMOTE": "true"}, root=checkout) == 0
    assert capsys.readouterr().out == ""


def test_main_in_the_cloud_enables_the_hook_and_prints_json(ss, checkout, tmp_path, monkeypatch, capsys):
    """With the install current but the hook off, main sets it and prints one JSON object."""
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    (purelib / ss.MARKER_NAME).write_text(ss.fingerprint(checkout) + "\n")
    monkeypatch.setattr(ss.sysconfig, "get_path", lambda name: str(purelib))
    monkeypatch.chdir(tmp_path)
    assert ss.main(env={"CLAUDE_CODE_REMOTE": "true"}, root=checkout) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["systemMessage"].endswith("Enabled the pre-push hook (core.hooksPath = .githooks).")
    hooks_path = subprocess.run(
        ["git", "-C", str(checkout), "config", "--get", "core.hooksPath"],
        capture_output=True, text=True, check=True,
    )
    assert hooks_path.stdout.strip() == ".githooks"


# --- the settings file ----------------------------------------------------------


def test_settings_run_this_script_with_a_timeout_above_the_install_limit(ss):
    """The hook's timeout must exceed the script's own install limit, so a slow
    install is reported by the script instead of the hook being cancelled."""
    settings = json.loads((REPO / ".claude" / "settings.json").read_text())
    (entry,) = settings["hooks"]["SessionStart"]
    (hook,) = entry["hooks"]
    assert hook["type"] == "command"
    assert hook["command"] == 'python3 "$CLAUDE_PROJECT_DIR/scripts/session_start.py"'
    assert hook["timeout"] > ss.INSTALL_TIMEOUT_S
