"""The pre-push hook refuses a commit whose message links to a Claude session.

CLAUDE.md ("Pushing and merging") forbids a session link in a commit message or
a PR description: it means nothing to a reader of the repository, and the
session can be private. ``.githooks/pre-push`` enforces it for the commits being
pushed. These tests run the real hook in a throwaway repository, with a ``PATH``
holding only the system tools, so ruff is not found and the lint gate after the
checks exits early; what they see is the session check and, for a push that
changes ``constraints.txt`` or ``pyproject.toml``, the check that the pins
resolve on every supported Python, run against a stub ``uv``.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).parents[2] / ".githooks" / "pre-push"
ZERO = "0" * 40
LINK = "https://claude.ai/code/session_0123456789abcdef"

pytestmark = pytest.mark.skipif(
    shutil.which("git", path="/usr/bin:/bin") is None or shutil.which("bash") is None,
    reason="needs git in /usr/bin or /bin and bash",
)


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repository with one clean commit, pushed to a bare remote as ``main``."""
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "T")
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "commit", "-q", "--allow-empty", "-m", "Base")
    _git(work, "push", "-q", "origin", "main")
    return work


def _commit(repo, message):
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _run_hook(repo, stdin, *, path="/usr/bin:/bin", fail_on=""):
    env = {**os.environ, "PATH": path, "FAIL_ON": fail_on}
    return subprocess.run(
        ["bash", str(HOOK), "origin", "url"],
        cwd=repo,
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
    )


def test_blocks_a_new_branch_whose_commit_links_a_session(repo):
    _commit(repo, "Clean change")
    bad = _commit(repo, f"Linked change\n\nClaude-Session: {LINK}")
    result = _run_hook(repo, f"refs/heads/b {bad} refs/heads/b {ZERO}\n")
    assert result.returncode == 1
    assert "link to a Claude session" in result.stderr
    assert "Linked change" in result.stderr
    assert "Clean change" not in result.stderr


def test_checks_only_the_commits_being_pushed(repo):
    old = _commit(repo, f"Already pushed\n\nClaude-Session: {LINK}")
    new = _commit(repo, "New and clean")
    result = _run_hook(repo, f"refs/heads/b {new} refs/heads/b {old}\n")
    assert "link to a Claude session" not in result.stderr
    assert result.returncode == 0


def test_passes_clean_commits_and_deletions(repo):
    clean = _commit(repo, "Clean\n\nCo-Authored-By: Someone <s@example.com>")
    assert _run_hook(repo, f"refs/heads/b {clean} refs/heads/b {ZERO}\n").returncode == 0
    assert _run_hook(repo, f"(delete) {ZERO} refs/heads/b {clean}\n").returncode == 0


# A stub uv that fails on the Python versions named in FAIL_ON, so the hook's
# constraints check can be driven without uv or the network
# (tests/unit/test_check_constraints.py tests the script itself).
STUB_UV = """\
#!/bin/sh
while [ $# -gt 0 ]; do
  [ "$1" = "--python-version" ] && version=$2
  shift
done
case " $FAIL_ON " in
  *" $version "*) echo "no solution on $version" >&2; exit 1 ;;
esac
"""

PYPROJECT = """\
[project]
name = "demo"
classifiers = ["Programming Language :: Python :: 3.11"]
"""


@pytest.fixture
def stub_path(tmp_path):
    """A PATH whose only uv is the stub."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(STUB_UV)
    (bin_dir / "uv").chmod(0o755)
    return f"{bin_dir}:/usr/bin:/bin"


def _commit_file(repo, name, text, message):
    (repo / name).write_text(text)
    _git(repo, "add", name)
    return _commit(repo, message)


def test_a_pin_change_that_does_not_resolve_blocks_the_push(repo, stub_path):
    _commit_file(repo, "pyproject.toml", PYPROJECT, "Add pyproject")
    sha = _commit_file(repo, "constraints.txt", "jax==0.12.1\n", "Bump jax")
    result = _run_hook(repo, f"refs/heads/b {sha} refs/heads/b {ZERO}\n",
                       path=stub_path, fail_on="3.11")
    assert result.returncode == 1
    assert "Python 3.11: does not resolve" in result.stderr
    assert "does not resolve on every supported Python" in result.stderr


def test_a_pin_change_that_resolves_passes(repo, stub_path):
    _commit_file(repo, "pyproject.toml", PYPROJECT, "Add pyproject")
    sha = _commit_file(repo, "constraints.txt", "jax==0.10.2\n", "Pin jax")
    result = _run_hook(repo, f"refs/heads/b {sha} refs/heads/b {ZERO}\n", path=stub_path)
    assert result.returncode == 0, result.stderr
    assert "Python 3.11: resolves" in result.stderr


def test_a_push_without_pin_changes_skips_the_resolve(repo, stub_path):
    base = _commit_file(repo, "pyproject.toml", PYPROJECT, "Add pyproject")
    sha = _commit_file(repo, "README.md", "text\n", "Docs only")
    # The stub would fail, so a run of the check would block the push.
    result = _run_hook(repo, f"refs/heads/b {sha} refs/heads/b {base}\n",
                       path=stub_path, fail_on="3.11")
    assert result.returncode == 0, result.stderr
    assert "resolves on every supported Python" not in result.stderr
