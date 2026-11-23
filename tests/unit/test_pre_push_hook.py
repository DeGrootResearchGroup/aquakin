"""The pre-push hook refuses a commit whose message links to a Claude session.

CLAUDE.md ("Pushing and merging") forbids a session link in a commit message or
a PR description: it means nothing to a reader of the repository, and the
session can be private. ``.githooks/pre-push`` enforces it for the commits being
pushed. These tests run the real hook in a throwaway repository, with a ``PATH``
holding only the system tools, so ruff is not found and the lint gate after the
check exits early; what they see is the session check alone.
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


def _run_hook(repo, stdin):
    env = {**os.environ, "PATH": "/usr/bin:/bin"}
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
