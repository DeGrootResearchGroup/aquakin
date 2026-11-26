"""The session-link check the pre-push hook and CI share refuses a linked commit.

``scripts/check_session_links.sh`` lists the commits in a ``git rev-list`` range
whose messages link to a Claude session and fails; the pre-push hook calls it
for each pushed ref, and the CI ``lint (ruff)`` job for a pull request's commits.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "check_session_links.sh"
HOOK = ROOT / ".githooks" / "pre-push"
LINK = "https://claude.ai/code/session_0123456789abcdef"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repository with a clean base commit, and the shared script and hook."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".githooks").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    shutil.copy2(HOOK, tmp_path / ".githooks" / HOOK.name)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _commit(repo, message):
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _check(repo, *rev_args):
    return subprocess.run(
        ["bash", "scripts/check_session_links.sh", *rev_args],
        cwd=repo, capture_output=True, text=True,
    )


def test_a_clean_range_passes(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "Fix a thing\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    done = _check(repo, f"{base}..HEAD")
    assert done.returncode == 0, done.stderr


def test_a_linked_commit_is_refused_and_named(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "Clean one")
    _commit(repo, f"Linked one\n\nClaude-Session: {LINK}")
    done = _check(repo, f"{base}..HEAD")
    assert done.returncode == 1
    assert "link to a Claude session" in done.stderr
    assert "Linked one" in done.stderr and "Clean one" not in done.stderr


def test_the_pre_push_hook_refuses_a_linked_commit(repo):
    """The hook runs the shared script on the pushed range, so it refuses the link
    before any of its other gates run."""
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, f"Linked\n\nClaude-Session: {LINK}")
    done = subprocess.run(
        ["bash", ".githooks/pre-push", "origin", "url"],
        cwd=repo, capture_output=True, text=True,
        input=f"refs/heads/x {head} refs/heads/x {base}\n",
    )
    assert done.returncode == 1
    assert "link to a Claude session" in done.stderr
