#!/usr/bin/env bash
#
# Refuse commits whose messages link to the Claude session that wrote them.
#
# A session link means nothing to a reader of the repository, and the session
# can be private (CLAUDE.md, "Pushing and merging"). This is the one check both
# the pre-push hook (.githooks/pre-push) and CI (the `lint (ruff)` job, on pull
# requests) run, so a link is refused even when the local hook is not active.
#
# Usage: scripts/check_session_links.sh <git rev-list arguments>...
#     scripts/check_session_links.sh origin/main..HEAD
#     scripts/check_session_links.sh <sha> --not --remotes
#
# Exits 0 when no commit in the range links to a session, 1 (listing them)
# otherwise.
set -euo pipefail

linked=""
for commit in $(git rev-list "$@"); do
  if git log -1 --format=%B "$commit" | grep -q 'claude\.ai/code/session_'; then
    linked+="  $(git log -1 --format='%h %s' "$commit")"$'\n'
  fi
done
if [ -n "$linked" ]; then
  echo "Blocked: these commit messages link to a Claude session:" >&2
  printf '%s' "$linked" >&2
  echo "Remove the link (git commit --amend, or git rebase -i for older commits) and push again." >&2
  exit 1
fi
