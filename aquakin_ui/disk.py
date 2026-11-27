"""Following the project's files as other programs change them.

An AI assistant working through ``aquakin mcp``, a text editor or a second
editor window may write the flowsheet or a study while this page shows it.
Every :data:`WATCH_SECONDS` the page looks at the project folder: an open
flowsheet whose file changed is shown again
(:func:`~aquakin_ui.files.reload_if_changed`), and the studies are listed
again when anything under the project folder changed.

Each function is bound onto :class:`~aquakin_ui.app.Editor` as a method
(``self`` is the editor).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquakin_ui.app import Editor

#: Seconds between two looks at the project's files.
WATCH_SECONDS = 1.5


def project_files(project: Path) -> tuple:
    """Every file under a project folder with its size and modification time.

    Folders whose name starts with a dot (``.git``) are left out. A file
    removed while the folder is read is left out too.

    Parameters
    ----------
    project : pathlib.Path
        The project folder.

    Returns
    -------
    tuple
        ``(relative path, mtime in ns, size)`` per file, sorted, so two calls
        compare equal exactly when no file was added, removed or rewritten.
    """
    entries = []
    for root, dirs, names in os.walk(project):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            path = Path(root) / name
            try:
                st = path.stat()
            except OSError:  # removed since the folder was listed
                continue
            entries.append((str(path.relative_to(project)), st.st_mtime_ns, st.st_size))
    return tuple(entries)


def follow_disk(self: Editor) -> None:
    """Look once: reload the open flowsheet if its file changed, and list the
    studies again if any file of the project changed since the last look."""
    self.reload_if_changed()
    project = self._project()
    seen = (project, project_files(project)) if project is not None else None
    if seen != self._project_seen:
        before, self._project_seen = self._project_seen, seen
        # The first look only records the folder: the studies were listed
        # when it was opened.
        if before is not None and before[0] == project:
            self.refresh_studies()


async def watch_disk(self: Editor) -> None:
    """Call :func:`follow_disk` every :data:`WATCH_SECONDS` while the page is served."""
    while True:
        await asyncio.sleep(WATCH_SECONDS)
        with self.state:
            self.follow_disk()
