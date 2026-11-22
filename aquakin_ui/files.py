"""Opening and saving the flowsheet's file, through aquakin's own spec reader and
writer (the ``read_spec`` and ``write_spec`` worker commands), and the
questions that keep unsaved or someone else's work from being lost.

Each function is bound onto :class:`~aquakin_ui.app.Editor` as a method
(``self`` is the editor).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from aquakin_ui.document import Flowsheet

if TYPE_CHECKING:
    from aquakin_ui.app import Editor


def open_spec(self: Editor, path=None, *, quiet: bool = False) -> None:
    """Read a spec with aquakin's reader (YAML 1.2) and show it."""
    path = Path(path or self.state.spec_path).resolve()

    def done(data):
        try:
            if isinstance(data, Exception):
                raise data
            doc = Flowsheet.from_spec(self.catalog, data, base_dir=path.parent)
        except Exception as exc:
            self.say(f"could not open {path}: {exc}")
            return
        self.doc = doc
        self._opened = (path, _digest(path))
        self._saved = self._snapshot()
        self.state.spec_path = str(path)
        self.state.selected = None
        self.state.fit_key += 1
        self._unit_checks.clear()
        self._plant_check = (None, None)
        self._history = []
        self.state.can_undo = False
        if not quiet:
            self.publish()
            self.say(f"opened {path}", error=False)
        self.refresh_studies()

    self._off_page(lambda: self.worker.request("read_spec", path=str(path)), done)


def _snapshot(self: Editor) -> str:
    return json.dumps(self.doc.to_spec(), sort_keys=True)


def _unsaved(self: Editor) -> bool:
    """Whether the flowsheet has edits its file does not hold (an empty new
    flowsheet has none)."""
    if self._saved is None:
        return bool(self.doc.units)
    return self._snapshot() != self._saved


def _spec_file(self: Editor) -> Path | None:
    """The flowsheet's file: the one last opened or saved here, else the path
    typed in the toolbar, else ``None`` (a new flowsheet with no name yet).

    The toolbar's path is also where Open reads and Save writes next, so it
    only becomes the flowsheet's file once one of them has gone through.
    """
    if self._opened is not None:
        return self._opened[0]
    typed = str(self.state.spec_path or "").strip()
    return Path(typed).resolve() if typed else None


def _apply_pending(self: Editor, pending: list[dict] | None) -> bool:
    """Apply the unit form's unapplied edits (``pending``, ``None`` when there
    are none) before an action that reads the flowsheet; ``False`` when they
    could not be applied, and the action should wait (the error, or the
    question about connections, is shown)."""
    if pending is None or self.state.selected not in self.doc.units:
        return True
    return self.apply_properties(pending)


def save_button(self: Editor, pending: list[dict] | None = None) -> None:
    """Save, with the unit form's unapplied edits applied first."""
    if self._apply_pending(pending):
        self.save()


def request_open(self: Editor, pending: list[dict] | None = None) -> None:
    """Open the path in the toolbar, asking first when there are unsaved edits."""
    if not self._apply_pending(pending):
        return
    target = str(self.state.spec_path or "").strip()
    if not target:
        self.say("type the path of a flowsheet file to open")
        return
    self._unless_unsaved(
        lambda: self.open(target),
        f"Open {Path(target).name} and leave the unsaved edits of this flowsheet?",
    )


def _unless_unsaved(self: Editor, then, question: str) -> None:
    """``then()`` at once when nothing is unsaved; otherwise ask, offering to
    save to the flowsheet's file first or to discard the edits."""
    if not self._unsaved():
        then()
        return
    self._after_unsaved = then
    self.state.unsaved_text = question
    self.state.unsaved_can_save = self._opened is not None
    self.state.unsaved_open = True


def answer_unsaved(self: Editor, choice: str) -> None:
    """The answer to the unsaved-edits question: ``"save"``, ``"discard"`` or
    ``"cancel"``."""
    self.state.unsaved_open = False
    then, self._after_unsaved = self._after_unsaved, None
    if then is None or choice == "cancel":
        return
    if choice == "save":
        if self._opened is None:
            # "Save first" is offered only for a flowsheet with a file of its own.
            self.say("this flowsheet has no file yet; type where to keep it, then Save")
            return
        self.save(then=then, path=self._opened[0])
    else:
        then()


def save(self: Editor, then=None, path=None) -> None:
    """Save through aquakin's own spec writer: settings checked and filled in,
    the aquakin version and file hashes recorded, outside model and influent
    files copied into the project, and the file replaced atomically; then
    ``then()``, when given and the save went through.

    ``path`` defaults to the path in the toolbar (so Save can save as)."""
    if path is None:
        typed = str(self.state.spec_path or "").strip()
        if not typed:
            self.say(
                "type where to keep the flowsheet in the path field (a .yaml file, "
                "in the folder of its project), then Save"
            )
            return
        path = typed
    path = Path(path).resolve()
    refusal = self._save_refusal(path)
    if refusal:
        self.say(refusal)
        return
    base = str(self.doc.base_dir) if self.doc.base_dir else str(path.parent)
    spec = self.doc.to_spec()  # read now, on the page's thread
    snapshot = json.dumps(spec, sort_keys=True)

    def done(written):
        if isinstance(written, Exception):
            self.say(f"could not save {path}: {written}")
            return
        # Copied files are now referenced inside the project; an edit made
        # while saving is kept rather than replaced by what was written.
        if json.dumps(self.doc.to_spec(), sort_keys=True) == snapshot:
            self.doc = Flowsheet.from_spec(self.catalog, written, base_dir=path.parent)
            self._saved = self._snapshot()
        self._opened = (path, _digest(path))
        self.state.spec_path = str(path)
        self._plant_check = (None, None)
        self.publish()
        self.say(f"saved {path}", error=False)
        if then is not None:
            then()

    self._off_page(
        lambda: self.worker.request("write_spec", spec=spec, path=str(path), base_dir=base),
        done,
    )


def _save_refusal(self: Editor, path: Path) -> str | None:
    """Why saving to ``path`` would lose someone else's work, or ``None``.

    Each window is a document of its own, so two can hold the same file:
    saving over a file that changed since this window opened it (or over
    another existing file) would silently replace that work."""
    if not path.exists():
        return None
    if self._opened is not None and self._opened[0] == path:
        if _digest(path) != self._opened[1]:
            return (
                f"{path.name} has changed on disk since it was opened here (another "
                f"window or program saved it); open it again, or save under another name"
            )
        return None
    return f"{path} already exists; open it, or save under another name"


def _digest(path: Path) -> str | None:
    """sha256 of a file's bytes (``None`` when it does not exist)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
