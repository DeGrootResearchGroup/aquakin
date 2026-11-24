"""The unit form's choices that redraw it, and sending its rows again.

A choice that changes which rows the form has (an aeration mode, a model) is
applied by Python, which sends the redrawn rows back. Until they arrive the
page marks the choice's row ``waiting`` and takes no input: anything entered
meanwhile would be overwritten by them. So rows always come back, the redrawn
form or, when the choice cannot be applied, the form as it was. Each function
is bound onto :class:`~aquakin_ui.app.Editor` as a method (``self`` is the
editor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aquakin_ui import forms

if TYPE_CHECKING:
    from aquakin_ui.app import Editor


def reshape_form(self: Editor, rows: list[dict], i: int, value) -> None:
    """A choice that changes which rows the form has (aeration mode, model).

    ``rows`` are the page's rows when the choice was made and ``i`` the
    choice's row. When the choice is not applied (a field cannot be read, or
    it waits on the user confirming the connections it removes, which then
    redraws the form) they are drawn again unchanged.
    """
    if not self.apply_properties(
        [{**r, "value": value} if j == i else r for j, r in enumerate(rows)]
    ):
        resend_rows(self, rows)
        return
    # The new rows may suggest values the old ones could not (the aerated
    # species of the unit's model): take them, so the unit builds at once.
    uid = self.state.selected
    try:
        _, settings = forms.read_rows(self.doc, uid, self.state.sel_rows)
    except ValueError:
        return
    if settings != self.doc.units[uid].settings:
        self.doc.apply(uid, settings, self.doc.units[uid].label)
        self.publish()


def resend_rows(self: Editor, rows: list[dict]) -> None:
    """Draw ``rows`` in the form again without their ``waiting`` marks.

    trame sends a value only when it differs from the one it last pushed, and
    the page keeps what it holds (edits just declined, a choice marked waiting)
    until it is sent something new, so the rows carry a new revision mark.
    """
    self._form_revision += 1
    rows = [{k: v for k, v in r.items() if k != "waiting"} for r in rows]
    if rows:
        rows[0] = {**rows[0], "revision": self._form_revision}
        self.state.sel_rows = rows
