"""Lists of records edited on the page: bound fields, and buttons that add and
remove a record.

A list inside page state (the study dialog's draft, a unit form's rows) is
nested, and the page does not redraw on a nested change: a record pushed onto
or spliced out of such a list does not appear or go, and a choice made inside
one is not shown. So every widget bound into one uses :func:`bound`, and every
add and remove ends with :func:`redraw`, which reassigns the state holding it.
The study dialog and the unit form share these, so the two cannot drift apart.
"""

from __future__ import annotations

from trame.widgets import vuetify3 as v3


def redraw(state: str) -> str:
    """The template statement that redraws ``state`` (a list or a mapping),
    prefixed with ``"; "`` to follow another statement."""
    return f"; {state} = Array.isArray({state}) ? [...{state}] : {{...{state}}}"


def _then(also: str) -> str:
    return f"{also}; " if also else ""


def bound(expr: str, state: str, also: str = "") -> dict:
    """The keywords that bind a widget to ``expr``, a value inside ``state``,
    redrawing ``state`` after each change; ``also`` is a statement run first
    (``"sel_dirty = true"``)."""
    return {
        "v_model": expr,
        "update_modelValue": f"{_then(also)}$nextTick(() => {{ {redraw(state)[2:]} }})",
    }


def remove_button(items: str, index: str, state: str, also: str = "") -> None:
    """A small cross that removes record ``index`` from the list ``items``."""
    v3.VBtn(
        icon="mdi-close",
        size="small",
        variant="text",
        click=f"{_then(also)}{items}.splice({index}, 1){redraw(state)}",
    )


def add_button(text: str, items: str, record: str, state: str, also: str = "") -> None:
    """A button that appends ``record`` (a JavaScript literal) to the list ``items``."""
    v3.VBtn(
        text,
        prepend_icon="mdi-plus",
        size="small",
        variant="text",
        classes="text-none",
        click=f"{_then(also)}{items}.push({record}){redraw(state)}",
    )
