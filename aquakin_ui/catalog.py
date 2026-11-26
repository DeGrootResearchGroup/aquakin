"""The unit catalogue as the interface holds it (plain data from the worker).

aquakin describes every unit kind (:mod:`aquakin.plant.catalog`); the interface
asks the worker for that description once and reads ports from it here, so a
port list follows a unit's settings without a round trip to aquakin.
"""

from __future__ import annotations

import re

# Docstring markup the help text comes with: ``code``, :role:`target`, *emphasis*.
_ROLE = re.compile(r":[a-z]+:`~?([^`<]*?)(?:\s*<[^>]*>)?`")
_CODE = re.compile(r"``([^`]*)``")
_EMPHASIS = re.compile(r"(?<![\w*])\*{1,2}([^*\s][^*]*?)\*{1,2}(?![\w*])")


def plain(text: str) -> str:
    """Docstring text as plain text: ``code`` and roles (:class:`Aeration`) lose
    their markup, as does *emphasis*; a role's ``~`` and dotted prefix go too."""
    text = _ROLE.sub(lambda m: m.group(1).rsplit(".", 1)[-1], text)
    text = _CODE.sub(r"\1", text)
    return _EMPHASIS.sub(r"\1", text)


class Catalog:
    """The unit kinds and shipped models, as ``aquakin serve`` describes them.

    Parameters
    ----------
    data : dict
        The worker's ``catalog`` reply.
    models : list of dict
        The worker's ``models`` reply (one ``{"name", "description"}`` each).
    """

    def __init__(self, data: dict, models: list[dict]):
        self.categories: list[str] = list(data["categories"])
        self.kinds: dict[str, dict] = {k["kind"]: _plain_kind(k) for k in data["kinds"]}
        self.shipped_models: list[str] = [m["name"] for m in models]

    def kind(self, kind: str) -> dict:
        """The description of one kind."""
        return self.kinds[kind]

    def settings(self, kind: str) -> list[dict]:
        """A kind's settings, in constructor order."""
        return self.kinds[kind]["settings"]

    def defaults(self, kind: str) -> dict:
        """The value of every setting of a kind when it is not given."""
        return {s["name"]: s["default"] for s in self.settings(kind)}

    def ports_for(self, kind: str, settings: dict) -> tuple[list[str], list[str]]:
        """The ``(inputs, outputs)`` port names of a unit with ``settings``."""
        full = self.defaults(kind) | settings
        out: dict[str, list[str]] = {"input": [], "output": []}
        for rule in self.kinds[kind]["ports"]:
            if not applies(rule.get("when"), full):
                names = []
            elif rule["form"] == "fixed":
                names = rule["names"]
            else:
                value = full.get(rule["field"])
                if value is None:
                    names = []
                elif rule["form"] == "name":
                    names = [value]
                else:
                    names = list(value)
            out[rule["side"]] += [str(n) for n in names]
        return out["input"], out["output"]


def applies(when: dict | None, settings: dict) -> bool:
    """Whether a setting or port rule with the catalogue's condition ``when``
    applies to a unit with ``settings``: always without one, else while the
    setting it names holds one of its ``values`` (an SBR's holding-volume
    settings and overflow port, with ``feed: storage``), or, when it lists none,
    while that setting is set (an expression controller's ``measured`` port,
    with a ``model``)."""
    if when is None:
        return True
    value = settings.get(when["setting"])
    return value is not None if "values" not in when else value in when["values"]


def _plain_kind(kind: dict) -> dict:
    """A kind with its summary and every setting's help as plain text."""
    return {
        **kind,
        "summary": plain(kind.get("summary", "")),
        "settings": [{**s, "help": plain(s.get("help", ""))} for s in kind["settings"]],
    }
