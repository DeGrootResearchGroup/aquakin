"""The unit catalogue as the interface holds it (plain data from the worker).

aquakin describes every unit kind (:mod:`aquakin.plant.catalog`); the interface
asks the worker for that description once and reads ports from it here, so a
port list follows a unit's settings without a round trip to aquakin.
"""

from __future__ import annotations


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
        self.kinds: dict[str, dict] = {k["kind"]: k for k in data["kinds"]}
        self.shipped_models: list[str] = [m["name"] for m in models]

    def kind(self, kind: str) -> dict:
        """The description of one kind."""
        return self.kinds[kind]

    def settings(self, kind: str) -> list[dict]:
        """A kind's settings, in constructor order."""
        return self.kinds[kind]["settings"]

    def setting(self, kind: str, name: str) -> dict:
        """One setting of a kind."""
        return next(s for s in self.settings(kind) if s["name"] == name)

    def defaults(self, kind: str) -> dict:
        """The value of every setting of a kind when it is not given."""
        return {s["name"]: s["default"] for s in self.settings(kind)}

    def ports_for(self, kind: str, settings: dict) -> tuple[list[str], list[str]]:
        """The ``(inputs, outputs)`` port names of a unit with ``settings``."""
        full = self.defaults(kind) | settings
        out: dict[str, list[str]] = {"input": [], "output": []}
        for rule in self.kinds[kind]["ports"]:
            if rule["form"] == "fixed":
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
