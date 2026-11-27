"""The model panel's draft, held as plain data: no page, no worker.

The panel shows a model's reactions (each ``rate`` and its stoichiometry) and
its parameters (``value``, ``units``, ``description``) as the worker's
``model_document`` gives them, every value as text. The page edits a copy of
that, the **draft**, and sends it whole; :func:`edits` turns it into the edit
set aquakin checks and writes (``aquakin.plant.model_edit``), holding only what
changed. Nothing here reads an expression or a number: a value goes to aquakin
as it was typed.

Each edited value has a **field**, the dotted path aquakin names it by in an
error: ``reactions.<reaction>.rate``, ``reactions.<reaction>.stoichiometry.<species>``,
``parameters.<name>.<field>`` and ``reactions.<reaction>.parameters.<name>.<field>``.
"""

from __future__ import annotations

import copy
import re

#: The parameter fields the panel edits, in the order it shows them.
PARAMETER_FIELDS = ("value", "units", "description")


def draft_of(document: dict) -> dict:
    """The draft the page edits, from the worker's ``model_document``."""
    return {
        "reactions": [
            {
                "name": r["name"],
                "description": r["description"],
                "rate": r["rate"],
                "stoichiometry": copy.deepcopy(r["stoichiometry"]),
            }
            for r in document["reactions"]
        ],
        "parameters": [{**p, "new": False} for p in document["parameters"]],
    }


def parameter_key(row: dict) -> str:
    """The field prefix of a parameter row (``parameters.<name>`` or
    ``reactions.<reaction>.parameters.<name>``)."""
    name = str(row["name"]).strip()
    if row.get("reaction"):
        return f"reactions.{row['reaction']}.parameters.{name}"
    return f"parameters.{name}"


def _changed(before: str, after) -> bool:
    return str(after).strip() != str(before).strip()


def edits(document: dict, draft: dict) -> dict:
    """The edit set the draft makes of ``document``: only what changed.

    A new parameter row is given with every field written, even with its name
    or value left empty, so aquakin names what is missing.
    """
    out: dict = {}
    before = {r["name"]: r for r in document["reactions"]}
    for r in draft["reactions"]:
        old = before[r["name"]]
        change: dict = {}
        if _changed(old["rate"], r["rate"]):
            change["rate"] = str(r["rate"]).strip()
        old_stoich = {e["species"]: e["value"] for e in old["stoichiometry"]}
        stoich = {
            e["species"]: str(e["value"]).strip()
            for e in r["stoichiometry"]
            if _changed(old_stoich[e["species"]], e["value"])
        }
        if stoich:
            change["stoichiometry"] = stoich
        if change:
            out.setdefault("reactions", {})[r["name"]] = change
    params = {(p["reaction"], p["name"]): p for p in document["parameters"]}
    for p in draft["parameters"]:
        name = str(p["name"]).strip()
        if p.get("new"):
            fields = {k: str(p[k]).strip() for k in PARAMETER_FIELDS if str(p[k]).strip()}
            out.setdefault("parameters", {})[name] = fields
            continue
        old = params[(p["reaction"], name)]
        fields = {k: str(p[k]).strip() for k in PARAMETER_FIELDS if _changed(old[k], p[k])}
        if not fields:
            continue
        if p["reaction"]:
            reaction = out.setdefault("reactions", {}).setdefault(p["reaction"], {})
            reaction.setdefault("parameters", {})[name] = fields
        else:
            out.setdefault("parameters", {})[name] = fields
    return out


def expressions(edits: dict) -> list[dict]:
    """Each edited expression, as a ``check_expression`` request: ``field``,
    ``expression``, ``kind`` and ``reaction``."""
    out = []
    for reaction, change in (edits.get("reactions") or {}).items():
        if "rate" in change:
            out.append(
                {
                    "field": f"reactions.{reaction}.rate",
                    "expression": change["rate"],
                    "kind": "rate",
                    "reaction": reaction,
                }
            )
        for species, value in (change.get("stoichiometry") or {}).items():
            out.append(
                {
                    "field": f"reactions.{reaction}.stoichiometry.{species}",
                    "expression": value,
                    "kind": "stoichiometry",
                    "reaction": reaction,
                }
            )
    return out


def added(draft: dict) -> list[str]:
    """The names of the parameters the draft adds."""
    return [str(p["name"]).strip() for p in draft["parameters"] if p.get("new")]


def add_parameter(draft: dict, name: str) -> dict:
    """The draft with a new model-level parameter row named ``name``."""
    out = copy.deepcopy(draft)
    out["parameters"].append(
        {"name": name, "reaction": None, "value": "", "units": "", "description": "", "new": True}
    )
    return out


def pointer(text: str, position: int | None) -> str:
    """``text`` with a caret under character ``position`` (empty without one),
    shown in a fixed-width font under the field."""
    if position is None:
        return ""
    return f"{text}\n{' ' * position}^"


def uses(references: dict) -> str:
    """What an expression uses, in one line (``species SS, XB_H · parameters muH``)."""
    parts = [f"{what} {', '.join(names)}" for what, names in references.items() if names]
    return " · ".join(parts)


def page_keys(draft: dict) -> dict[str, str]:
    """The key the page shows each field's error under, by field: ``rate.<i>``
    for reaction ``i``'s rate, ``stoich.<i>.<j>`` for its ``j``-th coefficient
    and ``param.<k>`` for parameter row ``k`` (keyed by the row's own prefix,
    :func:`parameter_key`, which every one of its fields starts with)."""
    keys = {}
    for i, r in enumerate(draft["reactions"]):
        keys[f"reactions.{r['name']}.rate"] = f"rate.{i}"
        for j, e in enumerate(r["stoichiometry"]):
            keys[f"reactions.{r['name']}.stoichiometry.{e['species']}"] = f"stoich.{i}.{j}"
    for k, p in enumerate(draft["parameters"]):
        keys.setdefault(parameter_key(p), f"param.{k}")
    return keys


def field_errors(errors: list[dict], draft: dict) -> tuple[dict, list[str]]:
    """aquakin's errors split into those shown under a field of the draft
    (``{page key: message}``, see :func:`page_keys`) and those about the model
    as a whole, or about a field the draft does not show."""
    keys = page_keys(draft)
    shown: dict[str, str] = {}
    general: list[str] = []
    for e in errors:
        parts = e["field"].split(".")
        prefixes = (".".join(parts[:n]) for n in range(len(parts), 0, -1))
        key = next((keys[f] for f in prefixes if f in keys), None)
        if key is not None:
            shown.setdefault(key, e["error"])
        else:
            general.append(e["error"] if e["field"] == "model" else f"{e['field']}: {e['error']}")
    return shown, general


def proposed_name(model: str, flowsheet: str) -> str:
    """The name offered for a model made from shipped ``model`` in the
    flowsheet named ``flowsheet``: ``asm1_bsm1`` for ``asm1`` in ``BSM1``."""
    tail = re.sub(r"[^0-9A-Za-z_]+", "_", flowsheet).strip("_").lower()
    return f"{model}_{tail or 'edited'}"
