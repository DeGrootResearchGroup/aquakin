"""The unit form: rows generated from a kind's catalogue settings, and back.

Pure functions, with no page in them. :func:`rows_for` turns one unit into the
rows the Unit panel shows (one per setting; a model's conditions and an
aeration's mode and value expand into several), and :func:`read_rows` turns the
edited rows back into the unit's label and settings. No setting, type or choice
is listed here: every row follows from the catalogue.
"""

from __future__ import annotations

import math

from aquakin_ui.document import Flowsheet

AERATION_MODES = ("none", "kla", "do_setpoint")
_READ_ONLY = ("unsupported", "reagent")


def _float_text(value: float) -> str:
    """The shortest text that reads back as exactly ``value`` (``240``, not ``240.0``).

    A field must never show a rounded value: Apply reads the text back, so a
    rounded one would silently replace the stored number.
    """
    text = repr(value)
    return text[:-2] if text.endswith(".0") else text


def _text(value) -> str:
    """A setting value as the text a field shows."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return _float_text(value)
    return str(value)


def _oxygen(species: list[str]) -> str | None:
    """The model's dissolved-oxygen species, by its usual names, or ``None``."""
    for name in ("SO", "SO2", "S_O2", "S_O", "DO", "O2"):
        if name in species:
            return name
    return next((s for s in species if s.upper().startswith(("SO", "S_O"))), None)


def _placeholder(s: dict) -> str:
    if s["required"]:
        return "required"
    if s["default"] in (None, {}, []):
        return "not set"
    return f"{_text(s['default'])} (default)"


def _row(key, label, widget="text", value="", **extra) -> dict:
    return {
        "key": key,
        "label": label,
        "widget": widget,
        "value": value,
        "items": [],
        "hint": "",
        "placeholder": "",
    } | extra


def rows_for(doc: Flowsheet, uid: str, model_info) -> list[dict]:
    """The form rows of one unit.

    Parameters
    ----------
    doc : Flowsheet
        The flowsheet.
    uid : str
        The unit's identifier.
    model_info : callable
        ``model_info(name)`` returns the worker's description of a model
        (species, conditions), or ``None`` when it cannot be read.

    Returns
    -------
    list of dict
        One row per field: ``key``, ``label``, ``widget`` (``text``,
        ``select``, ``multi``, ``readonly``), ``value``, ``items``, ``hint``,
        ``placeholder``.
    """
    u = doc.units[uid]
    model = u.settings.get("model")
    info = model_info(model) if model else None
    species = [s["name"] for s in info["species"]] if info else []
    rows = [
        _row("id", "identifier", "readonly", uid, hint="The unit's stable name in aquakin."),
        _row("label", "label", value=u.label or "", hint="Free display text.", placeholder=uid),
    ]
    for s in doc.catalog.settings(u.kind):
        key, value, t = s["name"], u.settings.get(s["name"]), s["type"]
        label = key + (" *" if s["required"] else "")
        common = {"hint": s["help"], "placeholder": _placeholder(s)}
        if t == "model":
            rows.append(_row(key, label, "select", value or "", items=doc.model_names(), **common))
        elif t == "conditions":
            for c in info["conditions"] if info else []:
                v = (value or {}).get(c["name"])
                default = c["default"]
                rows.append(
                    _row(
                        f"conditions.{c['name']}",
                        f"condition {c['name']}",
                        value=_text(v),
                        hint=s["help"],
                        placeholder=f"{_text(default)} (model default)"
                        if default is not None
                        else "required",
                    )
                )
        elif t == "aeration":
            mode = next(
                (m for m in AERATION_MODES[1:] if value and value.get(m) is not None), "none"
            )
            rows.append(
                _row(
                    "aeration.mode",
                    "aeration",
                    "select",
                    mode,
                    items=list(AERATION_MODES),
                    hint=s["help"],
                )
            )
            if mode != "none":
                rows.append(_row("aeration.value", mode, value=_text(value[mode]), hint=s["help"]))
            if mode != "none" and species:
                aerated = (value or {}).get("species")
                if aerated not in species:  # e.g. ASM1's SO on an ASM3 plant
                    aerated = _oxygen(species) or aerated
                rows.append(
                    _row(
                        "aeration.species",
                        "aerated species",
                        "select",
                        aerated or "",
                        items=species,
                        hint="The dissolved species the aeration transfers (oxygen).",
                    )
                )
        elif t == "species" and species:
            rows.append(_row(key, label, "select", value or "", items=species, **common))
        elif t == "species_list" and species:
            rows.append(_row(key, label, "multi", list(value or []), items=species, **common))
        elif t == "boolean":
            text = "" if value is None else str(value).lower()
            rows.append(_row(key, label, "select", text, items=["true", "false"], **common))
        elif t == "reagent":
            shown = (
                f"{value['label']}: {_text(value.get('composition', {}))}"
                if value
                else "set in the spec file"
            )
            rows.append(_row(key, label, "readonly", "", hint=s["help"], placeholder=shown))
        elif t == "unsupported":
            rows.append(
                _row(
                    key,
                    label,
                    "readonly",
                    "",
                    hint=s["help"],
                    placeholder=f"set from Python ({s['python_type']})",
                )
            )
        else:
            rows.append(_row(key, label, value=_text(value), **common))
    return rows


def _finite(text: str) -> float:
    """A finite number from a field's text; ``nan`` and overflowing values are refused."""
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a number") from None
    if not math.isfinite(value):  # "nan", "inf", "1e999"
        raise ValueError(f"{text!r} is not a finite number")
    return value


def _number(text: str):
    """A number setting's text: a finite number, or ``inf`` / ``-inf`` for no limit."""
    return text.lower() if text.lower() in ("inf", "-inf") else _finite(text)


def _parse(s: dict, text):
    """A field's text back to plain data, by setting type (empty means unset)."""
    t = s["type"]
    text = text.strip() if isinstance(text, str) else text
    if text in ("", None, []):
        return None
    if t == "number":
        return _number(text)
    if t == "integer":
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"{text!r} is not a whole number") from None
    if t == "boolean":
        return text == "true"
    if t in ("port_names", "species_list"):
        if isinstance(text, list):
            return list(text)
        return [p.strip() for p in text.split(",") if p.strip()]
    if t in ("port_map", "species_values", "number_map"):
        out = {}
        for pair in (p for p in text.split(",") if p.strip()):
            name, colon, value = pair.partition(":")
            if not colon or not name.strip():
                raise ValueError(f"write each entry as 'name: value' (not {pair.strip()!r})")
            out[name.strip()] = _number(value.strip())
        return out
    return text


def read_rows(doc: Flowsheet, uid: str, rows: list[dict]) -> tuple[str | None, dict]:
    """The label and settings the edited ``rows`` give one unit.

    Settings shown read-only, and the aeration fields the form does not show
    (controller gains, saturation, ...), keep the values the unit had.

    Raises
    ------
    ValueError
        If a field's text cannot be read as its setting's type; the message
        starts with the field's label.
    """
    labels = {r["key"]: r["label"].removesuffix(" *") for r in rows}
    try:
        return _read_rows(doc, uid, rows)
    except _FieldError as exc:
        raise ValueError(f"{labels.get(exc.key, exc.key)}: {exc.message}") from None


class _FieldError(Exception):
    def __init__(self, key: str, message: str):
        self.key, self.message = key, message


def _field(key: str, read, *args):
    """``read(*args)``, a field's error naming the field."""
    try:
        return read(*args)
    except ValueError as exc:
        raise _FieldError(key, str(exc)) from None


def _read_rows(doc: Flowsheet, uid: str, rows: list[dict]) -> tuple[str | None, dict]:
    u = doc.units[uid]
    values = {r["key"]: r["value"] for r in rows}
    settings: dict = {}
    for s in doc.catalog.settings(u.kind):
        name, t = s["name"], s["type"]
        if t == "conditions":
            if not any(k.startswith("conditions.") for k in values):
                # The model could not be described, so the form had no condition
                # rows: the unit keeps the conditions it has.
                if u.settings.get(name):
                    settings[name] = u.settings[name]
                continue
            conds = {
                k.split(".", 1)[1]: _field(k, _finite, str(v).strip())
                for k, v in values.items()
                if k.startswith("conditions.") and str(v).strip()
            }
            if conds:
                settings[name] = conds
        elif t == "aeration":
            mode = values.get("aeration.mode", "none")
            old = u.settings.get(name) or {}
            if mode != "none":
                # The value field belongs to the mode the form was drawn with: after
                # a switch of mode it holds the other mode's number, so it is not read.
                shown = next((m for m in AERATION_MODES[1:] if old.get(m) is not None), "none")
                raw = str(values.get("aeration.value", "")).strip() if mode == shown else ""
                value = _field("aeration.value", _finite, raw) if raw else (old.get(mode) or 0.0)
                kept = {k: v for k, v in old.items() if k not in AERATION_MODES[1:]}
                if str(values.get("aeration.species") or "").strip():
                    kept["species"] = values["aeration.species"]
                settings[name] = {**kept, mode: value}
            else:
                settings[name] = None
        elif t in _READ_ONLY:
            if name in u.settings:
                settings[name] = u.settings[name]
        elif name in values:
            v = _field(name, _parse, s, values[name])
            if v is not None:
                settings[name] = v
    label = str(values.get("label", "")).strip() or None
    return label, settings
