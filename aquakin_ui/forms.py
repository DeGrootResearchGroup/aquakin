"""The unit form: rows generated from a kind's catalogue settings, and back.

Pure functions, with no page in them. :func:`rows_for` turns one unit into the
rows the Unit panel shows (one per setting; a model's conditions, an
aeration's mode and value, and a reagent's model, label and composition expand
into several), and :func:`read_rows` turns the
edited rows back into the unit's label and settings. No setting, type or choice
is listed here: every row follows from the catalogue.
"""

from __future__ import annotations

import math
import re

from aquakin_ui.document import Flowsheet

AERATION_MODES = ("none", "kla", "do_setpoint")
_READ_ONLY = ("unsupported",)


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


#: Species-list settings that name what settles; their defaults are ASM1's.
_SOLIDS_SETTINGS = ("particulate_species", "settling_species")


def suggest_species(setting: dict, value, species: list[str]) -> list[str] | None:
    """The settling species to suggest for a unit on a model with ``species``, or
    ``None`` to keep what it has.

    A setting naming what settles defaults to ASM1's particulates (or to none):
    on another model, or once the model changed, it gets the model's own
    particulates, the X-named species (aquakin's own rule for solids).
    """
    if setting["name"] not in _SOLIDS_SETTINGS:
        return None
    current = list(value) if value is not None else list(setting.get("default") or [])
    if current and all(c in species for c in current):
        return None
    return [s for s in species if s.startswith("X")]


def _oxygen(species: list[str]) -> str | None:
    """The model's dissolved-oxygen species, by its usual names, or ``None``."""
    for name in ("SO", "SO2", "S_O2", "S_O", "DO", "O2"):
        if name in species:
            return name
    return next((s for s in species if s.upper().startswith(("SO", "S_O"))), None)


_UNIT_IN_HELP = re.compile(r"\(([^()]{1,40})\)|,\s*(m³/d|m³/h)\b")
_UNIT = re.compile(
    r"^(m³|m²|m|mm|K|°C|d|h|s|min|1/d|1/h|1/s|m³/d|m³/h|m/d|m/h|g/m³|kg/m³|mg/L|g/L|%"
    r"|m²/m³|mW/cm²|mJ/cm²|mJ/cm²/log|g_\w+/m³|kg_\w+/m³|mol/m³|kmol/m³)$"
)
#: Units on the plant's clock, written per day in the help; shown per the clock.
_PER_DAY = {"d": "{}", "1/d": "1/{}", "m³/d": "m³/{}", "m/d": "m/{}"}


def unit_in(help_text: str, clock: str = "d") -> str:
    """The unit a setting's help names, ``(m³)``, ``(g/m³, i.e. mg/L)`` or
    ``, m³/d``, or ``""``. aquakin runs every flow and time on the plant's clock,
    and the help writes them per day: they are shown per ``clock``."""
    for match in _UNIT_IN_HELP.finditer(help_text or ""):
        text = re.split(r"[,;:]", match.group(1) or match.group(2))[0].strip()
        text = text.replace("^3", "³").replace("^2", "²")
        if _UNIT.match(text):
            return _PER_DAY[text].format(clock) if text in _PER_DAY else text
    return ""


def _placeholder(s: dict) -> str:
    if s["required"]:
        return "required"
    if s["default"] in (None, {}, []):
        return "not set"
    return f"{_text(s['default'])} (default)"


def _row(key, label, widget="text", value: object = "", **extra) -> dict:
    return {
        "key": key,
        "label": label,
        "widget": widget,
        "value": value,
        "items": [],
        "hint": "",
        "placeholder": "",
        "suffix": "",
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
        ``select``, ``multi``, ``readonly``, ``composition``, ``records``),
        ``value``, ``items``, ``hint``, ``placeholder``, ``suffix``; ``reshape``
        marks a choice that changes which rows the form has. A ``composition``
        row's value is a list of ``{species, value}`` entries, its ``items`` the
        species to choose from and its ``units`` each species' unit. A
        ``records`` row's value is a list of records, one mapping of field to
        shown value each; its ``fields`` describe them, ``blank`` is a new
        record and ``item`` names one (``phase``).
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
        clock = (doc.meta.get("options") or {}).get("time_unit") or "d"
        common = {
            "hint": s["help"],
            "placeholder": _placeholder(s),
            "suffix": unit_in(s["help"], clock),
        }
        if t == "model":
            rows.append(
                _row(
                    key,
                    label,
                    "select",
                    value or "",
                    items=doc.model_names(),
                    reshape=True,
                    **common,
                )
            )
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
                        # aquakin's temperatures are in kelvin
                        suffix=c.get("unit") or ("K" if c["name"] == "T" else ""),
                        placeholder=f"{_text(default)} (model default)"
                        if default is not None
                        else "required",
                    )
                )
        elif t == "aeration":
            rows += _aeration_rows(s, value, info, clock)
        elif t == "species" and species:
            rows.append(_row(key, label, "select", value or "", items=species, **common))
        elif t == "species_list" and species:
            chosen = list(value or [])
            suggested = suggest_species(s, value, species)
            if suggested is not None:
                chosen = suggested
            rows.append(_row(key, label, "multi", chosen, items=species, **common))
        elif t == "boolean":
            text = "" if value is None else str(value).lower()
            rows.append(_row(key, label, "select", text, items=["true", "false"], **common))
        elif t == "reagent":
            rows += _reagent_rows(doc, s, value or {}, model_info)
        elif t == "records":
            rows.append(_records_row(s, value or [], clock))
        elif t == "tagged":
            rows += _tagged_rows(s, value or {}, clock)
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


def _aeration_rows(s: dict, value, info, clock: str) -> list[dict]:
    """The rows of an aeration: its mode, then (in a mode other than none) its
    value on the plant's ``clock`` and the species it aerates."""
    species = [sp["name"] for sp in info["species"]] if info else []
    mode = next((m for m in AERATION_MODES[1:] if value and value.get(m) is not None), "none")
    rows = [
        _row(
            "aeration.mode",
            "aeration",
            "select",
            mode,
            items=list(AERATION_MODES),
            hint=s["help"],
            reshape=True,
        )
    ]
    if mode != "none" and value:  # a mode other than none is read from value
        oxygen = value.get("species") or _oxygen(species) or ""
        units = {sp["name"]: sp.get("unit", "") for sp in info["species"]} if info else {}
        suffix = f"1/{clock}" if mode == "kla" else units.get(oxygen, "")
        rows.append(
            _row("aeration.value", mode, value=_text(value[mode]), hint=s["help"], suffix=suffix)
        )
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
    return rows


def _reagent_rows(doc: Flowsheet, s: dict, value: dict, model_info) -> list[dict]:
    """The rows of a dosed reagent: its model, its label and its composition.

    The composition lists the species of the reagent's own model, which is
    the dosed stream's, so a dosing unit (with no model of its own) is drawn
    with them; the model is a choice that redraws the form.
    """
    key = s["name"]
    model = value.get("model") or ""
    info = model_info(model) if model else None
    species = info["species"] if info else []
    composition = value.get("composition") or {}
    return [
        _row(
            f"{key}.model",
            f"{key} model" + (" *" if s["required"] else ""),
            "select",
            model,
            items=doc.model_names(),
            hint=f"{s['help']} Choose it first: the composition lists its species.",
            placeholder="required",
            reshape=True,
        ),
        _row(
            f"{key}.label",
            f"{key} label",
            value=value.get("label") or "",
            hint="A name for the reagent, used in messages.",
            placeholder="reagent (default)",
        ),
        _row(
            f"{key}.composition",
            f"{key} composition",
            "composition",
            [{"species": k, "value": _text(v)} for k, v in composition.items()],
            items=[sp["name"] for sp in species],
            units={sp["name"]: sp.get("unit", "") for sp in species},
            hint=(
                "The neat reagent's concentration of each species it contains; "
                "every species not listed is zero."
            ),
        ),
    ]


#: The choices of a true / false field that may also be left to its default.
_MAYBE = [
    {"title": "default", "value": ""},
    {"title": "true", "value": "true"},
    {"title": "false", "value": "false"},
]


def _checkbox(f: dict) -> bool:
    """Whether a record's field is a box to tick (true / false, never unset)."""
    return f["type"] == "boolean" and not f["optional"]


def _shown(f: dict, value):
    """A record field's value as its widget holds it: a box's true / false, else text."""
    if _checkbox(f):
        return bool(f["default"] if value is None else value)
    if f["type"] == "boolean":
        return "" if value is None else str(value).lower()
    return _text(value)


def _records_row(s: dict, value: list, clock: str) -> dict:
    """The row of a ``records`` setting: one record per entry of ``value``."""
    fields = [
        {
            **f,
            "suffix": unit_in(f["help"], clock),
            "label": f["name"] + (" *" if f["required"] else ""),
            "placeholder": _placeholder(f),
        }
        for f in s["fields"]
    ]
    return _row(
        s["name"],
        s["name"] + (" *" if s["required"] else ""),
        "records",
        [{f["name"]: _shown(f, r.get(f["name"])) for f in fields} for r in value],
        hint=s["help"],
        fields=fields,
        items=_MAYBE,
        blank={f["name"]: _shown(f, None) for f in fields},
        item=s["name"].removesuffix("s"),
    )


def _read_records(s: dict, entries: list[dict]) -> list[dict]:
    """A ``records`` setting's entries back to plain records; an entry left as
    it was added (every field at its blank) is skipped.

    Raises
    ------
    FieldError
        If a field cannot be read, or a required one is empty; the message
        names the entry and the field.
    """
    item = s["name"].removesuffix("s")
    blank = {f["name"]: _shown(f, None) for f in s["fields"]}
    out = []
    for i, entry in enumerate(entries):
        if all(entry.get(k, b) == b for k, b in blank.items()):
            continue
        own = str(entry.get("name") or "").strip()
        where = f"{item} {i + 1}" + (f" ({own})" if own else "")
        record: dict = {}
        for f in s["fields"]:
            raw = entry.get(f["name"], blank[f["name"]])
            if _checkbox(f):
                record[f["name"]] = bool(raw)
                continue
            try:
                value = _parse(f, raw)
            except ValueError as exc:
                raise FieldError(s["name"], f"{where}: {f['name']}: {exc}") from None
            if value is None and f["required"]:
                raise FieldError(s["name"], f"{where}: give {f['name']}")
            if value is not None or f["type"] == "boolean":
                record[f["name"]] = value
        out.append(record)
    return out


def _tagged_rows(s: dict, value: dict, clock: str) -> list[dict]:
    """The rows of a ``tagged`` setting: its kind (a choice that redraws the
    form), then the arguments of that kind."""
    name, kind = s["name"], value.get("kind") or ""
    rows = [
        _row(
            f"{name}.kind",
            name + (" *" if s["required"] else ""),
            "select",
            kind,
            items=list(s["fields"]),
            hint=s["help"],
            placeholder="required" if s["required"] else "not set",
            reshape=True,
        )
    ]
    for f in s["fields"].get(kind, []):
        label = f"{name} {f['name']}" + (" *" if f["required"] else "")
        rows.append(
            _row(
                f"{name}.{f['name']}",
                label,
                value=_text(value.get(f["name"])),
                hint=f["help"],
                placeholder=_placeholder(f),
                suffix=unit_in(f["help"], clock),
            )
        )
    return rows


def _read_tagged(s: dict, values: dict) -> dict | None:
    """A ``tagged`` setting from its rows: its kind and every argument given
    (an empty one takes its default), or ``None`` when no kind is chosen. The
    arguments of the kind the form was drawn with that the chosen kind also
    takes are kept, so switching kind keeps what both share."""
    name = s["name"]
    kind = str(values.get(f"{name}.kind") or "").strip()
    if not kind:
        return None
    out: dict = {"kind": kind}
    for f in s["fields"].get(kind, []):
        key = f"{name}.{f['name']}"
        if key in values:
            value = _field(key, _parse, f, values[key])
            if value is not None:
                out[f["name"]] = value
    return out


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


def _scheduled(text: str):
    """A value that may be a number or an expression of time.

    A number stays a number, so a constant feed is written to the spec exactly
    as it was before expressions existed; anything else is kept as the text the
    user typed, for aquakin to read (``aquakin_ui`` cannot read it -- the
    independence rule -- so an unreadable expression is reported by the check or
    by Save, not here).
    """
    try:
        return _number(text)
    except ValueError:
        return text


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
    if t == "scheduled_number":
        return _scheduled(text)
    if t == "scheduled_species_values":
        out = {}
        for pair in (p for p in text.split(",") if p.strip()):
            name, colon, value = pair.partition(":")
            if not colon or not name.strip():
                raise ValueError(f"write each entry as 'name: value' (not {pair.strip()!r})")
            out[name.strip()] = _scheduled(value.strip())
        return out
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
    FieldError
        If a field's text cannot be read as its setting's type (a ``ValueError``
        whose ``key`` names the row); the message starts with the field's label.
    """
    labels = {r["key"]: r["label"].removesuffix(" *") for r in rows}
    try:
        return _read_rows(doc, uid, rows)
    except FieldError as exc:
        raise FieldError(exc.key, f"{labels.get(exc.key, exc.key)}: {exc}") from None


class FieldError(ValueError):
    """A field of the form that cannot be read; ``key`` names its row."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


def _field(key: str, read, *args):
    """``read(*args)``, a field's error naming the field."""
    try:
        return read(*args)
    except ValueError as exc:
        raise FieldError(key, str(exc)) from None


def _read_rows(doc: Flowsheet, uid: str, rows: list[dict]) -> tuple[str | None, dict]:
    u = doc.units[uid]
    values = {r["key"]: r["value"] for r in rows}
    items = {r["key"]: r.get("items") for r in rows}
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
        elif t == "reagent":
            reagent = _read_reagent(name, values, items.get(f"{name}.composition") or [])
            if reagent is not None:
                settings[name] = reagent
        elif t == "records":
            records = _read_records(s, values.get(name) or [])
            if records:
                settings[name] = records
        elif t == "tagged":
            tagged = _read_tagged(s, values)
            if tagged is not None:
                settings[name] = tagged
        elif t in _READ_ONLY:
            if name in u.settings:
                settings[name] = u.settings[name]
        elif name in values:
            v = _field(name, _parse, s, values[name])
            if v is not None:
                settings[name] = v
    label = str(values.get("label", "")).strip() or None
    return label, settings


def _composition(entries: list[dict], species: list[str]) -> dict[str, float]:
    """A composition's entries (``{species, value}``, as the form holds them) as
    a mapping of species to a finite number; a wholly empty entry is skipped.

    ``species`` are those the reagent's model has (none when it could not be
    described: then no name is refused here, and aquakin's check reports one).
    """
    out: dict[str, float] = {}
    for entry in entries:
        name = str(entry.get("species") or "").strip()
        raw = entry.get("value")
        text = "" if raw is None else str(raw).strip()
        if not name and not text:
            continue
        if not name:
            raise ValueError(f"choose the species {text} is the concentration of")
        if species and name not in species:
            raise ValueError(f"{name!r} is not a species of the reagent's model")
        if name in out:
            raise ValueError(f"{name} is listed twice")
        if not text:
            raise ValueError(f"give {name} a concentration")
        out[name] = _finite(text)
    return out


def _read_reagent(name: str, values: dict, species: list[str]) -> dict | None:
    """A reagent setting from its rows, or ``None`` when they are all empty.

    Raises
    ------
    FieldError
        If a composition entry cannot be read, or a label or composition is
        given without the model they are written in.
    """
    key = f"{name}.composition"
    composition = _field(key, _composition, values.get(key) or [], species)
    model = str(values.get(f"{name}.model") or "").strip()
    label = str(values.get(f"{name}.label") or "").strip()
    if not model:
        if composition or label:
            raise FieldError(f"{name}.model", "choose the model the reagent is written in")
        return None
    return {"model": model, **({"label": label} if label else {}), "composition": composition}
