"""A data description of the plant unit types, for front ends.

A flowsheet editor needs to know, for every kind of unit it can place, what the
unit is called, which settings it takes (with their types, defaults and help),
and which ports it has. This module answers that from the unit classes
themselves: each unit's settings are read from its constructor signature and
its docstring's ``Parameters`` section, so a new constructor argument appears in
the catalogue without editing it. What the signature cannot say is declared
here, per kind:

- a label and a category,
- how the unit's ports follow from its settings (a :class:`PortRule` per side),
- which settings are mutually exclusive (``exactly_one_of``).

The port rules let a front end show a unit's ports while its settings are being
edited, without building it. :func:`check_unit` is the authority: it builds the
real unit from the same settings and reports either the error the constructor
raised or the ports the built unit has.

Everything returned by :func:`catalog` is plain JSON-compatible data (no JAX
arrays, no classes), so it can be sent to a front end in another process.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml

from aquakin.plant.clarifier import IdealClarifier
from aquakin.plant.control import PIController
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.plant.delay import HydraulicDelayUnit
from aquakin.plant.digester import ADM1DigesterUnit
from aquakin.plant.disinfection import ChlorineContactUnit, UVUnit
from aquakin.plant.dosing import DosingUnit
from aquakin.plant.errors import error_reply
from aquakin.plant.headspace import HeadspaceDigesterUnit
from aquakin.plant.ifas import IFASUnit
from aquakin.plant.mbr import MBRUnit
from aquakin.plant.mixer import MixerUnit, RatioSplitter, SetpointSplitter, ThresholdSplitter
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.sbr import SBRPhase, SBRUnit
from aquakin.plant.separators import IdealThickener
from aquakin.plant.settling import InterfaceSettling, LayeredSettling
from aquakin.plant.storage import StorageTank
from aquakin.plant.takacs import TakacsClarifier

__all__ = [
    "CATEGORIES",
    "SETTING_TYPES",
    "PortRule",
    "Setting",
    "UnitKind",
    "build_unit",
    "catalog",
    "check_unit",
    "model_info",
    "shipped_models",
    "unit_kind",
    "unit_kinds",
    "unit_settings",
    "validate_settings",
]

#: The categories a unit kind belongs to, in palette order.
CATEGORIES = ("boundary", "reactor", "separator", "flow", "treatment", "control")

#: The setting types a front end must be able to edit, and what each holds.
SETTING_TYPES = {
    "number": "a real number (``inf`` is written as the string ``'inf'``)",
    "integer": "an integer",
    "boolean": "true or false",
    "string": "free text",
    "model": "the name of a shipped kinetic model",
    "species": "one species name of the unit's model",
    "species_list": "a list of species names of the unit's model",
    "species_values": "a mapping of species name to a number (a composition)",
    "scheduled_number": (
        "a real number, or an expression of time as a string "
        "(``'18446 + 4000*sin(2*pi*t)'``; see :mod:`aquakin.plant.time_expr`)"
    ),
    "scheduled_species_values": (
        "a mapping of species name to a number or an expression of time (a composition "
        "that may follow a schedule)"
    ),
    "conditions": "a mapping of condition name to a number (e.g. temperature)",
    "aeration": "``None`` or a mapping of :class:`Aeration` arguments",
    "port_name": "the name of one port",
    "port_names": "a list of port names",
    "port_map": "a mapping of port name to a number (one port per key)",
    "number_map": "a mapping of name to a number (e.g. settling parameters)",
    "reagent": "a dosed reagent: ``{model, label, composition}``",
    "records": (
        "a list of mappings, each the arguments of one record (an SBR's phases: "
        "``{name, duration, feed, ...}``); its ``fields`` describe them"
    ),
    "tagged": (
        "a mapping naming a ``kind`` and its arguments (an SBR's settling model: "
        "``{kind: interface, v_settle, area}``); its ``fields`` describe each kind's"
    ),
    "unsupported": "a value that can only be given from Python for now",
}


@dataclass(frozen=True)
class PortRule:
    """How some of a unit's ports follow from its settings.

    Parameters
    ----------
    side : {"input", "output"}
        Which side of the unit the ports are on.
    form : {"name", "names", "keys", "fixed"}
        ``"name"``: the setting ``field`` holds one port name. ``"names"``: it
        holds a list of port names. ``"keys"``: it holds a mapping whose keys
        are the port names. ``"fixed"``: the ports are always ``names``.
    field : str, optional
        The setting the ports come from (not used by ``"fixed"``).
    names : tuple of str, optional
        The port names of a ``"fixed"`` rule.
    """

    side: str
    form: str
    field: str | None = None
    names: tuple[str, ...] = ()

    def ports(self, settings: dict) -> list[str]:
        """The port names this rule gives for ``settings``."""
        if self.form == "fixed":
            return list(self.names)
        value = settings.get(self.field)
        if value is None:
            return []
        if self.form == "name":
            return [str(value)]
        if self.form == "names":
            return [str(v) for v in value]
        return [str(k) for k in value]

    def to_json(self) -> dict:
        """The rule as plain data."""
        return {
            "side": self.side,
            "form": self.form,
            "field": self.field,
            "names": list(self.names),
        }


@dataclass(frozen=True)
class Setting:
    """One setting of a unit kind, as a front end edits it.

    Parameters
    ----------
    name : str
        The constructor argument it is passed as.
    type : str
        One of :data:`SETTING_TYPES`.
    required : bool
        Whether the unit cannot be built without it.
    default : object
        The default when not required (plain data; ``None`` when unset).
    optional : bool
        Whether ``None`` is an accepted value.
    help : str
        The argument's description from the unit's docstring.
    python_type : str
        The constructor's annotation, as written.
    fields : list or dict, optional
        What a ``records`` setting's records hold (a list of settings, as plain
        data), or for a ``tagged`` setting, that list for each kind.
    """

    name: str
    type: str
    required: bool
    default: Any = None
    optional: bool = False
    help: str = ""
    python_type: str = ""
    fields: Any = None

    def to_json(self) -> dict:
        """The setting as plain data."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class UnitKind:
    """One kind of unit a flowsheet can hold.

    Parameters
    ----------
    kind : str
        The identifier a flowsheet file uses (e.g. ``"cstr"``).
    label : str
        A short human name.
    category : str
        One of :data:`CATEGORIES`.
    summary : str
        The first paragraph of the unit's docstring.
    python_class : str
        The importable class (or, for an influent, the builder) it maps to.
    settings : tuple of Setting
        Every setting, in constructor order.
    ports : tuple of PortRule
        How the input and output ports follow from the settings.
    exactly_one_of : tuple of tuple of str
        Groups of settings of which exactly one must be given.
    """

    kind: str
    label: str
    category: str
    summary: str
    python_class: str
    settings: tuple[Setting, ...]
    ports: tuple[PortRule, ...]
    exactly_one_of: tuple[tuple[str, ...], ...] = ()

    def defaults(self) -> dict:
        """The settings a newly placed unit starts with (required ones blank)."""
        return {s.name: s.default for s in self.settings}

    def ports_for(self, settings: dict) -> tuple[list[str], list[str]]:
        """The ``(inputs, outputs)`` port names for ``settings``.

        Settings missing from ``settings`` take their defaults.
        """
        full = {**self.defaults(), **settings}
        inputs = [p for r in self.ports if r.side == "input" for p in r.ports(full)]
        outputs = [p for r in self.ports if r.side == "output" for p in r.ports(full)]
        return inputs, outputs

    def to_json(self) -> dict:
        """The kind as plain data."""
        return {
            "kind": self.kind,
            "label": self.label,
            "category": self.category,
            "summary": self.summary,
            "python_class": self.python_class,
            "settings": [s.to_json() for s in self.settings],
            "ports": [r.to_json() for r in self.ports],
            "exactly_one_of": [list(g) for g in self.exactly_one_of],
        }


# ---------------------------------------------------------------------------
# Reading a unit class
# ---------------------------------------------------------------------------

_SECTION_END = re.compile(r"^(Returns|Raises|Notes|Examples|Attributes|See Also|References)\s*$")


def _docstring_parts(cls) -> tuple[str, dict[str, str]]:
    """The summary paragraph and the per-parameter help of a NumPy docstring."""
    doc = inspect.getdoc(cls) or ""
    summary = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    params: dict[str, str] = {}
    lines = doc.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "Parameters") + 2
    except StopIteration:
        return summary, params
    current = None
    for ln in lines[start:]:
        if _SECTION_END.match(ln):
            break
        if ln and not ln.startswith(" "):
            current = ln.split(":", 1)[0].strip()
            for name in current.split(","):
                params.setdefault(name.strip(), "")
            current = [n.strip() for n in current.split(",")]
        elif current and ln.strip():
            for name in current:
                params[name] = (params[name] + " " + ln.strip()).strip()
    return summary, params


def _plain(value) -> Any:
    """``value`` as JSON-compatible data, or ``None`` if it has no plain form."""
    if getattr(value, "shape", None) == () and hasattr(value, "dtype"):
        value = value.item()  # a NumPy / JAX scalar
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            # Not a value a spec can hold: NaN is no setting (and reading it back
            # as "unset" would silently swap in a default).
            raise ValueError("a setting holds NaN")
        return value if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float)) for v in value):
        return [_plain(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        if all(isinstance(v, (int, float, str)) for v in value.values()):
            return {k: _plain(v) for k, v in value.items()}
    return None


#: Settings that hold a list of records: the annotation, the class of a record.
_RECORDS = {"Sequence[SBRPhase]": SBRPhase}
#: Settings that pick a class by ``kind``: the annotation, the classes by kind.
_TAGGED = {"SettlingModel": {"interface": InterfaceSettling, "layered": LayeredSettling}}


def _base(annotation: str) -> str:
    """An annotation without its ``None`` alternative."""
    return annotation.replace("| None", "").replace("None |", "").strip()


def _setting_type(name: str, annotation: str) -> tuple[str, bool]:
    """The front-end type of a constructor argument, and whether ``None`` is accepted."""
    optional = "None" in annotation
    base = _base(annotation)
    if base in _RECORDS:
        return "records", optional
    if base in _TAGGED:
        return "tagged", optional
    if base == "CompiledModel":
        return "model", optional
    if name == "conditions":
        return "conditions", optional
    if base == "Aeration":
        return "aeration", True
    if name in ("input_port_names",):
        return "port_names", optional
    if name.endswith("_port") and base == "str":
        return "port_name", optional
    if name in ("output_port_ratios", "output_port_flows"):
        return "port_map", optional
    if base == "Reagent":
        return "reagent", optional
    if re.fullmatch(r"dict\[str, float\]", base):
        return "number_map", optional
    if "species" in name and base == "str":
        return "species", optional
    if "species" in name and re.fullmatch(r"(Sequence|list|tuple)\[str(, \.\.\.)?\]", base):
        return "species_list", optional
    simple = {"float": "number", "int": "integer", "bool": "boolean", "str": "string"}
    if base in simple:
        return simple[base], optional
    return "unsupported", optional


def _generic_help(name: str, kind: str) -> str:
    """Help for the settings every unit shares, when its docstring omits them."""
    if kind == "model":
        return "Kinetic model the unit's streams are written in."
    if kind == "port_name":
        return f"Name of the {name[: -len('_port')].replace('_', ' ')} port."
    return ""


def _settings_of(cls, skip: tuple[str, ...] = ("name",)) -> tuple[Setting, ...]:
    """The settings of a unit class, read from its signature and docstring."""
    _, helps = _docstring_parts(cls)
    factories = {}
    if dataclasses.is_dataclass(cls):
        factories = {
            f.name: f.default_factory
            for f in dataclasses.fields(cls)
            if f.default_factory is not dataclasses.MISSING
        }
    out = []
    for p in inspect.signature(cls).parameters.values():
        if p.name in skip:
            continue
        annotation = (
            p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", "")
        )
        kind, optional = _setting_type(p.name, annotation)
        help_text = helps.get(p.name, "") or _generic_help(p.name, kind)
        required = p.default is inspect.Parameter.empty
        default = None
        if not required:
            raw = factories[p.name]() if p.name in factories else p.default
            default = _plain(raw)
        out.append(
            Setting(
                name=p.name,
                type=kind,
                required=required,
                default=default,
                optional=optional,
                help=help_text,
                python_type=annotation,
                fields=_fields(kind, annotation),
            )
        )
    return tuple(out)


def _private(cls) -> tuple[str, ...]:
    """The constructor arguments of ``cls`` that are not settings (``_``-named)."""
    return tuple(n for n in inspect.signature(cls).parameters if n.startswith("_"))


def _record_fields(cls) -> list[dict]:
    return [f.to_json() for f in _settings_of(cls, skip=_private(cls))]


def _fields(kind: str, annotation: str):
    """The ``fields`` of a ``records`` or ``tagged`` setting, else ``None``."""
    if kind == "records":
        return _record_fields(_RECORDS[_base(annotation)])
    if kind == "tagged":
        return {k: _record_fields(c) for k, c in _TAGGED[_base(annotation)].items()}
    return None


# ---------------------------------------------------------------------------
# Settings that are the constructor arguments of a class, picked by kind
# ---------------------------------------------------------------------------


def same_kind(default, value) -> bool:
    """Whether ``value`` has the type of a setting whose default is ``default``:
    true/false, a whole number, a finite number, or text."""
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    if isinstance(default, str):
        return isinstance(value, str)
    return isinstance(value, (bool, int, float, str))


def record_settings(value: dict, cls: type, what: str, owner: str, skip=()) -> dict:
    """``value``, a mapping of ``cls``'s constructor arguments, with every one
    filled in from its default and checked by type.

    Parameters
    ----------
    value : dict
        Argument name to plain value.
    cls : type
        The class whose constructor the arguments are for.
    what : str
        The setting, as messages name it (``"translator"``).
    owner : str
        What holds the arguments, as messages name it (``"translator asm1_to_adm1"``).
    skip : tuple of str, optional
        Constructor arguments that are not settings (given by the caller).

    Returns
    -------
    dict
        Every argument but those skipped, a float argument as a float.

    Raises
    ------
    ValueError
        If ``value`` names an argument ``cls`` does not take, holds a value of
        the wrong type, or leaves out an argument that has no default.
    """
    from aquakin.core.hints import did_you_mean

    params = {k: p for k, p in inspect.signature(cls).parameters.items() if k not in skip}
    full = {key: p.default for key, p in params.items()}
    for key, v in value.items():
        if key not in params:
            hint = did_you_mean(str(key), list(params))
            raise ValueError(f"'{key}' is not a setting of {owner}.{hint}")
        example = _example(params[key])
        if v is None and (params[key].default is None or "None" in str(params[key].annotation)):
            full[key] = None
            continue
        if not same_kind(example, v):
            raise ValueError(f"{what} setting '{key}' has the wrong type ({v!r})")
        full[key] = float(v) if isinstance(example, float) else v
    for key, v in full.items():
        if v is inspect.Parameter.empty:
            raise ValueError(f"{what} setting '{key}' is required")
    return full


#: A value of each plain annotation, to check a setting with no default against.
_EXAMPLES = {"bool": False, "int": 0, "float": 0.0, "str": ""}


def _example(p: inspect.Parameter):
    """A value of the type argument ``p`` takes: its default, or else one of
    its annotation (``None`` when that is not a plain type: any plain value)."""
    if p.default is not inspect.Parameter.empty and p.default is not None:
        return p.default
    annotation = p.annotation if isinstance(p.annotation, str) else p.annotation.__name__
    return _EXAMPLES.get(_base(annotation))


def kind_settings(value: dict, classes: dict, what: str, skip=()) -> tuple[type, dict]:
    """The class ``value``'s ``kind`` names among ``classes``, and ``value`` with
    every argument of that class filled in and checked (:func:`record_settings`).

    Raises
    ------
    ValueError
        If the kind is not one of ``classes``, or an argument is unknown or of
        the wrong type.
    """
    value = dict(value)
    kind = value.pop("kind", None)
    if kind not in classes:
        raise ValueError(f"{what} kind {kind!r} is not one of {sorted(classes)}")
    full = record_settings(value, classes[kind], what, f"{what} {kind}", skip)
    return classes[kind], {"kind": kind, **full}


def record_plain(obj, what: str, skip=()) -> dict:
    """The constructor arguments of ``obj`` (but ``skip``) as plain data.

    Raises
    ------
    ValueError
        If an argument has no plain form.
    """
    out: dict[str, Any] = {}
    for name in inspect.signature(type(obj)).parameters:
        if name in skip:
            continue
        raw = getattr(obj, name)
        value = _plain(raw)
        if value is None and raw is not None:
            raise ValueError(f"{what} setting '{name}' has no plain form")
        out[name] = value
    return out


def _entry(name: str, i: int, item) -> str:
    """Entry ``i`` of records setting ``name`` as messages name it, with its own
    name when it has one: ``'phases' entry 2 (react)``."""
    own = item.get("name") if isinstance(item, dict) else None
    return f"'{name}' entry {i + 1}" + (f" ({own})" if isinstance(own, str) else "")


def filled_value(s: Setting, value):
    """The value of a ``records`` or ``tagged`` setting with every argument of
    each record filled in from its default and checked by name and type.

    Parameters
    ----------
    s : Setting
        A setting of type ``records`` or ``tagged``.
    value : list or dict
        Its plain value: a list of mappings, or a mapping with a ``kind``.

    Returns
    -------
    list of dict or dict
        The records, or ``{kind, ...arguments}``.

    Raises
    ------
    ValueError
        If the value has the wrong shape, names an unknown kind or argument
        (with the entry it is in), or holds a value of the wrong type.
    """
    if s.type == "records":
        cls = _RECORDS[_base(s.python_type)]
        if not isinstance(value, list):
            raise ValueError(f"'{s.name}' must be a list, got {type(value).__name__}")
        out = []
        for i, item in enumerate(value):
            entry = _entry(s.name, i, item)
            if not isinstance(item, dict):
                raise ValueError(f"{entry} must be a mapping, got {type(item).__name__}")
            out.append(record_settings(item, cls, entry, entry, _private(cls)))
        return out
    classes = _TAGGED[_base(s.python_type)]
    if not isinstance(value, dict):
        raise ValueError(f"'{s.name}' must be a mapping, got {type(value).__name__}")
    skip = tuple(n for c in classes.values() for n in _private(c))
    return kind_settings(value, classes, f"'{s.name}'", skip)[1]


def kind_plain(obj, classes: dict, what: str, skip=()) -> dict:
    """``obj`` as ``{kind, ...arguments}``, its kind found in ``classes``.

    Raises
    ------
    ValueError
        If ``obj``'s class is none of ``classes``, or an argument has no plain form.
    """
    for kind, cls in classes.items():
        if type(obj) is cls:
            return {"kind": kind, **record_plain(obj, what, skip)}
    raise ValueError(f"{what} {type(obj).__name__} cannot be written in a spec")


def _port_rules(settings: tuple[Setting, ...], fixed: tuple[PortRule, ...]) -> tuple[PortRule, ...]:
    """Port rules read from the port settings, after any fixed rules.

    By the units' naming convention ``input_port`` / ``input_port_names`` name
    inputs and every other ``*_port`` (and the port-keyed mappings) names an
    output; the constructor order of those settings is the port order.
    """
    rules = list(r for r in fixed if r.side == "input")
    for s in settings:
        if s.type == "port_names":
            rules.append(PortRule("input", "names", s.name))
        elif s.type == "port_name" and s.name == "input_port":
            rules.append(PortRule("input", "name", s.name))
    for s in settings:
        if s.type == "port_map":
            rules.append(PortRule("output", "keys", s.name))
        elif s.type == "port_name" and s.name != "input_port":
            rules.append(PortRule("output", "name", s.name))
    rules += [r for r in fixed if r.side == "output"]
    return tuple(rules)


def _kind(kind, label, category, cls, *, fixed=(), exactly_one_of=(), defaults=None) -> UnitKind:
    summary, _ = _docstring_parts(cls)
    settings = _settings_of(cls)
    if defaults:
        settings = tuple(
            dataclasses.replace(s, default=defaults[s.name], required=False)
            if s.name in defaults
            else s
            for s in settings
        )
    return UnitKind(
        kind=kind,
        label=label,
        category=category,
        summary=summary,
        python_class=f"{cls.__module__}.{cls.__qualname__}",
        settings=settings,
        ports=_port_rules(settings, tuple(fixed)),
        exactly_one_of=tuple(tuple(g) for g in exactly_one_of),
    )


def _influent_kind() -> UnitKind:
    """The influent: a constant feed, or a time series read from a table file."""
    return UnitKind(
        kind="influent",
        label="Influent",
        category="boundary",
        summary=(
            "A plant inflow, wired to one unit input with Plant.add_influent: constant "
            "(Q, T, composition), a pattern written as expressions of time, or a time "
            "series read from a table file (series)."
        ),
        python_class="aquakin.plant.influent.InfluentSeries.constant",
        settings=(
            Setting(
                "model", "model", True, help="Kinetic model whose species the feed is written in."
            ),
            Setting(
                "Q",
                "scheduled_number",
                False,
                None,
                optional=True,
                help=(
                    "Volumetric flow rate of the feed (m³/d). A number, or an expression "
                    "of time for a feed that follows a pattern: "
                    "'18446 + 4000*sin(2*pi*t)', with t the plant's clock."
                ),
            ),
            Setting(
                "T",
                "scheduled_number",
                False,
                None,
                optional=True,
                help=(
                    "Influent temperature (K), a number or an expression of time; unset "
                    "leaves the feed temperature-agnostic."
                ),
            ),
            Setting(
                "composition",
                "scheduled_species_values",
                False,
                {},
                help=(
                    "Feed concentration of each species, in that species' unit (g/m³ for "
                    "most); species not listed are absent. Each may be a number or an "
                    "expression of time, so part of a feed can follow a pattern."
                ),
            ),
            Setting(
                "series",
                "string",
                False,
                None,
                optional=True,
                help=(
                    "A time-varying feed: the path of an influent table (columns t, Q, "
                    "optionally T, and species; see read_influent_table), relative to the "
                    "spec file. Given instead of Q, T and composition."
                ),
            ),
        ),
        ports=(PortRule("output", "fixed", names=("out",)),),
        exactly_one_of=(("Q", "series"),),
    )


_SPLITTER_IN = (PortRule("input", "fixed", names=("in",)),)


def _build_kinds() -> dict[str, UnitKind]:
    kinds = [
        _influent_kind(),
        _kind("cstr", "CSTR", "reactor", CSTRUnit, defaults={"input_port_names": ["in"]}),
        _kind("mbr", "Membrane bioreactor", "reactor", MBRUnit),
        _kind(
            "ifas",
            "IFAS / MBBR reactor",
            "reactor",
            IFASUnit,
            defaults={"input_port_names": ["in"]},
        ),
        _kind("sbr", "Sequencing batch reactor", "reactor", SBRUnit),
        _kind("digester", "Anaerobic digester", "reactor", ADM1DigesterUnit),
        _kind(
            "headspace_digester",
            "Digester with unit headspace",
            "reactor",
            HeadspaceDigesterUnit,
        ),
        _kind(
            "clarifier",
            "Secondary clarifier",
            "separator",
            TakacsClarifier,
            exactly_one_of=[("overflow_Q", "underflow_Q")],
        ),
        _kind("primary_clarifier", "Primary clarifier", "separator", PrimaryClarifier),
        _kind(
            "ideal_clarifier",
            "Ideal clarifier",
            "separator",
            IdealClarifier,
            exactly_one_of=[("overflow_Q", "underflow_Q")],
        ),
        _kind("thickener", "Thickener", "separator", IdealThickener),
        _kind(
            "mixer",
            "Mixer",
            "flow",
            MixerUnit,
            fixed=(PortRule("output", "fixed", names=("out",)),),
            defaults={"input_port_names": ["in1", "in2"]},
        ),
        _kind(
            "splitter",
            "Ratio splitter",
            "flow",
            RatioSplitter,
            fixed=_SPLITTER_IN,
            defaults={"output_port_ratios": {"out1": 0.5, "out2": 0.5}},
        ),
        _kind(
            "setpoint_splitter",
            "Setpoint splitter",
            "flow",
            SetpointSplitter,
            fixed=_SPLITTER_IN,
            defaults={"output_port_flows": {"set": 0.0}, "remainder_port": "rest"},
        ),
        _kind(
            "threshold_splitter",
            "Threshold splitter",
            "flow",
            ThresholdSplitter,
            fixed=_SPLITTER_IN,
            defaults={"threshold_port": "bypass", "remainder_port": "rest"},
        ),
        _kind("delay", "Hydraulic delay", "flow", HydraulicDelayUnit),
        _kind(
            "storage",
            "Storage tank",
            "flow",
            StorageTank,
            fixed=(PortRule("output", "fixed", names=("out", "bypass")),),
        ),
        _kind(
            "dosing",
            "Chemical dosing",
            "treatment",
            DosingUnit,
            exactly_one_of=[("flow", "setpoint")],
        ),
        _kind("uv", "UV disinfection", "treatment", UVUnit),
        _kind("chlorine_contact", "Chlorine contact tank", "treatment", ChlorineContactUnit),
        _kind("pi_controller", "PI controller", "control", PIController),
    ]
    return {k.kind: k for k in kinds}


_KINDS: dict[str, UnitKind] | None = None


def unit_kinds() -> dict[str, UnitKind]:
    """Every unit kind, keyed by its identifier, in palette order."""
    global _KINDS
    if _KINDS is None:
        _KINDS = _build_kinds()
    return _KINDS


def unit_kind(kind: str) -> UnitKind:
    """The unit kind named ``kind``.

    Raises
    ------
    KeyError
        If no kind has that name.
    """
    kinds = unit_kinds()
    if kind not in kinds:
        raise KeyError(f"unknown unit kind {kind!r}; known kinds: {sorted(kinds)}")
    return kinds[kind]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def shipped_models() -> list[dict]:
    """The shipped kinetic models: name and description, read without compiling.

    Returns
    -------
    list of dict
        One ``{"name", "description"}`` per shipped model YAML, sorted by name.
    """
    out = []
    for res in sorted(files("aquakin.models").iterdir(), key=lambda r: r.name):
        if not res.name.endswith(".yaml"):
            continue
        doc = yaml.safe_load(res.read_text()) or {}
        meta = doc.get("model", {}) if isinstance(doc, dict) else {}
        out.append(
            {
                "name": res.name[: -len(".yaml")],
                "description": " ".join(str(meta.get("description", "")).split()),
            }
        )
    return out


_FILE_MODELS: dict = {}


def load_model_file(path) -> Any:
    """A model read from a YAML file, cached until it or a file it extends changes.

    Parameters
    ----------
    path : str or pathlib.Path
        The model file.

    Returns
    -------
    CompiledModel
    """
    from pathlib import Path

    from aquakin.schema.loader import load_model_from_file, model_fingerprint

    p = Path(path).resolve()
    # Keyed by the content with the extends chain resolved, so an edit to a base
    # file is seen too.
    key = (str(p), model_fingerprint(p.read_text(), str(p), p.parent))
    if key not in _FILE_MODELS:
        _FILE_MODELS[key] = load_model_from_file(p)
    return _FILE_MODELS[key]


def model_info(name: str, path=None) -> dict:
    """What a front end needs about one model: its species and conditions.

    Parameters
    ----------
    name : str
        A shipped model's name.
    path : str or pathlib.Path, optional
        Read the model from this YAML file instead (a project model).

    Returns
    -------
    dict
        ``species`` (name, description, unit per species), ``conditions``
        (name, default, unit per condition) and the model's ``time_unit``.
    """
    from aquakin.schema.loader import load_model

    m = load_model_file(path) if path is not None else load_model(name)
    defaults = m.condition_defaults()
    conditions = []
    for c in m.conditions_required:
        conditions.append(
            {
                "name": c,
                "default": _plain(defaults.get(c)),
                "unit": m.condition_units.get(c, ""),
            }
        )
    return {
        "name": m.name,
        "time_unit": m.time_unit,
        "species": [
            {
                "name": s,
                "description": m.species_descriptions.get(s, ""),
                "unit": m.species_units.get(s, ""),
            }
            for s in m.species
        ],
        "conditions": conditions,
    }


# ---------------------------------------------------------------------------
# Building a unit from front-end settings
# ---------------------------------------------------------------------------


def _number(value):
    if isinstance(value, str) and value.strip().lower() in ("inf", "+inf", "-inf"):
        return float(value.strip().lower().replace("+", ""))
    return value


def _is_number(v) -> bool:
    """A number a setting may hold: finite, or infinite (``"inf"``) for "no limit"; never NaN."""
    if isinstance(v, str):
        return v.strip().lower() in ("inf", "+inf", "-inf")
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def _check_map(name, value, check, what):
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping, got {type(value).__name__}")
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(f"'{name}' key {k!r} must be a string ({what})")
        if not check(v):
            raise ValueError(f"'{name}.{k}' must be a number, got {v!r}")


def validate_settings(kind: UnitKind, settings: dict) -> None:
    """Check plain settings strictly against ``kind`` before anything is built.

    Every key must be one of the kind's settings (a misspelling would otherwise
    be ignored and its unit built without it), and every value must have its
    setting's type exactly: a boolean is ``true``/``false`` (not the string
    ``"false"``), an integer is not ``10.7``, map keys are strings.

    Raises
    ------
    ValueError
        Naming the first offending setting.
    """
    from aquakin.core.hints import did_you_mean

    if not isinstance(settings, dict):
        raise ValueError(f"settings must be a mapping, got {type(settings).__name__}")
    known = {s.name: s for s in kind.settings}
    for name, value in settings.items():
        if name not in known:
            hint = did_you_mean(str(name), list(known))
            raise ValueError(f"'{name}' is not a setting of {kind.kind}.{hint}")
        s = known[name]
        if value is None:
            if s.required:
                raise ValueError(f"'{name}' is required")
            continue
        t = s.type
        if t in ("number",):
            ok = _is_number(value)
        elif t == "scheduled_number":
            ok = _is_scheduled(name, value)
        elif t == "scheduled_species_values":
            # Checked key by key rather than through _check_map, so an
            # unreadable expression names the species that holds it.
            if not isinstance(value, dict):
                raise ValueError(f"'{name}' must be a mapping, got {type(value).__name__}")
            for species, entry in value.items():
                if not isinstance(species, str):
                    raise ValueError(f"'{name}' key {species!r} must be a string (a species)")
                if not _is_scheduled(f"{name}.{species}", entry):
                    raise ValueError(
                        f"'{name}.{species}' must be "
                        f"{SETTING_TYPES['scheduled_number'].split(' (')[0]}, got {entry!r}"
                    )
            ok = True
        elif t == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif t == "boolean":
            ok = isinstance(value, bool)
        elif t in ("string", "model", "species", "port_name"):
            ok = isinstance(value, str)
        elif t in ("port_names", "species_list"):
            ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        elif t in ("port_map", "number_map", "conditions", "species_values"):
            _check_map(name, value, _is_number, "a name")
            ok = True
        elif t == "aeration":
            ok = isinstance(value, dict)
            if ok:
                _check_aeration(name, value)
        elif t in ("records", "tagged"):
            filled_value(s, value)
            ok = True
        elif t == "reagent":
            ok = (
                isinstance(value, dict)
                and set(value) <= {"model", "label", "composition"}
                and isinstance(value.get("model"), str)
            )
            if ok:
                _check_map(
                    f"{name}.composition", value.get("composition", {}), _is_number, "a species"
                )
        else:  # unsupported
            raise ValueError(f"'{name}' ({s.python_type}) can only be set from Python for now")
        if not ok:
            raise ValueError(f"'{name}' must be {SETTING_TYPES[t].split(' (')[0]}, got {value!r}")


def _is_scheduled(name: str, value) -> bool:
    """Whether a value is a number or a readable expression of time.

    An expression is checked here, as the spec is read, so a typo names the
    setting that holds it rather than failing inside the first solve.
    """
    from aquakin.plant.time_expr import TimeExpressionError, compile_time_expression

    if _is_number(value):
        return True
    if not isinstance(value, str):
        return False
    try:
        compile_time_expression(value, what=f"'{name}'")
    except TimeExpressionError as exc:
        raise ValueError(exc.args[0]) from None
    return True


def _check_aeration(name: str, value: dict) -> None:
    """Check an aeration mapping's keys and value types against :class:`Aeration`."""
    from aquakin.core.hints import did_you_mean

    fields = {f.name: str(f.type) for f in dataclasses.fields(Aeration)}
    for k, v in value.items():
        if k not in fields:
            hint = did_you_mean(str(k), sorted(fields))
            raise ValueError(f"'{name}.{k}' is not an Aeration setting.{hint}")
        annotation = fields[k]
        if v is None:
            if "None" not in annotation:
                raise ValueError(f"'{name}.{k}' cannot be empty")
            continue
        if "bool" in annotation:
            ok, what = isinstance(v, bool), "true or false"
        elif "float" in annotation or "int" in annotation:
            ok, what = _is_number(v), "a number"
        elif "str" in annotation:
            ok, what = isinstance(v, str), "text"
        else:  # pragma: no cover - every Aeration field is one of the above
            ok, what = True, ""
        if not ok:
            raise ValueError(f"'{name}.{k}' must be {what}, got {v!r}")


def _arguments(kind: UnitKind, settings: dict, models: dict | None = None) -> dict:
    """Constructor arguments from front-end settings (missing ones defaulted).

    ``models`` maps model names to model objects to use instead of loading the
    shipped model of that name (a spec's project models).
    """
    from aquakin.schema.loader import load_model as _load_shipped

    def load_model(name):
        return models[name] if models and name in models else _load_shipped(name)

    validate_settings(kind, settings)

    given = {k: v for k, v in settings.items() if v is not None}
    groups = {n for g in kind.exactly_one_of for n in g}
    args: dict[str, Any] = {}
    for s in kind.settings:
        if s.name not in given:
            if s.required:
                if s.type == "unsupported":
                    raise ValueError(
                        f"'{s.name}' ({s.python_type}) can only be set from Python for now"
                    )
                raise ValueError(f"'{s.name}' is required")
            if s.name in groups:
                continue
            if s.type == "unsupported":
                continue
            value = s.default
        else:
            # validate_settings has refused a given value for a Python-only setting.
            value = given[s.name]
        if value is None:
            args[s.name] = None
        elif s.type == "model":
            args[s.name] = load_model(value)
        elif s.type == "number":
            args[s.name] = float(_number(value))
        elif s.type == "integer":
            args[s.name] = int(value)
        elif s.type == "aeration":
            args[s.name] = Aeration(**value) if value else None
        elif s.type in ("port_names", "species_list"):
            args[s.name] = list(value) if s.type == "port_names" else tuple(value)
        elif s.type in ("port_map", "conditions", "species_values", "number_map"):
            args[s.name] = {k: float(_number(v)) for k, v in value.items()}
        elif s.type == "records":
            cls = _RECORDS[_base(s.python_type)]
            args[s.name] = tuple(cls(**r) for r in filled_value(s, value))
        elif s.type == "tagged":
            full = dict(filled_value(s, value))
            args[s.name] = _TAGGED[_base(s.python_type)][full.pop("kind")](**full)
        elif s.type == "reagent":
            from aquakin.plant.dosing import Reagent

            spec = dict(value)
            args[s.name] = Reagent.from_species(
                load_model(spec["model"]),
                label=spec.get("label", "reagent"),
                **{k: float(v) for k, v in spec.get("composition", {}).items()},
            )
        else:
            args[s.name] = value
    for group in kind.exactly_one_of:
        chosen = [n for n in group if given.get(n) is not None]
        if len(chosen) != 1:
            raise ValueError(f"give exactly one of {', '.join(group)}")
    return args


def build_unit(kind: str, name: str, settings: dict, *, models: dict | None = None, base_dir=None):
    """Build a unit (or, for ``"influent"``, an influent series) from plain settings.

    Parameters
    ----------
    kind : str
        A unit kind identifier (see :func:`unit_kinds`).
    name : str
        The unit's name.
    settings : dict
        Setting values as plain data; missing settings take their defaults.
    models : dict, optional
        ``{name: CompiledModel}`` to use for model names instead of loading the
        shipped model of that name.
    base_dir : str or pathlib.Path, optional
        The directory an influent ``series`` path is relative to.

    Returns
    -------
    Unit or InfluentSeries
        The built unit, or the constant influent series for ``"influent"``.

    Raises
    ------
    KeyError
        If ``kind`` is unknown.
    ValueError
        If a required setting is missing or only Python can give it; the unit's
        own constructor errors propagate unchanged.
    """
    k = unit_kind(kind)
    args = _arguments(k, settings, models)
    if k.kind == "influent":
        return _influent(args, base_dir)
    return _class_of(k)(name=name, **args)


def _influent(args: dict, base_dir):
    from pathlib import Path

    from aquakin.plant.influent import InfluentSeries, read_influent_table
    from aquakin.plant.time_expr import is_time_expression

    if args.get("series") is not None:
        if args.get("T") is not None or args.get("composition"):
            raise ValueError("an influent 'series' gives Q, T and composition; set none of them")
        path = Path(args["series"])
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        return read_influent_table(path, args["model"])
    composition = args.get("composition", {})
    # Any expression among the values makes the whole feed a pattern; a feed of
    # plain numbers keeps building the sampled constant series it always did, so
    # every existing spec is read bit-identically.
    if any(is_time_expression(v) for v in (args["Q"], args.get("T"), *composition.values())):
        from aquakin.plant.influent import InfluentExpression

        return InfluentExpression(
            args["model"], Q=args["Q"], composition=composition, T=args.get("T")
        )
    return InfluentSeries.constant(args["model"], Q=args["Q"], T=args.get("T"), **composition)


def _model_name(model) -> str:
    """The shipped name of ``model``, which must be a shipped model as loaded by name.

    Recognised by the provenance ``load_model`` records on the model, so a
    modified copy (``dataclasses.replace``) or a model from a file is refused
    rather than written under a shipped name it does not match.
    """
    if tuple(getattr(model, "_aquakin_source", ()) or ())[:2] != ("shipped", model.name):
        raise ValueError(
            f"model '{model.name}' is not a shipped model as aquakin.load_model('{model.name}') "
            f"returns it (a model from a file, or modified), so it cannot be named in a spec"
        )
    return model.name


def _species_values(model, C, *, keep_zero=False) -> dict:
    return {
        sp: float(v)
        for sp, v in zip(model.species, [float(x) for x in C], strict=True)
        if keep_zero or v != 0.0
    }


def _setting_value(s: Setting, value, *, model_ref=None, full=False):
    """A unit attribute as the plain data of setting ``s``; ``ValueError`` if it has none.

    ``model_ref`` maps a model object to the name a spec refers to it by
    (default: its shipped name). With ``full``, an aeration is written with
    every field rather than only those that differ from the defaults.
    """
    model_ref = model_ref or _model_name
    if value is None:
        return None
    if s.type == "model":
        return model_ref(value)
    if s.type == "aeration":
        out = {}
        for f in dataclasses.fields(value):
            v = _plain(getattr(value, f.name))
            if not full and f.default is not dataclasses.MISSING:
                # The default as the aeration stores it (its normalization runs
                # on it too: stripped=None is kept as an empty tuple). A field
                # the aeration cannot do without (kla) keeps its declared default.
                try:
                    default = getattr(dataclasses.replace(value, **{f.name: f.default}), f.name)
                except ValueError:
                    default = f.default
                if v == _plain(default):
                    continue
            out[f.name] = v
        return out
    if s.type == "records":
        return [
            record_plain(r, f"'{s.name}' entry {i + 1}", _private(type(r)))
            for i, r in enumerate(value)
        ]
    if s.type == "tagged":
        return kind_plain(
            value, _TAGGED[_base(s.python_type)], f"'{s.name}'", _private(type(value))
        )
    if s.type == "reagent":
        return {
            "model": model_ref(value.model),
            "label": value.label,
            "composition": _species_values(value.model, value.composition),
        }
    plain = _plain(value)
    if plain is None:
        raise ValueError(f"'{s.name}' ({s.python_type}) has no plain form")
    return plain


def _kind_of(unit) -> UnitKind:
    for k in unit_kinds().values():
        if k.kind != "influent" and isinstance(unit, _class_of(k)) and type(unit) is _class_of(k):
            return k
    raise ValueError(f"no unit kind for {type(unit).__name__}")


def unit_settings(unit, *, full=False, model_ref=None) -> tuple[str, dict]:
    """The kind and the plain settings that rebuild ``unit`` with :func:`build_unit`.

    Parameters
    ----------
    unit : Unit
        A built unit.
    full : bool, optional
        Write every setting, defaults included, so the result does not depend
        on the defaults of the aquakin that reads it. By default, settings at
        their default are left out. Settings only Python can give are left out
        either way when they are at their default.
    model_ref : callable, optional
        Maps a model object to the name to write for it (default: its shipped
        name, refusing a model not loaded by name).

    Raises
    ------
    ValueError
        If the unit's class has no kind, or one of its settings differs from
        the default and has no plain form (such as an array of initial
        concentrations).
    """
    k = _kind_of(unit)
    out: dict[str, Any] = {}
    for s in k.settings:
        value = getattr(unit, s.name)
        if s.type == "unsupported":
            plain = _plain(value)
            if value is None or (plain is not None and plain == s.default):
                continue
            raise ValueError(f"unit '{unit.name}': '{s.name}' ({s.python_type}) has no plain form")
        try:
            plain = _setting_value(s, value, model_ref=model_ref, full=full)
        except ValueError as exc:
            raise ValueError(f"unit '{unit.name}': {exc}") from None
        if not full and not s.required and plain == s.default:
            continue
        out[s.name] = plain
    return k.kind, out


def check_unit(
    kind: str, name: str, settings: dict, model_files: dict | None = None, base_dir=None
) -> dict:
    """Build a unit from front-end settings and report what happened.

    Parameters
    ----------
    kind : str
        A unit kind identifier (see :func:`unit_kinds`).
    name : str
        The unit's name.
    settings : dict
        Setting values as the front end holds them (plain data).
    model_files : dict, optional
        ``{model name: YAML path}`` for project models the settings may name.
    base_dir : str or pathlib.Path, optional
        The directory an influent ``series`` path is relative to.

    Returns
    -------
    dict
        ``{"ok": True, "inputs": [...], "outputs": [...]}`` with the built
        unit's real ports, or ``{"ok": False, "error": message, "type": class
        name}`` with the reason it could not be built (see
        :func:`~aquakin.plant.errors.error_reply`).
    """
    try:
        models = {n: load_model_file(p) for n, p in (model_files or {}).items()}
        unit = build_unit(kind, name, settings, models=models, base_dir=base_dir)
        if kind == "influent":
            return {"ok": True, "inputs": [], "outputs": ["out"]}
        return {"ok": True, "inputs": list(unit.input_ports), "outputs": list(unit.output_ports)}
    except Exception as exc:  # the constructor's own message is the answer
        return {"ok": False, **error_reply(exc)}


def _class_of(kind: UnitKind):
    module, _, qual = kind.python_class.rpartition(".")
    mod = __import__(module, fromlist=[qual])
    return getattr(mod, qual)


def catalog() -> dict:
    """The whole catalogue as plain data, for a front end.

    Returns
    -------
    dict
        ``categories``, ``setting_types`` and ``kinds`` (one entry per
        :class:`UnitKind`, in palette order).
    """
    return {
        "categories": list(CATEGORIES),
        "setting_types": dict(SETTING_TYPES),
        "kinds": [k.to_json() for k in unit_kinds().values()],
    }
