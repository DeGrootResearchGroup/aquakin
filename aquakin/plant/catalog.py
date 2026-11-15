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
from aquakin.plant.ifas import IFASUnit
from aquakin.plant.mbr import MBRUnit
from aquakin.plant.mixer import MixerUnit, RatioSplitter, SetpointSplitter, ThresholdSplitter
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.sbr import SBRUnit
from aquakin.plant.separators import IdealThickener
from aquakin.plant.storage import StorageTank
from aquakin.plant.takacs import TakacsClarifier

__all__ = [
    "CATEGORIES",
    "SETTING_TYPES",
    "PortRule",
    "Setting",
    "UnitKind",
    "catalog",
    "check_unit",
    "model_info",
    "shipped_models",
    "unit_kind",
    "unit_kinds",
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
    "conditions": "a mapping of condition name to a number (e.g. temperature)",
    "aeration": "``None`` or a mapping of :class:`Aeration` arguments",
    "port_name": "the name of one port",
    "port_names": "a list of port names",
    "port_map": "a mapping of port name to a number (one port per key)",
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
    """

    name: str
    type: str
    required: bool
    default: Any = None
    optional: bool = False
    help: str = ""
    python_type: str = ""

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
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float)) for v in value):
        return [_plain(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        if all(isinstance(v, (int, float, str)) for v in value.values()):
            return {k: _plain(v) for k, v in value.items()}
    return None


def _setting_type(name: str, annotation: str) -> tuple[str, bool]:
    """The front-end type of a constructor argument, and whether ``None`` is accepted."""
    optional = "None" in annotation
    base = annotation.replace("| None", "").replace("None |", "").strip()
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
            )
        )
    return tuple(out)


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
    """The influent: a constant feed, built with :meth:`InfluentSeries.constant`."""
    return UnitKind(
        kind="influent",
        label="Influent",
        category="boundary",
        summary="A constant plant inflow, wired to one unit input with Plant.add_influent.",
        python_class="aquakin.plant.influent.InfluentSeries.constant",
        settings=(
            Setting(
                "model", "model", True, help="Kinetic model whose species the feed is written in."
            ),
            Setting("Q", "number", True, help="Volumetric flow rate."),
            Setting(
                "T",
                "number",
                False,
                None,
                optional=True,
                help="Influent temperature (Kelvin); unset leaves the feed temperature-agnostic.",
            ),
            Setting(
                "composition",
                "species_values",
                False,
                {},
                help="Feed concentration of each species; species not listed are absent.",
            ),
        ),
        ports=(PortRule("output", "fixed", names=("out",)),),
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


def model_info(name: str) -> dict:
    """What a front end needs about one model: its species and conditions.

    Parameters
    ----------
    name : str
        A shipped model's name.

    Returns
    -------
    dict
        ``species`` (name, description, unit per species), ``conditions``
        (name, default, unit per condition) and the model's ``time_unit``.
    """
    from aquakin.schema.loader import load_model

    m = load_model(name)
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


def _arguments(kind: UnitKind, settings: dict) -> dict:
    """Constructor arguments from front-end settings (missing ones defaulted)."""
    from aquakin.schema.loader import load_model

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
            value = given[s.name]
        if s.type == "unsupported":
            raise ValueError(f"'{s.name}' ({s.python_type}) can only be set from Python for now")
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
        elif s.type in ("port_map", "conditions", "species_values"):
            args[s.name] = {k: float(_number(v)) for k, v in value.items()}
        else:
            args[s.name] = value
    for group in kind.exactly_one_of:
        chosen = [n for n in group if given.get(n) is not None]
        if len(chosen) != 1:
            raise ValueError(f"give exactly one of {', '.join(group)}")
    return args


def check_unit(kind: str, name: str, settings: dict) -> dict:
    """Build a unit from front-end settings and report what happened.

    Parameters
    ----------
    kind : str
        A unit kind identifier (see :func:`unit_kinds`).
    name : str
        The unit's name.
    settings : dict
        Setting values as the front end holds them (plain data).

    Returns
    -------
    dict
        ``{"ok": True, "inputs": [...], "outputs": [...]}`` with the built
        unit's real ports, or ``{"ok": False, "error": message}`` with the
        reason it could not be built.
    """
    try:
        k = unit_kind(kind)
        args = _arguments(k, settings)
        if k.kind == "influent":
            from aquakin.plant.influent import InfluentSeries

            InfluentSeries.constant(
                args["model"], Q=args["Q"], T=args.get("T"), **args.get("composition", {})
            )
            return {"ok": True, "inputs": [], "outputs": ["out"]}
        cls = _class_of(k)
        unit = cls(name=name, **args)
        return {"ok": True, "inputs": list(unit.input_ports), "outputs": list(unit.output_ports)}
    except Exception as exc:  # the constructor's own message is the answer
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


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
