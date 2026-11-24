"""A plant flowsheet as plain data: :meth:`Plant.from_spec` and :meth:`Plant.to_spec`.

A *spec* is a JSON/YAML document that rebuilds a plant::

    format: 2
    aquakin: 0.2.0
    name: BSM1
    options: {recycle_passes: 3, recycle_tol: 1.0e-08, recycle_max_passes: 100}
    models:
      asm1: {source: shipped, sha256: 3f9c...}
      asm1_bsm2: {source: models/asm1_bsm2.yaml, sha256: 8a01...}
    files:
      data/feed.csv: 51b2...
    units:
      feed: {kind: influent, model: asm1, series: data/feed.csv, ...}
      tank1: {kind: cstr, label: Anoxic zone 1, model: asm1, volume: 1000.0, ...}
    connections:
      - inlet_mix.out -> tank1.inlet
      - {from: sludge_mix.out, to: digester.inlet, translator: {kind: asm1_to_adm1, ...}}
      - {from: reject_mix.out, to: front_mix.reject, seed: {Q: 0.0, T: 293.15, ...}}
    streams: {effluent: clarifier.overflow}
    influent_endpoint: inlet_mix.fresh
    effluent_endpoint: clarifier.overflow

**Identity.** A unit's key is its stable identifier: the plant's unit name,
used by every connection, parameter (``"<unit>.<setpoint>"``), stream and
result. ``label`` is free display text (kept in :attr:`Plant.labels`), so
relabelling a unit changes nothing that refers to it. Names follow one rule
(:data:`aquakin.plant.errors.NAME_PATTERN`).

**Written in full.** Every setting is written, defaults included, together
with the aquakin version that wrote the file, so a spec means the same plant
whatever the defaults of the aquakin that reads it.

**A project travels whole.** Every model a unit uses is listed under
``models``: a shipped model, or a model YAML file inside the spec's folder (it
may ``extends:`` a shipped model). A time-varying influent is a table file
(``series``, see :func:`~aquakin.plant.influent.read_influent_table`) inside
the folder too, its sha256 under ``files``. :func:`write_spec` copies a file
from elsewhere into ``models/`` or ``data/`` beside the spec, so no spec it
writes points outside its folder. Each model records the fingerprint of the
content it was built from; reading a spec whose model or table has changed
since it was written warns (:class:`SpecWarning`).

**Validation.** The document is validated by a Pydantic schema and each unit's
settings strictly against the unit catalogue (:mod:`aquakin.plant.catalog`):
an unknown key or a mistyped value is an error naming where it is. YAML is read
with YAML 1.2 rules for scalars (``1e-5`` is a number; ``yes``, ``on`` and dates
are text).

**Provenance.** :func:`spec_inputs` fingerprints what a spec simulates, one
entry per input (the flowsheet, each model, each influent table), each hashed
from its *current* content; :func:`spec_fingerprint` combines them. A result
records them to tell later whether it is stale, and which input changed.

A unit of kind ``influent`` is a feed, constant or a table file; its single
connection becomes :meth:`Plant.add_influent`. Units the plant derives itself
(the PI controllers a closed-loop :class:`Aeration` or a feedback
:class:`DosingUnit` requires) are not written; they are derived again. What a
spec cannot express yet is refused by :meth:`Plant.to_spec` with a
:class:`SpecError` naming it rather than dropped: a model that is neither
shipped nor read from a file, a heat-balance temperature model, and settings
only Python can give (arrays of initial concentrations, SBR phases, settling
models).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aquakin.plant.errors import NAME_PATTERN, error_reply

__all__ = [
    "SPEC_FORMAT",
    "TRANSLATORS",
    "SpecError",
    "SpecWarning",
    "aquakin_version",
    "check_spec",
    "load_spec_file",
    "plant_from_spec",
    "plant_to_spec",
    "spec_changes",
    "spec_fingerprint",
    "spec_inputs",
    "write_spec",
]

#: The spec format version this module writes and reads.
SPEC_FORMAT = 2

#: Translator kinds a spec can name.
TRANSLATORS = ("asm1_to_adm1", "adm1_to_asm1")

_OPTION_KEYS = ("time_unit", "recycle_passes", "recycle_tol", "recycle_max_passes")


class SpecError(ValueError):
    """A spec that cannot be read, or a plant that cannot be written as one."""


class SpecWarning(UserWarning):
    """A spec read with a model or table whose content changed since it was written."""


def _translators() -> dict:
    from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1

    return {"asm1_to_adm1": ASM1toADM1, "adm1_to_asm1": ADM1toASM1}


def aquakin_version() -> str:
    """The aquakin version, with the git commit for a development checkout.

    Returns
    -------
    str
        ``"0.2.0"`` for an installed release, ``"0.2.0+g1a2b3c4"`` when aquakin
        runs from a git checkout (``+dirty`` added when it has local changes).
    """
    import subprocess

    from aquakin import __version__

    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return __version__
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return __version__
    return f"{__version__}+g{commit}" + (".dirty" if dirty else "")


# ---------------------------------------------------------------------------
# YAML with YAML 1.2 scalars
# ---------------------------------------------------------------------------


class _Yaml12Loader(yaml.SafeLoader):
    """PyYAML's safe loader with YAML 1.2 core-schema scalars.

    PyYAML follows YAML 1.1, where ``yes``/``no``/``on``/``off`` are booleans,
    ``1e-5`` (no dot) is a string, ``010`` is octal and ``2026-01-01`` a date; in
    a spec those would silently turn a port named ``on`` into ``True`` or a
    concentration into text. Here booleans are ``true``/``false`` only, integers
    are decimal (or ``0o``/``0x``), floats include ``1e-5``, and dates stay text.
    """


_SCALAR_TAGS = (
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:timestamp",
)
_Yaml12Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag not in _SCALAR_TAGS]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
    list("-+0123456789"),
)
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _yaml12_int(loader, node):
    text = loader.construct_scalar(node)
    if text.startswith(("0o", "0x")):
        return int(text, 0)
    return int(text, 10)  # a leading zero is still decimal, as in YAML 1.2


_Yaml12Loader.add_constructor(
    "tag:yaml.org,2002:bool", lambda loader, node: loader.construct_scalar(node).lower() == "true"
)
_Yaml12Loader.add_constructor("tag:yaml.org,2002:int", _yaml12_int)


def load_spec_file(path) -> dict:
    """Read a spec file as plain data: ``.json`` as JSON, anything else as YAML 1.2.

    Parameters
    ----------
    path : str or pathlib.Path
        The spec file.

    Returns
    -------
    dict

    Raises
    ------
    SpecError
        If the file is not valid YAML (or JSON).
    """
    text = Path(path).read_text()
    try:
        if Path(path).suffix == ".json":
            return json.loads(text)
        return yaml.load(text, Loader=_Yaml12Loader)  # a SafeLoader subclass
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise SpecError(f"{path} is not a readable spec file: {exc}") from None


# ---------------------------------------------------------------------------
# The document schema
# ---------------------------------------------------------------------------

_STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _int_to_float(v):
    """Accept a whole number where a float is expected (``Q: 0``), nothing else."""
    return float(v) if isinstance(v, int) and not isinstance(v, bool) else v


class _Options(BaseModel):
    model_config = _STRICT
    # The plant's clock ("d", "h", ...); a spec written before plants declared
    # one leaves it out, and the models' common unit is used.
    time_unit: str | None = None
    recycle_passes: int = 3
    recycle_tol: float | None = 1e-8
    recycle_max_passes: int = 100

    _floats = field_validator("recycle_tol", mode="before")(_int_to_float)


class _ModelRef(BaseModel):
    model_config = _STRICT
    source: str  # "shipped", or a model YAML path relative to the spec file
    sha256: str | None = None
    # A frozen copy of a shipped model (a study's snapshot): the file holds the
    # shipped model as it was, so it keeps the shipped name.
    copy_of: Literal["shipped"] | None = None


class _Unit(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    kind: str
    label: str | None = None


class _Seed(BaseModel):
    model_config = _STRICT
    Q: float = 0.0
    T: float | None = None
    concentrations: dict[str, float] | Literal["default"] = "default"

    _floats = field_validator("Q", "T", mode="before")(_int_to_float)

    @field_validator("concentrations", mode="before")
    @classmethod
    def _int_values_to_float(cls, v):
        return {k: _int_to_float(x) for k, x in v.items()} if isinstance(v, dict) else v


class _Connection(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, populate_by_name=True, allow_inf_nan=False
    )
    source: str = Field(alias="from")
    to: str
    translator: dict[str, Any] | None = None
    seed: _Seed | None = None


def _check_identifier(what: str, name: str) -> str:
    if not NAME_PATTERN.match(name):
        raise ValueError(
            f"{what} {name!r} must start with a letter or '_' and hold only letters, digits, "
            f"'_' and '-'"
        )
    return name


class _Document(BaseModel):
    model_config = _STRICT
    format: Literal[2]
    aquakin: str | None = None
    name: str = "plant"
    options: _Options = _Options()
    models: dict[str, _ModelRef] = {}
    files: dict[str, str] = {}
    units: dict[str, _Unit] = {}
    connections: list[str | _Connection] = []
    streams: dict[str, str] = {}
    influent_endpoint: str | None = None
    effluent_endpoint: str | None = None
    layout: dict[str, Any] | None = None

    @field_validator("units")
    @classmethod
    def _unit_identifiers(cls, units):
        for uid in units:
            _check_identifier("unit identifier", uid)
        return units

    @field_validator("models")
    @classmethod
    def _model_names(cls, models):
        for name in models:
            _check_identifier("model name", name)
        return models


def _validate(data) -> _Document:
    if not isinstance(data, dict):
        raise SpecError("a spec must be a mapping")
    if data.get("format") != SPEC_FORMAT:
        raise SpecError(
            f"spec format {data.get('format')!r} is not supported (this aquakin reads format "
            f"{SPEC_FORMAT}); write the plant again with Plant.to_spec"
        )
    try:
        return _Document.model_validate(data)
    except ValidationError as exc:
        lines = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors(include_url=False)
        ]
        raise SpecError("invalid spec:\n  " + "\n  ".join(lines)) from None


def _parse_connection(entry, where: str) -> _Connection:
    if isinstance(entry, _Connection):
        return entry
    parts = [p.strip() for p in entry.split("->")]
    if len(parts) != 2 or not all(parts):
        raise SpecError(f"{where}: write a connection as 'unit.port -> unit.port', got {entry!r}")
    return _Connection(**{"from": parts[0], "to": parts[1]})


# ---------------------------------------------------------------------------
# Content fingerprints
# ---------------------------------------------------------------------------


def _model_fingerprint(path: Path) -> str:
    from aquakin.schema.loader import model_fingerprint

    try:
        text = path.read_text()
    except FileNotFoundError:
        raise SpecError(f"model file {path} does not exist") from None
    return model_fingerprint(text, str(path), path.parent)


def _shipped_fingerprint(name: str) -> str:
    from importlib.resources import files

    from aquakin.schema.loader import model_fingerprint

    return model_fingerprint((files("aquakin.models") / f"{name}.yaml").read_text(), name)


def _shipped_names() -> set[str]:
    from importlib.resources import files

    return {
        q.name.removesuffix(".yaml")
        for q in files("aquakin.models").iterdir()
        if q.name.endswith(".yaml") and not q.name.startswith("_")
    }


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        raise SpecError(f"{path} does not exist") from None


def _located(source: str, base_dir: Path | None, where: str) -> Path:
    path = Path(source)
    if not path.is_absolute():
        if base_dir is None:
            raise SpecError(
                f"{where}: the relative path {source!r} needs the spec to be read from a "
                f"file, or base_dir="
            )
        path = base_dir / path
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _load_models(doc: _Document, base_dir: Path | None) -> tuple[dict, dict]:
    """The spec's models by name, and the absolute path of each one read from a file."""
    from aquakin.schema.loader import load_model, load_model_from_file

    models, model_files = {}, {}
    for name, ref in doc.models.items():
        where = f"models.{name}"
        try:
            if ref.source == "shipped":
                model = load_model(name)
            else:
                path = _located(ref.source, base_dir, where)
                model = load_model_from_file(path)
                model_files[name] = path.resolve()
        except SpecError:
            raise
        except Exception as exc:
            raise SpecError(f"{where}: {type(exc).__name__}: {exc}") from None
        if ref.source != "shipped" and ref.copy_of is None and name in _shipped_names():
            raise SpecError(
                f"{where}: '{name}' is the name of a shipped model; give the model in "
                f"{ref.source!r} a name of its own"
            )
        if model.name != name:
            raise SpecError(
                f"{where}: the model in {ref.source!r} is named '{model.name}', not '{name}'"
            )
        digest = model._aquakin_source[2]
        if ref.sha256 is not None and ref.sha256 != digest:
            warnings.warn(
                f"model '{name}' ({ref.source}) has changed since this spec was written "
                f"(aquakin {doc.aquakin or 'unknown'}); the plant is built with the current model",
                SpecWarning,
                stacklevel=4,
            )
        models[name] = model
    return models, model_files


def _check_files(doc: _Document, base_dir: Path | None) -> None:
    for source, recorded in doc.files.items():
        path = _located(source, base_dir, f"files.{source}")
        if path.is_file() and _file_digest(path) != recorded:
            warnings.warn(
                f"{source} has changed since this spec was written "
                f"(aquakin {doc.aquakin or 'unknown'}); the plant is built with the current file",
                SpecWarning,
                stacklevel=4,
            )


def _unit_of(endpoint: str) -> str:
    return endpoint.split(".", 1)[0]


def _check_model_names(settings: dict, models: dict, where: str) -> None:
    """Every model a unit names must be one the spec lists."""
    names = [settings.get("model")]
    if isinstance(settings.get("reagent"), dict):
        names.append(settings["reagent"].get("model"))
    for name in names:
        if name is not None and name not in models:
            raise SpecError(f"{where}: model '{name}' is not listed under 'models'")


def _same_kind(default, value) -> bool:
    """Whether ``value`` has the type of a translator setting whose default is
    ``default``: true/false, a finite number, or text."""
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    if isinstance(default, str):
        return isinstance(value, str)
    return isinstance(value, (bool, int, float, str))


def _translator_settings(spec: dict, where: str) -> tuple[type, dict]:
    """The translator class and its settings, every one filled in, checked by type."""
    spec = dict(spec)
    kind = spec.pop("kind", None)
    classes = _translators()
    if kind not in classes:
        raise SpecError(f"{where}: translator kind {kind!r} is not one of {sorted(classes)}")
    cls = classes[kind]
    params = inspect.signature(cls).parameters
    full = {}
    for key, p in params.items():
        if key in ("source_model", "target_model"):
            continue
        full[key] = p.default
    for key, value in spec.items():
        if key not in full:
            raise SpecError(f"{where}: '{key}' is not a setting of translator {kind}")
        if not _same_kind(full[key], value):
            raise SpecError(f"{where}: translator setting '{key}' has the wrong type ({value!r})")
        full[key] = float(value) if isinstance(full[key], float) else value
    return cls, {"kind": kind, **full}


def _build_translator(spec: dict, src_model, dst_model, where: str):
    cls, full = _translator_settings(spec, where)
    full.pop("kind")
    try:
        return cls(source_model=src_model, target_model=dst_model, **full)
    except Exception as exc:
        raise SpecError(f"{where}: translator: {type(exc).__name__}: {exc}") from None


def _build_seed(seed: _Seed, model):
    import jax.numpy as jnp

    from aquakin.plant.streams import Stream

    if seed.concentrations == "default":
        C = model.default_concentrations()
    else:
        C = model.concentrations(seed.concentrations, base="zero")
    scalars = {"T": jnp.asarray(seed.T)} if seed.T is not None else {}
    return Stream(Q=jnp.asarray(seed.Q), C=C, model=model, scalars=scalars)


def _resolve(spec, base_dir):
    if isinstance(spec, (str, Path)):
        base_dir = Path(spec).parent if base_dir is None else Path(base_dir)
        spec = load_spec_file(spec)
    elif base_dir is not None:
        base_dir = Path(base_dir)
    return spec, base_dir


def plant_from_spec(cls, spec, *, base_dir=None):
    """Build a plant from a spec (a mapping, or a ``.yaml`` / ``.json`` file path).

    Bound as the classmethod :meth:`Plant.from_spec`.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The spec, or the path of a file holding it.
    base_dir : str or pathlib.Path, optional
        The directory model and table paths are relative to when ``spec`` is a
        mapping (a file's own directory otherwise).

    Returns
    -------
    Plant

    Raises
    ------
    SpecError
        If the spec is malformed, names an unknown kind, setting or model, or a
        unit cannot be built; the message says where (``units.tank1: ...``).

    Warns
    -----
    SpecWarning
        If a model or table differs from the fingerprint the spec recorded.
    """
    from aquakin.plant.catalog import build_unit

    spec, base_dir = _resolve(spec, base_dir)
    doc = _validate(spec)
    models, model_files = _load_models(doc, base_dir)
    _check_files(doc, base_dir)

    plant = cls(doc.name, **doc.options.model_dump())
    plant._spec_model_files = model_files
    influents = {}
    for uid, entry in doc.units.items():
        where = f"units.{uid}"
        settings = dict(entry.model_extra or {})
        _check_model_names(settings, models, where)
        try:
            built = build_unit(entry.kind, uid, settings, models=models, base_dir=base_dir)
        except Exception as exc:
            raise SpecError(f"{where}: {type(exc).__name__}: {exc}") from None
        if entry.kind == "influent":
            influents[uid] = built
        else:
            plant.add_unit(built)
        if entry.label is not None:
            plant.labels[uid] = entry.label

    fed = dict.fromkeys(influents, 0)
    for i, raw in enumerate(doc.connections):
        where = f"connections[{i}]"
        entry = _parse_connection(raw, where)
        src, dst = entry.source, entry.to
        src_unit, dst_unit = _unit_of(src), _unit_of(dst)
        try:
            dst_model = plant.units[dst_unit].model
            if src_unit in influents:
                fed[src_unit] += 1
                translator = None
                if entry.translator:
                    translator = _build_translator(
                        entry.translator, influents[src_unit].model, dst_model, where
                    )
                plant.add_influent(src_unit, influents[src_unit], to=dst, translator=translator)
                continue
            kwargs = {}
            if entry.translator:
                kwargs["translator"] = _build_translator(
                    entry.translator, plant.units[src_unit].model, dst_model, where
                )
            if entry.seed is not None:
                # A seed stands for the stream arriving at the destination, so it
                # is written in the destination's model (after any translator).
                kwargs["initial_value"] = _build_seed(entry.seed, dst_model)
            plant.connect(src, dst, **kwargs)
        except SpecError:
            raise
        except Exception as exc:
            raise SpecError(f"{where} ({src} -> {dst}): {type(exc).__name__}: {exc}") from None
    for name, n in fed.items():
        if n != 1:
            raise SpecError(f"influent '{name}' must feed exactly one unit input; it feeds {n}")

    for name, endpoint in doc.streams.items():
        try:
            plant.register_stream(name, endpoint)
        except Exception as exc:
            raise SpecError(f"streams.{name}: {type(exc).__name__}: {exc}") from None
    plant.influent_endpoint = doc.influent_endpoint
    plant.effluent_endpoint = doc.effluent_endpoint
    # Rebuilt plants of one spec compile the same programs, so they share them
    # (a study run, the editor's worker); a change to this plant ends that.
    plant._solve_cache.share_as((cls, _built_fingerprint(doc, models, influents)))
    return plant


# ---------------------------------------------------------------------------
# From a plant
# ---------------------------------------------------------------------------


class _ModelRefs:
    """Names each model object of a plant and records where it came from.

    A model read from a file (``load_model_from_file`` records the path and the
    content it was built from) is written as that file, a model ``load_model``
    loaded as shipped. Anything else (a modified copy) is refused, since naming
    it would build a different plant; so is a file edited since it was loaded.
    """

    def __init__(self, extra_files: dict):
        self._files = {name: Path(p).resolve() for name, p in (extra_files or {}).items()}
        self.used: dict[str, dict] = {}

    def __call__(self, model) -> str:
        name = model.name
        if name in self.used:
            return name
        source = tuple(getattr(model, "_aquakin_source", ()) or ())
        if source[:1] == ("file",) or name in self._files:
            if name in _shipped_names():
                raise SpecError(
                    f"model '{name}' is read from a file but has the name of a shipped model; "
                    f"give it a name of its own so a spec cannot mistake one for the other"
                )
            path = Path(source[1]) if source[:1] == ("file",) else self._files[name]
            loaded = source[2] if source[:1] == ("file",) else None
            current = _model_fingerprint(path)
            if loaded is not None and current != loaded:
                raise SpecError(
                    f"model file {path} has changed since model '{name}' was loaded from it; "
                    f"load it again so the spec describes the plant in memory"
                )
            self.used[name] = {"source": str(path), "sha256": current}
        elif source[:2] == ("shipped", name):
            self.used[name] = {"source": "shipped", "sha256": source[2]}
        else:
            raise SpecError(
                f"model '{name}' is neither a shipped model as aquakin.load_model('{name}') "
                f"returns it nor read from a model file (a modified copy, say); write it as a "
                f"model file and load it with aquakin.load_model_from_file"
            )
        return name


def _influent_entry(name, series, refs) -> tuple[dict, dict]:
    """An influent's unit entry, and ``{absolute table path: sha256}`` for a series."""
    import numpy as np

    from aquakin.plant.influent import InfluentExpression

    if isinstance(series, InfluentExpression):
        # A pattern is already the settings it was built from -- nothing is
        # sampled, so there is no table to write beside the spec.
        return {
            "kind": "influent",
            "model": refs(series.model),
            "Q": _scheduled(name, "Q", series.Q),
            "T": None if series.T is None else _scheduled(name, "T", series.T),
            "composition": {sp: _scheduled(name, sp, v) for sp, v in series.composition.items()},
            "series": None,
        }, {}

    Q, C = np.asarray(series.Q), np.asarray(series.C)
    T = None if series.T is None else np.asarray(series.T)
    constant = np.all(Q[0] == Q) and np.all(C[0] == C) and (T is None or np.all(T[0] == T))
    species = series.model.species
    out = {"kind": "influent", "model": refs(series.model)}
    if constant:
        composition = {sp: float(v) for sp, v in zip(species, C[0], strict=True) if v != 0.0}
        values = [float(Q[0]), *composition.values()] + ([] if T is None else [float(T[0])])
        if not all(math.isfinite(v) for v in values):
            raise SpecError(f"influent '{name}' holds a value that is not finite")
        return out | {
            "Q": float(Q[0]),
            "T": None if T is None else float(T[0]),
            "composition": composition,
            "series": None,
        }, {}
    source = tuple(getattr(series, "_aquakin_source", ()) or ())
    if source[:1] == ("file",):
        path = Path(source[1])
        if _file_digest(path) != source[2]:
            raise SpecError(
                f"influent table {path} has changed since influent '{name}' was read from it"
            )
        files = {str(path): source[2]}
    else:
        files = {}  # written by write_spec from the series in memory
        out["_series_object"] = series
    return out | {
        "Q": None,
        "T": None,
        "composition": {},
        "series": str(source[1]) if files else None,
    }, files


def _scheduled(name: str, key: str, value):
    """One value of a patterned influent, as a spec holds it: an expression stays
    the text it was written as, a number is written as a float."""
    if isinstance(value, str):
        return value
    value = float(value)
    if not math.isfinite(value):
        raise SpecError(f"influent '{name}' holds a value that is not finite ({key})")
    return value


def _translator_spec(t, where) -> dict:
    from aquakin.plant.catalog import _plain

    for kind, cls in _translators().items():
        if type(t) is cls:
            break
    else:
        raise SpecError(f"{where}: translator {type(t).__name__} cannot be written in a spec")
    out: dict[str, Any] = {"kind": kind}
    params = inspect.signature(cls).parameters
    for f in dataclasses.fields(t):
        if f.name in ("source_model", "target_model") or f.name not in params:
            continue
        value = _plain(getattr(t, f.name))
        if value is None:
            raise SpecError(f"{where}: translator setting '{f.name}' has no plain form")
        out[f.name] = value
    return out


def _seed_spec(stream, model, where) -> dict:
    import numpy as np

    if stream.model is not model:
        raise SpecError(f"{where}: a seed in another model than its destination unit's")
    extra = set(stream.scalars) - {"T"}
    if extra:
        raise SpecError(f"{where}: seed scalars {sorted(extra)} cannot be written in a spec")
    C = np.asarray(stream.C)
    return {
        "Q": float(stream.Q),
        "T": float(stream.scalars["T"]) if "T" in stream.scalars else None,
        "concentrations": {sp: float(v) for sp, v in zip(model.species, C, strict=True)},
    }


def plant_to_spec(plant, path=None, *, models=None) -> dict:
    """The plant as a spec that :meth:`Plant.from_spec` rebuilds.

    Bound as :meth:`Plant.to_spec`. Every setting is written in full.

    Parameters
    ----------
    plant : Plant
        The plant to write.
    path : str or pathlib.Path, optional
        Write the spec to this file with :func:`write_spec`, which copies the
        models and influent tables it uses into ``models/`` and ``data/`` beside
        it when they are not already in its folder, and writes paths relative to
        it. Without a path, model files are named by absolute path, and a
        time-varying influent read from no file is refused.
    models : dict, optional
        ``{model name: model YAML path}`` for a model that was not read with
        ``load_model_from_file`` but is that file's content.

    Returns
    -------
    dict
        The spec (plain data), as written when ``path`` is given.

    Raises
    ------
    SpecError
        If the plant holds something a spec cannot express yet (the message
        names it), or a model or table file changed since it was read.
    """
    from aquakin.plant.catalog import unit_settings
    from aquakin.plant.temperature import AlgebraicTemperature
    from aquakin.plant.translators import IdentityTranslator

    if type(plant.temperature_model) is not AlgebraicTemperature:
        raise SpecError(
            f"temperature model {type(plant.temperature_model).__name__} cannot be written "
            f"in a spec yet"
        )
    refs = _ModelRefs(models)
    derived = getattr(plant, "_derived_units", set())
    files: dict[str, str] = {}

    units: dict[str, dict] = {}
    for name, series in plant.influents.items():
        units[name], f = _influent_entry(name, series, refs)
        files.update(f)
    for name in plant._insertion_order:
        if name in derived:
            continue
        try:
            kind, settings = unit_settings(plant.units[name], full=True, model_ref=refs)
        except ValueError as exc:
            raise SpecError(str(exc)) from None
        units[name] = {"kind": kind, **settings}
    for name, label in plant.labels.items():
        if name in units:
            entry = units[name]
            units[name] = {"kind": entry.pop("kind"), "label": label, **entry}

    connections = []
    for i, c in enumerate(plant.connections):
        if c.to_unit in derived or c.from_unit in derived:
            continue
        where = f"connection {i}"
        src = f"{c.from_port}.out" if c.from_unit is None else f"{c.from_unit}.{c.from_port}"
        dst = f"{c.to_unit}.{c.to_port}"
        extra = {}
        if not isinstance(c.translator, IdentityTranslator):
            extra["translator"] = _translator_spec(c.translator, where)
        if c.initial_value is not None:
            extra["seed"] = _seed_spec(c.initial_value, plant.units[c.to_unit].model, where)
        connections.append({"from": src, "to": dst, **extra} if extra else f"{src} -> {dst}")

    spec: dict[str, Any] = {
        "format": SPEC_FORMAT,
        "aquakin": aquakin_version(),
        "name": plant.name,
        "options": {key: getattr(plant, key) for key in _OPTION_KEYS},
        "models": dict(refs.used),
        "files": files,
        "units": units,
        "connections": connections,
        "streams": dict(plant.named_streams),
        "influent_endpoint": plant.influent_endpoint,
        "effluent_endpoint": plant.effluent_endpoint,
    }
    if path is not None:
        return write_spec(spec, path)
    pending = [u for u, e in units.items() if "_series_object" in e]
    if pending:
        raise SpecError(
            f"influent '{pending[0]}' varies in time; write the spec to a file "
            f"(Plant.to_spec(path)) so its series can be written beside it"
        )
    return spec


# ---------------------------------------------------------------------------
# Writing a spec file (shared by Plant.to_spec and the flowsheet editor)
# ---------------------------------------------------------------------------


def _within(path: Path, folder: Path) -> Path | None:
    try:
        return path.resolve().relative_to(folder.resolve())
    except ValueError:
        return None


def _extends_of(text: str, where: str):
    """The ``extends:`` value of a model file's text (``None`` without one)."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise SpecError(f"{where} is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        return None
    meta = data.get("model")
    base = meta.get("extends") if isinstance(meta, dict) else None
    return base if base is not None else data.get("extends")


def _is_file_base(base) -> bool:
    """Whether an ``extends:`` value names a file (else a shipped model), as the loader reads it."""
    return isinstance(base, str) and ("/" in base or "\\" in base or base.endswith(".yaml"))


def _model_chain(src: Path, where: str) -> list[Path]:
    """A model file and every file it ``extends:``, in order (shipped bases end the chain)."""
    if not src.is_file():
        raise SpecError(f"{where}: model file {src} does not exist")
    chain = [src.resolve()]
    while True:
        base = _extends_of(chain[-1].read_text(), str(chain[-1]))
        if not _is_file_base(base):
            return chain
        nxt = Path(base)
        nxt = (nxt if nxt.is_absolute() else chain[-1].parent / nxt).resolve()
        if nxt in chain:
            raise SpecError(f"{where}: the extends chain of {src} returns to {nxt}")
        if not nxt.is_file():
            raise SpecError(f"{where}: {chain[-1]} extends {base!r}, which does not exist")
        chain.append(nxt)


_EXTENDS_LINE = re.compile(
    r"^(?P<head>[ \t]*extends[ \t]*:[ \t]*)(?P<q>['\"]?)[^'\"#\n]+?(?P=q)(?P<tail>[ \t]*(#.*)?)$",
    re.MULTILINE,
)


def _rewrite_extends(text: str, base: str, where: str) -> str:
    """``text`` with its one ``extends:`` line pointing at ``base`` (comments kept)."""
    found = _EXTENDS_LINE.findall(text)
    if len(found) != 1:
        raise SpecError(
            f"{where}: cannot find its one 'extends:' line to point it at the copied {base}"
        )
    return _EXTENDS_LINE.sub(lambda m: f"{m['head']}{base}{m['tail']}", text)


def _plan_model(name: str, src: Path, folder: Path, writes: list) -> str:
    """Where model ``name`` sits relative to the spec, adding any copies it needs to ``writes``.

    A model whose whole ``extends:`` chain lies inside the spec's folder stays where
    it is. Otherwise the whole chain is brought into ``models/`` (the model as
    ``<name>.yaml``, each base beside it, each file's ``extends:`` pointing at the
    next copy), so the project holds everything it is built from.
    """
    where = f"models.{name}"
    chain = _model_chain(src, where)
    if all(_within(p, folder) is not None for p in chain):
        return _within(src, folder).as_posix()
    dests = [folder / "models" / f"{name}.yaml"]
    used = {dests[0].name}
    for p in chain[1:]:
        candidate, i = f"{p.stem}.yaml", 2
        while candidate in used:
            candidate, i = f"{p.stem}_{i}.yaml", i + 1
        used.add(candidate)
        dests.append(folder / "models" / candidate)
    for i, (p, dest) in enumerate(zip(chain, dests)):
        text = p.read_text()
        if i + 1 < len(chain):
            text = _rewrite_extends(text, dests[i + 1].name, str(p))
        writes.append((dest, text.encode(), f"model '{name}'", p))
    return f"models/{name}.yaml"


def _check_writes(writes: list) -> None:
    """Refuse a planned copy that would replace a different file, or two that disagree."""
    planned: dict[Path, bytes] = {}
    for dest, content, what, src in writes:
        key = dest.resolve()
        if key in planned and planned[key] != content:
            raise SpecError(f"{what}: two different files would be written to {dest}")
        planned[key] = content
        if dest.exists() and dest.read_bytes() != content:
            if src is not None and src.resolve() == key:
                continue  # the file itself, with its extends: line updated
            raise SpecError(
                f"{what}: {dest} already exists with different content; move it or rename "
                f"the {what.split()[0]} so its file can be written"
            )


def _do_writes(writes: list) -> None:
    for dest, content, _, _ in writes:
        if dest.exists() and dest.read_bytes() == content:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)


def _series_bytes(series) -> bytes:
    """An in-memory influent series as the table file :func:`write_influent_table` writes."""
    from aquakin.plant.influent import write_influent_table

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "series.csv"
        write_influent_table(series, target)
        return target.read_bytes()


def _canonical_number(v):
    """A number setting in one spelling: a float, or ``"inf"`` / ``"-inf"``."""
    if isinstance(v, str):
        return v.strip().lower().replace("+", "")
    v = float(v)
    return v if math.isfinite(v) else ("inf" if v > 0 else "-inf")


def _canonical_settings(kind, settings: dict, where: str) -> dict:
    """``settings`` checked strictly against ``kind`` and with every number a float.

    So that ``1000`` and ``1000.0`` (a hand edit, a JSON writer) are one plant.
    ``None`` values (unset; required ones too, in a draft) are kept as they are.
    """
    from aquakin.plant.catalog import validate_settings

    try:
        validate_settings(kind, {k: v for k, v in settings.items() if v is not None})
    except ValueError as exc:
        raise SpecError(f"{where}: {exc}") from None
    types = {s.name: s.type for s in kind.settings}
    out = {}
    for k, v in settings.items():
        t = types.get(k)
        if v is None:
            out[k] = None
        elif t == "number":
            out[k] = _canonical_number(v)
        elif t in ("port_map", "number_map", "conditions", "species_values"):
            out[k] = {key: _canonical_number(x) for key, x in v.items()}
        elif t == "aeration":
            out[k] = {
                key: _canonical_number(x)
                if isinstance(x, (int, float)) and not isinstance(x, bool)
                else x
                for key, x in v.items()
            }
        elif t == "reagent":
            composition = {key: _canonical_number(x) for key, x in v.get("composition", {}).items()}
            out[k] = {**v, "composition": composition}
        else:
            out[k] = v
    return out


def write_spec(spec: dict, path, *, base_dir=None) -> dict:
    """Write a spec file that holds the whole plant, and return what was written.

    The one writer behind :meth:`Plant.to_spec` and the flowsheet editor's
    Save. It checks every unit's settings (a draft may leave required ones
    unset), fills them in from the catalogue defaults with every number a float,
    records the aquakin version and the current sha256 of every model and
    influent table, copies a model (with its whole ``extends:`` chain) or a
    table from outside the spec's folder into ``models/`` or
    ``data/<unit>.csv``, writes their paths relative to the spec, and replaces
    the file atomically (written to a temporary file, then renamed). Everything
    is checked before any file is touched, so a refused spec leaves the folder
    as it was.

    Parameters
    ----------
    spec : dict
        The spec to write (its relative paths are relative to ``base_dir``).
    path : str or pathlib.Path
        The spec file (``.json`` as JSON, anything else as YAML).
    base_dir : str or pathlib.Path, optional
        Where ``spec``'s relative paths point from (the editor's open folder);
        by default, the folder of ``path``.

    Returns
    -------
    dict
        The spec as written (plain JSON-compatible data).

    Raises
    ------
    SpecError
        If a unit kind or setting is unknown or of the wrong type (NaN included),
        a referenced file is missing, or a file to copy would overwrite another.
    """
    from aquakin.plant.catalog import unit_kind

    path = Path(path)
    folder = path.parent if str(path.parent) else Path(".")
    base = Path(base_dir) if base_dir is not None else folder
    spec = dict(spec)
    writes: list = []  # (destination, content, what, source file or None)

    models = {}
    for name, ref in (spec.get("models") or {}).items():
        _check_identifier_or_error("model name", name)
        source = ref.get("source", "shipped")
        if source == "shipped":
            models[name] = {"source": "shipped", "sha256": _shipped_fingerprint(name)}
            continue
        src = _located(source, base, f"models.{name}")
        relative = _plan_model(name, src, folder, writes)
        models[name] = {"source": relative, "sha256": _model_fingerprint(src)}
        if ref.get("copy_of") is not None:
            models[name]["copy_of"] = ref["copy_of"]

    files: dict[str, str] = {}
    units = {}
    for uid, entry in (spec.get("units") or {}).items():
        entry = dict(entry)
        kind = entry.pop("kind")
        label = entry.pop("label", None)
        series_obj = entry.pop("_series_object", None)
        entry.pop("series_sha256", None)  # format-2 drafts kept it as a setting
        try:
            k = unit_kind(kind)
        except KeyError as exc:
            raise SpecError(f"units.{uid}: {exc.args[0]}") from None
        settings = _canonical_settings(k, {**k.defaults(), **entry}, f"units.{uid}")
        if kind == "influent" and (settings.get("series") is not None or series_obj is not None):
            dest_rel = Path("data") / f"{uid}.csv"
            if series_obj is not None:
                content = _series_bytes(series_obj)
                writes.append((folder / dest_rel, content, f"influent '{uid}'", None))
                relative = dest_rel.as_posix()
            else:
                src = _located(settings["series"], base, f"units.{uid}.series")
                if not src.is_file():
                    raise SpecError(f"units.{uid}.series: {src} does not exist")
                content = src.read_bytes()
                rel = _within(src, folder)
                if rel is None:
                    writes.append((folder / dest_rel, content, f"influent '{uid}'", src))
                    rel = dest_rel
                relative = rel.as_posix()
            settings["series"] = relative
            files[relative] = hashlib.sha256(content).hexdigest()
        units[uid] = {"kind": kind, **({"label": label} if label else {}), **settings}

    out = {
        "format": SPEC_FORMAT,
        "aquakin": aquakin_version(),
        "name": spec.get("name", "plant"),
        "options": {**_Options().model_dump(), **(spec.get("options") or {})},
        "models": models,
        "files": files,
        "units": units,
        "connections": list(spec.get("connections") or []),
        "streams": dict(spec.get("streams") or {}),
        "influent_endpoint": spec.get("influent_endpoint"),
        "effluent_endpoint": spec.get("effluent_endpoint"),
    }
    if spec.get("layout"):
        out["layout"] = spec["layout"]
    try:
        json_text = json.dumps(out, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise SpecError(f"the spec holds a value a spec file cannot: {exc}") from None
    if path.suffix == ".json":
        text = json_text
    else:
        text = yaml.safe_dump(out, sort_keys=False, allow_unicode=True)
    _check_writes(writes)

    _do_writes(writes)
    folder.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def _check_identifier_or_error(what: str, name: str) -> None:
    try:
        _check_identifier(what, name)
    except ValueError as exc:
        raise SpecError(str(exc)) from None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _canonical_endpoint(endpoint: str, ports: dict, side: str) -> str:
    unit, _, port = endpoint.strip().partition(".")
    unit, port = unit.strip(), port.strip()
    if not port:
        found = ports.get(unit, ([], []))[0 if side == "input" else 1]
        if len(found) == 1:
            port = found[0]
    return f"{unit}.{port}"


def spec_inputs(spec, *, base_dir=None) -> dict[str, str]:
    """Fingerprints of what a spec simulates, one per input, from current content.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The spec, or the path of a file holding it.
    base_dir : str or pathlib.Path, optional
        Where a mapping's relative paths point from (a file's own folder otherwise).

    Returns
    -------
    dict
        ``{"flowsheet": ..., "model:<name>": ..., "table:<unit>": ...}``: the
        flowsheet (options, units in order with every setting, connections in a
        canonical form, streams and endpoints; not layout, labels, the plant's
        name or where files sit), each model by the fingerprint of its current
        content, and each influent table by the sha256 of its current bytes.
        A result records these to tell, later, whether it is stale and which
        input changed.
    """
    spec, base_dir = _resolve(spec, base_dir)

    def model_digest(name, ref):
        if ref.source == "shipped":
            return _shipped_fingerprint(name)
        return _model_fingerprint(_located(ref.source, base_dir, name))

    def table_digest(uid, source):
        return _file_digest(_located(source, base_dir, f"units.{uid}.series"))

    return _inputs_of(_validate(spec), model_digest, table_digest)


def _inputs_of(doc: _Document, model_digest, table_digest) -> dict[str, str]:
    """:func:`spec_inputs` of a validated spec, its models and tables digested
    by the given callables.

    Parameters
    ----------
    doc : _Document
        The validated spec.
    model_digest : callable
        ``model_digest(name, ref)``: the fingerprint of model ``name``.
    table_digest : callable
        ``table_digest(unit, source)``: the sha256 of the table an influent
        ``unit`` reads from ``source``.

    Returns
    -------
    dict
        As :func:`spec_inputs` returns.
    """
    from aquakin.plant.catalog import unit_kind

    inputs = {f"model:{name}": model_digest(name, ref) for name, ref in doc.models.items()}
    units, ports = [], {}
    for uid, entry in doc.units.items():
        try:
            kind = unit_kind(entry.kind)
        except KeyError as exc:
            raise SpecError(f"units.{uid}: {exc.args[0]}") from None
        extra = dict(entry.model_extra or {})
        extra.pop("series_sha256", None)
        settings = _canonical_settings(kind, {**kind.defaults(), **extra}, f"units.{uid}")
        ports[uid] = kind.ports_for(settings)
        if settings.get("series") is not None:
            inputs[f"table:{uid}"] = table_digest(uid, settings["series"])
            settings["series"] = "<table>"  # where the table sits does not matter
        units.append([uid, entry.kind, settings])
    connections = []
    for i, raw in enumerate(doc.connections):
        c = _parse_connection(raw, f"connections[{i}]")
        connections.append(
            {
                "from": _canonical_endpoint(c.source, ports, "output"),
                "to": _canonical_endpoint(c.to, ports, "input"),
                "translator": _translator_settings(c.translator, f"connections[{i}]")[1]
                if c.translator
                else None,
                "seed": c.seed.model_dump() if c.seed is not None else None,
            }
        )
    flowsheet = {
        "format": SPEC_FORMAT,
        "options": doc.options.model_dump(),
        "units": units,
        "connections": connections,
        "streams": doc.streams,
        "influent_endpoint": doc.influent_endpoint,
        "effluent_endpoint": doc.effluent_endpoint,
    }
    text = json.dumps(flowsheet, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"flowsheet": hashlib.sha256(text.encode()).hexdigest(), **inputs}


def spec_fingerprint(spec, *, base_dir=None) -> str:
    """One fingerprint of the plant a spec builds: :func:`spec_inputs` combined.

    It changes when what is simulated changes (a setting, a connection, unit
    order, the content of a model or influent table) and not otherwise (layout,
    labels, the plant's name, how a connection is spelled, where a file sits).
    The aquakin version is left out: record it beside the fingerprint.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The spec, or the path of a file holding it.
    base_dir : str or pathlib.Path, optional
        Where a mapping's relative paths point from.

    Returns
    -------
    str
        A 64-character hex digest.
    """
    return _combined(spec_inputs(spec, base_dir=base_dir))


def _combined(inputs: dict[str, str]) -> str:
    """One digest of :func:`spec_inputs`' fingerprints (:func:`spec_fingerprint`)."""
    text = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _built_fingerprint(doc: _Document, models: dict, influents: dict) -> str:
    """:func:`spec_fingerprint` of the plant :func:`plant_from_spec` built.

    The models and influent tables are digested as they were read (the
    fingerprint ``load_model`` / ``load_model_from_file`` and
    ``read_influent_table`` record on what they return), not as the files are
    now: a file edited after it was read, while the plant was built or before
    a cached shipped model was reloaded, must not give two different plants
    one fingerprint.

    Parameters
    ----------
    doc : _Document
        The validated spec.
    models : dict
        ``{name: CompiledModel}`` as loaded.
    influents : dict
        ``{unit: InfluentSeries}`` as built.

    Returns
    -------
    str
        A 64-character hex digest.
    """
    inputs = _inputs_of(
        doc,
        lambda name, ref: models[name]._aquakin_source[2],
        lambda uid, source: influents[uid]._aquakin_source[2],
    )
    return _combined(inputs)


def spec_changes(spec, *, base_dir=None) -> list[str]:
    """The models and tables whose content differs from what the spec recorded.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The spec, or the path of a file holding it.
    base_dir : str or pathlib.Path, optional
        Where a mapping's relative paths point from.

    Returns
    -------
    list of str
        One line per changed input (empty when nothing changed).
    """
    spec, base_dir = _resolve(spec, base_dir)
    doc = _validate(spec)
    changed = []
    for name, ref in doc.models.items():
        if ref.sha256 is None:
            continue
        if ref.source == "shipped":
            current = _shipped_fingerprint(name)
        else:
            current = _model_fingerprint(_located(ref.source, base_dir, name))
        if current != ref.sha256:
            changed.append(f"model '{name}' ({ref.source}) has changed")
    for source, recorded in doc.files.items():
        path = _located(source, base_dir, source)
        if not path.is_file():
            changed.append(f"{source} is missing")
        elif _file_digest(path) != recorded:
            changed.append(f"{source} has changed")
    return changed


def check_spec(spec, *, base_dir=None) -> dict:
    """Build a plant from ``spec`` and report :meth:`Plant.check` on it.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        As for :meth:`Plant.from_spec`.
    base_dir : str or pathlib.Path, optional
        As for :meth:`Plant.from_spec`.

    Returns
    -------
    dict
        ``{"ok": True, "unfed": [...], "dangling": [...], "recycles": [...]}``
        when the plant builds (``unfed`` input ports are the errors), or
        ``{"ok": False, "error": message, "type": class name}`` when it does
        not (see :func:`~aquakin.plant.errors.error_reply`).
    """
    from aquakin.plant.plant import Plant

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpecWarning)
            result = Plant.from_spec(spec, base_dir=base_dir).check()
    except Exception as exc:
        return {"ok": False, **error_reply(exc)}
    return {
        "ok": True,
        "unfed": list(result.unfed_ports),
        "dangling": list(result.dangling_outputs),
        "recycles": list(result.recycles),
    }
