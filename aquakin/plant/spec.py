"""A plant flowsheet as plain data: :meth:`Plant.from_spec` and :meth:`Plant.to_spec`.

A *spec* is a JSON/YAML document that rebuilds a plant::

    format: 2
    aquakin: 0.2.0
    name: BSM1
    options: {recycle_passes: 3, recycle_tol: 1.0e-08, recycle_max_passes: 100}
    models:
      asm1: {source: shipped, sha256: 3f9c...}
      asm1_bsm2: {source: models/asm1_bsm2.yaml, sha256: 8a01...}
    units:
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
relabelling a unit changes nothing that refers to it.

**Written in full.** Every setting is written, defaults included, together
with the aquakin version that wrote the file, so a spec means the same plant
whatever the defaults of the aquakin that reads it.

**Models.** Every model a unit uses is listed under ``models``, keyed by its
name: either a shipped model (``source: shipped``) or a model YAML file whose
path is relative to the spec file (it may ``extends:`` a shipped model).
``sha256`` fingerprints the model's content with its ``extends`` chain
resolved; reading a spec whose model has changed since it was written warns
(:class:`SpecWarning`).

**Validation.** The document is validated by a Pydantic schema and each unit's
settings strictly against the unit catalogue (:mod:`aquakin.plant.catalog`):
an unknown key or a mistyped value is an error naming where it is. YAML is read
with YAML 1.2 rules for booleans and numbers (``1e-5`` is a number, ``yes`` and
``on`` are strings).

A unit of kind ``influent`` is a constant feed: its single connection becomes
:meth:`Plant.add_influent`. Units the plant derives itself (the PI controllers
a closed-loop :class:`Aeration` or a feedback :class:`DosingUnit` requires) are
not written; they are derived again. What a spec cannot express yet is refused
by :meth:`Plant.to_spec` with a :class:`SpecError` naming it rather than
dropped: a time-varying influent, a model that is neither shipped nor given as
a file, a heat-balance temperature model, and settings only Python can give
(arrays of initial concentrations, SBR phases, settling models).
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "SPEC_FORMAT",
    "TRANSLATORS",
    "SpecError",
    "SpecWarning",
    "check_spec",
    "load_spec_file",
    "plant_from_spec",
    "plant_to_spec",
]

#: The spec format version this module writes and reads.
SPEC_FORMAT = 2

#: Translator kinds a spec can name.
TRANSLATORS = ("asm1_to_adm1", "adm1_to_asm1")

_OPTION_KEYS = ("recycle_passes", "recycle_tol", "recycle_max_passes")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


class SpecError(ValueError):
    """A spec that cannot be read, or a plant that cannot be written as one."""


class SpecWarning(UserWarning):
    """A spec read with a model whose content changed since the spec was written."""


def _translators() -> dict:
    from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1

    return {"asm1_to_adm1": ASM1toADM1, "adm1_to_asm1": ADM1toASM1}


# ---------------------------------------------------------------------------
# YAML with YAML 1.2 booleans and numbers
# ---------------------------------------------------------------------------


class _Yaml12Loader(yaml.SafeLoader):
    """PyYAML's safe loader with YAML 1.2 core-schema booleans, integers and floats.

    PyYAML follows YAML 1.1, where ``yes``/``no``/``on``/``off`` are booleans and
    ``1e-5`` (no dot) is a string; in a spec those would silently turn a port
    named ``on`` into ``True`` or a concentration into text.
    """


_SCALAR_TAGS = ("tag:yaml.org,2002:bool", "tag:yaml.org,2002:int", "tag:yaml.org,2002:float")
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
_Yaml12Loader.add_constructor(
    "tag:yaml.org,2002:bool", lambda loader, node: loader.construct_scalar(node).lower() == "true"
)
_Yaml12Loader.add_constructor(
    "tag:yaml.org,2002:int", lambda loader, node: int(loader.construct_scalar(node), 0)
)


def load_spec_file(path) -> dict:
    """Read a spec file as plain data: ``.json`` as JSON, anything else as YAML 1.2.

    Parameters
    ----------
    path : str or pathlib.Path
        The spec file.

    Returns
    -------
    dict
    """
    text = Path(path).read_text()
    if Path(path).suffix == ".json":
        return json.loads(text)
    return yaml.load(text, Loader=_Yaml12Loader)  # a SafeLoader subclass


# ---------------------------------------------------------------------------
# The document schema
# ---------------------------------------------------------------------------

_STRICT = ConfigDict(extra="forbid", strict=True)


def _int_to_float(v):
    """Accept a whole number where a float is expected (``Q: 0``), nothing else."""
    return float(v) if isinstance(v, int) and not isinstance(v, bool) else v


class _Options(BaseModel):
    model_config = _STRICT
    recycle_passes: int = 3
    recycle_tol: float | None = 1e-8
    recycle_max_passes: int = 100

    _floats = field_validator("recycle_tol", mode="before")(_int_to_float)


class _ModelRef(BaseModel):
    model_config = _STRICT
    source: str  # "shipped", or a model YAML path relative to the spec file
    sha256: str | None = None


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
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    source: str = Field(alias="from")
    to: str
    translator: dict[str, Any] | None = None
    seed: _Seed | None = None


class _Document(BaseModel):
    model_config = _STRICT
    format: Literal[2]
    aquakin: str | None = None
    name: str = "plant"
    options: _Options = _Options()
    models: dict[str, _ModelRef] = {}
    units: dict[str, _Unit] = {}
    connections: list[str | _Connection] = []
    streams: dict[str, str] = {}
    influent_endpoint: str | None = None
    effluent_endpoint: str | None = None
    layout: dict[str, Any] | None = None

    @field_validator("units")
    @classmethod
    def _identifiers(cls, units):
        for uid in units:
            if not _IDENTIFIER.match(uid):
                raise ValueError(
                    f"unit identifier {uid!r} may use letters, digits, '_' and '-' only "
                    f"(it is the name endpoints and parameters are written with)"
                )
        return units


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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _model_fingerprint(text: str, source: str, base_dir) -> str:
    """sha256 of a model's content with its ``extends`` chain merged in."""
    from aquakin.schema.loader import _resolve_inheritance

    data = _resolve_inheritance(yaml.safe_load(text), source, base_dir, seen=())
    canonical = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _shipped_fingerprint(name: str) -> str:
    text = (files("aquakin.models") / f"{name}.yaml").read_text()
    return _model_fingerprint(text, name, None)


def _file_fingerprint(path: Path) -> str:
    return _model_fingerprint(path.read_text(), str(path), path.parent)


def _load_models(doc: _Document, base_dir: Path | None) -> tuple[dict, dict]:
    """The spec's models by name, and the absolute path of each one read from a file."""
    from aquakin.schema.loader import load_model, load_model_from_file

    models, model_files = {}, {}
    for name, ref in doc.models.items():
        where = f"models.{name}"
        try:
            if ref.source == "shipped":
                model = load_model(name)
                digest = _shipped_fingerprint(name)
            else:
                path = Path(ref.source)
                if not path.is_absolute():
                    if base_dir is None:
                        raise SpecError(
                            f"{where}: the relative model file {ref.source!r} needs the spec to "
                            f"be read from a file, or base_dir="
                        )
                    path = base_dir / path
                model = load_model_from_file(path)
                digest = _file_fingerprint(path)
                model_files[name] = path.resolve()
        except SpecError:
            raise
        except Exception as exc:
            raise SpecError(f"{where}: {type(exc).__name__}: {exc}") from None
        if model.name != name:
            raise SpecError(
                f"{where}: the model in {ref.source!r} is named '{model.name}', not '{name}'"
            )
        if ref.sha256 is not None and ref.sha256 != digest:
            warnings.warn(
                f"model '{name}' ({ref.source}) has changed since this spec was written "
                f"(aquakin {doc.aquakin or 'unknown'}); the plant is built with the current model",
                SpecWarning,
                stacklevel=4,
            )
        models[name] = model
    return models, model_files


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


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


def _build_translator(spec: dict, src_model, dst_model, where: str):
    spec = dict(spec)
    kind = spec.pop("kind", None)
    classes = _translators()
    if kind not in classes:
        raise SpecError(f"{where}: translator kind {kind!r} is not one of {sorted(classes)}")
    cls = classes[kind]
    params = inspect.signature(cls).parameters
    for key, value in spec.items():
        if key not in params or key in ("source_model", "target_model"):
            raise SpecError(f"{where}: '{key}' is not a setting of translator {kind}")
        expect_bool = isinstance(params[key].default, bool)
        if isinstance(value, bool) != expect_bool or not isinstance(value, (int, float, str)):
            raise SpecError(f"{where}: translator setting '{key}' has the wrong type ({value!r})")
    try:
        return cls(source_model=src_model, target_model=dst_model, **spec)
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


def plant_from_spec(cls, spec, *, base_dir=None):
    """Build a plant from a spec (a mapping, or a ``.yaml`` / ``.json`` file path).

    Bound as the classmethod :meth:`Plant.from_spec`.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The spec, or the path of a file holding it.
    base_dir : str or pathlib.Path, optional
        The directory model file paths are relative to when ``spec`` is a
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
        If a model's content differs from the fingerprint the spec recorded.
    """
    from aquakin.plant.catalog import build_unit

    if isinstance(spec, (str, Path)):
        base_dir = Path(spec).parent if base_dir is None else Path(base_dir)
        spec = load_spec_file(spec)
    elif base_dir is not None:
        base_dir = Path(base_dir)
    doc = _validate(spec)
    models, model_files = _load_models(doc, base_dir)

    plant = cls(doc.name, **doc.options.model_dump())
    plant._spec_model_files = model_files
    influents = {}
    for uid, entry in doc.units.items():
        where = f"units.{uid}"
        settings = dict(entry.model_extra or {})
        _check_model_names(settings, models, where)
        try:
            built = build_unit(entry.kind, uid, settings, models=models)
        except Exception as exc:
            raise SpecError(f"{where}: {type(exc).__name__}: {exc}") from None
        if entry.kind == "influent":
            influents[uid] = built
        else:
            plant.add_unit(built)
        if entry.label is not None:
            plant.labels[uid] = entry.label

    fed = dict.fromkeys(influents, 0)
    for i, entry in enumerate(doc.connections):
        where = f"connections[{i}]"
        if isinstance(entry, str):
            parts = [p.strip() for p in entry.split("->")]
            if len(parts) != 2 or not all(parts):
                raise SpecError(
                    f"{where}: write a connection as 'unit.port -> unit.port', got {entry!r}"
                )
            entry = _Connection(**{"from": parts[0], "to": parts[1]})
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
    return plant


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class _ModelRefs:
    """Names each model object of a plant and records how it is sourced.

    A model is written as a file when its name is one of the given model files
    or it was read from one (``load_model_from_file`` records the path), and as
    shipped when ``load_model`` loaded it by name. Anything else (a modified
    copy) is refused, since naming it would build a different plant.
    """

    def __init__(self, model_files: dict, base_dir: Path | None):
        self._files = {name: Path(p) for name, p in model_files.items()}
        self._base_dir = base_dir
        self.used: dict[str, dict] = {}

    def __call__(self, model) -> str:
        name = model.name
        if name in self.used:
            return name
        source = getattr(model, "_aquakin_source", None)
        if name in self._files:
            path = self._files[name]
        elif source is not None and source[0] == "file":
            path = Path(source[1])
        else:
            path = None
        if path is not None:
            self.used[name] = {"source": self._relative(path), "sha256": _file_fingerprint(path)}
        elif source == ("shipped", name):
            self.used[name] = {"source": "shipped", "sha256": _shipped_fingerprint(name)}
        else:
            raise SpecError(
                f"model '{name}' is neither a shipped model as aquakin.load_model('{name}') "
                f"returns it nor read from a model file (a modified copy, say); write it as a "
                f"model file and pass Plant.to_spec(models={{'{name}': 'path/to/{name}.yaml'}})"
            )
        return name

    def _relative(self, path: Path) -> str:
        if self._base_dir is not None:
            try:
                return path.resolve().relative_to(self._base_dir.resolve()).as_posix()
            except ValueError:
                pass
        return path.resolve().as_posix()


def _influent_spec(name, series, refs) -> dict:
    import numpy as np

    Q, C = np.asarray(series.Q), np.asarray(series.C)
    T = None if series.T is None else np.asarray(series.T)
    constant = np.all(Q[0] == Q) and np.all(C[0] == C) and (T is None or np.all(T[0] == T))
    if not constant:
        raise SpecError(
            f"influent '{name}' varies in time; a spec holds constant influents only for now"
        )
    species = series.model.species
    return {
        "kind": "influent",
        "model": refs(series.model),
        "Q": float(Q[0]),
        "T": None if T is None else float(T[0]),
        "composition": {sp: float(v) for sp, v in zip(species, C[0], strict=True) if v != 0.0},
    }


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
        Also write the spec to this file (``.json`` as JSON, anything else as
        YAML). Model file paths are then written relative to its directory.
    models : dict, optional
        ``{model name: model YAML path}`` for models that are not shipped, so
        they can be named in the spec (a plant read by :meth:`from_spec`
        remembers its own).

    Returns
    -------
    dict
        The spec (plain data).

    Raises
    ------
    SpecError
        If the plant holds something a spec cannot express yet (the message
        names it).
    """
    from aquakin import __version__
    from aquakin.plant.catalog import unit_settings
    from aquakin.plant.temperature import AlgebraicTemperature
    from aquakin.plant.translators import IdentityTranslator

    if type(plant.temperature_model) is not AlgebraicTemperature:
        raise SpecError(
            f"temperature model {type(plant.temperature_model).__name__} cannot be written "
            f"in a spec yet"
        )
    base_dir = Path(path).parent if path is not None else None
    model_files = {**getattr(plant, "_spec_model_files", {}), **(models or {})}
    refs = _ModelRefs(model_files, base_dir)
    derived = getattr(plant, "_derived_units", set())

    units: dict[str, dict] = {}
    for name, series in plant.influents.items():
        units[name] = _influent_spec(name, series, refs)
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
        "aquakin": __version__,
        "name": plant.name,
        "options": {key: getattr(plant, key) for key in _OPTION_KEYS},
        "models": dict(refs.used),
        "units": units,
        "connections": connections,
        "streams": dict(plant.named_streams),
        "influent_endpoint": plant.influent_endpoint,
        "effluent_endpoint": plant.effluent_endpoint,
    }
    if path is not None:
        path = Path(path)
        if path.suffix == ".json":
            path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
        else:
            path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    return spec


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
        ``{"ok": False, "error": message}`` when it does not.
    """
    from aquakin.plant.plant import Plant

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpecWarning)
            result = Plant.from_spec(spec, base_dir=base_dir).check()
    except Exception as exc:
        message = str(exc) if isinstance(exc, SpecError) else f"{type(exc).__name__}: {exc}"
        return {"ok": False, "error": message}
    return {
        "ok": True,
        "unfed": list(result.unfed_ports),
        "dangling": list(result.dangling_outputs),
        "recycles": list(result.recycles),
    }
