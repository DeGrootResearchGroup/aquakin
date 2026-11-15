"""A plant flowsheet as plain data: :meth:`Plant.from_spec` and :meth:`Plant.to_spec`.

A *spec* is a JSON/YAML-compatible mapping that rebuilds a plant::

    format: 1
    name: BSM1
    units:
      tank1: {kind: cstr, model: asm1, volume: 1000.0, conditions: {T: 288.15}}
      ...
    connections:
      - inlet_mix.out -> tank1.in
      - {from: sludge_mix.out, to: digester.inlet, translator: {kind: asm1_to_adm1}}
      - {from: reject_mix.out, to: front_mix.reject, seed: {Q: 0.0, T: 293.15}}
    streams: {effluent: clarifier.overflow}
    influent_endpoint: inlet_mix.fresh
    effluent_endpoint: clarifier.overflow

Units are written with the kinds and settings of :mod:`aquakin.plant.catalog`
(settings at their default are left out). A unit of kind ``influent`` is a
constant feed: its single connection becomes :meth:`Plant.add_influent`. A
connection is a ``"unit.port -> unit.port"`` string, or a mapping when it needs
a state translator between two models or a recycle seed. ``options`` holds the
plant's recycle settings when they differ from the defaults, and ``layout`` (a
flowsheet editor's drawing positions) is kept by editors and ignored here.

Units the plant derives itself (the PI controllers a closed-loop
:class:`Aeration` or a feedback :class:`DosingUnit` requires) are not written:
they follow from those units' settings and are derived again when the rebuilt
plant is solved. What a spec cannot express yet is refused by
:meth:`Plant.to_spec` with a :class:`SpecError` naming it, rather than dropped: a
time-varying influent, a model loaded from a file, a non-default temperature
model, and settings only Python can give (arrays of initial concentrations, SBR
phases, settling models).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "SPEC_FORMAT",
    "TRANSLATORS",
    "SpecError",
    "check_spec",
    "plant_from_spec",
    "plant_to_spec",
]

#: The spec format version this module writes and reads.
SPEC_FORMAT = 1

_TOP_KEYS = {
    "format",
    "name",
    "options",
    "units",
    "connections",
    "streams",
    "influent_endpoint",
    "effluent_endpoint",
    "layout",
}
_OPTIONS = {"recycle_passes": 3, "recycle_tol": 1e-8, "recycle_max_passes": 100}


class SpecError(ValueError):
    """A spec that cannot be read, or a plant that cannot be written as one."""


def _translators() -> dict:
    from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1

    return {"asm1_to_adm1": ASM1toADM1, "adm1_to_asm1": ADM1toASM1}


#: Translator kinds a spec can name, read lazily: ``{kind: class}``.
TRANSLATORS = ("asm1_to_adm1", "adm1_to_asm1")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _load(spec) -> dict:
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        text = path.read_text()
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    else:
        data = spec
    if not isinstance(data, dict):
        raise SpecError("a spec must be a mapping")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise SpecError(f"unknown top-level key(s) {sorted(unknown)}; known: {sorted(_TOP_KEYS)}")
    fmt = data.get("format", SPEC_FORMAT)
    if fmt != SPEC_FORMAT:
        raise SpecError(f"spec format {fmt!r} is not supported (this aquakin reads {SPEC_FORMAT})")
    return data


def _connection(entry, i: int) -> dict:
    where = f"connections[{i}]"
    if isinstance(entry, str):
        parts = [p.strip() for p in entry.split("->")]
        if len(parts) != 2 or not all(parts):
            raise SpecError(
                f"{where}: write a connection as 'unit.port -> unit.port', got {entry!r}"
            )
        return {"from": parts[0], "to": parts[1], "where": where}
    if isinstance(entry, dict) and "from" in entry and "to" in entry:
        extra = set(entry) - {"from", "to", "translator", "seed"}
        if extra:
            raise SpecError(f"{where}: unknown key(s) {sorted(extra)}")
        return {**entry, "where": where}
    raise SpecError(f"{where}: a connection is a string or a mapping with 'from' and 'to'")


def _unit_name(endpoint: str) -> str:
    return endpoint.split(".", 1)[0]


def _build_translator(spec, src_model, dst_model, where):
    spec = dict(spec)
    kind = spec.pop("kind", None)
    classes = _translators()
    if kind not in classes:
        raise SpecError(f"{where}: translator kind {kind!r} is not one of {sorted(classes)}")
    try:
        return classes[kind](source_model=src_model, target_model=dst_model, **spec)
    except TypeError as exc:
        raise SpecError(f"{where}: translator: {exc}") from None


def _build_seed(spec, model, where):
    import jax.numpy as jnp

    from aquakin.plant.streams import Stream

    spec = dict(spec)
    extra = set(spec) - {"Q", "T", "concentrations"}
    if extra:
        raise SpecError(f"{where}: seed: unknown key(s) {sorted(extra)}")
    conc = spec.get("concentrations", "default")
    if conc == "default":
        C = model.default_concentrations()
    else:
        C = model.concentrations(conc, base="zero")
    scalars = {"T": jnp.asarray(float(spec["T"]))} if spec.get("T") is not None else {}
    return Stream(Q=jnp.asarray(float(spec.get("Q", 0.0))), C=C, model=model, scalars=scalars)


def plant_from_spec(cls, spec):
    """Build a plant from a spec (a mapping, or a ``.yaml`` / ``.json`` file path).

    Bound as the classmethod :meth:`Plant.from_spec`.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The spec, or the path of a file holding it.

    Returns
    -------
    Plant

    Raises
    ------
    SpecError
        If the spec is malformed, names an unknown kind, or a unit cannot be
        built from its settings; the message says where (``units.tank1: ...``).
    """
    from aquakin.plant.catalog import build_unit

    data = _load(spec)
    options = {**_OPTIONS, **data.get("options", {})}
    unknown = set(options) - set(_OPTIONS)
    if unknown:
        raise SpecError(f"options: unknown key(s) {sorted(unknown)}; known: {sorted(_OPTIONS)}")
    if options["recycle_tol"] is not None:
        options["recycle_tol"] = float(options["recycle_tol"])
    plant = cls(data.get("name", "plant"), **options)

    influents = {}
    for name, entry in (data.get("units") or {}).items():
        if not isinstance(entry, dict) or "kind" not in entry:
            raise SpecError(f"units.{name}: a unit is a mapping with a 'kind'")
        settings = {k: v for k, v in entry.items() if k != "kind"}
        try:
            built = build_unit(entry["kind"], name, settings)
        except Exception as exc:
            raise SpecError(f"units.{name}: {type(exc).__name__}: {exc}") from None
        if entry["kind"] == "influent":
            influents[name] = built
        else:
            plant.add_unit(built)

    fed = dict.fromkeys(influents, 0)
    for i, entry in enumerate(data.get("connections") or []):
        c = _connection(entry, i)
        src, dst, where = c["from"], c["to"], c["where"]
        src_unit = _unit_name(src)
        try:
            if src_unit in influents:
                fed[src_unit] += 1
                translator = None
                if c.get("translator"):
                    dst_model = plant.units[_unit_name(dst)].model
                    translator = _build_translator(
                        c["translator"], influents[src_unit].model, dst_model, where
                    )
                plant.add_influent(src_unit, influents[src_unit], to=dst, translator=translator)
                continue
            kwargs = {}
            if c.get("translator"):
                kwargs["translator"] = _build_translator(
                    c["translator"],
                    plant.units[src_unit].model,
                    plant.units[_unit_name(dst)].model,
                    where,
                )
            if c.get("seed"):
                # A seed stands for the stream arriving at the destination, so it
                # is written in the destination's model (after any translator).
                dst_model = plant.units[_unit_name(dst)].model
                kwargs["initial_value"] = _build_seed(c["seed"], dst_model, where)
            plant.connect(src, dst, **kwargs)
        except SpecError:
            raise
        except Exception as exc:
            raise SpecError(f"{where} ({src} -> {dst}): {type(exc).__name__}: {exc}") from None
    for name, n in fed.items():
        if n != 1:
            raise SpecError(f"influent '{name}' must feed exactly one unit input; it feeds {n}")

    for name, endpoint in (data.get("streams") or {}).items():
        try:
            plant.register_stream(name, endpoint)
        except Exception as exc:
            raise SpecError(f"streams.{name}: {type(exc).__name__}: {exc}") from None
    plant.influent_endpoint = data.get("influent_endpoint")
    plant.effluent_endpoint = data.get("effluent_endpoint")
    return plant


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _influent_spec(name, series) -> dict:
    import numpy as np

    from aquakin.plant.catalog import _model_name

    Q, C = np.asarray(series.Q), np.asarray(series.C)
    T = None if series.T is None else np.asarray(series.T)
    constant = np.all(Q[0] == Q) and np.all(C[0] == C) and (T is None or np.all(T[0] == T))
    if not constant:
        raise SpecError(
            f"influent '{name}' varies in time; a spec holds constant influents only for now"
        )
    out = {"kind": "influent", "model": _model_name(series.model), "Q": float(Q[0])}
    if T is not None:
        out["T"] = float(T[0])
    out["composition"] = {
        sp: float(v) for sp, v in zip(series.model.species, C[0], strict=True) if v != 0.0
    }
    return out


def _translator_spec(t, where) -> dict:
    for kind, cls in _translators().items():
        if type(t) is cls:
            break
    else:
        raise SpecError(f"{where}: translator {type(t).__name__} cannot be written in a spec")
    from aquakin.plant.catalog import _plain

    out = {"kind": kind}
    params = inspect.signature(cls).parameters
    for f in dataclasses.fields(t):
        if f.name in ("source_model", "target_model") or f.name not in params:
            continue
        value = _plain(getattr(t, f.name))
        default = params[f.name].default
        if default is not inspect.Parameter.empty and value == _plain(default):
            continue
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
    out: dict[str, Any] = {"Q": float(stream.Q)}
    if "T" in stream.scalars:
        out["T"] = float(stream.scalars["T"])
    C = np.asarray(stream.C)
    if not np.array_equal(C, np.asarray(model.default_concentrations())):
        out["concentrations"] = {
            sp: float(v) for sp, v in zip(model.species, C, strict=True) if v != 0.0
        }
    return out


def plant_to_spec(plant, path=None) -> dict:
    """The plant as a spec that :meth:`Plant.from_spec` rebuilds.

    Bound as :meth:`Plant.to_spec`.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Also write the spec to this file (``.json`` as JSON, anything else as YAML).

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
    from aquakin.plant.catalog import unit_settings
    from aquakin.plant.temperature import AlgebraicTemperature
    from aquakin.plant.translators import IdentityTranslator

    if type(plant.temperature_model) is not AlgebraicTemperature:
        raise SpecError(
            f"temperature model {type(plant.temperature_model).__name__} cannot be written "
            f"in a spec yet"
        )
    derived = getattr(plant, "_derived_units", set())
    spec: dict[str, Any] = {"format": SPEC_FORMAT, "name": plant.name}
    options = {}
    for key, default in _OPTIONS.items():
        value = getattr(plant, key)
        same = (
            value == default
            if value is None or default is None
            else math.isclose(float(value), float(default))
        )
        if not same:
            options[key] = value
    if options:
        spec["options"] = options

    units: dict[str, dict] = {}
    for name, series in plant.influents.items():
        units[name] = _influent_spec(name, series)
    for name in plant._insertion_order:
        if name in derived:
            continue
        try:
            kind, settings = unit_settings(plant.units[name])
        except ValueError as exc:
            raise SpecError(str(exc)) from None
        units[name] = {"kind": kind, **settings}
    spec["units"] = units

    connections = []
    for i, c in enumerate(plant.connections):
        if c.to_unit in derived or c.from_unit in derived:
            continue
        where = f"connection {i}"
        if c.from_unit is None:
            src = f"{c.from_port}.out"
        else:
            src = f"{c.from_unit}.{c.from_port}"
        dst = f"{c.to_unit}.{c.to_port}"
        extra = {}
        if not isinstance(c.translator, IdentityTranslator):
            extra["translator"] = _translator_spec(c.translator, where)
        if c.initial_value is not None:
            extra["seed"] = _seed_spec(c.initial_value, plant.units[c.to_unit].model, where)
        connections.append({"from": src, "to": dst, **extra} if extra else f"{src} -> {dst}")
    spec["connections"] = connections
    if plant.named_streams:
        spec["streams"] = dict(plant.named_streams)
    for key in ("influent_endpoint", "effluent_endpoint"):
        if getattr(plant, key):
            spec[key] = getattr(plant, key)

    if path is not None:
        path = Path(path)
        if path.suffix == ".json":
            path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
        else:
            path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    return spec


def check_spec(spec) -> dict:
    """Build a plant from ``spec`` and report :meth:`Plant.check` on it.

    Returns
    -------
    dict
        ``{"ok": True, "unfed": [...], "dangling": [...], "recycles": [...]}``
        when the plant builds (``unfed`` input ports are the errors), or
        ``{"ok": False, "error": message}`` when it does not.
    """
    from aquakin.plant.plant import Plant

    try:
        result = Plant.from_spec(spec).check()
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc) if isinstance(exc, SpecError) else f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": True,
        "unfed": list(result.unfed_ports),
        "dangling": list(result.dangling_outputs),
        "recycles": list(result.recycles),
    }
