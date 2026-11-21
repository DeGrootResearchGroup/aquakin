"""Time-varying influent streams.

:class:`InfluentSeries` reads a CSV of time-series influent data (one
row per timestep, columns for flow rate and each species concentration)
and exposes an AD-clean piecewise-linear interpolant ``.at(t)`` that
the plant's RHS calls to get the inlet stream at the current integration
time.

:func:`load_bsm1_influent` reads the **synthesised** BSM1 dry / rain / storm
influent files shipped under ``aquakin/plant/bsm/data/`` (they match the BSM1
statistical profile but are not the canonical IWA series).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.plant.streams import Stream, make_scalars
from aquakin.utils.units import CELSIUS_SPELLINGS, conversion_factor, parse_units

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


# Roles a ``column_map`` may target besides ``"t"`` / ``"Q"`` / ``"T"`` and
# direct ASM-species names: aggregate lab/SCADA measurements, fractionated into
# ASM1 states per row (see aquakin.plant.characterize.fractionate).
_AGGREGATE_ROLES = (
    "total_cod",
    "tkn",
    "ammonia",
    "nox",
    "alkalinity",
    "filtered_cod",
    "flocculated_filtered_cod",
    "soluble_inert_cod",
)


# BSM1 influent CSVs use this exact column order. ``t`` is days since
# the start of the simulation; ``Q`` is in m³/d; all concentrations are
# in g_COD/m³ or g_N/m³ per ASM1 conventions.
_BSM1_COLUMN_ORDER = [
    "t",
    "SI",
    "SS",
    "XI",
    "XS",
    "XB_H",
    "XB_A",
    "XP",
    "SO",
    "SNO",
    "SNH",
    "SND",
    "XND",
    "SALK",
    "Q",
]

# BSM2 files add a time-varying influent temperature ``T`` (degC) as the last
# column; load_bsm2_influent converts it to Kelvin.
_BSM2_COLUMN_ORDER = _BSM1_COLUMN_ORDER + ["T"]

# Units of the shipped BSM influent files (days, m3/d, ASM1 concentrations).
_BSM1_INFLUENT_UNITS = {
    "t": "d",
    "Q": "m3/d",
    **dict.fromkeys(("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP"), "g_COD/m3"),
    "SO": "g_O2/m3",
    **dict.fromkeys(("SNO", "SNH", "SND", "XND"), "g_N/m3"),
    "SALK": "mol/m3",
}
_BSM2_INFLUENT_UNITS = {**_BSM1_INFLUENT_UNITS, "T": "K"}


def _declared_for(model: CompiledModel, units: dict) -> dict:
    """The entries of a file's ``units`` that apply to ``model``: ``t``, ``Q``,
    ``T`` and the species the model declares (a model may use a subset of the
    file's columns)."""
    keep = {"t", "Q", "T", *model.species}
    return {k: v for k, v in units.items() if k in keep}


@dataclass
class InfluentSeries:
    """A time-series of influent data.

    Attributes
    ----------
    t : jnp.ndarray
        Sample times, shape ``(n_t,)``, strictly ascending.
    Q : jnp.ndarray
        Volumetric flow rate at each sample, shape ``(n_t,)``.
    C : jnp.ndarray
        Concentration at each sample, shape ``(n_t, n_species)`` where
        columns follow ``model.species`` ordering.
    model : CompiledModel
    T : jnp.ndarray, optional
        Influent temperature at each sample (Kelvin), shape ``(n_t,)``. When
        given, ``at(t)`` returns a stream carrying the interpolated temperature,
        which the plant propagates to temperature-dependent kinetics. ``None``
        (default) leaves the influent temperature-agnostic.
    units : dict[str, str], optional
        The units the data are given in, keyed by ``"t"``, ``"Q"``, ``"T"`` or a
        species name, e.g. ``{"t": "h", "Q": "m3/h", "T": "degC", "SNH":
        "mg/L"}``. The data are converted on construction to the units the plant
        works in: ``t`` to the model's time unit, ``Q`` to m3 per that unit,
        ``T`` to Kelvin (``degC`` is offset by 273.15), and each species to the
        model's declared unit. Only the scale changes; a unit measuring another
        quantity (moles where the model has grams, or another chemical currency)
        raises. Afterwards ``units`` holds the converted units (an entry whose
        model unit is blank or cannot be read keeps its declared unit, and its
        data are left as given). Any subset may be given; undeclared data are
        taken to be in the model's units already.
        ``None`` (default) declares nothing.
    """

    t: jnp.ndarray
    Q: jnp.ndarray
    C: jnp.ndarray
    model: CompiledModel
    T: jnp.ndarray | None = None
    units: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.t.ndim != 1:
            raise ValueError(f"t must be 1-D, got shape {self.t.shape}")
        if self.Q.shape != self.t.shape:
            raise ValueError(f"Q shape {self.Q.shape} does not match t shape {self.t.shape}")
        if self.C.ndim != 2 or self.C.shape[0] != self.t.shape[0]:
            raise ValueError(f"C shape {self.C.shape} expected ({self.t.shape[0]}, n_species)")
        if self.C.shape[1] != self.model.n_species:
            raise ValueError(
                f"C has {self.C.shape[1]} species columns but model has {self.model.n_species}"
            )
        if self.T is not None and self.T.shape != self.t.shape:
            raise ValueError(f"T shape {self.T.shape} does not match t shape {self.t.shape}")
        if self.units is not None:
            allowed = {"t", "Q", "T", *self.model.species}
            unknown = sorted(k for k in self.units if k not in allowed)
            if unknown:
                raise ValueError(
                    f"units declares {unknown}, which are not 't', 'Q', 'T' or a "
                    f"species of model '{self.model.name}'."
                )
            bad = sorted(k for k, v in self.units.items() if not isinstance(v, str))
            if bad:
                raise ValueError(f"units values must be strings; got non-strings for {bad}.")
            # A declared unit must be readable, or it could not be converted. A
            # Celsius temperature is recognized by name (``C`` alone parses as
            # the carbon currency).
            unreadable = sorted(
                k
                for k, v in self.units.items()
                if parse_units(v) is None
                and not (k == "T" and v.strip().lower() in CELSIUS_SPELLINGS)
            )
            if unreadable:
                raise ValueError(
                    f"units declares {unreadable} in units that cannot be read: "
                    + ", ".join(f"{k}={self.units[k]!r}" for k in unreadable)
                    + ". Use e.g. 'd', 'm3/d', 'K', 'g_N/m3' or 'mg/L'."
                )
            self._convert_to_model_units()

    def _convert_to_model_units(self) -> None:
        """Convert the declared data to the units the plant works in.

        ``t`` goes to the model's time unit, ``Q`` to m3 per that unit, ``T`` to
        Kelvin, and each species to the unit the model declares for it. Only
        the scale changes (a Celsius ``T`` is also offset): a declared unit that
        measures a different quantity -- a mass where the model has moles, or
        g NH4 where it has g N -- raises, since converting it would need a molar
        mass. Where the model's own unit is unknown (a blank or unreadable
        species unit, or no inferable time unit) there is nothing to convert to,
        and that entry is kept as declared. ``units`` then records the units the
        data are in. A factor of exactly 1 leaves the data untouched.
        """
        units = dict(self.units)
        time_unit = self.model.time_unit
        if time_unit is not None:
            for key, target in (("t", time_unit), ("Q", f"m3/{time_unit}")):
                factor = self._factor(units, key, target)
                if factor is not None and factor != 1.0:
                    setattr(self, key, getattr(self, key) * factor)
        if "T" in units:
            if units["T"].strip().lower() in CELSIUS_SPELLINGS:
                if self.T is not None:
                    self.T = self.T + 273.15
                units["T"] = "K"
            else:
                # Kelvin is the only other temperature unit that parses, so this
                # validates and records the unit; there is no scale to apply.
                self._factor(units, "T", "K")
        scale = np.ones(self.model.n_species)
        for species in self.model.species:
            factor = self._factor(units, species, self.model.species_units.get(species, ""))
            if factor is not None:
                scale[self.model.species_index[species]] = factor
        if np.any(scale != 1.0):
            self.C = self.C * jnp.asarray(scale)[None, :]
        self.units = units

    def _factor(self, units: dict, key: str, target: str) -> float | None:
        """Factor from the declared unit of ``key`` to ``target``, recording
        ``target`` in ``units``; ``None`` when nothing is declared or ``target``
        is unknown."""
        unit = units.get(key)
        if unit is None:
            return None
        try:
            factor = conversion_factor(unit, target)
        except ValueError as exc:
            raise ValueError(
                f"units declares '{key}' in '{unit}', which cannot be converted to "
                f"'{target}' as model '{self.model.name}' requires: {exc}"
            ) from None
        if factor is not None:
            units[key] = target
        return factor

    @classmethod
    def constant(
        cls, model, overrides=None, /, *, Q, base: str = "zero", T=None, units=None, **species
    ) -> InfluentSeries:
        """Build a constant-in-time influent from a feed composition.

        The composition is built with ``model.concentrations(overrides,
        base=base, **species)`` -- ``base="zero"`` by default, so an unspecified
        species is absent from the feed rather than at its YAML reference value.
        The series carries two identical samples, so ``at(t)`` returns the same
        constant stream at every time.

        Parameters
        ----------
        model : CompiledModel
            Kinetic model whose species ordering ``C`` follows.
        overrides : dict[str, float], optional
            Species name -> feed concentration. Positional-only.
        Q : float
            Constant volumetric flow rate.
        base : {"zero", "defaults"}, optional
            Composition base; defaults to ``"zero"``.
        T : float, optional
            Constant feed temperature (Kelvin); ``None`` leaves it agnostic.
        units : dict[str, str], optional
            The units ``Q``, ``T`` and the concentrations are given in, converted
            to the model's units on construction (see :class:`InfluentSeries`).
        **species : float
            Convenience overrides for identifier-safe species names.

        Returns
        -------
        InfluentSeries

        Examples
        --------
        >>> InfluentSeries.constant(net, {"SS": 60.0, "SNH": 25.0}, Q=18446.0)
        >>> InfluentSeries.constant(net, SS=400.0, Q=2.0)
        """
        C = model.concentrations(overrides, base=base, **species)
        # Two identical samples -> a genuine constant: jnp.interp clamps outside
        # the range and interpolates a flat line within it, so any solve horizon
        # sees the same value.
        t = jnp.asarray([0.0, 1.0e9])
        Q_arr = jnp.full((2,), float(Q))
        C_arr = jnp.tile(C, (2, 1))
        T_arr = None if T is None else jnp.full((2,), float(T))
        return cls(t=t, Q=Q_arr, C=C_arr, model=model, T=T_arr, units=units)

    def in_time_unit(self, time_unit: str | None) -> InfluentSeries:
        """This series with ``t`` and ``Q`` expressed in ``time_unit``.

        A plant whose time unit differs from the model's uses this to put the
        sample times and flows on its own clock. The series' current time unit
        is the declared ``units["t"]`` (``units["Q"]`` for the flow), else the
        model's. Returns ``self`` unchanged when ``time_unit`` is ``None``, when
        the current unit is unknown, or when nothing needs converting.

        Parameters
        ----------
        time_unit : str or None
            The time unit to express the series in, e.g. ``"h"``.

        Returns
        -------
        InfluentSeries
            A new series (same model, concentrations and temperature), or
            ``self``.
        """
        if time_unit is None:
            return self
        units = dict(self.units or {})
        native = self.model.time_unit
        t_unit = units.get("t", native)
        q_unit = units.get("Q", None if native is None else f"m3/{native}")
        t_factor = None if t_unit is None else conversion_factor(t_unit, time_unit)
        q_factor = None if q_unit is None else conversion_factor(q_unit, f"m3/{time_unit}")
        if t_factor in (None, 1.0) and q_factor in (None, 1.0):
            return self
        t, Q = self.t, self.Q
        if t_factor is not None:
            t = t * t_factor
            units["t"] = time_unit
        if q_factor is not None:
            Q = Q * q_factor
            units["Q"] = f"m3/{time_unit}"
        # Built without ``units`` so construction does not convert back to the
        # model's time unit; the record of what the data are in is set after.
        series = dataclasses.replace(self, t=t, Q=Q, units=None)
        series.units = units
        return series

    def at(self, t: jnp.ndarray) -> Stream:
        """Return the influent :class:`Stream` at time ``t``.

        Linearly interpolates between samples. Outside the range, clamps
        to the endpoint values (``jnp.interp`` semantics).
        """
        Q_t = jnp.interp(t, self.t, self.Q)
        # Interpolate every species column in one vmapped op (over the species
        # axis) rather than a Python loop of n_species separate interp calls.
        C_t = jax.vmap(lambda col: jnp.interp(t, self.t, col), in_axes=1)(self.C)
        T_t = None if self.T is None else jnp.interp(t, self.t, self.T)
        return Stream(Q=Q_t, C=C_t, model=self.model, scalars=make_scalars(T=T_t))


def read_influent_csv(
    path: str | Path,
    model: CompiledModel,
    *,
    column_order: list[str] | None = None,
    column_map: dict | None = None,
    fractions=None,
    delimiter: str | None = None,
    units: dict[str, str] | None = None,
) -> InfluentSeries:
    """Read a CSV influent file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    model : CompiledModel
        Kinetic model whose species ordering ``C`` is built against.
    column_order : list[str], optional
        Positional column layout, used when ``column_map`` is not given. The
        first column is time ``t``, the column named ``"Q"`` is the flow, and
        every other name must be a species the model declares. Defaults to the
        standard BSM1 ordering.
    column_map : dict, optional
        Map of role -> CSV header name, for an arbitrary-header file (a lab /
        SCADA export) -- no renaming the file. Roles are ``"t"``, ``"Q"``,
        optional ``"T"``, any ASM species name (mapped directly), and the
        aggregate measurements ``total_cod`` / ``tkn`` / ``ammonia`` / ``nox`` /
        ``alkalinity`` / ``filtered_cod`` / ``flocculated_filtered_cod`` /
        ``soluble_inert_cod``. When aggregates are mapped, each row is
        fractionated into ASM1 states (see
        :func:`aquakin.plant.characterize.fractionate`); a directly-mapped
        species overrides its fractionated value, and any species neither mapped
        nor produced is zero. Requires a header row in the file.
    fractions : InfluentFractions, optional
        Fractionation parameters for the aggregate columns (defaults to the SUMO
        Sumo1 raw-influent values).
    delimiter : str, optional
        Field delimiter. Defaults to auto-sniffing (whitespace and comma both
        work).
    units : dict[str, str], optional
        The units the file's data are in, keyed by ``"t"``, ``"Q"``, ``"T"`` or a
        species name. The data are converted to the model's units on
        construction (see :class:`InfluentSeries`).

    Returns
    -------
    InfluentSeries
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Influent file not found: {p}")
    series = _influent_from_text(
        p.read_text(encoding="utf-8"),
        model,
        column_order=column_order,
        column_map=column_map,
        fractions=fractions,
        delimiter=delimiter,
        source=str(p),
    )
    return series if units is None else dataclasses.replace(series, units=dict(units))


def write_influent_table(series: InfluentSeries, path: str | Path) -> Path:
    """Write an influent as a table :func:`read_influent_table` reads back exactly.

    The table is comma-separated with a header row: ``t``, ``Q``, ``T`` (when the
    series carries a temperature), then every species of the series' model.
    Numbers are written with full precision, so reading the file back gives the
    same series bit for bit.

    Parameters
    ----------
    series : InfluentSeries
        The influent.
    path : str | Path
        The file to write (its directory is created if needed).

    Returns
    -------
    Path
        The file written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns = ["t", "Q"] + (["T"] if series.T is not None else []) + list(series.model.species)
    data = [np.asarray(series.t), np.asarray(series.Q)]
    if series.T is not None:
        data.append(np.asarray(series.T))
    data = np.column_stack(data + [np.asarray(series.C)])
    lines = [",".join(columns)] + [",".join(repr(float(v)) for v in row) for row in data]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def read_influent_table(path: str | Path, model: CompiledModel) -> InfluentSeries:
    """Read an influent table written by :func:`write_influent_table` (or by hand).

    The file has a header row naming its columns: ``t`` (time, in the model's
    time unit), ``Q`` (flow), optionally ``T`` (temperature, K), and any of the
    model's species; species not given are zero. The series records the file it
    was read from, so a flowsheet spec can refer to it.

    Parameters
    ----------
    path : str | Path
        The table.
    model : CompiledModel
        The model whose species the columns name.

    Returns
    -------
    InfluentSeries

    Raises
    ------
    ValueError
        If ``t`` or ``Q`` is missing, or a column is neither ``T`` nor a species.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Influent table not found: {p}")
    header, data = _parse_named_table(p.read_text(encoding="utf-8"), None)
    for required in ("t", "Q"):
        if required not in header:
            raise ValueError(f"influent table {p} has no '{required}' column")
    pos = {name: i for i, name in enumerate(header)}
    unknown = [h for h in header if h not in ("t", "Q", "T") and h not in model.species_index]
    if unknown:
        raise ValueError(
            f"influent table {p}: column(s) {unknown} are not species of model '{model.name}'"
        )
    C = np.zeros((data.shape[0], model.n_species))
    for name, i in pos.items():
        if name in model.species_index:
            C[:, model.species_index[name]] = data[:, i]
    series = InfluentSeries(
        t=jnp.asarray(data[:, pos["t"]]),
        Q=jnp.asarray(data[:, pos["Q"]]),
        C=jnp.asarray(C),
        model=model,
        T=jnp.asarray(data[:, pos["T"]]) if "T" in pos else None,
    )
    # The file it was read from and the sha256 of the bytes it was built from.
    import hashlib

    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    series._aquakin_source = ("file", str(p.resolve()), digest)
    return series


def _influent_from_text(
    text: str,
    model: CompiledModel,
    *,
    column_order: list[str] | None = None,
    column_map: dict | None = None,
    fractions=None,
    delimiter: str | None = None,
    source: str = "<text>",
) -> InfluentSeries:
    """Parse influent CSV / whitespace text into an :class:`InfluentSeries`.

    Shared by :func:`read_influent_csv` (file path) and
    :func:`load_bsm1_influent` (in-package data), so neither round-trips the
    data through a temporary file. ``source`` only labels error messages.
    """
    if column_map is not None:
        return _influent_from_column_map(text, model, column_map, fractions, delimiter, source)
    if column_order is None:
        column_order = _BSM1_COLUMN_ORDER

    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip a leading column-name header (only checked before any data row).
        if not rows and _looks_like_header(line):
            continue
        tokens = _tokenize(line, delimiter, expected_n=len(column_order))
        if len(tokens) != len(column_order):
            raise ValueError(
                f"Influent row '{raw_line}' has {len(tokens)} fields but "
                f"column_order specifies {len(column_order)}."
            )
        try:
            rows.append([float(tok) for tok in tokens])
        except ValueError as exc:
            raise ValueError(f"Influent row '{raw_line}' has non-numeric field: {exc}") from exc

    if not rows:
        raise ValueError(f"Influent source {source} contained no data rows.")

    data = jnp.asarray(rows)  # shape (n_t, n_cols)

    # Find indices for t and Q within column_order.
    t_idx = column_order.index("t")
    if "Q" not in column_order:
        raise ValueError("column_order must contain 'Q'")
    Q_idx = column_order.index("Q")

    # Build the (n_t, n_species) C matrix, gathering each declared species
    # from its column.
    species_idx_in_file: list[int] = []
    for sp in model.species:
        if sp not in column_order:
            raise ValueError(
                f"Influent file is missing species column '{sp}' "
                f"(declared by model '{model.name}')."
            )
        species_idx_in_file.append(column_order.index(sp))
    species_idx_arr = jnp.asarray(species_idx_in_file)

    t = data[:, t_idx]
    Q = data[:, Q_idx]
    C = data[:, species_idx_arr]
    # An optional 'T' column carries the influent temperature, in the file's own
    # units (the caller converts if needed -- e.g. load_bsm2_influent's degC).
    T = data[:, column_order.index("T")] if "T" in column_order else None

    return InfluentSeries(t=t, Q=Q, C=C, model=model, T=T)


def _tokenize(line: str, delimiter: str | None, expected_n: int | None = None) -> list[str]:
    """Split one line into trimmed, non-empty tokens.

    With an explicit ``delimiter``, split on it. Otherwise prefer comma-splitting
    when it yields ``expected_n`` fields (or, when no count is required, whenever
    the line contains a comma), falling back to whitespace. This is the single
    tokenizer for both the positional and headered influent parsers.
    """
    if delimiter is not None:
        return [tok.strip() for tok in line.split(delimiter) if tok.strip()]
    comma = [tok.strip() for tok in line.split(",") if tok.strip()]
    if expected_n is not None:
        return comma if len(comma) == expected_n else line.split()
    return comma if "," in line else line.split()


def _looks_like_header(line: str) -> bool:
    """True if a line is a column-name header rather than a numeric data row: it
    contains a letter and at least one token that does not parse as a number."""
    if not any(c.isalpha() for c in line):
        return False
    tokens = line.replace(",", " ").split()
    try:
        [float(tok) for tok in tokens]
    except ValueError:
        return True  # a non-numeric token -> header
    return False  # all numeric (e.g. scientific notation) -> data


def _parse_named_table(text: str, delimiter: str | None):
    """Parse a headered table: the first non-comment line is the column-name
    header, the rest are numeric rows. Returns ``(header, data (n_t, n_cols))``."""
    header = None
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = _tokenize(line, delimiter)
        if header is None:
            header = toks
            continue
        try:
            rows.append([float(tok) for tok in toks])
        except ValueError as exc:
            raise ValueError(f"Influent row '{raw}' has a non-numeric field: {exc}") from exc
    if header is None or not rows:
        raise ValueError("column_map requires a header row and at least one data row.")
    return header, np.asarray(rows)


def _influent_from_column_map(
    text: str,
    model,
    column_map: dict,
    fractions,
    delimiter,
    source: str,
) -> InfluentSeries:
    """Build an :class:`InfluentSeries` from an arbitrary-header CSV via a
    role -> header ``column_map``, fractionating any mapped aggregate columns
    into ASM1 states per row. See :func:`read_influent_csv`."""
    from aquakin.plant.characterize import fractionate

    header, data = _parse_named_table(text, delimiter)
    pos = {name: i for i, name in enumerate(header)}

    def col(role: str):
        name = column_map[role]
        if name not in pos:
            raise ValueError(
                f"column_map role '{role}' -> column '{name}' is not in the "
                f"file header {header} (source {source})."
            )
        return data[:, pos[name]]

    for required in ("t", "Q"):
        if required not in column_map:
            raise ValueError(f"column_map must map the '{required}' role.")
    t = col("t")
    Q = col("Q")
    T = col("T") if "T" in column_map else None

    # Aggregate measurement columns -> per-row fractionation into ASM1 states.
    aggregates = {r: col(r) for r in _AGGREGATE_ROLES if r in column_map}
    produced: dict = {}
    if aggregates:
        for need in ("total_cod", "tkn"):
            if need not in aggregates:
                raise ValueError(
                    f"column_map maps aggregate measurements but not '{need}', "
                    f"which the influent fractionation requires."
                )
        kw = dict(aggregates)
        if fractions is not None:
            kw["fractions"] = fractions
        produced = fractionate(**kw)

    n_t = data.shape[0]
    C = np.zeros((n_t, model.n_species))
    for sp in model.species:
        if sp in column_map:  # a directly-mapped species
            C[:, model.species_index[sp]] = col(sp)
        elif sp in produced:  # a fractionated state
            C[:, model.species_index[sp]] = np.asarray(produced[sp])
        # otherwise left at zero (zero-based influent)

    return InfluentSeries(
        t=jnp.asarray(t),
        Q=jnp.asarray(Q),
        C=jnp.asarray(C),
        model=model,
        T=None if T is None else jnp.asarray(T),
    )


def load_bsm1_influent(profile: str, model: CompiledModel) -> InfluentSeries:
    """Load one of the synthesised BSM1 influent files.

    Parameters
    ----------
    profile : {"dry", "rain", "storm"}
        Which BSM1 influent to load.
    model : CompiledModel
        ASM1 model (the influent species map to ASM1 state ordering).

    Returns
    -------
    InfluentSeries

    Notes
    -----
    These files are **synthesised** (``scripts/generate_bsm1_influent.py``):
    they match the BSM1 statistical dry / rain / storm profile (Copp 2002;
    Alex et al. 2008) but are **not** the canonical IWA series, so headline
    EQI / OCI numbers from them are not reproducible against groups using the
    official files -- replace ``BSM1_<profile>.csv`` under
    ``aquakin/plant/bsm/data/`` with the official file for published
    comparisons. Times are in days, flow in m³/d, concentrations in g_COD/m³
    or g_N/m³ following ASM1 conventions.
    """
    if profile not in ("dry", "rain", "storm"):
        raise ValueError(f"profile must be 'dry', 'rain', or 'storm'; got {profile!r}")
    resource = files("aquakin.plant.bsm.data") / f"BSM1_{profile}.csv"
    if not resource.is_file():
        raise FileNotFoundError(
            f"BSM1 influent file 'BSM1_{profile}.csv' not found in package data."
        )
    # Parse the package data directly from text -- no temporary-file round-trip.
    series = _influent_from_text(
        resource.read_text(encoding="utf-8"),
        model,
        source=f"BSM1_{profile}.csv",
    )
    return dataclasses.replace(series, units=_declared_for(model, _BSM1_INFLUENT_UNITS))


def load_bsm2_influent(profile: str, model: CompiledModel) -> InfluentSeries:
    """Load one of the BSM2 influent files.

    Parameters
    ----------
    profile : {"dry", "rain", "storm"}
        Which BSM2 influent to load.
    model : CompiledModel
        ASM1 model (the water-line influent species map to ASM1 ordering).

    Returns
    -------
    InfluentSeries

    Notes
    -----
    These files are **synthesised** (``scripts/generate_bsm2_influent.py``): they
    follow the BSM2 constant-influent composition (Gernaey et al. 2014) plus a
    diurnal flow / load pattern, but are not the canonical 609-day IWA series.
    The layout is the BSM1 columns plus a time-varying influent temperature
    ``T`` (stored in degC; returned as ``InfluentSeries.T`` in Kelvin), which
    drives the ASM1 temperature corrections seasonally -- pair the model with
    :func:`aquakin.plant.bsm.bsm2_asm1_model` so the kinetics reference the
    BSM2 15 degC base. TSS is omitted (it is derived, not a state).
    """
    if profile not in ("dry", "rain", "storm"):
        raise ValueError(f"profile must be 'dry', 'rain', or 'storm'; got {profile!r}")
    resource = files("aquakin.plant.bsm.data") / f"BSM2_{profile}.csv"
    if not resource.is_file():
        raise FileNotFoundError(
            f"BSM2 influent file 'BSM2_{profile}.csv' not found in package data."
        )
    series = _influent_from_text(
        resource.read_text(encoding="utf-8"),
        model,
        column_order=_BSM2_COLUMN_ORDER,
        source=f"BSM2_{profile}.csv",
    )
    # The file stores temperature in degC; reactors expect Kelvin.
    return dataclasses.replace(
        series, T=series.T + 273.15, units=_declared_for(model, _BSM2_INFLUENT_UNITS)
    )


@dataclass
class InfluentExpression:
    """An influent whose flow, temperature and composition are expressions of time.

    The counterpart of :class:`InfluentSeries` for a feed that follows a
    *pattern* rather than a measured record: a diurnal flow, a weekly load, a
    step test. Each value is read by :mod:`aquakin.plant.time_expr`, so it is a
    plain string in a flowsheet spec -- it fingerprints, diffs and round-trips,
    and needs no table file beside it::

        InfluentExpression(
            model,
            Q="18446 + 4000*sin(2*pi*t)",
            composition={"SNH": "31.6 + 8*sin(2*pi*t - 0.5)", "SS": 69.5},
        )

    A value may equally be a plain number, which costs nothing extra, so a
    partly-patterned feed needs no ceremony for the parts that are constant.

    ``t`` is the plant's clock. The expressions are written on
    ``time_unit`` (the model's when not given) and :meth:`in_time_unit` puts
    them on the plant's, so an expression written in days still means days on a
    plant running on hours.

    Attributes
    ----------
    model : CompiledModel
        Kinetic model whose species the composition names.
    Q : str or float
        Volumetric flow, as an expression of ``t`` or a constant.
    composition : dict[str, str or float], optional
        Species name -> expression or constant. A species not named is absent
        from the feed (zero), as with ``InfluentSeries.constant(base="zero")``.
    T : str or float, optional
        Feed temperature; ``None`` (default) leaves the influent
        temperature-agnostic, as in :class:`InfluentSeries`.
    units : dict[str, str], optional
        The units the values are given in, keyed by ``"t"``, ``"Q"``, ``"T"``
        or a species name -- read exactly as :class:`InfluentSeries` reads them,
        except that the conversion is applied as the expression is evaluated
        rather than to stored data (a string cannot be scaled in place).
    time_unit : str, optional
        The clock the expressions are written on; the model's when ``None``.
        ``units["t"]`` sets it too, and wins when both are given.

    See Also
    --------
    InfluentSeries : the same feed as a measured record of samples.
    aquakin.plant.time_expr : the expression language, and what it can call.
    """

    model: CompiledModel
    Q: str | float = 0.0
    composition: dict[str, str | float] | None = None
    T: str | float | None = None
    units: dict[str, str] | None = None
    time_unit: str | None = None
    #: The clock ``at()`` is called on; set by :meth:`in_time_unit`, else the
    #: expressions' own, so a standalone influent needs no plant to be read.
    clock: str | None = None

    def __post_init__(self) -> None:
        self.composition = dict(self.composition or {})
        unknown = sorted(s for s in self.composition if s not in self.model.species_index)
        if unknown:
            from aquakin.core.hints import did_you_mean

            hint = did_you_mean(unknown[0], list(self.model.species))
            raise ValueError(
                f"composition names {unknown}, which are not species of model "
                f"'{self.model.name}'.{hint}"
            )
        self._check_units()
        # Read every expression now, so a typo is a load-time error naming the
        # field rather than a failure inside the first solve.
        self._compile()

    def _check_units(self) -> None:
        if not self.units:
            return
        allowed = {"t", "Q", "T", *self.model.species}
        unknown = sorted(k for k in self.units if k not in allowed)
        if unknown:
            raise ValueError(
                f"units declares {unknown}, which are not 't', 'Q', 'T' or a "
                f"species of model '{self.model.name}'."
            )
        bad = sorted(k for k, v in self.units.items() if not isinstance(v, str))
        if bad:
            raise ValueError(f"units values must be strings; got non-strings for {bad}.")
        unreadable = sorted(
            k
            for k, v in self.units.items()
            if parse_units(v) is None and not (k == "T" and v.strip().lower() in CELSIUS_SPELLINGS)
        )
        if unreadable:
            raise ValueError(
                f"units declares {unreadable} in units that cannot be read: "
                + ", ".join(f"{k}={self.units[k]!r}" for k in unreadable)
                + ". Use e.g. 'd', 'm3/d', 'K', 'g_N/m3' or 'mg/L'."
            )

    @property
    def _expression_clock(self) -> str | None:
        """The clock the expressions are written on."""
        return (self.units or {}).get("t") or self.time_unit or self.model.time_unit

    def _scale(self, key: str, target: str) -> float:
        """Factor from the declared unit of ``key`` to ``target`` (1 when nothing
        is declared, or the target is unknown)."""
        unit = (self.units or {}).get(key)
        if unit is None or not target:
            return 1.0
        try:
            factor = conversion_factor(unit, target)
        except ValueError as exc:
            raise ValueError(
                f"units declares '{key}' in '{unit}', which cannot be converted to "
                f"'{target}' as model '{self.model.name}' requires: {exc}"
            ) from None
        return 1.0 if factor is None else factor

    def _compile(self) -> None:
        """Build the callables ``at`` evaluates, with the unit factors folded in.

        A declared unit cannot be applied to the stored value the way
        :class:`InfluentSeries` scales its arrays -- the value is a string -- so
        it is applied to the result instead, which is the same number.
        """
        from aquakin.plant.time_expr import compile_time_expression

        expr_clock = self._expression_clock
        caller_clock = self.clock or expr_clock
        time_scale = 1.0
        if expr_clock is not None and caller_clock is not None and caller_clock != expr_clock:
            factor = conversion_factor(caller_clock, expr_clock)
            if factor is not None:
                time_scale = factor

        def read(value, what, out_scale=1.0, offset=0.0):
            fn = compile_time_expression(value, time_scale=time_scale, what=what)
            if out_scale == 1.0 and offset == 0.0:
                return fn
            return lambda t, f=fn, s=out_scale, o=offset: f(t) * s + o

        flow_target = "" if caller_clock is None else f"m3/{caller_clock}"
        # Q is declared per the expressions' own clock; it leaves per the
        # caller's, so the two conversions compose.
        q_scale = self._scale("Q", f"m3/{expr_clock}" if expr_clock else "")
        if expr_clock is not None and caller_clock is not None and flow_target:
            per_clock = conversion_factor(f"m3/{expr_clock}", flow_target)
            if per_clock is not None:
                q_scale *= per_clock
        self._Q_fn = read(self.Q, "the flow", q_scale)

        self._T_fn = None
        if self.T is not None:
            celsius = (self.units or {}).get("T", "").strip().lower() in CELSIUS_SPELLINGS
            t_scale = 1.0 if celsius else self._scale("T", "K")
            self._T_fn = read(self.T, "the temperature", t_scale, 273.15 if celsius else 0.0)

        self._C_fns = []
        for species, value in self.composition.items():
            scale = self._scale(species, self.model.species_units.get(species, ""))
            self._C_fns.append(
                (self.model.species_index[species], read(value, f"species '{species}'", scale))
            )

    def in_time_unit(self, time_unit: str | None) -> InfluentExpression:
        """This influent read on the clock ``time_unit``.

        The expressions are unchanged -- they still mean what they said on their
        own clock; only the ``t`` they are given, and the flow they return, are
        converted. Returns ``self`` when there is nothing to change.
        """
        if time_unit is None or time_unit == (self.clock or self._expression_clock):
            return self
        return dataclasses.replace(self, clock=time_unit)

    def at(self, t: jnp.ndarray) -> Stream:
        """Return the influent :class:`Stream` at time ``t``.

        Every value is evaluated at ``t``; a species the composition does not
        name is zero.
        """
        t = jnp.asarray(t, dtype=float)
        C = jnp.zeros(self.model.n_species)
        for index, fn in self._C_fns:
            C = C.at[index].set(jnp.asarray(fn(t), dtype=float))
        T = None if self._T_fn is None else jnp.asarray(self._T_fn(t), dtype=float)
        return Stream(
            Q=jnp.asarray(self._Q_fn(t), dtype=float),
            C=C,
            model=self.model,
            scalars=make_scalars(T=T),
        )
