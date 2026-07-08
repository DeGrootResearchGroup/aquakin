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
    """

    t: jnp.ndarray
    Q: jnp.ndarray
    C: jnp.ndarray
    model: CompiledModel
    T: jnp.ndarray | None = None

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

    @classmethod
    def constant(
        cls, model, overrides=None, /, *, Q, base: str = "zero", T=None, **species
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
        return cls(t=t, Q=Q_arr, C=C_arr, model=model, T=T_arr)

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

    Returns
    -------
    InfluentSeries
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Influent file not found: {p}")
    return _influent_from_text(
        p.read_text(encoding="utf-8"),
        model,
        column_order=column_order,
        column_map=column_map,
        fractions=fractions,
        delimiter=delimiter,
        source=str(p),
    )


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
    return _influent_from_text(
        resource.read_text(encoding="utf-8"),
        model,
        source=f"BSM1_{profile}.csv",
    )


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
    return dataclasses.replace(series, T=series.T + 273.15)
