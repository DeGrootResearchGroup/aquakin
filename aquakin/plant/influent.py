"""Time-varying influent streams.

:class:`InfluentSeries` reads a CSV of time-series influent data (one
row per timestep, columns for flow rate and each species concentration)
and exposes an AD-clean piecewise-linear interpolant ``.at(t)`` that
the plant's RHS calls to get the inlet stream at the current integration
time.

:func:`load_bsm1_influent` reads the canonical BSM1 dry / rain / storm
influent files shipped under ``aquakin/plant/bsm/data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Union

import jax.numpy as jnp

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


# BSM1 influent CSVs use this exact column order. ``t`` is days since
# the start of the simulation; ``Q`` is in m³/d; all concentrations are
# in g_COD/m³ or g_N/m³ per ASM1 conventions.
_BSM1_COLUMN_ORDER = [
    "t", "SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP",
    "SO", "SNO", "SNH", "SND", "XND", "SALK", "Q",
]


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
        columns follow ``network.species`` ordering.
    network : CompiledNetwork
    """

    t: jnp.ndarray
    Q: jnp.ndarray
    C: jnp.ndarray
    network: "CompiledNetwork"

    def __post_init__(self) -> None:
        if self.t.ndim != 1:
            raise ValueError(f"t must be 1-D, got shape {self.t.shape}")
        if self.Q.shape != self.t.shape:
            raise ValueError(
                f"Q shape {self.Q.shape} does not match t shape {self.t.shape}"
            )
        if self.C.ndim != 2 or self.C.shape[0] != self.t.shape[0]:
            raise ValueError(
                f"C shape {self.C.shape} expected ({self.t.shape[0]}, n_species)"
            )
        if self.C.shape[1] != self.network.n_species:
            raise ValueError(
                f"C has {self.C.shape[1]} species columns but network has "
                f"{self.network.n_species}"
            )

    def at(self, t: jnp.ndarray) -> Stream:
        """Return the influent :class:`Stream` at time ``t``.

        Linearly interpolates between samples. Outside the range, clamps
        to the endpoint values (``jnp.interp`` semantics).
        """
        Q_t = jnp.interp(t, self.t, self.Q)
        # Interpolate each species column separately, then stack.
        n_species = self.network.n_species
        C_t = jnp.stack(
            [jnp.interp(t, self.t, self.C[:, j]) for j in range(n_species)]
        )
        return Stream(Q=Q_t, C=C_t, network=self.network)


def read_influent_csv(
    path: Union[str, Path],
    network: "CompiledNetwork",
    *,
    column_order: list[str] | None = None,
    delimiter: str | None = None,
) -> InfluentSeries:
    """Read a CSV influent file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    network : CompiledNetwork
        Kinetic network whose species ordering ``C`` is built against.
    column_order : list[str], optional
        Column order in the file. The first column is treated as time
        ``t`` and the column named ``"Q"`` provides the flow rate; every
        other name must be a species declared by the network. Defaults to
        the standard BSM1 ordering.
    delimiter : str, optional
        Field delimiter. Defaults to auto-sniffing (whitespace and comma
        both work).

    Returns
    -------
    InfluentSeries
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Influent file not found: {p}")
    return _influent_from_text(
        p.read_text(encoding="utf-8"), network,
        column_order=column_order, delimiter=delimiter, source=str(p),
    )


def _influent_from_text(
    text: str,
    network: "CompiledNetwork",
    *,
    column_order: list[str] | None = None,
    delimiter: str | None = None,
    source: str = "<text>",
) -> InfluentSeries:
    """Parse influent CSV / whitespace text into an :class:`InfluentSeries`.

    Shared by :func:`read_influent_csv` (file path) and
    :func:`load_bsm1_influent` (in-package data), so neither round-trips the
    data through a temporary file. ``source`` only labels error messages.
    """
    if column_order is None:
        column_order = _BSM1_COLUMN_ORDER

    # Auto-detect delimiter by trying commas first, then whitespace.
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # If the first non-comment line is alphabetic and matches the
        # expected header, skip it.
        if rows == [] and any(c.isalpha() for c in line):
            # Could be a header; only accept if all tokens are non-numeric.
            tokens = line.replace(",", " ").split()
            try:
                [float(t) for t in tokens]
                # All numeric → this is data, not a header.
            except ValueError:
                continue  # header line; skip
        if delimiter == ",":
            tokens = [tok.strip() for tok in line.split(",") if tok.strip()]
        elif delimiter is not None:
            tokens = [tok.strip() for tok in line.split(delimiter) if tok.strip()]
        else:
            # Try comma first; if it doesn't give the expected count,
            # fall back to whitespace.
            comma_tokens = [tok.strip() for tok in line.split(",") if tok.strip()]
            if len(comma_tokens) == len(column_order):
                tokens = comma_tokens
            else:
                tokens = line.split()
        if len(tokens) != len(column_order):
            raise ValueError(
                f"Influent row '{raw_line}' has {len(tokens)} fields but "
                f"column_order specifies {len(column_order)}."
            )
        try:
            rows.append([float(tok) for tok in tokens])
        except ValueError as exc:
            raise ValueError(
                f"Influent row '{raw_line}' has non-numeric field: {exc}"
            ) from exc

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
    for sp in network.species:
        if sp not in column_order:
            raise ValueError(
                f"Influent file is missing species column '{sp}' "
                f"(declared by network '{network.name}')."
            )
        species_idx_in_file.append(column_order.index(sp))
    species_idx_arr = jnp.asarray(species_idx_in_file)

    t = data[:, t_idx]
    Q = data[:, Q_idx]
    C = data[:, species_idx_arr]

    return InfluentSeries(t=t, Q=Q, C=C, network=network)


def load_bsm1_influent(profile: str, network: "CompiledNetwork") -> InfluentSeries:
    """Load one of the canonical BSM1 influent files.

    Parameters
    ----------
    profile : {"dry", "rain", "storm"}
        Which BSM1 reference influent to load.
    network : CompiledNetwork
        ASM1 network (the influent species map to ASM1 state ordering).

    Returns
    -------
    InfluentSeries

    Notes
    -----
    These files follow the BSM1 specification (Copp 2002; Alex et al.
    2008). Times are in days, flow in m³/d, concentrations in g_COD/m³
    or g_N/m³ following ASM1 conventions.
    """
    if profile not in ("dry", "rain", "storm"):
        raise ValueError(
            f"profile must be 'dry', 'rain', or 'storm'; got {profile!r}"
        )
    resource = files("aquakin.plant.bsm.data") / f"BSM1_{profile}.csv"
    if not resource.is_file():
        raise FileNotFoundError(
            f"BSM1 influent file 'BSM1_{profile}.csv' not found in package data."
        )
    # Parse the package data directly from text -- no temporary-file round-trip.
    return _influent_from_text(
        resource.read_text(encoding="utf-8"), network,
        source=f"BSM1_{profile}.csv",
    )
