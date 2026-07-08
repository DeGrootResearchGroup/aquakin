"""Streams: the data passed between units in a plant flowsheet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.integrate._common import PlottableSolutionMixin, _HasNamedSpecies
from aquakin.plant._constants import EPS_Q

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


@dataclass(frozen=True)
class Stream:
    """A flow stream — bulk volumetric rate plus a concentration vector.

    Streams are produced by a unit's ``compute_outputs`` and consumed by
    downstream units' ``rhs`` / ``compute_outputs`` calls. They are
    intentionally immutable per evaluation: a connection delivers the
    upstream output directly, with optional :class:`StateTranslator`
    interposed for cross-model mappings.

    Attributes
    ----------
    Q : jnp.ndarray
        Volumetric flow rate (scalar), units must be consistent across the
        plant (typically m³/d for BSM-family plants).
    C : jnp.ndarray
        Concentration vector, shape ``(n_species,)`` where species ordering
        is ``model.species``.
    model : CompiledModel
        The kinetic model whose species ordering applies to ``C``.
    scalars : Mapping[str, jnp.ndarray]
        Per-stream **side-channel** scalars carried algebraically through the
        flowsheet alongside ``Q``/``C`` -- a single open-ended map rather than a
        fixed field per quantity, so a new carried scalar needs no new field, no
        ``with_*`` copier and no ``mixed_*`` combiner. Multi-inlet units combine
        them with the one shared :func:`mixed_scalars`; pass-through units forward
        them unchanged (``scalars=s_in.scalars``). A scalar **absent from the map**
        means the stream does not carry it (an agnostic feed / a zero-flow recycle
        seed); its presence is a static structural property (consistent across RHS
        calls), so callers stay jit-safe. The two used today are:

        * ``"T"`` -- stream temperature (K); reactors read their inlet temperature
          from it for temperature-dependent kinetics and fall back to their static
          condition when it is absent.
        * ``"org"`` -- indicator-organism density (e.g. CFU/100 mL) for
          disinfection; a disinfection unit reduces it by the computed
          log-inactivation and falls back to its design ``inlet_density`` when it
          is absent.

        Read a scalar with ``stream.scalars.get(name)`` (``None`` if not carried);
        build the map for a leaf stream with :func:`make_scalars`, which drops the
        agnostic (``None``) entries.
    """

    Q: jnp.ndarray
    C: jnp.ndarray
    model: CompiledModel
    scalars: Mapping[str, jnp.ndarray] = field(default_factory=dict)

    def mass_flow(self) -> jnp.ndarray:
        """Per-species mass flow rate ``Q * C``, shape ``(n_species,)``."""
        return self.Q * self.C

    def with_C(self, C: jnp.ndarray) -> Stream:
        """Return a new stream with a new ``C`` vector, everything else (including
        the side-channel ``scalars``) preserved."""
        return replace(self, C=C)

    def with_Q(self, Q: jnp.ndarray) -> Stream:
        """Return a new stream with a new flow rate, everything else (including the
        side-channel ``scalars``) preserved."""
        return replace(self, Q=Q)


def make_scalars(**values) -> dict:
    """Assemble a :attr:`Stream.scalars` map, dropping the agnostic (``None``) entries.

    A leaf stream (an influent sample, a recycle seed) builds its side-channel map
    from named quantities that may or may not be present -- e.g.
    ``make_scalars(T=T_t)`` where ``T_t`` is ``None`` for a temperature-agnostic
    feed. Absence, not a stored ``None``, is how :func:`mixed_scalars` tells a
    stream does not carry a scalar, so ``None`` values are omitted rather than
    stored. Not specific to any scalar name -- pass whichever the stream carries."""
    return {k: v for k, v in values.items() if v is not None}


#: The side-channel scalars combined by default: temperature (a heat balance) and
#: indicator-organism density (an indicator mass balance).
_FIRST_CLASS_SCALARS = ("T", "org")


def mixed_scalars(inputs: dict[str, Stream], names, keys=_FIRST_CLASS_SCALARS) -> dict:
    """Flow-weighted outlet value for each side-channel scalar a unit's inlets carry.

    The single shared rule every multi-inlet unit (mixer, CSTR, clarifier,
    digester) uses to combine inlet side-channel scalars -- temperature ``T`` (a
    heat balance), indicator density ``org`` (a mass balance), and any future
    scalar -- so the convention cannot drift between units or between scalars.

    For each name in ``keys``, only the inlets that actually carry it
    (``name in stream.scalars``) are combined; an inlet that does not -- a
    temperature-agnostic feed, or a zero-flow recycle seed -- is **ignored**, not
    allowed to poison the result. (A single agnostic inlet used to force the whole
    mix to ``None``; for a recycle loop seeded with an agnostic zero-flow stream
    that disabled propagation around the entire loop.) A scalar that **no** inlet
    carries is omitted from the returned map entirely, so ``name in result`` is the
    same static structural property the old per-scalar ``None`` return was (callers
    stay jit-safe). The result drops straight into a downstream ``Stream``'s
    ``scalars=``.

    Zero-flow-safe: the weighting divides by the carriers' total flow, but if that
    is ~zero (every carrier momentarily at zero flow) it falls back to their plain
    mean instead of dividing by ~0, which would otherwise collapse a temperature
    toward 0 K and feed a garbage value to any Arrhenius correction downstream.

    Parameters
    ----------
    inputs : dict[str, Stream]
        The unit's inlet streams keyed by input-port name.
    names : iterable of str
        The input-port names to combine (the unit's ``input_port_names``).
    keys : iterable of str, optional
        The scalar names to combine (default: the first-class ``T`` and ``org``).

    Returns
    -------
    dict[str, jnp.ndarray]
        The flow-weighted value of each combined scalar at least one inlet carries.
    """
    out = {}
    for key in keys:
        carriers = [
            (inputs[n].Q, inputs[n].scalars[key]) for n in names if key in inputs[n].scalars
        ]
        if carriers:
            out[key] = _flow_weighted_scalar(carriers)
    return out


def _flow_weighted_scalar(carriers) -> jnp.ndarray:
    """Flow-weighted mean of a per-stream scalar over the streams that carry it.

    ``carriers`` is a list of ``(Q, value)``. Divides the flow-weighted sum by the
    carriers' total flow, falling back to the plain mean when that total is ~zero
    (every carrier momentarily at zero flow) rather than dividing by ~0. The shared
    kernel behind :func:`mixed_scalars` -- a heat balance for ``T``, an indicator
    mass balance for ``org``."""
    Q_total = jnp.zeros(())
    weighted = jnp.zeros(())
    for q, v in carriers:
        Q_total = Q_total + q
        weighted = weighted + q * v
    mean = sum(v for _, v in carriers) / len(carriers)
    return jnp.where(Q_total > EPS_Q, weighted / (Q_total + EPS_Q), mean)


def total_flow(flows) -> jnp.ndarray:
    """Total flow ``Σ Q`` over an iterable of per-port flows.

    ``flows`` is any iterable of scalar flows -- the callers pass a generator of
    either inlet-stream flows (``inputs[n].Q``) or the concentration-free flow-map
    values (``input_flows[n]``), so the one summation rule serves a unit's
    ``compute_outputs`` and its ``flow_outputs``. The flow sibling of
    :func:`mixed_scalars` / :func:`mixed_feed`.
    """
    total = jnp.zeros(())
    for q in flows:
        total = total + q
    return total


def mixed_feed(inputs: dict[str, Stream], names) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(Q_total, C_in)`` for a Q-weighted multi-inlet feed.

    The total inflow ``Σ Q`` and the flow-weighted inlet concentration
    ``Σ(Q·C) / Σ(Q)`` -- the shared rule every well-mixed multi-inlet unit (the
    ADM1 digester, the primary-clarifier holding tank, the IFAS bulk) uses to form
    the feed its dilution term drives toward. The concentration companion to
    :func:`mixed_scalars` (which combines the side-channel scalars) and
    :func:`total_flow`. The division is guarded by the shared ``EPS_Q`` so a
    momentarily zero total inflow yields ``0`` rather than ``inf`` -- matching the
    ``/(Q_total + 1e-12)`` guard these units carried inline.

    Parameters
    ----------
    inputs : dict[str, Stream]
        The unit's inlet streams keyed by input-port name.
    names : iterable of str
        The input-port names to combine (the unit's ``input_port_names``).

    Returns
    -------
    Q_total : jnp.ndarray
        Scalar total inflow.
    C_in : jnp.ndarray
        ``(n_species,)`` flow-weighted inlet concentration.
    """
    Q_total = jnp.zeros(())
    mass = jnp.zeros(())
    for n in names:
        s = inputs[n]
        Q_total = Q_total + s.Q
        mass = mass + s.Q * s.C
    return Q_total, mass / (Q_total + EPS_Q)


def split_by_capture(
    C_in: jnp.ndarray,
    part_mask: jnp.ndarray,
    capture_frac: jnp.ndarray,
    Q_in: jnp.ndarray,
    Q_under: jnp.ndarray,
    Q_over: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mass-conserving capture partition of a feed into underflow + overflow.

    A fraction ``capture_frac`` of each *particulate* species' inflowing mass
    (``part_mask == 1``) is captured to the underflow and the rest to the
    overflow; solubles (``part_mask == 0``) pass through at the inlet
    concentration into both outlets (the flow split carries their partition).
    Returns the two outlet concentration vectors ``(C_under, C_over)``.

    The particulate outlet concentrations are the captured / escaped mass divided
    by the (separately determined) outlet flow, guarded by ``EPS_Q`` against a
    zero outlet flow. This is the fixed-capture-fraction separation the ideal
    secondary clarifier uses. (The ideal ``%TSS`` thickener is the *same*
    partition with ``capture_frac`` equal to its solids-removal fraction, but it
    writes it as per-species thickening-factor scales because its outlet *flows*
    are concentration-dependent; the Otterpohl primary clarifier differs
    genuinely -- it partitions its well-mixed *state*, not the inflow.)
    """
    sol_mask = 1.0 - part_mask
    mass_in_p = Q_in * C_in * part_mask
    sol_C = C_in * sol_mask
    C_under = sol_C + capture_frac * mass_in_p / (Q_under + EPS_Q)
    C_over = sol_C + (1.0 - capture_frac) * mass_in_p / (Q_over + EPS_Q)
    return C_under, C_over


@dataclass(frozen=True)
class StreamSeries(_HasNamedSpecies, PlottableSolutionMixin):
    """A stream's flow and concentration trajectory over time.

    Returned by :meth:`Plant.stream`, which reconstructs a named output stream
    (e.g. the clarifier effluent) from a solution's saved states -- the plant
    integrates unit *states*, not the inter-unit streams, so the effluent is
    recomputed after the fact.

    Attributes
    ----------
    t : jnp.ndarray
        Save times, shape ``(n_t,)``.
    Q : jnp.ndarray
        Volumetric flow rate at each time, shape ``(n_t,)``.
    C : jnp.ndarray
        Concentration over time, shape ``(n_t, n_species)`` in the model's
        species ordering.
    model : CompiledModel
        The kinetic model whose species ordering applies to ``C``.
    org : jnp.ndarray, optional
        Indicator-organism density trajectory, shape ``(n_t,)``, when the stream
        carries one (e.g. downstream of a disinfection unit); ``None`` otherwise.
    """

    t: jnp.ndarray
    Q: jnp.ndarray
    C: jnp.ndarray
    model: CompiledModel
    org: jnp.ndarray | None = None

    # C_named / C_named_many / final_named / .final come from _HasNamedSpecies
    # (shared with the reactor solutions), keyed off .C and .model.

    def to_dataframe(self, *, units_in_columns: bool = False):
        """Return the stream trajectory as a pandas ``DataFrame``.

        One row per save time, indexed by time ``t``, with a flow column ``Q``
        followed by one column per species (in model ordering).

        Parameters
        ----------
        units_in_columns : bool, optional
            If ``True``, append ``" [unit]"`` to each species column label;
            otherwise columns are bare species names and per-species units are
            stored in ``df.attrs["units"]``.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If pandas (an optional dependency) is not installed.
        """
        from aquakin.integrate._common import build_dataframe

        columns = [(sp, self.C[:, j]) for j, sp in enumerate(self.model.species)]
        units = {sp: self.model.units_of(sp) for sp in self.model.species}
        return build_dataframe(
            self.t,
            columns,
            index_name="t",
            units=units,
            units_in_columns=units_in_columns,
            extra=[("Q", self.Q)],
        )

    def to_csv(self, path_or_buf=None, *, units_in_columns: bool = True, **kwargs):
        """Write the stream trajectory to CSV (delegates to :meth:`to_dataframe`).

        ``units_in_columns`` defaults to ``True`` so the written file is
        self-describing (a CSV cannot carry ``df.attrs``). Extra keyword
        arguments are forwarded to ``pandas.DataFrame.to_csv``.
        """
        return self.to_dataframe(units_in_columns=units_in_columns).to_csv(path_or_buf, **kwargs)
