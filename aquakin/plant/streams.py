"""Streams: the data passed between units in a plant flowsheet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.integrate._common import _HasNamedSpecies

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


@dataclass(frozen=True)
class Stream:
    """A flow stream — bulk volumetric rate plus a concentration vector.

    Streams are produced by a unit's ``compute_outputs`` and consumed by
    downstream units' ``rhs`` / ``compute_outputs`` calls. They are
    intentionally immutable per evaluation: a connection delivers the
    upstream output directly, with optional :class:`StateTranslator`
    interposed for cross-network mappings.

    Attributes
    ----------
    Q : jnp.ndarray
        Volumetric flow rate (scalar), units must be consistent across the
        plant (typically m³/d for BSM-family plants).
    C : jnp.ndarray
        Concentration vector, shape ``(n_species,)`` where species ordering
        is ``network.species``.
    network : CompiledNetwork
        The kinetic network whose species ordering applies to ``C``.
    T : jnp.ndarray, optional
        Stream temperature (scalar, Kelvin). Carried algebraically through the
        flowsheet: mixers flow-weight it (a heat balance) and pass-through units
        propagate it unchanged, so a reactor can read its inlet temperature and
        feed it to temperature-dependent kinetics. ``None`` (the default) means
        the stream is temperature-agnostic; reactors then fall back to their
        static condition, so existing plants are unaffected. ``None``-ness is a
        static structural property (consistent across RHS calls), so it is
        jit-safe.
    """

    Q: jnp.ndarray
    C: jnp.ndarray
    network: "CompiledNetwork"
    T: "jnp.ndarray | None" = None

    def mass_flow(self) -> jnp.ndarray:
        """Per-species mass flow rate ``Q * C``, shape ``(n_species,)``."""
        return self.Q * self.C

    def with_C(self, C: jnp.ndarray) -> "Stream":
        """Return a new stream with the same Q/T/network but a new C vector."""
        return Stream(Q=self.Q, C=C, network=self.network, T=self.T)

    def with_Q(self, Q: jnp.ndarray) -> "Stream":
        """Return a new stream with the same C/T/network but a new flow rate."""
        return Stream(Q=Q, C=self.C, network=self.network, T=self.T)

    def with_T(self, T: "jnp.ndarray | None") -> "Stream":
        """Return a new stream with the same Q/C/network but a new temperature."""
        return Stream(Q=self.Q, C=self.C, network=self.network, T=T)


_EPS_Q = 1e-12  # guard the flow-weighted division when total inflow is ~zero


def mixed_temperature(inputs: "dict[str, Stream]", names) -> "jnp.ndarray | None":
    """Flow-weighted outlet temperature for a unit's inlet streams (a heat balance).

    The single shared rule every multi-inlet unit (mixer, CSTR, clarifier,
    digester) uses to combine inlet temperatures, so the convention cannot drift
    between them.

    Only the inlets that actually carry a temperature (``Stream.T is not None``)
    are combined; an inlet with ``T is None`` -- a temperature-agnostic feed, or a
    zero-flow recycle seed -- is **ignored**, not allowed to poison the result.
    (A single ``None`` inlet used to force the whole mix to ``None``; for a recycle
    loop seeded with a temperature-agnostic zero-flow stream that disabled
    temperature propagation around the entire loop.) Returns ``None`` only when
    *no* inlet carries a temperature, i.e. a fully temperature-agnostic mix.

    Zero-flow-safe: the weighting divides by the carriers' total flow, but if that
    is ~zero (every temperature-carrying inlet momentarily at zero flow) it falls
    back to their plain mean instead of dividing by ~0, which would otherwise
    collapse the temperature toward 0 K and feed a garbage value to any Arrhenius
    correction downstream.

    Parameters
    ----------
    inputs : dict[str, Stream]
        The unit's inlet streams keyed by input-port name.
    names : iterable of str
        The input-port names to combine (the unit's ``input_port_names``).

    Returns
    -------
    jnp.ndarray or None
        The flow-weighted temperature (scalar), or ``None`` if no inlet carries
        one. ``None``-ness is a static structural property (it depends only on
        which inlets carry a temperature), so callers stay jit-safe.
    """
    carriers = [inputs[n] for n in names if inputs[n].T is not None]
    if not carriers:
        return None
    Q_total = jnp.zeros(())
    heat = jnp.zeros(())
    for s in carriers:
        Q_total = Q_total + s.Q
        heat = heat + s.Q * s.T
    mean_T = sum(s.T for s in carriers) / len(carriers)
    return jnp.where(Q_total > _EPS_Q, heat / (Q_total + _EPS_Q), mean_T)


@dataclass(frozen=True)
class StreamSeries(_HasNamedSpecies):
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
        Concentration over time, shape ``(n_t, n_species)`` in the network's
        species ordering.
    network : CompiledNetwork
        The kinetic network whose species ordering applies to ``C``.
    """

    t: jnp.ndarray
    Q: jnp.ndarray
    C: jnp.ndarray
    network: "CompiledNetwork"

    # C_named / C_named_many / final_named / .final come from _HasNamedSpecies
    # (shared with the reactor solutions), keyed off .C and .network.

    def to_dataframe(self, *, units_in_columns: bool = False):
        """Return the stream trajectory as a pandas ``DataFrame``.

        One row per save time, indexed by time ``t``, with a flow column ``Q``
        followed by one column per species (in network ordering).

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

        columns = [(sp, self.C[:, j]) for j, sp in enumerate(self.network.species)]
        units = {sp: self.network.units_of(sp) for sp in self.network.species}
        return build_dataframe(
            self.t, columns, index_name="t", units=units,
            units_in_columns=units_in_columns, extra=[("Q", self.Q)],
        )

    def to_csv(self, path_or_buf=None, *, units_in_columns: bool = True, **kwargs):
        """Write the stream trajectory to CSV (delegates to :meth:`to_dataframe`).

        ``units_in_columns`` defaults to ``True`` so the written file is
        self-describing (a CSV cannot carry ``df.attrs``). Extra keyword
        arguments are forwarded to ``pandas.DataFrame.to_csv``.
        """
        return self.to_dataframe(units_in_columns=units_in_columns).to_csv(
            path_or_buf, **kwargs
        )
