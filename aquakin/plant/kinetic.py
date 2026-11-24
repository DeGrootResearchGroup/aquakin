"""Shared reaction and operating-temperature pieces of the kinetic plant units.

Every unit that evaluates a kinetic model in its ``rhs`` (the CSTR, MBR, SBR,
IFAS/MBBR tank and ADM1 digester) needs the same four things: the model's
conditions checked and broadcast to the length-1 arrays the rate functions
index, the temperature its kinetics run at, the reaction term itself, and (for
the units whose temperature an operator sets) a way to change that temperature.
:class:`KineticUnit` and :class:`AdjustableTemperatureUnit` hold the one
implementation of each, so the units cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.streams import mixed_scalars
from aquakin.plant.temperature import OPERATING_T_SIGNAL
from aquakin.plant.units import ModelTimeScaling

if TYPE_CHECKING:
    from aquakin.plant.streams import Stream


class KineticUnit(ModelTimeScaling):
    """Mixin for a plant unit whose ``rhs`` evaluates a kinetic model.

    A consumer sets ``self.model`` and ``self.name`` and calls
    :meth:`_init_conditions` from its ``__post_init__``. Its ``rhs`` then builds
    the reaction term with :meth:`_reaction`, which is the model's canonical
    :meth:`~aquakin.core.model.CompiledModel.dCdt` (rate inputs clipped at zero
    when the model asks for it, net consumption throttled by the positivity
    limiter) scaled to the plant's time unit, and adds its own transport on top.
    A unit whose kinetics follow the plant temperature takes its rate conditions
    from :meth:`_operating_conditions`.
    """

    def _init_conditions(self, conditions: Mapping[str, float]) -> None:
        """Check ``conditions`` covers the model and store the rate-call arrays.

        Each condition the model declares is broadcast to a length-1 array in
        ``self._condition_arrays`` so the rate functions index it with
        ``loc_idx=0``, the same convention as :class:`~aquakin.BatchReactor`.

        Raises
        ------
        ValueError
            If a condition the model requires is not in ``conditions``.
        """
        missing = sorted(set(self.model.conditions_required) - set(conditions))
        if missing:
            raise ValueError(
                f"{type(self).__name__} '{self.name}' is missing required condition "
                f"values for: {missing}. Provided: {sorted(conditions)}"
            )
        self._condition_arrays = {
            cname: jnp.asarray([float(conditions[cname])])
            for cname in self.model.conditions_required
        }

    def _operating_conditions(self, inputs: dict[str, Stream], signals: dict | None):
        """The rate conditions at the unit's operating temperature.

        The operating temperature is, in order: the temperature the plant hands
        the unit through the control-signal bus (its own lagged temperature
        state under a heat-balance temperature model); else the flow-weighted
        inlet temperature (the algebraic default: the well-mixed tank is at its
        inlet temperature, with no thermal lag); else none, and the static
        conditions stand. It replaces the ``T`` condition when the model has one.

        Returns
        -------
        conditions : dict[str, jnp.ndarray]
            The condition arrays for the rate call.
        T : jnp.ndarray or float or None
            The operating temperature, else the static ``T`` condition (``None``
            when there is neither) -- the temperature an aeration correction
            uses.
        """
        T_op = None if signals is None else signals.get(OPERATING_T_SIGNAL)
        if T_op is None:
            T_op = mixed_scalars(inputs, self.input_ports, keys=("T",)).get("T")
        if T_op is None:
            return self._condition_arrays, self.conditions.get("T")
        if "T" in self._condition_arrays:
            return {**self._condition_arrays, "T": jnp.reshape(T_op, (1,))}, T_op
        return self._condition_arrays, T_op

    def _reaction_conditions(self, inputs: dict[str, Stream], signals: dict | None) -> dict:
        """The conditions this unit's ``rhs`` evaluates its reaction term at.

        The operating conditions (:meth:`_operating_conditions`) by default; a
        unit held at its static conditions (the heated digester, the SBR)
        overrides this, and its ``rhs`` reads it too. The mass balance's reaction
        integral calls it with the inputs and signals the solve's ``rhs`` saw, so
        it integrates the reaction the unit integrated.
        """
        return self._operating_conditions(inputs, signals)[0]

    def _reaction(self, C: jnp.ndarray, params: jnp.ndarray, conditions: dict) -> jnp.ndarray:
        """The reaction term ``model.dCdt`` at ``C``, per the plant's time unit."""
        return self._scale_rates(self.model.dCdt(C, params, conditions, 0))

    # ----- the mass balance's reaction contract (units.ReactionBalanceUnit) -----
    #: Components whose reaction loss leaves as gas: an activated-sludge reactor
    #: reduces nitrate to N₂ (N) and oxidizes COD with it or with oxygen (COD).
    reaction_gas_components: frozenset = frozenset({"COD", "N"})

    def reaction_term(
        self,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None,
    ) -> jnp.ndarray:
        """The reaction term the unit's ``rhs`` adds, at the conditions it used.

        Parameters
        ----------
        state : jnp.ndarray
            The unit's state; its leading ``n_species`` entries are the
            concentrations the reaction sees (any trailing non-species state,
            such as an MBR's fouling resistance, is not).
        inputs : dict[str, Stream]
            The unit's input streams, as the plant's RHS assembly hands them.
        params : jnp.ndarray
            The unit's parameter vector.
        signals : dict or None
            The control-signal bus.

        Returns
        -------
        jnp.ndarray
            ``dC/dt`` from the reactions, shape ``(n_species,)``, per the
            plant's time unit.
        """
        C = state[: self.model.n_species]
        return self._reaction(C, params, self._reaction_conditions(inputs, signals))

    def reaction_volumes(self, state: jnp.ndarray, params: jnp.ndarray):
        """The volume each species' reaction term acts in: the unit's fixed
        ``volume`` for every species.

        Parameters
        ----------
        state : jnp.ndarray
            The unit's state (unused: the volume is fixed).
        params : jnp.ndarray
            The unit's parameter vector (unused).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_species,)`` (m³).
        """
        import numpy as np

        return np.full(self.model.n_species, float(self.volume))


class AdjustableTemperatureUnit(KineticUnit):
    """A :class:`KineticUnit` whose static operating temperature can be set.

    Gives the unit :meth:`set_temperature`, so it satisfies the
    :class:`~aquakin.plant.units.TemperatureSettable` capability and
    :meth:`Plant.set_temperature <aquakin.plant.plant.Plant.set_temperature>`
    reaches it. A unit held at a fixed temperature (the heated digester) uses
    :class:`KineticUnit` alone, so the plant-wide call leaves it alone.
    """

    def set_temperature(self, temperature_K: float) -> None:
        """Set this reactor's static operating temperature (Kelvin).

        Updates the ``T`` condition (and its rate-evaluation array) in place, so
        a re-solve runs the kinetics -- including any Arrhenius
        ``temperature_corrections`` -- at the new temperature. A no-op for a
        model that declares no ``T`` condition. The plant clears its compiled-
        solve cache after calling this; on a bare unit, rebuild any cached solve.
        """
        if "T" not in self.model.conditions_required:
            return
        self.conditions = {**self.conditions, "T": float(temperature_K)}
        self._condition_arrays = {
            **self._condition_arrays,
            "T": jnp.asarray([float(temperature_K)]),
        }
