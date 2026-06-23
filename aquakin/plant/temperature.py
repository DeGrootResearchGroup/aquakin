"""Selectable plant temperature models.

A plant's reactor temperature can be handled two ways, and the choice is a
``TemperatureModel`` strategy held by the :class:`~aquakin.plant.plant.Plant`:

- :class:`AlgebraicTemperature` (the default) -- temperature is *instantaneous*:
  each unit's outlet temperature is the flow-weighted mix of its inlets, with no
  thermal storage. A reactor runs its kinetics at its (flow-weighted) inlet
  temperature. This is the historic behaviour; it carries **zero** extra state
  and is a pure no-op (every existing plant and validated steady state is
  unchanged).

- :class:`HeatBalanceTemperature` -- each finite-volume liquid unit carries its
  temperature as a **dynamic state** governed by the completely-mixed first-order
  heat balance ``V dT/dt = Q_in (T_in - T)``, so the reactor temperature *lags*
  and *damps* the influent temperature. This is the BSM2 protocol's treatment
  (Jeppsson et al. 2007: "temperature dynamics in each reactor with a defined
  volume are modelled by a first-order system based on the heat content (T*V) of
  the wastewater ... except for the digester, for which the temperature is
  fixed"). It matters because plant recycles trap heat, so the effective thermal
  time constant ``V_total / Q_fresh`` can be hours -- comparable to diurnal
  forcing -- which an algebraic instantaneous model cannot represent.

The temperature states (if any) are appended as one contiguous block at the tail
of the flat plant state vector, mirroring the ``FlowSetpoint`` parameter-block
append: existing per-unit state slices keep their indices, so warm-starts and
``states_by_unit`` are unaffected. The plant orchestrates; the model supplies
the tracked-unit list, the initial temperatures, and the state derivative.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp

# Reserved control-signal key under which the plant threads a unit's operating
# temperature (its lagged tank state, under a heat-balance model) into the unit's
# ``rhs``. A reactor reads it for its kinetics/aeration, falling back to the
# flow-weighted inlet temperature when absent (the algebraic default / standalone
# use). Lives here so both ``plant`` and the units can import it without a cycle.
OPERATING_T_SIGNAL = "__operating_T__"

# Default reference temperature (K) for a tracked unit that declares no operating
# temperature of its own (e.g. a clarifier). 15 degC is the BSM2 AS reference.
_DEFAULT_REF_T_K = 288.15


def _unit_reference_T(unit) -> float:
    """The unit's declared static operating temperature (K), else the default.

    Reactors carry a ``conditions`` dict with a ``"T"`` entry; non-reactive
    volume units (clarifiers, settlers) need not, so fall back to the AS
    reference. Only used to seed the temperature state -- the heat balance
    relaxes it to the resolved inlet temperature within ``V/Q`` hours.
    """
    conditions = getattr(unit, "conditions", None) or {}
    T = conditions.get("T") if hasattr(conditions, "get") else None
    if T is None:
        return _DEFAULT_REF_T_K
    return float(jnp.asarray(T).reshape(-1)[0])


class TemperatureModel(ABC):
    """Strategy for how a plant resolves unit temperatures.

    Concrete models declare which units carry a temperature state, the initial
    values, and the state derivative. The base class derives ``state_size`` from
    :meth:`tracked_units` so a subclass need only implement the three methods.
    """

    @abstractmethod
    def tracked_units(self, plant) -> list[str]:
        """Names of units that carry a dynamic temperature state, in unit order."""

    def state_size(self, plant) -> int:
        """Length of the appended temperature-state block."""
        return len(self.tracked_units(plant))

    @abstractmethod
    def initial_state(self, plant) -> jnp.ndarray:
        """Initial temperatures (K) for the tracked units, shape ``(state_size,)``."""

    @abstractmethod
    def state_rhs(self, plant, temp_state, inlet_by_unit: dict) -> jnp.ndarray:
        """``dT/dt`` for the temperature block, shape ``(state_size,)``.

        Parameters
        ----------
        temp_state : jnp.ndarray
            Current temperatures of the tracked units (in ``tracked_units`` order).
        inlet_by_unit : dict
            ``{unit_name: (Q_in, T_in)}`` -- the resolved total inlet flow and
            flow-weighted inlet temperature of each tracked unit. ``T_in`` is
            ``None`` for a temperature-agnostic inlet (the model then holds that
            state fixed).
        """


class AlgebraicTemperature(TemperatureModel):
    """Instantaneous flow-weighted temperature (the default, historic behaviour).

    Carries no state and overrides nothing -- units self-compute their outlet
    temperature from their inlets exactly as before, so a plant using this model
    is byte-for-byte identical to the pre-``TemperatureModel`` plant.
    """

    def tracked_units(self, plant) -> list[str]:
        return []

    def initial_state(self, plant) -> jnp.ndarray:
        return jnp.zeros((0,))

    def state_rhs(self, plant, temp_state, inlet_by_unit: dict) -> jnp.ndarray:
        return jnp.zeros((0,))


class HeatBalanceTemperature(TemperatureModel):
    """Per-unit dynamic temperature via a completely-mixed first-order heat balance.

    Every finite-volume liquid unit (one exposing a positive ``volume``) that is
    not temperature-fixed (the heated digester sets ``temperature_fixed = True``)
    carries its temperature as a state with ``V dT/dt = Q_in (T_in - T)``. The
    unit's reactor kinetics then run at this lagged tank temperature rather than
    the instantaneous inlet temperature, and its outlet stream leaves at it.
    """

    def tracked_units(self, plant) -> list[str]:
        order = getattr(plant, "_unit_order", None) or list(plant.units)
        tracked = []
        for name in order:
            unit = plant.units[name]
            volume = getattr(unit, "volume", None)
            if volume is None or float(volume) <= 0.0:
                continue
            if getattr(unit, "temperature_fixed", False):
                continue
            tracked.append(name)
        return tracked

    def initial_state(self, plant) -> jnp.ndarray:
        names = self.tracked_units(plant)
        if not names:
            return jnp.zeros((0,))
        return jnp.asarray([_unit_reference_T(plant.units[n]) for n in names],
                           dtype=float)

    def state_rhs(self, plant, temp_state, inlet_by_unit: dict) -> jnp.ndarray:
        # The tracked-unit list and their volumes are fixed for the plant's
        # lifetime, so read the values the plant precomputed at layout time
        # rather than rebuilding tracked_units / re-reading float(volume) here.
        names = plant._temperature_units
        if not names:
            return jnp.zeros((0,))
        volumes = plant._temperature_volumes
        deriv = []
        for i, name in enumerate(names):
            Q_in, T_in = inlet_by_unit.get(name, (None, None))
            T = temp_state[i]
            if T_in is None or Q_in is None:
                # Temperature-agnostic inlet: hold the state fixed.
                deriv.append(jnp.zeros(()))
                continue
            deriv.append((Q_in / volumes[i]) * (T_in - T))
        return jnp.stack(deriv)
