"""Differentiable flow setpoints (RAS / recycle / wastage pumps, clarifier splits).

A flow setpoint -- a recycle or wastage pump flow, a clarifier underflow, a
primary-sludge fraction -- is consumed in **two** plant code paths: the recycle
flow-network resolution (:meth:`Plant._resolve_flows` -> ``unit.flow_outputs``)
and the material-stream sweep (:meth:`Plant._sweep_outputs` ->
``unit.compute_outputs``). Reading a raw ``float`` in each place duplicates the
value and risks the two desyncing. :class:`FlowSetpoint` is the single source of
truth: both paths call ``resolve(flow_params)`` on the same object.

Because it resolves its value from the unit's slice of the plant parameter
vector (when the unit runs inside a plant), the setpoint is a first-class
*parameter* -- :meth:`Plant.steady_state` and dynamic solves are differentiable
w.r.t. it (the implicit-function-theorem / autodiff sees it), so SRT and
recycle-ratio design sweeps work. Used standalone (no plant), ``resolve`` returns
the fixed default, so a unit constructed on its own behaves exactly as a plain
float setpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class FlowSetpoint:
    """One flow setpoint, resolvable from the plant parameter vector or a default.

    Attributes
    ----------
    default : float
        The setpoint value used when the unit runs outside a plant (or before
        the flow-parameter block is built) -- the historic fixed-pump behaviour.
    local_pos : int
        Position of this setpoint within the unit's flow-parameter slice (the
        tail of ``params_unit`` after the kinetic block). Assigned by the unit at
        construction in a fixed order.
    """

    default: float
    local_pos: int

    def resolve(self, flow_params: jnp.ndarray) -> jnp.ndarray:
        """The live setpoint: the plant parameter when present, else the default.

        Parameters
        ----------
        flow_params : jnp.ndarray
            The unit's flow-parameter slice (the tail of ``params_unit`` after
            its kinetic block). Empty when the unit is used outside a plant; the
            ``shape`` test is static, so this is jit/grad safe.
        """
        if flow_params.shape[0] > self.local_pos:
            return flow_params[self.local_pos]
        return jnp.asarray(self.default)


class FlowParameterized:
    """Mixin for units carrying differentiable :class:`FlowSetpoint` s.

    A unit lists its setpoints (in a fixed order) from :meth:`_flow_setpoints`;
    the plant reads their defaults via :meth:`flow_param_defaults` to size and
    seed a flow-parameter block, and the unit reads the live values from its
    slice of ``params_unit`` via :meth:`_flow_params`.
    """

    def _flow_setpoints(self) -> "dict[str, FlowSetpoint]":
        """Ordered ``name -> FlowSetpoint`` for this unit (override per unit)."""
        return {}

    def _flow_params(self, params_unit) -> jnp.ndarray:
        """The unit's flow-parameter slice: ``params_unit`` after the kinetic block.

        Returns an empty array when ``params_unit`` is ``None`` (a unit exercised
        standalone with no parameters), so every setpoint resolves to its default.
        """
        if params_unit is None:
            return jnp.zeros((0,))
        net = getattr(self, "network", None)
        n_kinetic = net.n_params if net is not None else 0
        return params_unit[n_kinetic:]

    def flow_param_defaults(self) -> list[float]:
        """Ordered default setpoint values (the unit's flow-parameter block)."""
        return [sp.default for sp in self._flow_setpoints().values()]

    def flow_param_local_names(self) -> list[str]:
        """Ordered setpoint names, used to address the flow parameters by name."""
        return list(self._flow_setpoints().keys())
