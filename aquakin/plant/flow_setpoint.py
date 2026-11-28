"""Differentiable flow setpoints (RAS / recycle / wastage pumps, clarifier splits).

A flow setpoint -- a recycle or wastage pump flow, a clarifier underflow, a
primary-sludge fraction -- is consumed in **two** plant code paths: the recycle
flow-model resolution (:meth:`Plant._resolve_flows` -> ``unit.flow_outputs``)
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

A setpoint may instead be written as an expression of the plant clock
(``"385 + 75*sin(2*pi*t)"``, see :mod:`aquakin.core.time_expr`): an
:class:`ExpressionSetpoint`, resolved through the same ``resolve(flow_params, t)``
call. It takes no slot in the parameter vector, and its value is clipped into
the setpoint's range (a flow to ``>= 0``, a fraction to ``[0, 1]``) as it is
evaluated, since no check made when the unit is built can bound an arbitrary
expression over a whole run. :func:`build_setpoints` builds a unit's setpoints
of either kind and numbers the parameter slots, so every unit does it one way.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax.numpy as jnp

from aquakin.core.time_expr import compile_time_expression, is_time_expression

#: The range a flow setpoint is clipped into when it is an expression.
FLOW_RANGE = (0.0, math.inf)
#: The range a fractional setpoint (a primary-sludge fraction) is clipped into.
FRACTION_RANGE = (0.0, 1.0)


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

    def resolve(self, flow_params: jnp.ndarray, t=None) -> jnp.ndarray:
        """The live setpoint: the plant parameter when present, else the default.

        Parameters
        ----------
        flow_params : jnp.ndarray
            The unit's flow-parameter slice (the tail of ``params_unit`` after
            its kinetic block). Empty when the unit is used outside a plant; the
            ``shape`` test is static, so this is jit/grad safe.
        t : jnp.ndarray, optional
            The time. A constant setpoint does not read it; it is taken so that
            every setpoint resolves through one call (see
            :class:`ExpressionSetpoint`).

        Returns
        -------
        jnp.ndarray
            The setpoint, a scalar.
        """
        if flow_params.shape[0] > self.local_pos:
            return flow_params[self.local_pos]
        return jnp.asarray(self.default)

    def initial(self) -> float:
        """The setpoint's value at the start of a run: its default."""
        return self.default


@dataclass(frozen=True)
class ExpressionSetpoint:
    """A setpoint written as an expression of the plant clock.

    Attributes
    ----------
    expression : str
        The expression of ``t`` (see :mod:`aquakin.core.time_expr`), on the
        plant's clock: ``t`` in the plant's time unit and a flow in m³ per that
        unit, never converted.
    value_range : tuple of float
        ``(lower, upper)``: the value is clipped into this range as it is
        evaluated (:data:`FLOW_RANGE` or :data:`FRACTION_RANGE`).
    what : str
        What the setpoint is, for error messages (``"IdealClarifier 'c':
        'underflow_Q'"``).

    Raises
    ------
    ValueError
        If the expression cannot be read, or its value at ``t = 0`` is not finite.
    """

    expression: str
    value_range: tuple[float, float] = FLOW_RANGE
    what: str = "setpoint"
    _fn: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fn = compile_time_expression(self.expression, what=self.what)
        object.__setattr__(self, "_fn", fn)
        if not math.isfinite(float(fn(jnp.asarray(0.0)))):
            raise ValueError(f"{self.what}: {self.expression!r} is not finite at t = 0")

    def resolve(self, flow_params: jnp.ndarray, t) -> jnp.ndarray:
        """The setpoint at time ``t``, clipped into its range.

        Parameters
        ----------
        flow_params : jnp.ndarray
            The unit's flow-parameter slice; not read (an expression has no
            parameter slot), taken so every setpoint resolves through one call.
        t : jnp.ndarray or None
            The time on the plant's clock.

        Returns
        -------
        jnp.ndarray
            The setpoint, a scalar.

        Raises
        ------
        ValueError
            If ``t`` is ``None`` (a unit used outside a plant, with no clock).
        """
        if t is None:
            raise ValueError(
                f"{self.what} is an expression of time, so it needs the time "
                "(pass a FlowContext with t to flow_outputs)"
            )
        lower, upper = self.value_range
        return jnp.clip(jnp.asarray(self._fn(t), dtype=float), lower, upper)

    def initial(self) -> float:
        """The setpoint's value at ``t = 0``, clipped into its range."""
        return float(self.resolve(jnp.zeros((0,)), jnp.asarray(0.0)))


#: A unit's setpoint of either kind.
Setpoint = FlowSetpoint | ExpressionSetpoint


def build_setpoints(
    values: Mapping[str, float | str], *, owner: str, value_range=FLOW_RANGE
) -> dict[str, Setpoint]:
    """A unit's setpoints from their values, in order.

    A number becomes a :class:`FlowSetpoint` (a slot in the unit's
    flow-parameter block, numbered in order over the numbers alone); an
    expression of time becomes an :class:`ExpressionSetpoint`, which has no
    slot.

    Parameters
    ----------
    values : mapping of str to float or str
        Setpoint name to its value, in the unit's fixed order.
    owner : str
        The unit, for error messages (``"IdealClarifier 'clarifier'"``).
    value_range : tuple of float, optional
        The range an expression's value is clipped into (default
        :data:`FLOW_RANGE`).

    Returns
    -------
    dict of str to Setpoint
        In the order of ``values``.

    Raises
    ------
    ValueError
        If an expression cannot be read, or is not finite at ``t = 0``.
    """
    out: dict[str, Setpoint] = {}
    pos = 0
    for name, value in values.items():
        if is_time_expression(value):
            out[name] = ExpressionSetpoint(
                str(value), value_range=value_range, what=f"{owner}: '{name}'"
            )
        else:
            out[name] = FlowSetpoint(float(value), pos)
            pos += 1
    return out


class FlowParameterized:
    """Mixin for units carrying differentiable :class:`FlowSetpoint` s.

    A unit lists its setpoints (in a fixed order) from :meth:`_flow_setpoints`;
    the plant reads their defaults via :meth:`flow_param_defaults` to size and
    seed a flow-parameter block, and the unit reads the live values from its
    slice of ``params_unit`` via :meth:`_flow_params`.
    """

    def _flow_setpoints(self) -> dict[str, Setpoint]:
        """Ordered ``name -> setpoint`` for this unit, of either kind (override per unit)."""
        return {}

    def _parameter_setpoints(self) -> dict[str, FlowSetpoint]:
        """The setpoints that are parameters (the numbers), in slot order."""
        return {n: sp for n, sp in self._flow_setpoints().items() if isinstance(sp, FlowSetpoint)}

    def _flow_params(self, params_unit) -> jnp.ndarray:
        """The unit's flow-parameter slice: ``params_unit`` after the kinetic block.

        Returns an empty array when ``params_unit`` is ``None`` (a unit exercised
        standalone with no parameters), so every setpoint resolves to its default.
        """
        if params_unit is None:
            return jnp.zeros((0,))
        net = getattr(self, "model", None)
        n_kinetic = net.n_params if net is not None else 0
        return params_unit[n_kinetic:]

    def flow_param_defaults(self) -> list[float]:
        """Ordered default setpoint values (the unit's flow-parameter block)."""
        return [sp.default for sp in self._parameter_setpoints().values()]

    def flow_param_local_names(self) -> list[str]:
        """Ordered setpoint names, used to address the flow parameters by name."""
        return list(self._parameter_setpoints().keys())

    def scheduled_setpoints(self) -> tuple[str, ...]:
        """The names of the setpoints written as expressions of time.

        A plant with any re-probes its recycle maps on every evaluation (see
        :class:`~aquakin.plant.units.ScheduledSetpointUnit`).
        """
        return tuple(
            n for n, sp in self._flow_setpoints().items() if isinstance(sp, ExpressionSetpoint)
        )
