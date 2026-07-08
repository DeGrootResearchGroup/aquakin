"""Time schedules for plant operating setpoints.

A :class:`PiecewiseConstantSchedule` gives a setpoint that steps between fixed
values at scheduled times -- e.g. the BSM2 wastage-flow timer, which alternates
the waste pump between a low and a high rate over the year to manage the sludge
inventory. The evaluation is ``jit`` / AD-safe (a ``searchsorted`` gather, no
data-dependent control flow), so it can drive a flow inside the monolithic plant
solve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass(frozen=True)
class PiecewiseConstantSchedule:
    """A setpoint that holds constant between scheduled step times.

    ``values[i]`` applies on the interval ``[t_breaks[i-1], t_breaks[i])``; the
    first value applies before ``t_breaks[0]`` and the last from ``t_breaks[-1]``
    onward, so ``len(values) == len(t_breaks) + 1``.

    Parameters
    ----------
    t_breaks : sequence of float
        Strictly increasing step times.
    values : sequence of float
        The held values; one more than ``t_breaks``.

    Examples
    --------
    >>> # 300 before day 182, 450 from day 182 to 364, 300 after.
    >>> s = PiecewiseConstantSchedule([182.0, 364.0], [300.0, 450.0, 300.0])
    >>> float(s.at(100.0)), float(s.at(200.0)), float(s.at(400.0))
    (300.0, 450.0, 300.0)
    """

    t_breaks: Sequence[float]
    values: Sequence[float]

    # Converted JAX arrays for the AD-safe ``at`` gather, derived from the public
    # sequences at construction. Declared (not stashed) so the value object stays
    # frozen and the shadow fields are part of the dataclass contract.
    _t_breaks: jnp.ndarray = field(init=False, repr=False, compare=False)
    _values: jnp.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_t_breaks", jnp.asarray(self.t_breaks, dtype=float))
        object.__setattr__(self, "_values", jnp.asarray(self.values, dtype=float))
        if self._values.shape[0] != self._t_breaks.shape[0] + 1:
            raise ValueError(
                "PiecewiseConstantSchedule: len(values) must be len(t_breaks)+1; "
                f"got {self._values.shape[0]} and {self._t_breaks.shape[0]}."
            )
        if self._t_breaks.shape[0] and bool(jnp.any(jnp.diff(self._t_breaks) <= 0)):
            raise ValueError("PiecewiseConstantSchedule: t_breaks must be strictly increasing.")

    def shifted(self, delta: float) -> PiecewiseConstantSchedule:
        """A copy with every held value offset by ``delta`` (same step times)."""
        return PiecewiseConstantSchedule(self._t_breaks, self._values + float(delta))

    @property
    def min_value(self) -> float:
        return float(jnp.min(self._values))

    @property
    def max_value(self) -> float:
        return float(jnp.max(self._values))

    def at(self, t: jnp.ndarray) -> jnp.ndarray:
        """The scheduled value at time ``t`` (scalar in, scalar out)."""
        idx = jnp.searchsorted(self._t_breaks, jnp.asarray(t), side="right")
        return self._values[idx]
