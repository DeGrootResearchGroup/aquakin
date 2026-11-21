"""Guarded numerics shared between aquakin's expression languages.

A function whose derivative is non-finite somewhere on its domain needs the
same guard wherever it is evaluated. The rate language (``core/nodes.py``,
reached through ``core/parser.py``) and the plant's time-expression language
both offer ``sqrt`` and ``log``, and both are evaluated inside a jitted,
differentiated right-hand side -- so the guard belongs in one place rather
than once per language, or the same spelling means two different things
depending on where it was written.

Each guard is a **double-where**: the live branch is exact, and off the domain
the expression is evaluated at a finite stand-in so the masked branch
contributes a zero tangent rather than a NaN one. The naive
``where(ok, f(x), fallback)`` is not enough on its own -- reverse-mode AD still
back-propagates through the unused branch, so the *argument* has to be guarded
too.

What each returns off its domain is a modelling decision, not a mechanical
one, and the two differ deliberately: see each function.

``_safe_ratio`` deliberately stays in ``core/nodes.py``. It guards the Monod
family and ``safe_div``, which exist only in the rate language, so it is not
shared and moving it would churn those nodes for no second consumer.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["_safe_log", "_safe_sqrt"]


def _safe_log(x):
    """``log(x)`` whose *gradient* stays finite at and below zero.

    ``d/dx log(x) = 1/x`` is infinite at ``x == 0`` and the value is NaN below
    it, so a reverse-mode pass through a depleted state poisons the whole
    parameter gradient. The double-where evaluates the logarithm at a finite
    stand-in off the domain, so the live branch is exact and the masked branch
    contributes a zero -- not a NaN -- tangent.

    The *value* off the domain is ``-inf``, not a finite stand-in: a logarithm
    has no finite limit at zero to adopt, and inventing one would silently
    change the model. The guard therefore protects the gradient only, and the
    caller is responsible for keeping the argument positive. In a rate
    expression that means ``clip_negative_states: true``, or writing the term
    so the argument cannot reach zero (``log(1 + [X]/K)``); on a clock, where
    every solve starts at ``t == 0``, it means ``log(1 + t)`` rather than
    ``log(t)``. A ``-inf`` value fails the forward solve, which no gradient
    guard can or should hide.
    """
    positive = x > 0.0
    safe = jnp.where(positive, x, 1.0)
    return jnp.where(positive, jnp.log(safe), -jnp.inf)


def _safe_sqrt(x):
    """``sqrt(x)`` whose gradient stays finite at zero.

    ``d/dx sqrt(x) = 1/(2 sqrt(x))`` is infinite at ``x == 0``. That is not a
    corner case in either language: a half-order kinetic term sits there
    whenever its species depletes, and an expression in the plant clock sits
    there at the first step of *every* solve. The double-where keeps the value
    exact (``sqrt(0) == 0``) while the derivative at and below zero is 0 rather
    than ``inf``; a transiently negative argument returns 0 rather than NaN.

    Same reasoning as the constant-exponent ``powc`` kernel, which keeps
    ``[X] ** 0.5`` differentiable at ``X == 0``.

    The failure this prevents is quiet. An unguarded ``sqrt`` leaves the
    forward solve correct -- the trajectory is unchanged -- and breaks only the
    gradient, where ``0 * inf = nan`` in the backward linearization diverges
    the adjoint instead of returning NaN. The error surfaces as an exhausted
    solver step budget, which points at the integrator rather than at the
    expression that caused it.
    """
    positive = x > 0.0
    safe = jnp.where(positive, x, 1.0)
    return jnp.where(positive, jnp.sqrt(safe), 0.0)
