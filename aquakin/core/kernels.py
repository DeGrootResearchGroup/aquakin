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

These are public: a rate function a user registers (a subclass of
:class:`aquakin.RateFunction`) gets none of the built-in guards, and uses these
in its ``op`` wherever its arithmetic can reach a singular derivative.

``_safe_ratio`` deliberately stays in ``core/nodes.py``. It guards the Monod
family and ``safe_div``, which exist only in the rate language, so it is not
shared and moving it would churn those nodes for no second consumer.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

__all__ = ["safe_log", "safe_pow", "safe_sqrt"]


def safe_log(x):
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


def safe_sqrt(x):
    """``sqrt(x)`` whose gradient stays finite at zero.

    ``d/dx sqrt(x) = 1/(2 sqrt(x))`` is infinite at ``x == 0``. That is not a
    corner case in either language: a half-order kinetic term sits there
    whenever its species depletes, and an expression in the plant clock sits
    there at the first step of *every* solve. The double-where keeps the value
    exact (``sqrt(0) == 0``) while the derivative at and below zero is 0 rather
    than ``inf``; a transiently negative argument returns 0 rather than NaN.

    :func:`safe_pow` guards the same singularity reached through the ``**``
    operator, so ``sqrt([X])`` and ``[X] ** 0.5`` agree. (The ``powc`` kernel in
    ``core/vector_kernel.py`` does *not* help here: it keeps a constant exponent
    static so the generic ``pow`` JVP's ``base**exp * log(base)`` term is pruned,
    which is a different NaN and leaves the base derivative infinite.)

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


def safe_pow(base, exp):
    """``base ** exp`` whose gradient stays finite where the exponent makes it
    infinite.

    ``d/dx x**e = e * x**(e - 1)`` is infinite at ``x == 0`` whenever ``e < 1``
    -- so ``[X] ** 0.5`` carries exactly the singularity :func:`safe_sqrt`
    exists for, and the two spellings of half-order kinetics have to agree.

    **Every value is unchanged.** The guard fires only where the base is exactly
    zero *and* the exponent is below one, and returns ``0.0 ** exp`` there,
    which is what the unguarded expression already evaluated to: ``0`` for a
    positive exponent, ``1`` for a zero one, ``inf`` for a negative one. What
    changes is the derivative at that point, from infinite to 0 -- the same
    compromise :func:`safe_sqrt` makes. ``x ** 1`` therefore keeps its
    derivative of 1 and ``x ** 2`` is untouched: neither is singular, so neither
    is guarded.

    A **negative** base with a fractional exponent still yields NaN, as it must.
    It has no real value, and quietly turning it into a number would hide a
    modelling error rather than report it.

    A **calibrated** exponent is covered too, by the same branch: with the
    exponent traced, ``x ** p`` at ``x == 0`` and ``p == 0.5`` gives a NaN
    tangent unguarded and 0 here. Above one it is finite either way, since JAX's
    own ``pow`` JVP already guards the ``log(base)`` term (measured on JAX
    0.10.2; ``core/vector_kernel.py``'s ``powc``, which keeps a constant
    exponent static, was written against a version that did not).
    """
    base = jnp.asarray(base)
    exp = jnp.asarray(exp)
    singular = (base == 0.0) & (exp < 1.0)
    safe = jnp.where(singular, 1.0, base)
    # The singular branch is a limit value and carries no gradient by design, so
    # it is computed off the tape. That keeps the value *exactly* what the
    # unguarded expression gave -- including the sign of a signed zero, where
    # (-0.0) ** -1 is -inf and not inf -- without the infinite base derivative
    # or, for a traced exponent, the 0 * -inf of the log term.
    limit = lax.stop_gradient(base) ** lax.stop_gradient(exp)
    return jnp.where(singular, limit, safe**exp)
