"""Applying an ``observation_fn`` to a trajectory, shared by calibration and the
Fisher-information path."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp


def observation_takes_time(observation_fn) -> bool:
    """Whether ``observation_fn`` has the time-aware signature
    ``(t, C, params)`` rather than ``(C, params)``.

    The rule, on the positional parameters of the signature (those that can be
    filled by position), is:

    - exactly three of them *required* (no default): the time-aware form
      ``observation_fn(t, C, params)``. A time-aware function must therefore
      require all of ``t``, ``C`` and ``params``; further parameters may have
      defaults.
    - at most two required and at least two in all: the ``(C, params)`` form,
      called as ``observation_fn(C, params)`` (so ``f(C, p=None)`` and
      ``f(C, p, scale=2.0)`` are this form).
    - a ``*args`` parameter, or a signature that cannot be inspected: the
      ``(C, params)`` form.

    Raises
    ------
    TypeError
        If ``observation_fn`` is not callable, or its signature fits neither
        form (fewer than two positional parameters, or more than three
        required).
    """
    if not callable(observation_fn):
        raise TypeError(
            "observation_fn must be a callable (C, params) -> y or (t, C, params) -> y."
        )
    try:
        sig = inspect.signature(observation_fn)
    except (TypeError, ValueError):
        return False  # no introspectable signature: the (C, params) form
    params = list(sig.parameters.values())
    if any(q.kind is inspect.Parameter.VAR_POSITIONAL for q in params):
        return False
    positional = [
        q
        for q in params
        if q.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    n_req = sum(1 for q in positional if q.default is inspect.Parameter.empty)
    if n_req == 3:
        return True
    if n_req <= 2 and len(positional) >= 2:
        return False
    raise TypeError(
        "observation_fn must take (C, params), or require (t, C, params), as "
        f"positional arguments; got a signature {sig}."
    )


def observe(ys, p, species_idx, observation_fn=None, t=None):
    """Observed outputs of a ``(n_t, n_species)`` trajectory: the
    ``species_idx`` columns, or ``observation_fn`` applied at each time --
    ``observation_fn(C, p)``, or ``observation_fn(t, C, p)`` with the sample
    times ``t`` ``(n_t,)`` for a time-aware function."""
    if observation_fn is None:
        return ys[:, species_idx]
    if observation_takes_time(observation_fn):
        if t is None:
            raise ValueError("A time-aware observation_fn(t, C, params) needs the sample times.")
        return jax.vmap(lambda tt, c: jnp.atleast_1d(observation_fn(tt, c, p)))(
            jnp.asarray(t, dtype=float), ys
        )
    return jax.vmap(lambda c: jnp.atleast_1d(observation_fn(c, p)))(ys)


def observation_size(observation_fn, C0, p, t0: float = 0.0) -> int:
    """Number of outputs of ``observation_fn``, checked to be a 1-D vector
    (or a scalar) on the model's state and parameter vectors (and the time
    ``t0`` for a time-aware function)."""
    C0 = jnp.asarray(C0, dtype=float)
    if observation_takes_time(observation_fn):
        y = jnp.asarray(observation_fn(jnp.asarray(float(t0)), C0, jnp.asarray(p)))
    else:
        y = jnp.asarray(observation_fn(C0, jnp.asarray(p)))
    if y.ndim > 1:
        raise ValueError(
            f"observation_fn must return a scalar or 1-D vector per time point; got shape {y.shape}."
        )
    return int(y.size)
