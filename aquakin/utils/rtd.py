"""Residence-time distribution (RTD) analysis on tracer-response data.

All functions accept paired ``(t, C)`` arrays from a tracer impulse experiment
(or simulated equivalent), where ``C`` is the outlet tracer concentration as a
function of time ``t``. Inputs may be NumPy or JAX arrays; outputs are JAX
arrays for compatibility with the rest of the library.
"""

from __future__ import annotations

import jax.numpy as jnp


def _ensure_1d(t, C):
    t = jnp.asarray(t)
    C = jnp.asarray(C)
    if t.ndim != 1 or C.ndim != 1:
        raise ValueError(
            f"t and C must be 1-D, got shapes {t.shape} and {C.shape}"
        )
    if t.shape != C.shape:
        raise ValueError(f"t and C must have the same length, got {t.shape} vs {C.shape}")
    return t, C


def E_curve(t, C) -> jnp.ndarray:
    """
    Normalised residence time distribution ``E(t) = C(t) / integral(C dt)``.

    Parameters
    ----------
    t : array-like, shape (n,)
        Sample times (ascending).
    C : array-like, shape (n,)
        Tracer concentration at each time. Must integrate to a positive
        value over the interval; an all-zero (or net-negative) tracer raises
        ``ValueError``.

    Returns
    -------
    jnp.ndarray, shape (n,)
        ``E(t)`` with units of inverse time, normalised so that
        ``integral(E dt) = 1``.
    """
    t, C = _ensure_1d(t, C)
    total = jnp.trapezoid(C, t)
    if not float(total) > 0:
        raise ValueError(
            f"Tracer response integrates to {float(total):g}; E_curve requires "
            f"a positive integral."
        )
    return C / total


def F_curve(t, C) -> jnp.ndarray:
    """
    Cumulative residence time distribution ``F(t) = integral_0^t E(s) ds``.

    Computed by trapezoidal cumulative integration of ``E``. The first entry
    is ``0`` and the last entry is ``1`` (up to numerical error).
    """
    t, C = _ensure_1d(t, C)
    E = E_curve(t, C)
    increments = 0.5 * (E[1:] + E[:-1]) * jnp.diff(t)
    return jnp.concatenate([jnp.zeros(1, dtype=increments.dtype), jnp.cumsum(increments)])


def mean_residence_time(t, C) -> jnp.ndarray:
    """First moment of the RTD: ``<t> = integral(t E(t) dt)``."""
    t, C = _ensure_1d(t, C)
    E = E_curve(t, C)
    return jnp.trapezoid(t * E, t)


def variance(t, C) -> jnp.ndarray:
    """Second central moment of the RTD: ``sigma^2 = integral((t-<t>)^2 E dt)``."""
    t, C = _ensure_1d(t, C)
    E = E_curve(t, C)
    mean = jnp.trapezoid(t * E, t)
    return jnp.trapezoid((t - mean) ** 2 * E, t)


def percentile_time(t, C, q: float) -> jnp.ndarray:
    """
    Time at which the cumulative RTD reaches ``q``.

    Parameters
    ----------
    t, C : array-like
        Tracer response.
    q : float
        Fraction in ``[0, 1]``. ``q = 0.1`` gives the 10th-percentile
        residence time (``t10``).

    Returns
    -------
    jnp.ndarray (scalar)
        The time at which ``F(t) = q``, by linear interpolation of the
        strictly-increasing portion of ``F``. Plateaus where ``E = 0`` are
        masked out before interpolation so a constant-zero tail does not
        bias the result toward the array tail.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q}")
    t, _ = _ensure_1d(t, C)
    F = F_curve(t, C)
    # Drop runs where F is flat (E=0): keep the first occurrence of each
    # plateau so interp sees a strictly-increasing-by-points domain.
    keep = jnp.concatenate([jnp.asarray([True]), jnp.diff(F) > 0])
    return jnp.interp(jnp.asarray(q), F[keep], t[keep])


def morrill_index(t, C) -> jnp.ndarray:
    """
    Morrill dispersion index ``t90 / t10``.

    A perfect plug-flow reactor has Morrill = 1; a completely mixed reactor
    (CSTR) has Morrill = ln(0.1) / ln(0.9) ~= 21.85. Drinking-water
    disinfection guidelines often target Morrill <= 2 for "near plug flow".
    """
    return percentile_time(t, C, 0.9) / percentile_time(t, C, 0.1)
