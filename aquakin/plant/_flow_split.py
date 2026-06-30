"""Shared controlled-split flow logic for the secondary clarifiers.

A clarifier (``IdealClarifier``, ``TakacsClarifier``) splits one inlet into an
overflow (clarified effluent) and an underflow (thickened RAS + wastage). One of
the two is a *fixed setpoint* (a controlled pump flow) and the other is the free
remainder, so the uncontrolled flow tracks the feed (``Q_e = Q_f - Q_u``, the
BSM convention). Both units shared this logic verbatim; it lives here so they
cannot drift.
"""

from __future__ import annotations

import jax.numpy as jnp


def _min_setpoint(q) -> float:
    """Lower bound of a setpoint that may be a float or a time schedule."""
    return q.min_value if hasattr(q, "min_value") else float(q)


def validate_controlled_split(name: str, overflow_Q, underflow_Q) -> None:
    """Validate a controlled overflow/underflow split at construction.

    Exactly one of ``overflow_Q`` / ``underflow_Q`` is the controlled setpoint;
    the other is the free remainder. Both must be non-negative. A setpoint may
    be a constant or a time schedule (any object exposing ``min_value``); a
    schedule is validated by its minimum.

    Raises
    ------
    ValueError
        If both or neither is given, or a given value is negative.
    """
    if (overflow_Q is None) == (underflow_Q is None):
        raise ValueError(f"{name}: supply exactly one of overflow_Q or underflow_Q.")
    if overflow_Q is not None and _min_setpoint(overflow_Q) < 0:
        raise ValueError(f"{name}: overflow_Q must be non-negative; got {overflow_Q}")
    if underflow_Q is not None and _min_setpoint(underflow_Q) < 0:
        raise ValueError(f"{name}: underflow_Q must be non-negative; got {underflow_Q}")


def split_controlled_flows(overflow_Q, underflow_Q, Q_in: jnp.ndarray, clamp: bool):
    """Return ``(Q_over, Q_under)`` from the inlet flow.

    The *controlled* flow is fixed (``underflow_Q`` for a flow-controlled
    underflow pump, else ``overflow_Q``); the other is the remainder, so the free
    flow tracks the feed -- the BSM convention ``Q_e = Q_f - Q_u``. Each setpoint
    is a scalar already resolved at the current time (the caller evaluates any
    schedule), so it may be a traced value -- hence ``jnp.asarray`` rather than
    ``float``.
    ``clamp=True`` (concentration / settling stage) keeps both flows in
    ``[0, Q_in]`` so a transient feed below the setpoint never makes a
    negative-flow stream; ``clamp=False`` (the linear flow rule) leaves the split
    affine so ``Plant._resolve_flows`` is exact.
    """
    if underflow_Q is not None:
        Q_over = Q_in - jnp.asarray(underflow_Q)
    else:
        Q_over = jnp.asarray(overflow_Q)
    if clamp:
        Q_over = jnp.clip(Q_over, 0.0, Q_in)
    return Q_over, Q_in - Q_over
