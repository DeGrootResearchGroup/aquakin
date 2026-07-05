"""Parameter transforms shared by calibration and sensitivity screening.

A fitted or screened parameter lives in two spaces: the **physical** space it is
declared in (a positive rate, a fraction in ``(0, 1)``, an unconstrained value)
and the **unconstrained** space where its prior is Gaussian and the optimiser /
DGSM operates. These helpers map between them and provide the chain-rule factor
that carries a sensitivity from one space to the other.

Three transform kinds are supported, matching the schema's ``transform:`` field:

- ``"positive_log"`` -- ``physical = exp(u)`` (a positive quantity; prior Gaussian
  in log space).
- ``"logit"`` -- ``physical = sigmoid(u)`` (a fraction in ``(0, 1)``; prior
  Gaussian in log-odds space).
- ``"none"`` -- identity (already unconstrained).

Every function is backend-agnostic through ``xp``: pass ``jax.numpy`` (the
default) on the differentiable calibration path, or ``numpy`` for the host-side
DGSM sampling loop. The results are identical to machine precision either way.
"""

from __future__ import annotations

import jax.numpy as jnp

VALID_TRANSFORMS = ("none", "positive_log", "logit")


def _check(transform: str) -> None:
    if transform not in VALID_TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; choose from {VALID_TRANSFORMS}.")


def to_unconstrained(value, transform: str, xp=jnp):
    """Physical parameter -> unconstrained space (where its prior is Gaussian)."""
    _check(transform)
    if transform == "positive_log":
        return xp.log(value)
    if transform == "logit":
        return xp.log(value / (1.0 - value))
    return value  # none


def from_unconstrained(u, transform: str, xp=jnp):
    """Inverse of :func:`to_unconstrained`: unconstrained value -> physical."""
    _check(transform)
    if transform == "positive_log":
        return xp.exp(u)
    if transform == "logit":
        return 1.0 / (1.0 + xp.exp(-u))
    return u  # none


def dphysical_dunconstrained(physical, transform: str, xp=jnp):
    """``d(physical)/d(unconstrained)`` expressed through the *physical* value.

    This is the chain-rule factor that converts a physical sensitivity
    ``dg/d(physical)`` into the unconstrained-space ``dg/du`` (the DGSM screen),
    and the delta-method Jacobian mapping an unconstrained-space covariance back
    to physical std devs (the Laplace posterior). Given ``physical`` rather than
    the unconstrained value it saves recomputing the forward map:

    - ``positive_log`` -- ``d(e^u)/du = e^u = physical``.
    - ``logit`` -- ``sigmoid'(u) = s (1 - s) = physical (1 - physical)``.
    - ``none`` -- ``1``.
    """
    _check(transform)
    if transform == "positive_log":
        return physical
    if transform == "logit":
        return physical * (1.0 - physical)
    return xp.ones_like(physical)  # none
