"""Per-component absolute tolerance for the adaptive solvers.

The noise floor lives in ``core`` because both the model (``CompiledModel.atol``)
and every integrator build it; ``aquakin.integrate._common`` re-exports it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def default_atol(
    scale_like, reference=None, *, atol_factor: float = 1e-6, floor_frac: float = 1e-6
):
    """Per-component absolute tolerance scaled off the states' operating magnitudes.

    The error test every adaptive solver uses weights each component by
    ``atol_i + rtol*|y_i|``; ``atol_i`` is the **noise floor** below which
    component ``i`` is treated as negligible. When components span very different
    scales (e.g. ADM1 from ~1e-13 to ~17, or an OH radical at ~1e-12 beside a
    bulk reactant at ~1e-4) a single scalar floor is wrong for most of them, so
    the floor is set **per component** -- the SUNDIALS "vector atol" guidance and
    Hairer & Wanner's rule of ``atol_i`` proportional to the typical magnitude of
    component ``i``.

    Returns ``atol_i = atol_factor * max(|scale_i|, |reference_i|, floor_frac*char)``,
    where ``char = max_j(typical_j)`` is the system's bulk magnitude (so a
    component whose typical value is ~0, e.g. a product not present initially,
    gets a small floor tied to the system scale rather than ``atol_i = 0``, which
    the solver literature explicitly warns against).

    Parameters
    ----------
    scale_like : array
        A representative state vector -- typically the initial condition ``C0`` /
        ``y0`` (the operating magnitudes).
    reference : array, optional
        A second magnitude source merged in via elementwise max -- typically the
        model's ``default_concentrations`` (the YAML reference values), so a
        component that starts at 0 but has a nonzero reference is still floored
        sensibly.
    atol_factor : float
        Fraction of each component's typical magnitude used as its noise floor.
    floor_frac : float
        Floor for near-zero components, as a fraction of the bulk scale ``char``.

    Returns
    -------
    jnp.ndarray
        Per-component absolute tolerance, same shape as ``scale_like``.
    """
    typ = jnp.abs(jnp.asarray(scale_like, dtype=float))
    if reference is not None:
        typ = jnp.maximum(typ, jnp.abs(jnp.asarray(reference, dtype=float)))
    char = jnp.max(typ)
    # When every magnitude is zero there is no scale to floor against, so the
    # near-zero floor floor_frac*char would itself be 0 -- returning atol_i = 0,
    # which the solver literature warns against (the invariant this floor exists
    # to uphold). Fall back to unit scale in that case. Identity for any input
    # with a nonzero magnitude (the common path), since char > 0 there.
    char = jnp.where(char > 0.0, char, 1.0)
    typ = jnp.maximum(typ, floor_frac * char)
    # An absolute tolerance is a solver noise floor, never a differentiated
    # quantity, so detach it from autodiff. This matters when ``scale_like`` is the
    # initial state ``y0`` and the gradient is taken *with respect to* ``y0``: a
    # traced atol would otherwise be baked into the integrator's step controller
    # and -- inside the discrete-adjoint (``gradient="stable_adjoint"``) custom-VJP
    # forward, which re-runs ``diffrax.diffeqsolve`` -- escape that inner solve as
    # a leaked tracer (issue #420). ``stop_gradient`` collapses the ``y0``-derived
    # tracer to its concrete primal; it is the identity for a concrete
    # ``scale_like`` (the common path), so every existing solve is unchanged.
    return jax.lax.stop_gradient(atol_factor * typ)
