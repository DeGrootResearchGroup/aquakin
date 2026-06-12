"""Shared internals for the integrator submodules.

Not part of the public API. Reactors depend on this; this module depends only
on JAX and Diffrax.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, runtime_checkable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from aquakin.core.network import CompiledNetwork


def validate_t_eval(t_eval_arr: jnp.ndarray, t0: float, t1: float) -> None:
    """Validate output times before handing them to ``SaveAt(ts=...)``.

    Diffrax silently returns NaN for save times outside ``[t0, t1]`` or a
    non-ascending sequence, so check here for a clear error instead. Value
    checks run only for concrete (non-traced) ``t_eval``; a traced array
    (e.g. differentiating with respect to the save times) skips them.

    Parameters
    ----------
    t_eval_arr : jnp.ndarray
        Candidate save times.
    t0, t1 : float
        Integration interval bounds.

    Raises
    ------
    ValueError
        If ``t_eval`` is not 1-D, lies outside ``[t0, t1]``, or is not
        strictly ascending.
    """
    if t_eval_arr.ndim != 1:
        raise ValueError(
            f"t_eval must be 1-D; got shape {tuple(t_eval_arr.shape)}."
        )
    if isinstance(t_eval_arr, jax.core.Tracer):
        return
    t_eval_np = np.asarray(t_eval_arr)
    if t_eval_np.size == 0:
        return
    lo, hi = float(t_eval_np.min()), float(t_eval_np.max())
    if lo < t0 or hi > t1:
        raise ValueError(
            f"t_eval must lie within t_span [{t0}, {t1}]; got values in "
            f"[{lo}, {hi}]."
        )
    if t_eval_np.size > 1 and not np.all(np.diff(t_eval_np) > 0):
        raise ValueError("t_eval must be strictly ascending.")


class _HasNamedSpecies:
    """Mixin: provides ``C_named`` given a ``.C`` array and ``.network``.

    Solution dataclasses inherit from this to share the species-by-name
    accessor without duplicating the implementation.
    """

    C: jnp.ndarray  # set by the dataclass subclass
    network: CompiledNetwork  # set by the dataclass subclass

    def C_named(self, species: str) -> jnp.ndarray:
        """Return the trajectory of a single species by name."""
        if species not in self.network.species_index:
            raise KeyError(
                f"Unknown species '{species}'. Available: {self.network.species}"
            )
        return self.C[:, self.network.species_index[species]]


@runtime_checkable
class Reactor(Protocol):
    """Structural type for reactors usable by ``sensitivity`` / ``fit``.

    All concrete reactors expose ``network`` and ``solve``. Batch/PFR also
    expose ``conditions``; particle reactors expose ``track`` instead.
    Callers that need condition gradients should narrow to a reactor with a
    ``conditions`` attribute.
    """

    network: CompiledNetwork

    def solve(self, *args, **kwargs):  # pragma: no cover - protocol stub
        ...


def _coerce_atol(atol, n_species: int):
    """Validate and normalise an ``atol`` argument.

    Returns either the original scalar or a ``(n_species,)`` JAX array.
    Raises ``ValueError`` if an array of the wrong shape is supplied.
    """
    arr = jnp.asarray(atol)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape != (n_species,):
        raise ValueError(
            f"atol array must have shape ({n_species},), got {arr.shape}"
        )
    return arr


def _run_diffeqsolve(
    rhs: Callable,
    *,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    args,
    saveat: diffrax.SaveAt,
    rtol: float,
    atol,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int = 100_000,
    dtmax: float | None = None,
    event: "diffrax.Event | None" = None,
):
    """Wrapper around the canonical Kvaerno5 + PIDController + adjoint setup.

    All reactors call this with their own ``rhs``. Adjusting the default
    solver, controller, or adjoint here changes behaviour for every reactor.

    ``dtmax`` caps the integrator step size. It is ``None`` (uncapped) by
    default, which is fastest for plain forward solves. For *reverse-mode*
    differentiation of a stiff network it must be set. An L-stable solver may
    take steps far larger than the fastest reaction timescale and simply damp
    the unresolved fast modes in the primal (which stays accurate). The two AD
    modes then diverge: **forward mode** (``jax.jvp`` / ``jax.jacfwd``) stays
    finite at any step, losing only accuracy when the fast modes are
    unresolved; **reverse mode** (``jax.grad``, the discrete adjoint) returns
    **non-finite** values above a step-size threshold, an overflow in the
    backward accumulation governed by the per-step stiffness ``gamma*dt*||J||``
    (not by operator conditioning). Capping ``dtmax`` to a small multiple of
    the fastest reaction timescale bounds that product; the resulting reverse
    gradient is finite and matches both forward mode and finite differences.
    This is reverse-mode-specific and independent of the adjoint flavour. See
    the "Differentiating stiff networks" discussion in CLAUDE.md.
    """
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
        event=event,
    )


def solve_chemistry(
    network: CompiledNetwork,
    C0: jnp.ndarray,
    params: jnp.ndarray,
    *,
    cond_fn: Callable[[jnp.ndarray], Mapping[str, jnp.ndarray]],
    saveat: diffrax.SaveAt,
    t0,
    t1,
    rtol: float,
    atol,
    adjoint: diffrax.AbstractAdjoint | None = None,
    dtmax: float | None = None,
    max_steps: int = 100_000,
    rate_scale=None,
):
    """The canonical chemistry sub-solve shared by every reactor.

    Hoists the (parameter-dependent) stoichiometry out of the per-step RHS ---
    so dynamic coefficients are evaluated once per solve, not per step --- builds
    the right-hand side ``dC/dt = rate_scale * dCdt(C, params, cond_fn(t))`` and
    runs the Kvaerno5 + ``PIDController`` solve via :func:`_run_diffeqsolve`.

    The reactors differ only in three traced-time choices, passed in here:

    - ``cond_fn(t)`` returns the condition arrays at independent-variable value
      ``t``. A batch / CFD cell passes a constant dict; a PFR or particle track
      passes an interpolation of its spatially / temporally varying fields.
    - ``rate_scale`` (``None`` = identity) multiplies the rate, e.g. ``1/velocity``
      for the steady-state PFR whose independent variable is axial position.
    - ``saveat`` / ``t0`` / ``t1`` select the output points and the span.

    Returns the diffrax ``Solution``; callers read ``sol.ts`` / ``sol.ys`` (or
    ``sol.ys[-1]`` for a single-endpoint step).
    """
    stoich = network.compute_stoich(params)

    if rate_scale is None:
        def rhs(t, C, args):
            return network.dCdt(C, args, cond_fn(t), 0, stoich=stoich)
    else:
        def rhs(t, C, args):
            return network.dCdt(C, args, cond_fn(t), 0, stoich=stoich) * rate_scale

    return _run_diffeqsolve(
        rhs,
        t0=t0,
        t1=t1,
        y0=C0,
        args=params,
        saveat=saveat,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        dtmax=dtmax,
        max_steps=max_steps,
    )


def _interp_fields_to_scalar(
    t: jnp.ndarray,
    t_grid: jnp.ndarray,
    fields: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Linearly interpolate every field to a single-location array at ``t``.

    Returns a fresh dict whose values are shape-``(1,)`` arrays. Reactors
    pair this with ``loc_idx=0`` to satisfy the canonical rate-callable
    signature.
    """
    return {
        name: jnp.asarray([jnp.interp(t, t_grid, arr)])
        for name, arr in fields.items()
    }
