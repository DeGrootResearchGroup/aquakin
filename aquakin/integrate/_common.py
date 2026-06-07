"""Shared internals for the integrator submodules.

Not part of the public API. Reactors depend on this; this module depends only
on JAX and Diffrax.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, runtime_checkable

import diffrax
import jax.numpy as jnp

from aquakin.core.network import CompiledNetwork


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
):
    """Wrapper around the canonical Kvaerno5 + PIDController + adjoint setup.

    All reactors call this with their own ``rhs``. Adjusting the default
    solver, controller, or adjoint here changes behaviour for every reactor.

    ``dtmax`` caps the integrator step size. It is ``None`` (uncapped) by
    default, which is fastest for plain forward solves. For *differentiating*
    a stiff network it must be set: an L-stable solver may take steps far
    larger than the fastest reaction timescale and silently damp the
    unresolved fast modes in the primal, but the sensitivity of those modes is
    then ill-resolved and the differentiated solve returns non-finite values
    (in both forward and reverse mode). Capping ``dtmax`` to a small multiple
    of the fastest reaction timescale resolves it; the resulting gradients
    match finite differences.
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
