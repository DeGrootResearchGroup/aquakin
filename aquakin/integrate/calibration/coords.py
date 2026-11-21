"""The free-initial-condition coordinates: per-pool bounds, log-space pools, the
fixed-total simplex (log-ratio) coordinates and their names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._transforms import (
    dphysical_dunconstrained,
    from_unconstrained,
    to_unconstrained,
)
from aquakin.integrate.calibration.config import FreeICConfig

if TYPE_CHECKING:
    from aquakin.integrate.calibration.problem import _CalibrationProblem


_DEFAULT_IC_BOUNDS = (1e-3, 1e4)


def _free_ic_fields(free_ic) -> tuple[list | None, tuple | dict, float | None]:
    """Unpack an optional :class:`FreeICConfig` into ``(species, bounds,
    prior_log_std)`` for the internal problem resolver."""
    if free_ic is None:
        return None, _DEFAULT_IC_BOUNDS, None
    if isinstance(free_ic, FreeICConfig):
        return free_ic.species, free_ic.bounds, free_ic.prior_log_std
    raise TypeError(f"free_ic must be a FreeICConfig or None; got {type(free_ic).__name__}.")


def _per_species_ic_bounds(species, bounds) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` arrays, one entry per fitted pool, from a shared ``(lo, hi)``
    pair or a ``{species: (lo, hi)}`` dict (missing species get the default).
    Each pair must satisfy ``0 < lo < hi``."""
    species = list(species)
    if isinstance(bounds, dict):
        unknown = [s for s in bounds if s not in species]
        if unknown:
            raise KeyError(
                f"FreeICConfig.bounds names species that are not fitted: {unknown}; "
                f"fitted species are {species}."
            )
        pairs = [bounds.get(s, _DEFAULT_IC_BOUNDS) for s in species]
    else:
        pairs = [bounds] * len(species)
    lo, hi = [], []
    for s, pair in zip(species, pairs):
        if np.ndim(pair) != 1 or len(pair) != 2:
            raise ValueError(f"ic_bounds must be a (lo, hi) pair; got {pair!r} for '{s}'.")
        if not (0.0 < float(pair[0]) < float(pair[1])):
            raise ValueError(f"ic_bounds must satisfy 0 < lo < hi; got {pair} for '{s}'.")
        lo.append(float(pair[0]))
        hi.append(float(pair[1]))
    return np.asarray(lo), np.asarray(hi)


def _resolve_ic_coords(species, ic_bounds, fixed_total, pools_per_dataset):
    """The free-IC coordinate setup shared by the reactor and plant resolvers.

    ``species`` names the fitted pools and ``pools_per_dataset`` holds their
    starting values in each dataset's initial state. Returns ``(ic_bounds,
    ic_center_full, fixed_total)``: the per-pool ``(lo, hi)`` arrays (the
    bounds as given when nothing is fitted or for a split, which uses none),
    the unconstrained start of every pool coordinate (log pools, or the
    log-ratios of a ``fixed_total`` split, which is what the ic prior pulls
    toward), and ``fixed_total`` switched off when nothing is fitted."""
    species = list(species)
    if len(set(species)) != len(species):
        raise ValueError(f"free_ic species must be distinct; got {species}.")
    m_ic = len(species)
    fixed_total = bool(fixed_total and m_ic)
    if not m_ic:
        return ic_bounds, jnp.zeros(0), False
    if fixed_total:
        if isinstance(ic_bounds, dict) or tuple(ic_bounds) != _DEFAULT_IC_BOUNDS:
            raise ValueError(
                "FreeICConfig(fixed_total=True) fits the split of a fixed total on the "
                "simplex, so it takes no bounds; leave bounds at its default."
            )
    else:
        ic_bounds = _per_species_ic_bounds(species, ic_bounds)
    blocks = []
    for ds, pools in enumerate(pools_per_dataset):
        pools = np.asarray(pools, dtype=float)
        if fixed_total:
            if np.any(pools <= 0.0):
                raise ValueError(
                    f"dataset {ds}: with FreeICConfig(fixed_total=True) every pool in "
                    f"{species} must be positive in C0 (their sum is the fixed "
                    f"total, their split the starting guess); got {pools.tolist()}."
                )
            blocks.append(np.log(pools[1:]) - np.log(pools[0]))
        else:
            blocks.append(np.log(np.clip(pools, ic_bounds[0], ic_bounds[1])))
    return ic_bounds, jnp.asarray(np.concatenate(blocks)), fixed_total


def _simplex_pools(z, total):
    """Pools ``total * softmax([0, z])``: positive, summing to ``total``, from the
    ``m - 1`` log-ratio coordinates ``z_j = log(c_j / c_0)``."""
    logits = jnp.concatenate([jnp.zeros(1, dtype=jnp.result_type(z, float)), z])
    return total * jax.nn.softmax(logits)


def _simplex_fraction_cov(z, cov_z) -> np.ndarray:
    """Delta-method covariance of the fractions ``softmax([0, z])`` from the
    covariance of the log-ratio coordinates ``z``."""
    f = np.asarray(jax.nn.softmax(jnp.concatenate([jnp.zeros(1), jnp.asarray(z)])))
    J = (np.diag(f) - np.outer(f, f))[:, 1:]  # d f / d z
    return J @ np.asarray(cov_z) @ J.T


def _to_unconstrained(value: jnp.ndarray, transform: str) -> jnp.ndarray:
    return to_unconstrained(value, transform)


def _from_unconstrained(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    return from_unconstrained(theta, transform)


def _jacobian_physical_wrt_theta(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    """``dp/dtheta`` at the given ``theta``, used for the delta-method std."""
    return dphysical_dunconstrained(from_unconstrained(theta, transform), transform)


def _joint_names(problem: _CalibrationProblem) -> list[str]:
    """Row/column names of the joint rate + initial-pool Hessian: the rate
    names, then ``"ic[k].<species>"`` for each dataset ``k``'s fitted pools, or
    ``"ic[k].<c_j>:<c_0>"`` for the log-ratio coordinates of a fixed-total
    split."""
    return list(problem.free_params) + _ic_coordinate_names(
        problem.free_ic, problem.n_datasets, problem.ic_fixed_total
    )


def _ic_coordinate_names(species, n_datasets: int, fixed_total: bool) -> list[str]:
    """Names of the fitted initial-pool coordinates, dataset by dataset."""
    species = list(species)
    per = [f"{s}:{species[0]}" for s in species[1:]] if fixed_total else species
    return [f"ic[{k}].{c}" for k in range(n_datasets) for c in per]


def _ic_theta_from_pools(problem: _CalibrationProblem, C0s) -> jnp.ndarray:
    """Unconstrained free-IC coordinates of the pools in ``C0s`` (one initial
    state per dataset), laid out as ``problem.ic_center_full`` is."""
    idx = np.asarray(problem.ic_species_idx)
    blocks = []
    for C0_i in C0s:
        pools = np.asarray(C0_i, dtype=float)[idx]
        if problem.ic_fixed_total:
            blocks.append(np.log(pools[1:]) - np.log(pools[0]))
        else:
            lo, hi = problem.ic_bounds
            blocks.append(np.log(np.clip(pools, lo, hi)))
    return jnp.asarray(np.concatenate(blocks))
