"""Shared quasi-Monte-Carlo sampling and callback-evaluation helpers.

The pieces the design-of-experiments workflows have in common:

* :func:`_unit_sample` -- draw low-discrepancy points in the unit cube (scrambled
  Sobol / Latin hypercube / plain random), reused by :func:`aquakin.monte_carlo`
  (mapped through each input's inverse CDF) and by :func:`aquakin.optimize_design`
  (quasi-random multistart starts).
* :func:`_eval_fn_over` / :func:`_resolve_output_names` -- evaluate an
  ``fn(x) -> output`` callback over a stack of input rows and name its output
  columns, shared by :func:`aquakin.monte_carlo` and
  :func:`aquakin.compare_scenarios`.

Kept in one internal module so the ``monte_carlo`` / ``scenarios`` / ``design``
workflow modules stay self-contained and none has to import another just for a
sampler.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate.sensitivity import _sobol_sample


def _unit_sample(d, n_samples, sampler, seed):
    """Draw ``n`` low-discrepancy points in the unit cube ``[0,1]^d``."""
    if sampler == "sobol":
        Z, n = _sobol_sample(np.zeros(d), np.ones(d), d, n_samples, seed)
        return np.asarray(Z), n
    if sampler == "lhs":
        from scipy.stats import qmc

        U = qmc.LatinHypercube(d=d, seed=seed).random(n=n_samples)
        return np.asarray(U), int(U.shape[0])
    if sampler == "random":
        U = np.random.default_rng(seed).random((n_samples, d))
        return U, int(n_samples)
    raise ValueError(f"unknown sampler '{sampler}'; use 'sobol', 'lhs' or 'random'.")


def _eval_fn_over(fn, X, batched):
    """Evaluate ``fn`` over the rows of ``X`` (shape (n, d)); return a
    ``(n, m)`` output array and the per-row finiteness mask. Non-finite rows are
    kept in place (caller filters) so inputs and outputs stay aligned."""
    f_arr = jax.jit(lambda x: jnp.atleast_1d(jnp.asarray(fn(x), dtype=float)))
    Xj = jnp.asarray(X)
    if batched:
        Y = np.asarray(jax.vmap(f_arr)(Xj))
    else:
        Y = np.stack([np.asarray(f_arr(x)) for x in Xj])
    finite = np.isfinite(Y).all(axis=1)
    return Y, finite


def _resolve_output_names(output_names, m):
    if output_names is None:
        return [f"y{i}" for i in range(m)] if m > 1 else ["output"]
    if len(output_names) != m:
        raise ValueError(f"output_names has {len(output_names)} entries but fn returns m={m}.")
    return list(output_names)
