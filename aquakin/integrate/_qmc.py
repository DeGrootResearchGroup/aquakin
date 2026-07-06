"""Shared quasi-Monte-Carlo sampling and callback-evaluation helpers.

The pieces the design-of-experiments workflows and the DGSM screen have in
common:

* :func:`_sobol_sample` / :func:`_sobol_normal_sample` -- draw scrambled-Sobol
  points in an input box or from independent normals, used by
  :func:`aquakin.dgsm` (and the plant DGSM screens) and, via
  :func:`_unit_sample`, by the design-of-experiments workflows.
* :func:`_unit_sample` -- draw low-discrepancy points in the unit cube (scrambled
  Sobol / Latin hypercube / plain random), reused by :func:`aquakin.monte_carlo`
  (mapped through each input's inverse CDF) and by :func:`aquakin.optimize_design`
  (quasi-random multistart starts).
* :func:`_eval_fn_over` / :func:`_resolve_output_names` -- evaluate an
  ``fn(x) -> output`` callback over a stack of input rows and name its output
  columns, shared by :func:`aquakin.monte_carlo` and
  :func:`aquakin.compare_scenarios`.

Kept in one internal module so the ``monte_carlo`` / ``scenarios`` / ``design``
workflow modules and the DGSM screen stay self-contained and none has to import
another just for a sampler.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def _sobol_sample(lo, hi, d, n_samples, seed):
    """Draw scrambled-Sobol points in the input box.

    ``n_samples`` is rounded to the nearest power of two (Sobol sequences are
    balanced there). Returns ``(Z, n_drawn)`` with ``Z`` of shape
    ``(n_drawn, d)``.
    """
    from scipy.stats import qmc

    n_pow = max(1, round(math.log2(max(n_samples, 2))))
    U = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(n_pow)
    Z = lo[None, :] + (hi - lo)[None, :] * U
    return Z, int(Z.shape[0])


def _sobol_normal_sample(mean, std, d, n_samples, seed):
    """Draw scrambled-Sobol points from independent normals ``N(mean_j, std_j^2)``.

    Maps the low-discrepancy unit points through the inverse normal CDF, so the
    design is a quasi-Monte-Carlo sample of the Gaussian rather than of a box.
    This is the input distribution for a DGSM screen under Gaussian (prior)
    inputs, whose Sobol total-index bound carries the Poincare constant
    ``std_j^2`` in place of the uniform ``(b_j-a_j)^2 / pi^2`` (Sobol & Kucherenko
    2010, Sec. 8; Lamboni et al. 2013, Thm 3.1). ``mean``/``std`` are length-``d``
    (in the space where the input is Gaussian -- e.g. log-parameter space for a
    positive rate); ``n_samples`` is rounded to a power of two. Returns
    ``(Z, n_drawn)`` with ``Z`` of shape ``(n_drawn, d)``.
    """
    from scipy.stats import norm, qmc

    n_pow = max(1, round(math.log2(max(n_samples, 2))))
    U = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(n_pow)
    U = np.clip(U, 1e-12, 1.0 - 1e-12)  # avoid +/-inf at the 0/1 endpoints
    mean = np.asarray(mean)
    std = np.asarray(std)
    Z = mean[None, :] + std[None, :] * norm.ppf(U)
    return Z, int(Z.shape[0])


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
