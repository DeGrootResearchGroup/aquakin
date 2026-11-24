"""Variance-based global sensitivity: Sobol first-order and total indices.

:func:`sobol_indices` decomposes the variance of a model output over its
uncertain inputs. For each input ``z_j`` it estimates

- the **first-order index** ``S_j = Var(E[f | z_j]) / Var(f)``, the fraction of
  the output variance ``z_j`` explains on its own, with the Saltelli et al.
  (2010) estimator applied to the mean-centred output, and
- the **total-order index** ``S_j^tot = E[Var(f | z_~j)] / Var(f)``, the
  fraction it explains including every interaction, with the Jansen (1999)
  estimator,

so ``S_j^tot - S_j`` is the share of the variance ``z_j`` explains only through
interactions with other inputs. Confidence intervals come from a bootstrap over
the base sample.

This is the variance-based companion of :func:`aquakin.dgsm`, which bounds
``S_j^tot`` from above from one gradient per sample but cannot give ``S_j`` or
the interactions. The price is ``N (d + 2)`` evaluations of the model for ``d``
inputs against about ``N`` gradients for DGSM, so this estimator is opt-in;
:func:`sobol_indices_cost` quotes the count before a run.

The input specification (``(low, high)`` or uniform / normal / lognormal
mappings) is the one :func:`aquakin.monte_carlo` reads, the scrambled-Sobol
draw and the batched evaluation of ``fn`` come from :mod:`aquakin.integrate._qmc`,
shared with :func:`aquakin.dgsm` and :func:`aquakin.monte_carlo`.

References
----------
Jansen, M. J. W. (1999). Analysis of variance designs for model output.
*Computer Physics Communications* 117, 35-43.

Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M. & Tarantola, S.
(2010). Variance based sensitivity analysis of model output. Design and
estimator for the total sensitivity index. *Computer Physics Communications*
181, 259-270.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._qmc import (
    _eval_fn_over,
    _resolve_output_names,
    _sobol_log2_size,
    _unit_sample,
)
from aquakin.integrate.monte_carlo import _map_unit_sample, _resolve_distributions

_log = logging.getLogger(__name__)

# Bootstrap replicates for the confidence intervals. They reuse the evaluations
# already made, so they cost no solves; 1000 puts the Monte-Carlo error of a
# 2.5 % / 97.5 % percentile well below the interval width.
_N_BOOTSTRAP = 1000
# Replicates resampled per batch: bounds the ``(batch, d, N)`` temporary at a
# few tens of MB for d ~ 20 inputs and N ~ 4096 samples.
_BOOTSTRAP_BATCH = 64


@dataclass
class SobolResult:
    """Result of :func:`sobol_indices` for one output.

    Attributes
    ----------
    input_names : list[str]
        Names of the uncertain inputs, matching the rows of every array.
    first_order : np.ndarray
        First-order Sobol index ``S_j`` of each input, shape ``(d,)``: the
        fraction of the output variance the input explains on its own.
    total_order : np.ndarray
        Total-order Sobol index ``S_j^tot``, shape ``(d,)``: the fraction it
        explains including its interactions with the other inputs. Directly
        comparable with :attr:`DGSMResult.sobol_total_bound`, which bounds it
        from above.
    first_order_ci, total_order_ci : np.ndarray
        Bootstrap percentile confidence intervals at level ``confidence``,
        shape ``(d, 2)`` (lower, upper).
    output_variance : float
        Variance of the output over the sample.
    n_samples : int
        Size ``N`` of the base sample (a power of two).
    n_evaluations : int
        Evaluations of ``fn`` made, ``N (d + 2)``.
    n_valid : int
        Base-sample rows kept: a row is dropped when any of its ``d + 2``
        evaluations is non-finite in this output.
    confidence : float
        Level of the confidence intervals.
    seed : int
        Seed of the sampler and the bootstrap; fixing it makes the result
        reproducible.
    output_name : str or None
        Name of the output for a vector-valued ``fn``; ``None`` for a scalar one.
    """

    input_names: list[str]
    first_order: np.ndarray
    total_order: np.ndarray
    first_order_ci: np.ndarray
    total_order_ci: np.ndarray
    output_variance: float
    n_samples: int
    n_evaluations: int
    n_valid: int
    confidence: float
    seed: int
    output_name: str | None = None

    @property
    def interaction(self) -> np.ndarray:
        """The share each input explains only through interactions,
        ``S_j^tot - S_j``, shape ``(d,)``."""
        return self.total_order - self.first_order

    def ranked(self) -> list[tuple[str, float]]:
        """Return ``(name, total_order)`` pairs sorted by decreasing total index."""
        pairs = [(n, float(s)) for n, s in zip(self.input_names, self.total_order)]
        return sorted(pairs, key=lambda kv: kv[1], reverse=True)


def sobol_indices_cost(n_inputs: int, n_samples: int = 512) -> int:
    """Number of model evaluations :func:`sobol_indices` will make.

    The Saltelli design evaluates the model on two base samples and on one
    hybrid sample per input, ``N (d + 2)`` evaluations, where ``N`` is
    ``n_samples`` rounded to the nearest power of two. Call this before a run
    to see what it costs: each evaluation of a reactor model is one forward
    solve.

    Parameters
    ----------
    n_inputs : int
        Number of uncertain inputs ``d``.
    n_samples : int, optional
        The ``n_samples`` that will be passed to :func:`sobol_indices`.

    Returns
    -------
    int
        The number of evaluations of ``fn``.

    Raises
    ------
    ValueError
        If ``n_inputs`` is less than 1.

    Examples
    --------
    >>> aquakin.sobol_indices_cost(17, 1000)   # N rounds to 1024
    19456
    """
    if n_inputs < 1:
        raise ValueError(f"n_inputs must be at least 1; got {n_inputs}.")
    return (2 ** _sobol_log2_size(n_samples)) * (n_inputs + 2)


def _saltelli_design(ppfs, n_samples, seed):
    """Stack the Saltelli design ``[A; B; AB_0; ...; AB_{d-1}]``.

    One scrambled-Sobol draw of ``2d`` dimensions is split into two independent
    base samples ``A`` and ``B`` (shape ``(N, d)``), each mapped through the
    inputs' inverse CDFs; ``AB_j`` is ``A`` with column ``j`` taken from ``B``.
    Returns ``(X, N)`` with ``X`` of shape ``(N (d + 2), d)``.
    """
    d = len(ppfs)
    U, n = _unit_sample(2 * d, n_samples, "sobol", seed)
    A = _map_unit_sample(U[:, :d], ppfs)
    B = _map_unit_sample(U[:, d:], ppfs)
    AB = np.repeat(A[None, :, :], d, axis=0)  # (d, N, d)
    for j in range(d):
        AB[j, :, j] = B[:, j]
    return np.concatenate([A, B, AB.reshape(d * n, d)]), n


def _sobol_estimates(fA, fB, fAB):
    """First- and total-order indices from evaluations on the Saltelli design.

    ``fA`` and ``fB`` have shape ``(..., N)`` and ``fAB`` shape ``(..., d, N)``
    (leading axes are batch axes, e.g. bootstrap replicates). The mean ``f0``
    and variance ``V`` are taken over ``A`` and ``B`` together;
    ``S_j = mean((fB - f0) (fAB_j - fA)) / V`` (Saltelli et al. 2010, on the
    centred output) and ``S_j^tot = mean((fA - fAB_j)^2) / (2 V)`` (Jansen
    1999). Centring leaves the estimate's expectation unchanged, since
    ``E[fAB_j - fA] = 0``, but makes it invariant to a constant added to ``f``:
    without it the estimator's variance grows with ``f0^2``. Returns
    ``(S, S_tot, V)`` with ``S`` and ``S_tot`` of shape ``(..., d)``.
    """
    both = np.concatenate([fA, fB], axis=-1)
    var = both.var(axis=-1)
    diff = fAB - fA[..., None, :]
    centred_B = fB - both.mean(axis=-1, keepdims=True)
    num_first = (centred_B[..., None, :] * diff).mean(axis=-1)
    num_total = 0.5 * (diff**2).mean(axis=-1)
    # A zero-variance output gives 0/0 here; the caller reports it (with a warning).
    with np.errstate(divide="ignore", invalid="ignore"):
        return num_first / var[..., None], num_total / var[..., None], var


def _bootstrap_intervals(fA, fB, fAB, confidence, rng):
    """Bootstrap percentile intervals for ``S`` and ``S_tot``, each ``(d, 2)``.

    Resamples the ``N`` base-sample rows with replacement (a row carries its
    ``A``, ``B`` and every ``AB_j`` evaluation together, so the design's
    pairing is kept) and recomputes both estimators on each replicate.
    """
    n = fA.shape[-1]
    first, total = [], []
    for start in range(0, _N_BOOTSTRAP, _BOOTSTRAP_BATCH):
        idx = rng.integers(0, n, size=(min(_BOOTSTRAP_BATCH, _N_BOOTSTRAP - start), n))
        S, ST, _ = _sobol_estimates(fA[idx], fB[idx], np.moveaxis(fAB[:, idx], 0, 1))
        first.append(S)
        total.append(ST)
    tail = 50.0 * (1.0 - confidence)
    q = (tail, 100.0 - tail)
    return (
        np.percentile(np.concatenate(first), q, axis=0).T,
        np.percentile(np.concatenate(total), q, axis=0).T,
    )


def _check_sobol_arguments(d, confidence):
    if d < 1:
        raise ValueError("sobol_indices needs at least one input distribution.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1; got {confidence}.")


def _output_result(fA, fB, fAB, name, common, rng):
    """Estimate the indices of one output column and wrap them in a SobolResult.

    Drops every base-sample row with a non-finite evaluation in this output;
    ``common`` carries the fields shared by every output's result.
    """
    d = fAB.shape[0]
    keep = np.isfinite(fA) & np.isfinite(fB) & np.isfinite(fAB).all(axis=0)
    n_keep = int(keep.sum())
    which = f" {name!r}" if name is not None else ""
    if n_keep < 2:
        raise RuntimeError(
            f"sobol_indices needs >= 2 base-sample rows with finite evaluations of "
            f"output{which}; got {n_keep}/{keep.size}. fn returns NaN or inf over the "
            "sampled inputs: check it at the edges of the distributions, or narrow "
            "them to where the model is defined."
        )
    fA, fB, fAB = fA[keep], fB[keep], fAB[:, keep]
    S, ST, var = _sobol_estimates(fA, fB, fAB)
    if var > 0:
        S_ci, ST_ci = _bootstrap_intervals(fA, fB, fAB, common["confidence"], rng)
    else:
        # Every finite evaluation gave the same value: the indices are 0/0. Report
        # zeros but warn, so an empty ranking is not read as "no input matters".
        warnings.warn(
            f"sobol_indices output{which} has zero "
            "variance over the sampled inputs; the Sobol indices are undefined and "
            "reported as 0. The output may be saturated, clipped, or insensitive "
            "to every input.",
            stacklevel=3,
        )
        S, ST = np.zeros(d), np.zeros(d)
        S_ci, ST_ci = np.zeros((d, 2)), np.zeros((d, 2))
    return SobolResult(
        first_order=S,
        total_order=ST,
        first_order_ci=S_ci,
        total_order_ci=ST_ci,
        output_variance=float(var),
        n_valid=n_keep,
        output_name=name,
        **common,
    )


def sobol_indices(
    fn: Callable,
    distributions: dict | Sequence,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    n_samples: int = 512,
    seed: int = 0,
    confidence: float = 0.95,
    batched: bool = True,
) -> SobolResult | list[SobolResult]:
    """First-order and total Sobol indices by the Saltelli / Jansen estimators.

    Draws two independent scrambled-Sobol base samples ``A`` and ``B`` of the
    inputs and, for each input ``j``, the hybrid ``AB_j`` (``A`` with column
    ``j`` from ``B``), and evaluates ``fn`` on all of them. From those
    evaluations it estimates the first-order index ``S_j`` (Saltelli et al.
    2010) and the total-order index ``S_j^tot`` (Jansen 1999) of every input,
    with bootstrap confidence intervals.

    **Cost.** ``N (d + 2)`` evaluations of ``fn`` for ``d`` inputs and a base
    sample of ``N``; for a reactor model each is a forward solve (no gradient).
    :func:`aquakin.dgsm` needs about ``N`` gradient evaluations for the same
    ``d`` and bounds ``S_j^tot`` without giving ``S_j``; use it to screen many
    inputs, and this estimator for the decomposition on a reduced input set or
    a cheap model. :func:`sobol_indices_cost` gives the count before a run.

    Parameters
    ----------
    fn : callable
        ``fn(x) -> output`` mapping an input vector ``x`` (shape ``(d,)``, in the
        order of ``distributions``) to a scalar or an ``(m,)`` vector output. As
        in :func:`aquakin.monte_carlo`, ``fn`` builds the parameters / initial
        state and runs the solve itself, under :func:`jax.jit`, so it must be
        traceable. It need not be differentiable.
    distributions : mapping or sequence
        One distribution per input, as for :func:`aquakin.monte_carlo`: a
        ``name -> spec`` mapping (the keys name the inputs) or a sequence of
        specs. A spec is a ``(low, high)`` tuple (uniform) or a mapping
        ``{"dist": ...}``: ``uniform(low, high)``, ``normal(mean, std)`` or
        ``lognormal(mean, std)`` (mean and std in physical space). The indices
        assume independent inputs.
    input_names : sequence of str, optional
        Input names; default the mapping keys, else ``z0, z1, ...``.
    output_names : sequence of str, optional
        Names of the ``m`` outputs of a vector-valued ``fn``; default
        ``y0, y1, ...``. Ignored for a scalar ``fn``.
    n_samples : int, optional
        Base sample size ``N``, rounded to the nearest power of two. The
        estimators' error shrinks like ``1/sqrt(N)``; read the width of the
        confidence intervals to choose it.
    seed : int, optional
        Seed of the scrambled-Sobol sampler and of the bootstrap. Fixing it (the
        default ``0``) makes the result exactly reproducible.
    confidence : float, optional
        Level of the bootstrap percentile confidence intervals, in ``(0, 1)``.
        The 1000 bootstrap replicates reuse the evaluations already made, so
        they cost no solves.
    batched : bool, optional
        Evaluate the whole design through one :func:`jax.vmap` (default), or one
        call per point (lower peak memory for a large design).

    Returns
    -------
    SobolResult or list[SobolResult]
        One :class:`SobolResult` for a scalar ``fn``, or one per output, in
        order and each carrying its ``output_name``, for a vector ``fn``.

    Raises
    ------
    ValueError
        For no inputs, a malformed distribution, mismatched ``input_names`` /
        ``output_names``, or ``confidence`` outside
        ``(0, 1)``.
    RuntimeError
        If fewer than two base-sample rows give finite evaluations of an output.

    Examples
    --------
    >>> res = aquakin.sobol_indices(lambda z: z[0] + 2.0 * z[1],
    ...                             [(0.0, 1.0), (0.0, 1.0)], n_samples=256)
    >>> [round(float(s), 2) for s in res.first_order]   # exact: 0.2, 0.8
    [0.2, 0.8]
    """
    names, ppfs = _resolve_distributions(distributions, input_names)
    d = len(ppfs)
    _check_sobol_arguments(d, confidence)
    X, n = _saltelli_design(ppfs, n_samples, seed)
    vector = len(jax.eval_shape(lambda x: jnp.asarray(fn(x)), jnp.asarray(X[0])).shape) == 1
    _log.info("sobol_indices: %d evaluations of fn (N=%d, d=%d)", X.shape[0], n, d)

    Y, _ = _eval_fn_over(fn, X, batched)  # (N (d + 2), m); finiteness is per output below
    m = Y.shape[1]
    out_names: list[str | None] = list(_resolve_output_names(output_names, m)) if vector else [None]
    common = dict(
        input_names=list(names),
        n_samples=n,
        n_evaluations=int(X.shape[0]),
        confidence=float(confidence),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    fAB = Y[2 * n :].reshape(d, n, m)
    results = [
        _output_result(Y[:n, i], Y[n : 2 * n, i], fAB[:, :, i], name, common, rng)
        for i, name in enumerate(out_names)
    ]
    return results if vector else results[0]
