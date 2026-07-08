"""Monte-Carlo uncertainty propagation.

:func:`monte_carlo` propagates uncertain inputs (each with its own distribution)
through a model and reports the output ensemble and its percentiles.

It shares the same contract as :func:`aquakin.dgsm`: the caller supplies a
function ``fn(x) -> output`` that maps an input *vector* (named by ``input_names``)
to a scalar or vector output, building the params / initial state / conditions and
calling ``reactor.solve`` / ``plant.solve`` itself. The output may be any
JAX-differentiable quantity -- an effluent concentration, an EQI / OCI metric, a
removal efficiency. The sampler reuses the scrambled-Sobol quasi-MC sequence from
:mod:`aquakin.integrate._qmc` (shared with :func:`aquakin.dgsm`), adds
Latin-hypercube and plain random sampling, and maps the low-discrepancy unit
points through each input's inverse CDF (uniform / normal / lognormal), so
non-uniform marginals still get a low-discrepancy design.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from aquakin.integrate._qmc import _eval_fn_over, _resolve_output_names, _unit_sample

# --- distributions -----------------------------------------------------------
# A per-input distribution is given either as a ``(low, high)`` tuple (uniform)
# or a mapping ``{"dist": ..., ...}``. Supported: uniform(low, high),
# normal(mean, std), lognormal(mean, std) -- mean/std in PHYSICAL space.


def _ppf(spec) -> Callable[[np.ndarray], np.ndarray]:
    """Return the inverse-CDF (quantile function) ``u in [0,1] -> value`` for one
    input's distribution spec, used to map a low-discrepancy unit sample to the
    distribution by inverse-transform sampling."""
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        spec = {"dist": "uniform", "low": spec[0], "high": spec[1]}
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(
            f"distribution must be a (low, high) tuple or a mapping with a "
            f"'dist' key; got {spec!r}."
        )
    kind = spec["dist"]
    if kind == "uniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        if not hi > lo:
            raise ValueError(f"uniform needs high > low; got ({lo}, {hi}).")
        return lambda u: lo + (hi - lo) * u
    if kind == "normal":
        from scipy.stats import norm

        m, s = float(spec["mean"]), float(spec["std"])
        if s <= 0:
            raise ValueError("normal needs std > 0.")
        return lambda u: norm.ppf(u, loc=m, scale=s)
    if kind == "lognormal":
        from scipy.stats import norm

        m, s = float(spec["mean"]), float(spec["std"])  # physical mean / std
        if m <= 0 or s <= 0:
            raise ValueError("lognormal needs mean > 0 and std > 0.")
        sigma = math.sqrt(math.log1p((s / m) ** 2))  # log-space sigma
        mu = math.log(m) - 0.5 * sigma**2  # log-space mu
        return lambda u: np.exp(mu + sigma * norm.ppf(u))
    raise ValueError(f"unknown distribution '{kind}'; use 'uniform', 'normal' or 'lognormal'.")


def _normalise_distributions(distributions):
    """Return ``(input_names, ppfs)`` from a list of specs or a name->spec dict."""
    if isinstance(distributions, dict):
        names = list(distributions.keys())
        specs = list(distributions.values())
    else:
        names = None
        specs = list(distributions)
    ppfs = [_ppf(s) for s in specs]
    return names, ppfs


# --- Monte-Carlo -------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Result of :func:`monte_carlo`: the sampled input/output ensemble.

    Attributes
    ----------
    input_names, output_names : list[str]
        Names of the inputs (columns of ``samples``) and outputs (columns of
        ``outputs``).
    samples : np.ndarray
        ``(n_valid, d)`` sampled input vectors (in physical space) with a finite
        output.
    outputs : np.ndarray
        ``(n_valid, m)`` model outputs.
    n_drawn, n_valid : int
        Points drawn / kept (a non-finite output -- a failed/clipped solve -- is
        dropped).
    sampler, seed : str, int
        Sampler used and its seed (fixing it makes the result reproducible).
    """

    input_names: list[str]
    output_names: list[str]
    samples: np.ndarray
    outputs: np.ndarray
    n_drawn: int
    n_valid: int
    sampler: str
    seed: int

    def _col(self, name: str) -> np.ndarray:
        if name not in self.output_names:
            raise KeyError(f"unknown output '{name}'; have {self.output_names}.")
        return self.outputs[:, self.output_names.index(name)]

    def output_named(self, name: str) -> np.ndarray:
        """The ``(n_valid,)`` ensemble of one output by name."""
        return self._col(name)

    def mean(self) -> np.ndarray:
        """Per-output mean, shape ``(m,)``."""
        return self.outputs.mean(axis=0)

    def std(self) -> np.ndarray:
        """Per-output standard deviation, shape ``(m,)``."""
        return self.outputs.std(axis=0)

    def percentiles(self, q: Sequence[float] = (2.5, 50.0, 97.5)) -> np.ndarray:
        """Per-output percentiles, shape ``(len(q), m)``."""
        return np.percentile(self.outputs, np.asarray(q), axis=0)

    def summary(self, q: Sequence[float] = (2.5, 50.0, 97.5)) -> str:
        """A human-readable table of mean / std / percentiles per output."""
        mean, std = self.mean(), self.std()
        pct = self.percentiles(q)
        head = (
            f"Monte-Carlo ({self.sampler}, {self.n_valid}/{self.n_drawn} valid, seed {self.seed})"
        )
        cols = ["output", "mean", "std"] + [f"p{g:g}" for g in q]
        rows = [cols]
        for i, name in enumerate(self.output_names):
            rows.append(
                [
                    name,
                    f"{mean[i]:.4g}",
                    f"{std[i]:.4g}",
                    *[f"{pct[k, i]:.4g}" for k in range(len(q))],
                ]
            )
        w = [max(len(r[c]) for r in rows) for c in range(len(cols))]
        body = "\n".join("  ".join(r[c].ljust(w[c]) for c in range(len(cols))) for r in rows)
        return head + "\n" + body


def monte_carlo(
    fn: Callable,
    distributions: dict | Sequence,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    n_samples: int = 128,
    sampler: str = "sobol",
    seed: int = 0,
    batched: bool = True,
) -> MonteCarloResult:
    """Propagate uncertain inputs through ``fn`` and return the output ensemble.

    Parameters
    ----------
    fn : callable
        ``fn(x) -> output`` mapping an input vector ``x`` (shape ``(d,)``, in the
        order of ``distributions``) to a scalar or ``(m,)`` vector output. As in
        :func:`aquakin.dgsm`, ``fn`` builds the params / initial state and runs
        the solve itself.
    distributions : mapping or sequence
        One distribution per input, either a ``name -> spec`` mapping (then the
        keys are the input names) or a sequence of specs. Each spec is a
        ``(low, high)`` tuple (uniform) or a mapping ``{"dist": ...}`` --
        ``uniform(low, high)``, ``normal(mean, std)`` or ``lognormal(mean, std)``
        (mean / std in physical space).
    input_names, output_names : sequence of str, optional
        Names for the input columns (defaults to the mapping keys or ``z0..``)
        and output columns (defaults to ``output`` / ``y0..``).
    n_samples : int
        Number of points to draw. For ``sampler='sobol'`` it is rounded to the
        nearest power of two.
    sampler : {'sobol', 'lhs', 'random'}
        Low-discrepancy scrambled Sobol (default), Latin hypercube, or plain
        pseudo-random. Sampling is in the unit cube and mapped to the marginals
        by inverse-transform, so non-uniform inputs still get a good design.
    seed : int
        Sampler seed; fixing it makes the result reproducible.
    batched : bool
        Evaluate the whole sample through one :func:`jax.vmap` (default) or one
        call per point (lower peak memory).

    Returns
    -------
    MonteCarloResult
    """
    names, ppfs = _normalise_distributions(distributions)
    d = len(ppfs)
    if input_names is not None:
        if len(input_names) != d:
            raise ValueError(
                f"input_names has {len(input_names)} entries but there are {d} distributions."
            )
        names = list(input_names)
    elif names is None:
        names = [f"z{j}" for j in range(d)]

    U, n_drawn = _unit_sample(d, n_samples, sampler, seed)
    # Map each column through its inverse CDF (inverse-transform sampling).
    X = np.empty_like(U)
    for j in range(d):
        X[:, j] = ppfs[j](U[:, j])

    Y, finite = _eval_fn_over(fn, X, batched)
    Xv, Yv = X[finite], Y[finite]
    return MonteCarloResult(
        input_names=names,
        output_names=_resolve_output_names(output_names, Yv.shape[1]),
        samples=Xv,
        outputs=Yv,
        n_drawn=n_drawn,
        n_valid=int(Xv.shape[0]),
        sampler=sampler,
        seed=seed,
    )
