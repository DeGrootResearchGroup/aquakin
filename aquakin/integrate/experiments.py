"""Scenario comparison and Monte-Carlo uncertainty propagation.

These turn the library's per-solve primitives into the two engineering workflows
a process modeller actually delivers:

* :func:`compare_scenarios` -- run a model under several named input sets (design
  options, operating points) and tabulate the resulting KPIs side by side.
* :func:`monte_carlo` -- propagate uncertain inputs (each with its own
  distribution) through the model and report the output ensemble and its
  percentiles.

Both share the same contract as :func:`aquakin.dgsm`: the caller supplies a
function ``fn(x) -> output`` that maps an input *vector* (named by ``input_names``)
to a scalar or vector output, building the params / initial state / conditions and
calling ``reactor.solve`` / ``plant.solve`` itself. The output may be any
JAX-differentiable quantity -- an effluent concentration, an EQI / OCI metric, a
removal efficiency. The Monte-Carlo sampler reuses the scrambled-Sobol quasi-MC
sequence from :mod:`aquakin.integrate.sensitivity`, adds Latin-hypercube and plain
random sampling, and maps the low-discrepancy unit points through each input's
inverse CDF (uniform / normal / lognormal), so non-uniform marginals still get a
low-discrepancy design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate.sensitivity import _sobol_sample


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
            f"'dist' key; got {spec!r}.")
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
        m, s = float(spec["mean"]), float(spec["std"])   # physical mean / std
        if m <= 0 or s <= 0:
            raise ValueError("lognormal needs mean > 0 and std > 0.")
        sigma = math.sqrt(math.log1p((s / m) ** 2))      # log-space sigma
        mu = math.log(m) - 0.5 * sigma ** 2              # log-space mu
        return lambda u: np.exp(mu + sigma * norm.ppf(u))
    raise ValueError(
        f"unknown distribution '{kind}'; use 'uniform', 'normal' or 'lognormal'.")


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


# --- shared evaluation -------------------------------------------------------

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
        raise ValueError(
            f"output_names has {len(output_names)} entries but fn returns m={m}.")
    return list(output_names)


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
        head = (f"Monte-Carlo ({self.sampler}, {self.n_valid}/{self.n_drawn} "
                f"valid, seed {self.seed})")
        cols = ["output", "mean", "std"] + [f"p{g:g}" for g in q]
        rows = [cols]
        for i, name in enumerate(self.output_names):
            rows.append([name, f"{mean[i]:.4g}", f"{std[i]:.4g}",
                         *[f"{pct[k, i]:.4g}" for k in range(len(q))]])
        w = [max(len(r[c]) for r in rows) for c in range(len(cols))]
        body = "\n".join("  ".join(r[c].ljust(w[c]) for c in range(len(cols)))
                         for r in rows)
        return head + "\n" + body


def monte_carlo(
    fn: Callable,
    distributions: Union[dict, Sequence],
    *,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
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
                f"input_names has {len(input_names)} entries but there are "
                f"{d} distributions.")
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
        samples=Xv, outputs=Yv,
        n_drawn=n_drawn, n_valid=int(Xv.shape[0]), sampler=sampler, seed=seed,
    )


# --- Scenario comparison -----------------------------------------------------

@dataclass
class ScenarioComparison:
    """Result of :func:`compare_scenarios`: KPIs per named scenario.

    Attributes
    ----------
    scenario_names : list[str]
        The scenarios, row order of ``outputs`` / ``inputs``.
    input_names, output_names : list[str]
        Names of the input and output columns.
    inputs : np.ndarray
        ``(n_scenarios, d)`` input vector used for each scenario.
    outputs : np.ndarray
        ``(n_scenarios, m)`` outputs.
    """
    scenario_names: list[str]
    input_names: list[str]
    output_names: list[str]
    inputs: np.ndarray
    outputs: np.ndarray

    def _col(self, name: str) -> np.ndarray:
        if name not in self.output_names:
            raise KeyError(f"unknown output '{name}'; have {self.output_names}.")
        return self.outputs[:, self.output_names.index(name)]

    def output_named(self, name: str) -> np.ndarray:
        """The output ``name`` across scenarios, shape ``(n_scenarios,)``."""
        return self._col(name)

    def best(self, output: str, *, minimize: bool = True) -> str:
        """The scenario name with the lowest (or highest) value of ``output``."""
        col = self._col(output)
        idx = int(np.argmin(col) if minimize else np.argmax(col))
        return self.scenario_names[idx]

    def table(self) -> str:
        """A human-readable KPI table, one row per scenario."""
        cols = ["scenario"] + list(self.output_names)
        rows = [cols]
        for i, name in enumerate(self.scenario_names):
            rows.append([name] + [f"{self.outputs[i, k]:.4g}"
                                  for k in range(len(self.output_names))])
        w = [max(len(r[c]) for r in rows) for c in range(len(cols))]
        return "\n".join("  ".join(r[c].ljust(w[c]) for c in range(len(cols)))
                         for r in rows)


def compare_scenarios(
    fn: Callable,
    scenarios: dict,
    *,
    input_names: Sequence[str],
    baseline: Optional[Sequence[float]] = None,
    output_names: Optional[Sequence[str]] = None,
    batched: bool = True,
) -> ScenarioComparison:
    """Run ``fn`` under several named scenarios and tabulate the outputs.

    Parameters
    ----------
    fn : callable
        ``fn(x) -> output`` as in :func:`monte_carlo` / :func:`aquakin.dgsm`.
    scenarios : dict
        ``name -> overrides`` where ``overrides`` is either a full input vector
        (length ``d``) or a mapping ``{input_name: value}`` applied on top of
        ``baseline`` (so a scenario only states what it changes). An empty
        mapping ``{}`` is the baseline itself.
    input_names : sequence of str
        Names of the ``d`` inputs (defines the vector order and the override
        keys).
    baseline : sequence of float, optional
        The nominal input vector that mapping-style overrides modify. Required if
        any scenario uses ``{input_name: value}`` overrides; defaults to zeros.
    output_names : sequence of str, optional
        Output column names.
    batched : bool
        vmap the scenarios (default) or evaluate one at a time.

    Returns
    -------
    ScenarioComparison
    """
    input_names = list(input_names)
    d = len(input_names)
    base = (np.zeros(d) if baseline is None
            else np.asarray(baseline, dtype=float))
    if base.shape != (d,):
        raise ValueError(f"baseline must have shape ({d},); got {base.shape}.")
    idx = {n: i for i, n in enumerate(input_names)}

    names = list(scenarios.keys())
    X = np.empty((len(names), d))
    for r, name in enumerate(names):
        ov = scenarios[name]
        if isinstance(ov, dict):
            x = base.copy()
            for k, v in ov.items():
                if k not in idx:
                    raise KeyError(
                        f"scenario '{name}' overrides unknown input '{k}'; "
                        f"inputs are {input_names}.")
                x[idx[k]] = float(v)
        else:
            x = np.asarray(ov, dtype=float)
            if x.shape != (d,):
                raise ValueError(
                    f"scenario '{name}' vector must have shape ({d},); "
                    f"got {x.shape}.")
        X[r] = x

    Y, finite = _eval_fn_over(fn, X, batched)
    if not finite.all():
        bad = [names[i] for i in range(len(names)) if not finite[i]]
        raise ValueError(f"scenario(s) gave a non-finite output: {bad}.")
    return ScenarioComparison(
        scenario_names=names, input_names=input_names,
        output_names=_resolve_output_names(output_names, Y.shape[1]),
        inputs=X, outputs=Y,
    )
