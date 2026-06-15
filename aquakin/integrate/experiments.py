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


# --- Standardized KPI comparison ---------------------------------------------

@dataclass
class KPIComparison:
    """A side-by-side KPI table over several named results.

    The standardized-report companion to :func:`compare_scenarios`: where that
    runs a model and tabulates a fixed output *vector*, this assembles a table
    from heterogeneous **report objects** (a :class:`BSM2Evaluation`, a
    :class:`CarbonFootprint`, an :class:`OperatingCost`, or any object exposing a
    ``kpis()`` mapping -- or a plain ``{name: value}`` dict) already computed per
    scenario. The KPI columns are the union of every report's keys, in
    first-seen order; a KPI a given report does not provide is left blank.

    Attributes
    ----------
    names : list[str]
        The result names (table rows).
    kpi_names : list[str]
        The KPI labels (table columns), union over all results.
    values : dict
        ``name -> {kpi: value}`` for every result.
    """

    names: list[str]
    kpi_names: list[str]
    values: dict

    def column(self, kpi: str) -> dict:
        """The ``{name: value}`` map for one KPI across results."""
        if kpi not in self.kpi_names:
            raise KeyError(f"unknown KPI '{kpi}'; have {self.kpi_names}.")
        return {n: self.values[n].get(kpi, float("nan")) for n in self.names}

    def best(self, kpi: str, *, minimize: bool = True) -> str:
        """The result name with the lowest (or highest) value of ``kpi``."""
        col = self.column(kpi)
        finite = {n: v for n, v in col.items() if v == v}  # drop NaNs
        if not finite:
            raise ValueError(f"KPI '{kpi}' has no finite value across results.")
        return (min if minimize else max)(finite, key=finite.get)

    def table(self) -> str:
        """A human-readable KPI table, one column per result."""
        rows = [["KPI", *self.names]]
        for kpi in self.kpi_names:
            row = [kpi]
            for n in self.names:
                v = self.values[n].get(kpi)
                row.append("" if v is None else f"{v:.4g}")
            rows.append(row)
        w = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        return "\n".join("  ".join(r[c].ljust(w[c]) for c in range(len(r)))
                         for r in rows)

    def __str__(self) -> str:
        return self.table()


def _kpis_of(report) -> dict:
    """Extract a ``{kpi: value}`` mapping from a report object or plain dict."""
    if isinstance(report, dict):
        return dict(report)
    kpis = getattr(report, "kpis", None)
    if callable(kpis):
        return dict(kpis())
    raise TypeError(
        f"a KPI report must be a dict or expose a kpis() method; got "
        f"{type(report).__name__}.")


def kpi_comparison(reports: dict) -> KPIComparison:
    """Tabulate KPIs from several named report objects side by side.

    Parameters
    ----------
    reports : dict
        ``name -> report``, where each report is a result object exposing a
        ``kpis()`` method (:class:`BSM2Evaluation`, :class:`CarbonFootprint`,
        :class:`OperatingCost`, ...) or a plain ``{kpi: value}`` mapping. The KPI
        columns are the union of every report's keys, in first-seen order.

    Returns
    -------
    KPIComparison

    Examples
    --------
    >>> kpi_comparison({
    ...     "baseline": evaluation_a,
    ...     "low-DO":   evaluation_b,
    ... }).table()  # doctest: +SKIP
    """
    names = list(reports.keys())
    per_name = {n: _kpis_of(reports[n]) for n in names}
    kpi_names: list = []
    for n in names:
        for k in per_name[n]:
            if k not in kpi_names:
                kpi_names.append(k)
    return KPIComparison(names=names, kpi_names=kpi_names, values=per_name)


# --- Constrained design optimization -----------------------------------------

@dataclass
class Constraint:
    """An inequality constraint on a design optimization.

    ``fn(x)`` (a scalar function of the design vector, e.g. an effluent
    concentration or a total-nitrogen metric) must satisfy ``lower <= fn(x) <=
    upper``. Give an ``upper`` (a permit ceiling), a ``lower`` (a floor), or
    both. ``fn`` should be JAX-differentiable -- its gradient is taken by autodiff
    and handed to the optimizer.
    """
    fn: Callable
    upper: Optional[float] = None
    lower: Optional[float] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.upper is None and self.lower is None:
            raise ValueError("a Constraint needs an 'upper' and/or 'lower' bound.")


@dataclass
class OptimizeResult:
    """Result of :func:`optimize_design`.

    Attributes
    ----------
    input_names : list[str]
        Names of the design variables (order of ``x``).
    x : np.ndarray
        The optimal design vector.
    objective : float
        Objective value at ``x`` (in the original sense -- already un-negated for
        a ``maximize`` run).
    constraint_values : dict
        ``name -> fn(x)`` for every constraint at the optimum.
    feasible : bool
        Whether every constraint is satisfied at ``x`` (within ``constraint_tol``).
    success : bool
        The optimizer reported convergence for the chosen start.
    message : str
        Optimizer status message.
    n_iter, n_starts : int
        Iterations of the winning run; number of multistart runs.
    """
    input_names: list[str]
    x: np.ndarray
    objective: float
    constraint_values: dict
    feasible: bool
    success: bool
    message: str
    n_iter: int
    n_starts: int

    @property
    def x_named(self) -> dict:
        """The optimal design as a ``name -> value`` dict."""
        return {n: float(v) for n, v in zip(self.input_names, self.x)}

    def report(self) -> str:
        lines = [f"optimize_design: objective = {self.objective:.6g} "
                 f"({'feasible' if self.feasible else 'INFEASIBLE'}, "
                 f"{'converged' if self.success else 'not converged'})"]
        for n, v in self.x_named.items():
            lines.append(f"  {n} = {v:.6g}")
        for n, v in self.constraint_values.items():
            lines.append(f"  [{n}] = {v:.6g}")
        return "\n".join(lines)


def _jax_value_and_grad(f):
    """Return numpy ``value(x)`` and ``grad(x)`` callables for a scalar JAX fn."""
    vg = jax.jit(jax.value_and_grad(lambda x: jnp.asarray(f(x), dtype=float).reshape(())))
    def value(x):
        return float(vg(jnp.asarray(x, dtype=float))[0])
    def grad(x):
        return np.asarray(vg(jnp.asarray(x, dtype=float))[1], dtype=float)
    return value, grad


def optimize_design(
    objective: Callable,
    bounds: Sequence,
    *,
    input_names: Optional[Sequence[str]] = None,
    constraints: Sequence[Constraint] = (),
    x0: Optional[Sequence[float]] = None,
    maximize: bool = False,
    method: str = "SLSQP",
    n_starts: int = 1,
    seed: int = 0,
    tol: float = 1e-6,
    constraint_tol: float = 1e-4,
) -> OptimizeResult:
    """Minimise (or maximise) an objective over a bounded design space subject to
    inequality constraints, using autodiff gradients.

    The canonical use is "size a design to a permit at minimum cost": ``objective``
    is an operational-cost / energy metric and each :class:`Constraint` is an
    effluent-quality ceiling. ``objective`` and the constraint functions share the
    ``fn(x) -> scalar`` contract of :func:`monte_carlo` -- they build the params /
    initial state and run the solve themselves -- and must be JAX-differentiable,
    since their gradients are taken by autodiff and passed to the optimizer (a
    gradient-based, constrained NLP solver via SciPy).

    Parameters
    ----------
    objective : callable
        ``objective(x) -> scalar`` to minimise (or maximise; see ``maximize``).
    bounds : sequence of (low, high)
        Box bounds for each design variable, length ``d``.
    input_names : sequence of str, optional
        Design-variable names (defaults to ``x0..``).
    constraints : sequence of Constraint
        Inequality constraints ``lower <= c.fn(x) <= upper``.
    x0 : sequence of float, optional
        Starting point. Defaults to the box centre; with ``n_starts > 1`` it is
        ignored in favour of quasi-random starts.
    maximize : bool
        Maximise instead of minimise.
    method : str
        SciPy constrained method (default ``"SLSQP"``; ``"trust-constr"`` also
        works with bounds + constraints).
    n_starts : int
        Multistart count -- quasi-random (Sobol) starts in the box; the best
        feasible optimum is returned. Escapes local minima on multimodal designs.
    seed : int
        Seed for the multistart sampler (reproducible).
    tol, constraint_tol : float
        Optimizer tolerance and the slack within which a constraint counts as
        satisfied when judging feasibility / picking the multistart winner.

    Returns
    -------
    OptimizeResult
    """
    from scipy.optimize import minimize

    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    d = len(bounds)
    if input_names is None:
        input_names = [f"x{j}" for j in range(d)]
    elif len(input_names) != d:
        raise ValueError(
            f"input_names has {len(input_names)} entries but bounds has d={d}.")
    input_names = list(input_names)
    sign = -1.0 if maximize else 1.0

    obj_val, obj_grad = _jax_value_and_grad(lambda x: sign * objective(x))
    con_val_grad = [(c, *_jax_value_and_grad(c.fn)) for c in constraints]

    # SciPy inequality constraints: g(x) >= 0. upper -> upper - fn >= 0;
    # lower -> fn - lower >= 0. Jacobians come from the autodiff gradient.
    scipy_cons = []
    for c, cval, cgrad in con_val_grad:
        if c.upper is not None:
            scipy_cons.append({"type": "ineq",
                               "fun": (lambda x, cv=cval, u=c.upper: u - cv(x)),
                               "jac": (lambda x, cg=cgrad: -cg(x))})
        if c.lower is not None:
            scipy_cons.append({"type": "ineq",
                               "fun": (lambda x, cv=cval, lo=c.lower: cv(x) - lo),
                               "jac": (lambda x, cg=cgrad: cg(x))})

    # Starting points: x0 (or box centre) for a single start; quasi-random
    # otherwise.
    if n_starts > 1:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        U, _ = _unit_sample(d, n_starts, "sobol", seed)
        starts = lo[None, :] + (hi - lo)[None, :] * U[:n_starts]
    else:
        if x0 is None:
            starts = np.array([[0.5 * (b[0] + b[1]) for b in bounds]])
        else:
            starts = np.asarray(x0, dtype=float).reshape(1, d)

    def _feasible(x):
        for c, cval, _ in con_val_grad:
            v = cval(x)
            if c.upper is not None and v > c.upper + constraint_tol:
                return False
            if c.lower is not None and v < c.lower - constraint_tol:
                return False
        return True

    best = None
    for s in starts:
        res = minimize(obj_val, np.asarray(s, dtype=float), jac=obj_grad,
                       bounds=bounds, constraints=scipy_cons, method=method,
                       tol=tol)
        feas = _feasible(res.x)
        # Prefer a feasible point; among same feasibility, the lower objective.
        key = (not feas, float(res.fun))
        if best is None or key < best[0]:
            best = (key, res, feas)

    _, res, feas = best
    cvals = {(c.name or f"c{i}"): float(cval(res.x))
             for i, (c, cval, _) in enumerate(con_val_grad)}
    return OptimizeResult(
        input_names=input_names,
        x=np.asarray(res.x, dtype=float),
        objective=sign * float(res.fun),
        constraint_values=cvals,
        feasible=bool(feas),
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(getattr(res, "nit", 0)),
        n_starts=int(starts.shape[0]),
    )
