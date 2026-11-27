"""Experiment design for discriminating between two rival models.

When two model structures both fit the data in hand, the next experiment should
be the one whose outcome tells them apart: where their predictions differ by
more than the measurement noise and the uncertainty of their fitted parameters
can explain. For a candidate design with predictions ``y1`` and ``y2`` of the
observed quantities at the sampling times, each model at its fitted
parameters, the Buzzi-Ferraris (Box-Hill) criterion is

    T = (y1 - y2)^T (2 Sigma + V1 + V2)^{-1} (y1 - y2),

where ``Sigma`` is the diagonal measurement-noise covariance and
``Vi = Ji Covi Ji^T`` model ``i``'s prediction covariance: ``Ji`` is the
sensitivity of its predictions to its fitted parameters, in the unconstrained
coordinates its fit ran in, and ``Covi`` the fit's Laplace ``posterior_cov``
in the same coordinates. The Hunter-Reiner criterion is the same with
``V1 = V2 = 0``. Larger is better for both.

The sensitivities and predictions come from the same evaluation the Fisher
information uses (:class:`aquakin.integrate.fisher._FisherSetup`), and the
design is optimized by the driver of :func:`aquakin.optimize_experiment`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._common import cached_jitted_solver
from aquakin.integrate.calibration.problem import _ReactorForwardModel
from aquakin.integrate.experiment_design import (
    CandidateScore,
    ExperimentDesignResult,
    _design_window,
    _designed_experiments,
    _slsqp_design,
    _start_design,
    _structure_key,
)
from aquakin.integrate.fisher import Experiment, _FisherSetup

DISCRIMINATION_CRITERIA = ("buzzi_ferraris", "hunter_reiner")


@dataclass(frozen=True)
class DesignConfig:
    """What :func:`optimize_discrimination` may change, and how it searches.

    Parameters
    ----------
    optimize_times : bool
        Optimize the sampling times (default ``True``).
    time_bounds : tuple of float, optional
        ``(lo, hi)`` range for the sampling times, in the experiments' time
        unit. Defaults to ``(0, latest sampling time)``. Each experiment is
        integrated to ``hi``.
    min_spacing : float
        Minimum gap between consecutive samples of an experiment, in the same
        unit. Default 0 (samples only have to stay in order).
    initial_bounds : dict, optional
        Initial concentrations to optimize, as ``{species: (lo, hi)}`` (every
        experiment) or ``{(experiment_index, species): (lo, hi)}``.
    max_iter : int
        SLSQP iteration limit.
    tol : float
        SLSQP tolerance on the criterion, relative to its starting value.
    """

    optimize_times: bool = True
    time_bounds: tuple[float, float] | None = None
    min_spacing: float = 0.0
    initial_bounds: dict | None = None
    max_iter: int = 100
    tol: float = 1e-9


class _Rival(NamedTuple):
    """One model of the pair, checked: its reactor, its fit, what the fit
    observed (``("species", names)`` or ``("observation_fn", fn)``) and the
    fit's noise level per observed output (``None`` when it has none)."""

    reactor: object
    fit: object
    observes: tuple
    noise: np.ndarray | None


def _describe(observes) -> str:
    kind, what = observes
    if kind == "species":
        return f"species {list(what)}"
    return f"observation_fn {getattr(what, '__qualname__', repr(what))}"


def _fit_noise(fit, problem) -> np.ndarray | None:
    """The fit's noise standard deviation per observed output: its
    ``sigma_estimate``, else the ``sigma`` it was given when that is one value
    or one per output (the same for every dataset); ``None`` otherwise."""
    if fit.sigma_estimate is not None:
        return np.asarray(fit.sigma_estimate, dtype=float)
    given = problem.fit_sigma
    n = problem.n_observed
    levels = []
    for s in given if isinstance(given, (list, tuple)) else [given]:
        a = None if s is None else np.asarray(s, dtype=float)
        if a is None or a.ndim > 1 or (a.ndim == 1 and a.shape != (n,)):
            return None
        levels.append(np.broadcast_to(a, (n,)))
    if any(not np.array_equal(levels[0], lv) for lv in levels[1:]):
        return None
    return np.array(levels[0])


def _rival(model, label: str, criterion: str) -> _Rival:
    """Check one ``(reactor, fit)`` pair.

    Raises
    ------
    ValueError
        If ``model`` is not a pair, the fit did not come from
        :func:`aquakin.calibrate` on a reactor, the reactor is on another model
        than the fit, or Buzzi-Ferraris has no Laplace posterior to use.
    """
    if not (isinstance(model, tuple) and len(model) == 2):
        raise ValueError(f"{label} must be a (reactor, CalibrationResult) pair.")
    reactor, fit = model
    record = getattr(fit, "_fit_record", None)
    if record is None or not isinstance(record.forward_model, _ReactorForwardModel):
        raise ValueError(
            f"{label}: discrimination design needs the CalibrationResult that "
            "aquakin.calibrate returned for a reactor; this one carries no reactor fit."
        )
    problem = record.problem
    if reactor.model is not problem.model:
        raise ValueError(f"{label}: the reactor is on a different model from its fit's.")
    if criterion == "buzzi_ferraris" and fit.posterior_cov is None:
        raise ValueError(
            f"{label}: criterion='buzzi_ferraris' needs the fit's Laplace posterior; "
            "calibrate with laplace=True, or use criterion='hunter_reiner'."
        )
    if problem.observation_fn is not None:
        observes = ("observation_fn", problem.observation_fn)
    else:
        idx = np.asarray(problem.obs_species_indices)
        observes = ("species", tuple(problem.model.species[int(i)] for i in idx))
    return _Rival(reactor, fit, observes, _fit_noise(fit, problem))


class _Pair(NamedTuple):
    """The checked pair of rivals and the design's common noise level."""

    rivals: tuple
    sigma: object


def _resolve_pair(model1, model2, criterion, sigma) -> _Pair:
    """Check the two models against each other and settle the noise level.

    ``sigma`` overrides the noise; otherwise it is the root mean of the two
    fits' variances, so ``2 Sigma`` is the sum of their noise covariances.

    Raises
    ------
    ValueError
        For an unknown criterion, models on different species lists, models
        observing different quantities, or no noise level to use.
    """
    if criterion not in DISCRIMINATION_CRITERIA:
        raise ValueError(
            f"Unknown criterion {criterion!r}; choose one of {DISCRIMINATION_CRITERIA}."
        )
    species = [list(m[0].model.species) if isinstance(m, tuple) else None for m in (model1, model2)]
    if None not in species and species[0] != species[1]:
        raise ValueError(
            "The two models must share one species list, since the design's "
            f"initial states apply to both; model 1 has {species[0]}, model 2 has {species[1]}."
        )
    r1, r2 = _rival(model1, "model 1", criterion), _rival(model2, "model 2", criterion)
    if r1.observes != r2.observes:
        raise ValueError(
            "The two models must observe the same quantities; model 1 observes "
            f"{_describe(r1.observes)}, model 2 observes {_describe(r2.observes)}."
        )
    if sigma is None:
        if r1.noise is None or r2.noise is None:
            raise ValueError(
                "A fit has no noise level of one value per observed output (it was "
                "fitted without sigma, or with a per-observation sigma); pass sigma=."
            )
        sigma = np.sqrt(0.5 * (r1.noise**2 + r2.noise**2))
    return _Pair((r1, r2), sigma)


def _prepare(ex: Experiment, e: int, pair: _Pair) -> Experiment:
    """``ex`` observing what the fits observed, at the common noise level.

    Raises
    ------
    ValueError
        If ``ex`` observes something else or carries its own reactor.
    """
    kind, what = pair.rivals[0].observes
    observed, fn = (list(what), None) if kind == "species" else (None, what)
    if (ex.observed_species is not None and list(ex.observed_species) != observed) or (
        ex.observation_fn is not None and ex.observation_fn is not fn
    ):
        raise ValueError(
            f"experiment {e}: leave observed_species and observation_fn unset or "
            f"match the fits, which observe {_describe(pair.rivals[0].observes)}."
        )
    if ex.reactor is not None:
        raise ValueError(
            f"experiment {e}: each model is solved on its own reactor; leave reactor unset."
        )
    return replace(ex, observed_species=observed, observation_fn=fn, sigma=pair.sigma)


def _setups(pair: _Pair, exps, t_end) -> tuple:
    """One :class:`_FisherSetup` per model over ``exps``, estimating each fit's
    free parameters in the coordinates (transforms) the fit ran in."""
    return tuple(
        _FisherSetup(
            r.reactor,
            exps,
            r.fit.parameter_names,
            params=r.fit.params,
            transforms=dict(zip(r.fit.parameter_names, r.fit.transforms)),
            t_end=t_end,
        )
        for r in pair.rivals
    )


def _fitted(pair: _Pair) -> tuple:
    """``(params, covs)``: each fit's parameter vector and its posterior
    covariance (zeros when the fit has none, for Hunter-Reiner)."""
    params = tuple(jnp.asarray(r.fit.params, dtype=float) for r in pair.rivals)
    covs = []
    for r in pair.rivals:
        d = len(r.fit.parameter_names)
        cov = r.fit.posterior_cov
        covs.append(jnp.zeros((d, d)) if cov is None else jnp.asarray(cov, dtype=float))
    return params, tuple(covs)


def _discrimination_value(setups, fitted, tobs, C0s, criterion):
    """The criterion of the design ``(tobs, C0s)`` at the fits ``fitted``
    (:func:`_fitted`'s ``(params, covs)``). Pure JAX.

    Both setups scale their predictions by the same noise ``Sigma^{1/2}``, so
    with ``d = Sigma^{-1/2} (y1 - y2)`` and ``Ji`` the scaled sensitivities,
    Buzzi-Ferraris is ``d^T (2 I + J1 Cov1 J1^T + J2 Cov2 J2^T)^{-1} d`` and
    Hunter-Reiner ``d^T d / 2``. An unmeasured (masked) entry is 0 in ``d`` and
    in every ``Ji``, so it adds nothing.
    """
    ps, covs = fitted
    if criterion == "hunter_reiner":
        d = setups[0].outputs(ps[0], tobs, C0s) - setups[1].outputs(ps[1], tobs, C0s)
        return 0.5 * jnp.dot(d, d)
    ys, prediction_cov = [], 0.0
    for setup, p, cov in zip(setups, ps, covs):
        y, J = setup.outputs_and_jacobian(p, tobs, C0s)
        ys.append(y)
        prediction_cov = prediction_cov + J @ cov @ J.T
    d = ys[0] - ys[1]
    A = 2.0 * jnp.eye(d.shape[0]) + prediction_cov
    return jnp.dot(d, jnp.linalg.solve(A, d))


def _cached(key, setups, build):
    """``build()``'s program from the shared compiled-solver cache, under
    ``key`` plus both setups' structure keys (holding their models alive)."""
    full_key = (*key, setups[0].structure_key(), setups[1].structure_key())
    return cached_jitted_solver(full_key, build, *setups[0].models(), *setups[1].models())


def _discrimination_program(setups, layout, criterion):
    """The compiled criterion of a design vector and its gradient.

    The program takes ``(x, fitted, tobs, C0s, lo, span, conc_lo, conc_hi)``
    and bakes in only what both setups' structure keys, the layout and the
    criterion fix, so a repeat call of the same structure compiles nothing (as
    :func:`aquakin.integrate.experiment_design._design_program` does).

    Returns
    -------
    callable
        ``program(...) -> (value, gradient)`` with respect to ``x``.
    """

    def value(x, fitted, tobs, C0s, lo, span, conc_lo, conc_hi):
        tobs, C0s = layout.apply(x, tobs, C0s, lo, span, conc_lo, conc_hi)
        return _discrimination_value(setups, fitted, tobs, C0s, criterion)

    def build():
        return jax.jit(lambda x, *args: (value(x, *args), jax.jacfwd(value)(x, *args)))

    return _cached(("optimize_discrimination", layout, criterion), setups, build)


def optimize_discrimination(
    model1: tuple,
    model2: tuple,
    experiments: Experiment | list[Experiment],
    *,
    criterion: str = "buzzi_ferraris",
    sigma=None,
    design: DesignConfig = DesignConfig(),
) -> ExperimentDesignResult:
    """Optimize sampling times and initial concentrations to tell two models apart.

    Each model is a reactor with the calibration of its parameters. The
    design maximizes how far apart the two models' predictions of the
    observed quantities are, measured against the noise and, for
    Buzzi-Ferraris, against the uncertainty of each model's prediction:

        T = (y1 - y2)^T (2 Sigma + V1 + V2)^{-1} (y1 - y2),

    with ``yi`` model ``i``'s predictions at its fitted parameters,
    ``Sigma`` the diagonal noise covariance and ``Vi = Ji Covi Ji^T`` the
    covariance of model ``i``'s predictions under its Laplace posterior.
    ``criterion="hunter_reiner"`` drops ``V1`` and ``V2``. The criterion is
    differentiated with respect to the design by JAX and optimized by SLSQP
    under box bounds and a minimum spacing, as in
    :func:`aquakin.optimize_experiment`. The number of samples per
    experiment is fixed by its ``t_obs``, which is also the starting point.

    Parameters
    ----------
    model1, model2 : tuple of (Reactor, CalibrationResult)
        Each rival model: a reactor and the result :func:`aquakin.calibrate`
        returned for it. The two models must have the same species list and
        their fits must observe the same quantities (the same
        ``observed_species``, or the same ``observation_fn`` object).
    experiments : Experiment or list of Experiment
        The experiments to design. Each one measures what the fits observed:
        leave its ``observed_species`` and ``observation_fn`` unset (or equal
        to the fits') and its ``reactor`` unset. Its ``sigma`` is not used.
    criterion : {"buzzi_ferraris", "hunter_reiner"}
        ``"buzzi_ferraris"`` (default) weighs the gap by the noise and both
        models' prediction covariance, and needs each fit's Laplace posterior;
        ``"hunter_reiner"`` weighs it by the noise alone. Larger is better.
    sigma : float or array-like, optional
        Measurement standard deviation, a scalar or one value per observed
        output. By default the root mean square of the fits' noise levels,
        per output (each fit's ``sigma_estimate``, or the ``sigma`` it was
        given), so that ``2 Sigma`` is the sum of the two noise covariances.
    design : DesignConfig, optional
        What to optimize (sampling times, initial concentrations), the
        bounds and spacing, and the SLSQP settings.

    Returns
    -------
    ExperimentDesignResult
        The designed experiments and the criterion at the optimum and at the
        start.

    Raises
    ------
    ValueError
        For an unknown criterion, a model that is not a reactor with its fit,
        models on different species or observing different quantities, a fit
        without a usable noise level and no ``sigma``, Buzzi-Ferraris without
        a Laplace posterior, an experiment that observes something else or
        carries a reactor, nothing to optimize, or a starting design outside
        its bounds or spacing.

    Examples
    --------
    >>> res = aquakin.optimize_discrimination(
    ...     (reactor1, fit1), (reactor2, fit2),
    ...     aquakin.Experiment(C0, t_obs=jnp.linspace(1, 5, 4), sigma=0.01),
    ...     design=aquakin.DesignConfig(time_bounds=(0.5, 60.0), min_spacing=2.0))
    >>> res.experiments[0].t_obs
    """
    pair = _resolve_pair(model1, model2, criterion, sigma)
    exps = [experiments] if isinstance(experiments, Experiment) else list(experiments)
    if not exps:
        raise ValueError("experiments must contain at least one Experiment.")
    model = pair.rivals[0].reactor.model
    window = _design_window(
        exps,
        model,
        design.optimize_times,
        design.time_bounds,
        design.min_spacing,
        design.initial_bounds,
    )
    prepared = [_prepare(ex, e, pair) for e, ex in enumerate(exps)]
    setups = _setups(pair, prepared, [window.t_end] * len(exps))
    x0, layout, conc_lo, conc_hi = _start_design(
        exps, window.ib, model, 0, design.optimize_times, window.time_bounds, window.gap
    )
    program = _discrimination_program(setups, layout, criterion)
    bounds = (window.lo, window.span, conc_lo, conc_hi)
    args = (_fitted(pair), setups[0].tobs, setups[0].C0, *bounds)

    def value_and_grad(x):
        return program(jnp.asarray(x), *args)

    v0 = float(value_and_grad(x0)[0])
    x, res = _slsqp_design(
        value_and_grad, x0, v0, -1.0, layout, window, design.max_iter, design.tol
    )
    tobs, C0s = layout.apply(jnp.asarray(x), setups[0].tobs, setups[0].C0, *bounds)
    return ExperimentDesignResult(
        experiments=_designed_experiments(exps, 0, tobs, C0s),
        value=float(value_and_grad(x)[0]),
        initial_value=v0,
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(res.nit),
    )


def rank_discrimination(
    model1: tuple,
    model2: tuple,
    candidates: list[Experiment],
    *,
    criterion: str = "buzzi_ferraris",
    sigma=None,
) -> list[CandidateScore]:
    """Rank candidate experiments by how well each tells two models apart.

    Each candidate is scored alone by the criterion of
    :func:`optimize_discrimination`.

    Parameters
    ----------
    model1, model2 : tuple of (Reactor, CalibrationResult)
        The rival models, as for :func:`optimize_discrimination`.
    candidates : list of Experiment
        The experiments to choose between, set up as for
        :func:`optimize_discrimination`.
    criterion : {"buzzi_ferraris", "hunter_reiner"}
        See :func:`optimize_discrimination`. Larger is better.
    sigma : float or array-like, optional
        Measurement standard deviation, as for :func:`optimize_discrimination`.

    Returns
    -------
    list of CandidateScore
        Best candidate first; ``improvement`` is ``None``.

    Raises
    ------
    ValueError
        As for :func:`optimize_discrimination`, or for an empty list.
    """
    pair = _resolve_pair(model1, model2, criterion, sigma)
    if not candidates:
        raise ValueError("candidates must contain at least one Experiment.")
    prepared = [_prepare(c, e, pair) for e, c in enumerate(candidates)]
    # Validate every candidate; those of one structure then share one
    # compiled evaluator, integrated to the latest end among them.
    checked = [_setups(pair, [c], None)[0] for c in prepared]
    groups: dict = {}
    for i, c in enumerate(prepared):
        groups.setdefault(_structure_key(c), []).append(i)
    fitted = _fitted(pair)
    values = [0.0] * len(candidates)
    for idxs in groups.values():
        t_end = max(checked[i].t_end[-1] for i in idxs)
        setups = _setups(pair, [prepared[idxs[0]]], [t_end])
        evaluate = _cached(
            ("rank_discrimination", criterion),
            setups,
            lambda setups=setups: jax.jit(
                lambda fitted, tobs, C0s: _discrimination_value(
                    setups, fitted, tobs, C0s, criterion
                )
            ),
        )
        for i in idxs:
            values[i] = float(evaluate(fitted, checked[i].tobs, checked[i].C0))
    scores = [CandidateScore(index=i, value=v, improvement=None) for i, v in enumerate(values)]
    return sorted(scores, key=lambda c: -c.value)
