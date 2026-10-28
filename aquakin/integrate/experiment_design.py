"""Optimal experiment design on the Fisher information.

A design is scored by a scalar criterion of its Fisher information ``F`` (see
:mod:`aquakin.integrate.fisher`), computed from the eigenvalues ``w`` of ``F``:

- ``"D"``: ``log det F = sum(log w)`` -- larger is better. Minimises the volume
  of the confidence ellipsoid and is unchanged by rescaling the parameters.
- ``"A"``: ``trace(F^{-1}) = sum(1 / w)`` -- smaller is better. The summed
  variance bound.
- ``"E"``: ``min(w)`` -- larger is better. Information along the worst-determined
  direction.
- ``"modified_E"``: ``max(w) / min(w)`` -- smaller is better. The condition
  number; 1 means equally well determined in every direction.

With ``subset=`` the criterion is applied to the information about those
quantities alone, treating the rest as nuisance parameters: the Schur
complement ``F_ss - F_sn F_nn^{-1} F_ns`` (D on a subset is Ds-optimality).

Because the parameters are only known approximately before an experiment,
every scoring function accepts ``param_draws`` -- parameter vectors drawn from
a prior or a calibration posterior -- and averages the criterion over them
(robust, or pseudo-Bayesian, design).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from aquakin.integrate._common import Reactor, native_time_factor
from aquakin.integrate.fisher import Experiment, FisherInformation, _FisherSetup
from aquakin.integrate.identifiability import _NULL_RTOL

CRITERIA = ("D", "A", "E", "modified_E")
_LARGER_IS_BETTER = {"D": True, "A": False, "E": True, "modified_E": False}


def _check_criterion(criterion: str) -> None:
    if criterion not in CRITERIA:
        raise ValueError(f"Unknown criterion {criterion!r}; choose one of {CRITERIA}.")


def _subset_indices(names, subset):
    if subset is None:
        return None, None
    for n in subset:
        if n not in names:
            raise KeyError(f"Unknown subset name '{n}'. Available: {names}")
    sub = [names.index(n) for n in subset]
    return jnp.asarray(sub, dtype=int), jnp.asarray(
        [i for i in range(len(names)) if i not in sub], dtype=int
    )


def _criterion(F, criterion, sub=None, nui=None):
    """Natural criterion value of ``F`` (on the ``sub`` block, with ``nui`` as
    nuisance). A singular matrix gives the worst value (``-inf`` / ``+inf``, and
    0 for E). Singular uses the same rule as :func:`aquakin.identifiability`:
    the smallest eigenvalue is at most ``_NULL_RTOL`` of the largest, so a
    round-off eigenvalue of either sign counts as zero."""
    F = 0.5 * (F + F.T)
    if sub is not None:
        Fss = F[jnp.ix_(sub, sub)]
        if nui.shape[0]:
            Fsn = F[jnp.ix_(sub, nui)]
            Fss = Fss - Fsn @ jnp.linalg.solve(F[jnp.ix_(nui, nui)], Fsn.T)
        F = 0.5 * (Fss + Fss.T)
    w = jnp.linalg.eigvalsh(F)
    ok = w[0] > _NULL_RTOL * w[-1]
    safe = jnp.where(ok, w, 1.0)
    if criterion == "D":
        return jnp.where(ok, jnp.sum(jnp.log(safe)), -jnp.inf)
    if criterion == "A":
        return jnp.where(ok, jnp.sum(1.0 / safe), jnp.inf)
    if criterion == "E":
        return jnp.where(ok, w[0], 0.0)
    return jnp.where(ok, safe[-1] / safe[0], jnp.inf)


def design_criterion(
    fim: FisherInformation | np.ndarray,
    criterion: str = "D",
    *,
    subset: list[str] | None = None,
    names: list[str] | None = None,
) -> float:
    """Scalar design criterion of a Fisher information matrix.

    Parameters
    ----------
    fim : FisherInformation or array_like
        The information, from :func:`aquakin.fisher_information` or as a bare
        ``(n, n)`` matrix.
    criterion : {"D", "A", "E", "modified_E"}
        ``"D"`` (log det, larger is better), ``"A"`` (trace of the inverse,
        smaller is better), ``"E"`` (smallest eigenvalue, larger is better) or
        ``"modified_E"`` (condition number, smaller is better).
    subset : list of str, optional
        Score only the information about these quantities, treating the others
        as nuisance parameters (the Schur complement). Needs names: from the
        ``FisherInformation``, or ``names=`` for a bare matrix.
    names : list of str, optional
        Row names of a bare matrix, for ``subset``.

    Returns
    -------
    float
        The criterion value. A singular matrix gives ``-inf`` for D, ``+inf``
        for A and modified-E, and 0 for E. A matrix counts as singular when its
        smallest eigenvalue is at most 1e-10 of its largest, the rule
        :func:`aquakin.identifiability` uses for an exactly non-identifiable
        direction, so round-off eigenvalues of either sign are treated as zero.
    """
    _check_criterion(criterion)
    if isinstance(fim, FisherInformation):
        F, names = fim.matrix, fim.names
    else:
        F = fim
    if subset is not None and names is None:
        raise ValueError("Pass names= to score a subset of a bare matrix.")
    sub, nui = _subset_indices(names, subset)
    return float(_criterion(jnp.asarray(F, dtype=float), criterion, sub, nui))


def _mean_criterion(setup, draws, tobs, C0s, criterion, sub, nui):
    """Criterion averaged over the parameter draws (a ``(n_draw, n_params)``
    array). Pure JAX."""
    vals = jax.vmap(lambda p: _criterion(setup.information(p, tobs, C0s), criterion, sub, nui))(
        draws
    )
    return jnp.mean(vals)


def _draws(setup, param_draws):
    if param_draws is None:
        return setup.p0[None, :]
    draws = jnp.atleast_2d(jnp.asarray(param_draws, dtype=float))
    if draws.shape[1] != setup.p0.shape[0]:
        raise ValueError(
            f"param_draws has {draws.shape[1]} columns; the model has "
            f"{setup.p0.shape[0]} parameters."
        )
    return draws


def _structure_key(ex: Experiment) -> tuple:
    """What fixes the shape of an experiment's compiled evaluation: everything
    except its sampling-time values and initial state."""
    sigma = np.asarray(ex.sigma, dtype=float)
    obs = tuple(ex.observed_species) if ex.observed_species is not None else None
    mask = getattr(ex, "mask", None)
    mask = None if mask is None else np.asarray(mask, dtype=bool)
    return (
        id(ex.reactor),
        obs,
        id(getattr(ex, "observation_fn", None)),
        int(np.asarray(ex.t_obs).shape[0]),
        ex.time_unit,
        sigma.shape,
        sigma.tobytes(),
        None if mask is None else (mask.shape, mask.tobytes()),
    )


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's score from :func:`rank_experiments`.

    Attributes
    ----------
    index : int
        Position of the candidate in the list passed in.
    value : float
        Criterion of the base design plus this candidate (averaged over any
        parameter draws).
    improvement : float or None
        How much the candidate improves on the base design alone, signed so
        that positive is better for every criterion. ``None`` without a base.
    """

    index: int
    value: float
    improvement: float | None


def rank_experiments(
    reactor: Reactor,
    candidates: list[Experiment],
    free_params: list[str],
    *,
    base: list[Experiment] | None = None,
    criterion: str = "D",
    subset: list[str] | None = None,
    params: jnp.ndarray | None = None,
    param_draws: jnp.ndarray | None = None,
    transforms: dict[str, str] | None = None,
    free_ic=None,
    priors: dict[str, tuple[float, float]] | None = None,
    use_priors: bool = False,
) -> list[CandidateScore]:
    """Rank candidate experiments by the information they add to a base design.

    Each candidate is scored as ``criterion(base + [candidate])``.

    Parameters
    ----------
    reactor : Reactor
        As for :func:`aquakin.fisher_information`.
    candidates : list of Experiment
        The experiments to choose between.
    free_params : list of str
        Rate parameters to estimate.
    base : list of Experiment, optional
        Experiments already run or planned. With a base, each score also
        reports its improvement over the base alone.
    criterion : {"D", "A", "E", "modified_E"}
        See :func:`design_criterion`.
    subset : list of str, optional
        Names to score (others are nuisance), as for :func:`design_criterion`.
        A candidate's own pools are named ``ic[len(base)].<species>``.
    params : jnp.ndarray, optional
        Parameter vector to evaluate at (default: the model defaults).
    param_draws : jnp.ndarray, optional
        ``(n_draw, n_params)`` parameter vectors, e.g. from a prior or a
        calibration posterior. The criterion is averaged over them (robust
        design); ``params`` is then ignored.
    transforms, free_ic, priors, use_priors
        As for :func:`aquakin.fisher_information`.

    Returns
    -------
    list of CandidateScore
        Best candidate first.
    """
    _check_criterion(criterion)
    base = list(base or [])
    if not candidates:
        raise ValueError("candidates must contain at least one Experiment.")
    common = dict(
        params=params, transforms=transforms, free_ic=free_ic, priors=priors, use_priors=use_priors
    )

    def _score(exps):
        setup = _FisherSetup(reactor, exps, free_params, **common)
        sub, nui = _subset_indices(setup.names, subset)
        draws = _draws(setup, param_draws)
        return float(_mean_criterion(setup, draws, setup.tobs, setup.C0, criterion, sub, nui))

    # Candidates that differ only in sampling times and initial state share one
    # compiled evaluator, with those passed as arguments; compiling per
    # candidate would cost a nested-derivative solve compilation each.
    n_base = len(base)
    checked = [_FisherSetup(reactor, base + [c], free_params, **common) for c in candidates]
    groups: dict = {}
    for i, c in enumerate(candidates):
        groups.setdefault(_structure_key(c), []).append(i)
    values = [0.0] * len(candidates)
    for idxs in groups.values():
        t_end = max(checked[i].t_end[-1] for i in idxs)
        setup = _FisherSetup(
            reactor,
            base + [candidates[idxs[0]]],
            free_params,
            t_end=[None] * n_base + [t_end],
            **common,
        )
        sub, nui = _subset_indices(setup.names, subset)
        draws = _draws(setup, param_draws)
        evaluate = jax.jit(
            lambda tobs, C0, setup=setup, sub=sub, nui=nui, draws=draws: _mean_criterion(
                setup,
                draws,
                setup.tobs[:n_base] + [tobs],
                setup.C0[:n_base] + [C0],
                criterion,
                sub,
                nui,
            )
        )
        for i in idxs:
            values[i] = float(evaluate(checked[i].tobs[-1], checked[i].C0[-1]))

    sign = 1.0 if _LARGER_IS_BETTER[criterion] else -1.0
    base_value = _score(base) if base else None
    scores = [
        CandidateScore(
            index=i, value=v, improvement=None if base_value is None else sign * (v - base_value)
        )
        for i, v in enumerate(values)
    ]
    return sorted(scores, key=lambda c: -sign * c.value)


@dataclass
class ExperimentDesignResult:
    """Result of :func:`optimize_experiment`.

    Attributes
    ----------
    experiments : list of Experiment
        The optimised experiments (same order as passed in), with their
        sampling times and initial states at the optimum.
    value : float
        Criterion at the optimum (averaged over any parameter draws).
    initial_value : float
        Criterion of the starting design.
    success : bool
        Whether the optimiser reported convergence.
    message : str
        The optimiser's status message.
    n_iter : int
        Optimiser iterations.
    """

    experiments: list[Experiment]
    value: float
    initial_value: float
    success: bool
    message: str
    n_iter: int


def _initial_bounds_map(initial_bounds, n_exp, model):
    """``{(e, species): (lo, hi)}`` from a species- or (e, species)-keyed dict."""
    out = {}
    for key, (lo, hi) in (initial_bounds or {}).items():
        targets = [(e, key) for e in range(n_exp)] if isinstance(key, str) else [tuple(key)]
        for e, s in targets:
            if s not in model.species_index:
                raise KeyError(f"Unknown species '{s}' in initial_bounds.")
            if not 0 <= e < n_exp:
                raise ValueError(f"initial_bounds refers to experiment {e}; there are {n_exp}.")
            if not lo < hi:
                raise ValueError(f"initial_bounds for {s!r} must satisfy lo < hi; got {(lo, hi)}.")
            out[(e, s)] = (float(lo), float(hi))
    return out


def optimize_experiment(
    reactor: Reactor,
    experiments: Experiment | list[Experiment],
    free_params: list[str],
    *,
    criterion: str = "D",
    subset: list[str] | None = None,
    optimize_times: bool = True,
    time_bounds: tuple[float, float] | None = None,
    min_spacing: float = 0.0,
    initial_bounds: dict | None = None,
    base: list[Experiment] | None = None,
    params: jnp.ndarray | None = None,
    param_draws: jnp.ndarray | None = None,
    transforms: dict[str, str] | None = None,
    free_ic=None,
    priors: dict[str, tuple[float, float]] | None = None,
    use_priors: bool = False,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> ExperimentDesignResult:
    """Optimise sampling times and initial concentrations for a design criterion.

    The criterion is differentiated with respect to the design by JAX and
    optimised by SLSQP under box bounds and a minimum spacing between samples.
    The number of samples per experiment is fixed by its ``t_obs``, which is
    also the starting point.

    Parameters
    ----------
    reactor : Reactor
        As for :func:`aquakin.fisher_information`.
    experiments : Experiment or list of Experiment
        The experiments to optimise.
    free_params : list of str
        Rate parameters to estimate.
    criterion : {"D", "A", "E", "modified_E"}
        See :func:`design_criterion`.
    subset : list of str, optional
        Names to score (others are nuisance). With a ``base``, the optimised
        experiments' pools are named from ``ic[len(base)]`` on.
    optimize_times : bool
        Optimise the sampling times (default ``True``).
    time_bounds : tuple of float, optional
        ``(lo, hi)`` range for the sampling times, in the experiments' time
        unit. Defaults to ``(0, latest sampling time)``. Each optimised
        experiment is integrated to ``hi``.
    min_spacing : float
        Minimum gap between consecutive samples of an experiment, in the same
        unit. Default 0 (samples only have to stay in order).
    initial_bounds : dict, optional
        Initial concentrations to optimise, as ``{species: (lo, hi)}`` (every
        optimised experiment) or ``{(experiment_index, species): (lo, hi)}``.
    base : list of Experiment, optional
        Fixed experiments whose information is included but not optimised.
    params, param_draws
        Parameter vector, or ``(n_draw, n_params)`` draws to average the
        criterion over (robust design), as for :func:`rank_experiments`.
    transforms, free_ic, priors, use_priors
        As for :func:`aquakin.fisher_information`.
    max_iter : int
        SLSQP iteration limit.
    tol : float
        SLSQP tolerance on the criterion, relative to its starting value.

    Returns
    -------
    ExperimentDesignResult

    Raises
    ------
    ValueError
        If nothing is to be optimised, the starting design violates the bounds
        or spacing, or its information is singular for the criterion.

    Examples
    --------
    >>> res = aquakin.optimize_experiment(
    ...     reactor, aquakin.Experiment(C0, t_obs=jnp.linspace(1, 30, 6), sigma=0.02,
    ...                                 observed_species=["O3"]),
    ...     ["r1.k1", "r2.k2"], time_bounds=(0.5, 60.0), min_spacing=2.0,
    ...     initial_bounds={"O3": (0.5, 5.0)})
    >>> res.experiments[0].t_obs, res.experiments[0].C0
    """
    _check_criterion(criterion)
    exps = [experiments] if isinstance(experiments, Experiment) else list(experiments)
    base = list(base or [])
    if not exps:
        raise ValueError("experiments must contain at least one Experiment.")
    model = reactor.model
    n_base, n_exp = len(base), len(exps)
    ib = _initial_bounds_map(initial_bounds, n_exp, model)
    if not optimize_times and not ib:
        raise ValueError("Nothing to optimise: set optimize_times and/or initial_bounds.")

    units = {ex.time_unit for ex in exps}
    if len(units) > 1:
        raise ValueError("The optimised experiments must share one time_unit.")
    factor = native_time_factor(model.time_unit, units.pop())
    if time_bounds is None:
        time_bounds = (0.0, max(float(np.max(np.asarray(ex.t_obs))) for ex in exps))
    lo, hi = float(time_bounds[0]), float(time_bounds[1])
    if not 0.0 <= lo < hi:
        raise ValueError(f"time_bounds must satisfy 0 <= lo < hi; got {time_bounds}.")
    span = hi - lo
    gap = max(float(min_spacing), 1e-6 * span)

    setup = _FisherSetup(
        reactor,
        base + exps,
        free_params,
        params=params,
        transforms=transforms,
        free_ic=free_ic,
        priors=priors,
        use_priors=use_priors,
        t_end=[None] * n_base + [hi * factor] * n_exp,
    )
    sub, nui = _subset_indices(setup.names, subset)
    draws = _draws(setup, param_draws)

    # Design vector x in [0, 1]: each optimised experiment's times, then each
    # optimised initial concentration, scaled by its bounds.
    x0, time_slices, conc_vars = [], [], list(ib.items())
    if optimize_times:
        for e, ex in enumerate(exps):
            t = np.asarray(ex.t_obs, dtype=float)
            if t.min() < lo or t.max() > hi:
                raise ValueError(f"experiment {e}: t_obs lies outside time_bounds {time_bounds}.")
            if t.size > 1 and np.min(np.diff(t)) < gap * (1 - 1e-9):
                raise ValueError(f"experiment {e}: t_obs violates the minimum spacing {gap}.")
            time_slices.append(slice(len(x0), len(x0) + t.size))
            x0.extend(((t - lo) / span).tolist())
    for (e, s), (clo, chi) in conc_vars:
        v = float(np.asarray(exps[e].C0)[model.species_index[s]])
        if not clo <= v <= chi:
            raise ValueError(f"experiment {e}: C0[{s!r}] = {v} lies outside {(clo, chi)}.")
        x0.append((v - clo) / (chi - clo))
    x0 = np.asarray(x0)

    def _design(x):
        tobs, C0s = list(setup.tobs), list(setup.C0)
        if optimize_times:
            for e, sl in enumerate(time_slices):
                tobs[n_base + e] = (lo + x[sl] * span) * factor
        for k, ((e, s), (clo, chi)) in enumerate(conc_vars):
            i = n_base + e
            j = len(x) - len(conc_vars) + k
            C0s[i] = C0s[i].at[model.species_index[s]].set(clo + x[j] * (chi - clo))
        return tobs, C0s

    def _value(x):
        tobs, C0s = _design(x)
        return _mean_criterion(setup, draws, tobs, C0s, criterion, sub, nui)

    value_and_grad = jax.jit(lambda x: (_value(x), jax.jacfwd(_value)(x)))
    v0 = float(value_and_grad(jnp.asarray(x0))[0])
    if not np.isfinite(v0) or (criterion == "E" and v0 <= 0.0):
        raise ValueError(
            f"The starting design's information is singular for criterion {criterion!r}; "
            "start from a design that can estimate every quantity (check it with "
            "aquakin.identifiability)."
        )
    sign = -1.0 if _LARGER_IS_BETTER[criterion] else 1.0
    scale = abs(v0) if v0 != 0.0 else 1.0

    def _objective(x):
        v, g = value_and_grad(jnp.asarray(x))
        return sign * float(v) / scale, sign * np.asarray(g) / scale

    constraints = []
    for sl in time_slices:
        n = sl.stop - sl.start
        if n > 1:
            A = np.zeros((n - 1, len(x0)))
            idx = np.arange(sl.start, sl.stop - 1)
            A[np.arange(n - 1), idx] = -span
            A[np.arange(n - 1), idx + 1] = span
            constraints.append(
                {"type": "ineq", "fun": lambda x, A=A: A @ x - gap, "jac": lambda x, A=A: A}
            )
    res = minimize(
        _objective,
        x0,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(x0),
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol},
    )
    x = np.clip(res.x, 0.0, 1.0)
    tobs, C0s = _design(jnp.asarray(x))
    out = [
        replace(ex, t_obs=np.asarray(tobs[n_base + e]) / factor, C0=np.asarray(C0s[n_base + e]))
        for e, ex in enumerate(exps)
    ]
    return ExperimentDesignResult(
        experiments=out,
        value=float(_value(jnp.asarray(x))),
        initial_value=v0,
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(res.nit),
    )
