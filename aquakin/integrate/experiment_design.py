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

from aquakin.integrate._common import Reactor, cached_jitted_solver, native_time_factor
from aquakin.integrate._linalg import _NULL_RTOL
from aquakin.integrate.fisher import Experiment, FisherInformation, _FisherSetup

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


def _criterion(F, criterion, sub=None, nui=None, rtol=_NULL_RTOL):
    """Natural criterion value of ``F`` (on the ``sub`` block, with ``nui`` as
    nuisance). A singular matrix gives the worst value (``-inf`` / ``+inf``, and
    0 for E). Singular uses the same rule as :func:`aquakin.identifiability`:
    the smallest eigenvalue is at most ``rtol`` (default ``_NULL_RTOL``) of the
    largest, so a round-off eigenvalue of either sign counts as zero. A
    ridge-regularized matrix, whose eigenvalues are bounded away from zero by
    construction, is scored with ``rtol=0``."""
    F = 0.5 * (F + F.T)
    if sub is not None:
        Fss = F[jnp.ix_(sub, sub)]
        if nui.shape[0]:
            Fsn = F[jnp.ix_(sub, nui)]
            Fss = Fss - Fsn @ jnp.linalg.solve(F[jnp.ix_(nui, nui)], Fsn.T)
        F = 0.5 * (Fss + Fss.T)
    w = jnp.linalg.eigvalsh(F)
    ok = w[0] > rtol * w[-1]
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


@dataclass(frozen=True)
class _DesignLayout:
    """Where the entries of :func:`optimize_experiment`'s design vector go.

    The design vector ``x`` lies in ``[0, 1]``: each optimised experiment's
    sampling times, then each optimised initial concentration, scaled by its
    bounds. The layout is static (it fixes the compiled program's shape and is
    part of its cache key); the bounds themselves are arguments of
    :meth:`apply`.

    Attributes
    ----------
    n_base : int
        Number of fixed base experiments before the optimised ones.
    time_slices : tuple of (int, int)
        ``(start, stop)`` of each optimised experiment's times in ``x``; empty
        when the times are not optimised.
    conc : tuple of (int, int)
        ``(experiment, species index)`` of each optimised initial
        concentration, in the order of the last entries of ``x``.
    """

    n_base: int
    time_slices: tuple
    conc: tuple

    def apply(self, x, tobs, C0s, lo, span, conc_lo, conc_hi):
        """The design's sampling times and initial states at ``x``.

        Parameters
        ----------
        x : jnp.ndarray, shape (n_x,)
            The scaled design vector.
        tobs, C0s : list of jnp.ndarray
            Every experiment's sampling times and initial state (base first);
            the optimised entries are replaced.
        lo, span : float
            Lower end and width of the sampling-time window.
        conc_lo, conc_hi : jnp.ndarray, shape (n_conc,)
            Bounds of each optimised initial concentration.

        Returns
        -------
        tobs, C0s : list of jnp.ndarray
            New lists with the optimised entries set from ``x``.
        """
        tobs, C0s = list(tobs), list(C0s)
        for e, (start, stop) in enumerate(self.time_slices):
            tobs[self.n_base + e] = lo + x[start:stop] * span
        for k, (e, j) in enumerate(self.conc):
            i = self.n_base + e
            xk = x[len(x) - len(self.conc) + k]
            C0s[i] = C0s[i].at[j].set(conc_lo[k] + xk * (conc_hi[k] - conc_lo[k]))
        return tobs, C0s


def _start_design(exps, ib, model, n_base, optimize_times, time_bounds, gap):
    """The starting design vector of :func:`optimize_experiment` and its layout.

    Parameters
    ----------
    exps : list of Experiment
        The experiments to optimise.
    ib : dict
        ``{(experiment, species): (lo, hi)}`` from :func:`_initial_bounds_map`.
    model : CompiledModel
        The reactor's model (for species indices).
    n_base : int
        Number of fixed base experiments.
    optimize_times : bool
        Whether the sampling times are optimised.
    time_bounds : tuple of float
        ``(lo, hi)`` sampling-time window, as the caller gave it.
    gap : float
        Minimum spacing between consecutive samples.

    Returns
    -------
    x0 : np.ndarray
        The starting design vector, in ``[0, 1]``.
    layout : _DesignLayout
    conc_lo, conc_hi : jnp.ndarray
        Bounds of each optimised initial concentration.

    Raises
    ------
    ValueError
        If a starting time or concentration lies outside its bounds, or the
        starting times violate the minimum spacing.
    """
    lo, hi = float(time_bounds[0]), float(time_bounds[1])
    span = hi - lo
    x0, time_slices, conc_vars = [], [], list(ib.items())
    if optimize_times:
        for e, ex in enumerate(exps):
            t = np.asarray(ex.t_obs, dtype=float)
            if t.min() < lo or t.max() > hi:
                raise ValueError(f"experiment {e}: t_obs lies outside time_bounds {time_bounds}.")
            if t.size > 1 and np.min(np.diff(t)) < gap * (1 - 1e-9):
                raise ValueError(f"experiment {e}: t_obs violates the minimum spacing {gap}.")
            time_slices.append((len(x0), len(x0) + t.size))
            x0.extend(((t - lo) / span).tolist())
    for (e, s), (clo, chi) in conc_vars:
        v = float(np.asarray(exps[e].C0)[model.species_index[s]])
        if not clo <= v <= chi:
            raise ValueError(f"experiment {e}: C0[{s!r}] = {v} lies outside {(clo, chi)}.")
        x0.append((v - clo) / (chi - clo))
    layout = _DesignLayout(
        n_base=n_base,
        time_slices=tuple(time_slices),
        conc=tuple((e, model.species_index[s]) for (e, s), _ in conc_vars),
    )
    conc_lo = jnp.asarray([b[0] for _, b in conc_vars], dtype=float)
    conc_hi = jnp.asarray([b[1] for _, b in conc_vars], dtype=float)
    return np.asarray(x0), layout, conc_lo, conc_hi


def _design_program(setup, layout, criterion, sub, nui):
    """The compiled criterion and its design gradient, shared across calls.

    The program takes ``(x, draws, tobs, C0s, lo, span, conc_lo, conc_hi)``:
    the design vector, the parameter draws, every experiment's sampling times
    and initial state, and the design bounds. It bakes in only what
    ``setup.structure_key()``, the layout, the criterion and the subset fix, so
    it is cached on those in the shared compiled-solver cache, and a repeat
    :func:`optimize_experiment` call of the same structure compiles nothing.

    Parameters
    ----------
    setup : _FisherSetup
        The resolved Fisher-information evaluation.
    layout : _DesignLayout
        Where the design vector's entries go.
    criterion : str
        The design criterion.
    sub, nui : jnp.ndarray or None
        Scored and nuisance indices, from :func:`_subset_indices`.

    Returns
    -------
    callable
        ``program(x, draws, tobs, C0s, lo, span, conc_lo, conc_hi) -> (value,
        gradient)``, the criterion averaged over the draws and its gradient
        with respect to ``x``.
    """

    def value(x, draws, tobs, C0s, lo, span, conc_lo, conc_hi):
        tobs, C0s = layout.apply(x, tobs, C0s, lo, span, conc_lo, conc_hi)
        return _mean_criterion(setup, draws, tobs, C0s, criterion, sub, nui)

    def build():
        return jax.jit(lambda x, *args: (value(x, *args), jax.jacfwd(value)(x, *args)))

    key = (
        "optimize_experiment",
        setup.structure_key(),
        layout,
        criterion,
        _index_key(sub),
        _index_key(nui),
    )
    return cached_jitted_solver(key, build, *setup.models())


def _index_key(idx) -> tuple | None:
    """A hashable form of an index array from :func:`_subset_indices`."""
    return None if idx is None else tuple(int(i) for i in np.asarray(idx))


def _spacing_constraints(layout, n_x, span, gap):
    """SLSQP linear constraints keeping each experiment's consecutive samples
    at least ``gap`` apart.

    Parameters
    ----------
    layout : _DesignLayout
        Where each experiment's times sit in the design vector.
    n_x : int
        Length of the design vector.
    span : float
        Width of the sampling-time window (the scale from ``x`` to time).
    gap : float
        Minimum spacing, in time.

    Returns
    -------
    list of dict
        One ``{"type": "ineq", "fun", "jac"}`` constraint per experiment with
        more than one sample.
    """
    constraints = []
    for start, stop in layout.time_slices:
        n = stop - start
        if n > 1:
            A = np.zeros((n - 1, n_x))
            idx = np.arange(start, stop - 1)
            A[np.arange(n - 1), idx] = -span
            A[np.arange(n - 1), idx + 1] = span
            constraints.append(
                {"type": "ineq", "fun": lambda x, A=A: A @ x - gap, "jac": lambda x, A=A: A}
            )
    return constraints


@dataclass(frozen=True)
class _DesignWindow:
    """The checked design controls of an optimized design.

    Attributes
    ----------
    ib : dict
        ``{(experiment, species): (lo, hi)}`` from :func:`_initial_bounds_map`.
    time_bounds : tuple of float
        ``(lo, hi)`` sampling-time window, in the experiments' time unit.
    lo, span : float
        Lower end and width of the window.
    gap : float
        Minimum spacing between consecutive samples (at least ``1e-6 * span``).
    t_end : float
        End of every optimized experiment's solve, in the model's native time
        unit: the window's upper end.
    """

    ib: dict
    time_bounds: tuple
    lo: float
    span: float
    gap: float
    t_end: float


def _design_window(exps, model, optimize_times, time_bounds, min_spacing, initial_bounds):
    """Check the design controls of an optimized design.

    Parameters
    ----------
    exps : list of Experiment
        The experiments to optimize.
    model : CompiledModel
        The model they run on (species and native time unit).
    optimize_times : bool
        Whether the sampling times are optimized.
    time_bounds : tuple of float or None
        ``(lo, hi)`` window for the sampling times; ``None`` for
        ``(0, latest sampling time)``.
    min_spacing : float
        Minimum gap between consecutive samples.
    initial_bounds : dict or None
        Initial concentrations to optimize, as :func:`optimize_experiment`
        takes them.

    Returns
    -------
    _DesignWindow

    Raises
    ------
    ValueError
        If nothing is to be optimized, the experiments do not share one time
        unit, or the window is not ``0 <= lo < hi``.
    """
    ib = _initial_bounds_map(initial_bounds, len(exps), model)
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
    return _DesignWindow(
        ib=ib,
        time_bounds=time_bounds,
        lo=lo,
        span=span,
        gap=max(float(min_spacing), 1e-6 * span),
        t_end=hi * factor,
    )


def _slsqp_design(value_and_grad, x0, v0, sign, layout, window, max_iter, tol):
    """Optimize a design vector by SLSQP within its box and spacing constraints.

    Parameters
    ----------
    value_and_grad : callable
        ``x -> (value, gradient)`` of the criterion at the design vector.
    x0 : np.ndarray
        The starting design vector, in ``[0, 1]``.
    v0 : float
        The criterion at ``x0``; the objective is divided by its magnitude.
    sign : float
        ``-1`` for a criterion that is better larger, ``+1`` for one better
        smaller (SLSQP minimizes).
    layout : _DesignLayout
        Where each experiment's times sit in the design vector.
    window : _DesignWindow
        The sampling-time window and minimum spacing.
    max_iter : int
        SLSQP iteration limit.
    tol : float
        SLSQP tolerance on the scaled objective.

    Returns
    -------
    x : np.ndarray
        The optimized design vector, clipped to ``[0, 1]``.
    res : scipy.optimize.OptimizeResult
        SLSQP's result.
    """
    scale = abs(v0) if v0 != 0.0 else 1.0

    def _objective(x):
        v, g = value_and_grad(x)
        return sign * float(v) / scale, sign * np.asarray(g) / scale

    res = minimize(
        _objective,
        x0,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(x0),
        constraints=_spacing_constraints(layout, len(x0), window.span, window.gap),
        options={"maxiter": max_iter, "ftol": tol},
    )
    return np.clip(res.x, 0.0, 1.0), res


def _designed_experiments(exps, n_base, tobs, C0s):
    """``exps`` with the sampling times and initial states of the design.

    Parameters
    ----------
    exps : list of Experiment
        The optimized experiments.
    n_base : int
        Number of fixed experiments before them in ``tobs`` and ``C0s``.
    tobs, C0s : list of jnp.ndarray
        Every experiment's designed sampling times and initial state.

    Returns
    -------
    list of Experiment
    """
    return [
        replace(ex, t_obs=np.asarray(tobs[n_base + e]), C0=np.asarray(C0s[n_base + e]))
        for e, ex in enumerate(exps)
    ]


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
    window = _design_window(exps, model, optimize_times, time_bounds, min_spacing, initial_bounds)
    setup = _FisherSetup(
        reactor,
        base + exps,
        free_params,
        params=params,
        transforms=transforms,
        free_ic=free_ic,
        priors=priors,
        use_priors=use_priors,
        t_end=[None] * n_base + [window.t_end] * n_exp,
    )
    sub, nui = _subset_indices(setup.names, subset)
    x0, layout, conc_lo, conc_hi = _start_design(
        exps, window.ib, model, n_base, optimize_times, window.time_bounds, window.gap
    )
    # Everything the design changes is an argument of the compiled program, so
    # a repeat call with the same structure reuses it (see _design_program).
    program = _design_program(setup, layout, criterion, sub, nui)
    bounds = (window.lo, window.span, conc_lo, conc_hi)
    args = (_draws(setup, param_draws), setup.tobs, setup.C0, *bounds)

    def value_and_grad(x):
        return program(jnp.asarray(x), *args)

    v0 = float(value_and_grad(x0)[0])
    if not np.isfinite(v0) or (criterion == "E" and v0 <= 0.0):
        raise ValueError(
            f"The starting design's information is singular for criterion {criterion!r}; "
            "start from a design that can estimate every quantity (check it with "
            "aquakin.identifiability)."
        )
    sign = -1.0 if _LARGER_IS_BETTER[criterion] else 1.0
    x, res = _slsqp_design(value_and_grad, x0, v0, sign, layout, window, max_iter, tol)
    tobs, C0s = layout.apply(jnp.asarray(x), setup.tobs, setup.C0, *bounds)
    return ExperimentDesignResult(
        experiments=_designed_experiments(exps, n_base, tobs, C0s),
        value=float(value_and_grad(x)[0]),
        initial_value=v0,
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(res.nit),
    )
