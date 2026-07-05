"""Profile-likelihood identifiability analysis.

The rigorous companion to the Laplace covariance from :func:`aquakin.calibrate`.
A profile likelihood fixes one quantity --- a rate parameter or an initial
concentration --- at each value on a grid, re-optimises *all the other* free
quantities, and traces the best attainable objective. The confidence interval is
the range over which that profile rises by less than a likelihood-ratio
threshold above its minimum.

Unlike the Laplace approximation (a local quadratic at the optimum), the profile
is exact for non-quadratic and non-identifiable parameters: a parameter the data
cannot pin shows up as a flat or one-sided profile that never crosses the
threshold, i.e. an open confidence interval --- precisely the diagnosis a
quadratic approximation cannot give.

Each grid point is a single :func:`aquakin.calibrate` call with the profiled
quantity pinned, so the calibrate options (multistart, the Gauss-Newton
optimiser, free initial conditions, priors) all flow through. A warm-started
continuation sweep keeps consecutive grid points in the same local minimum, so
the profile is smooth rather than jagged on multimodal landscapes; a polish pass
re-fits any point a better-fitting neighbour can improve.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Optional

import jax.numpy as jnp
import numpy as np

from aquakin.integrate._common import Reactor
from aquakin.integrate.calibrate import (
    FreeICConfig,
    OptimizerConfig,
    calibrate,
)


@dataclass
class ProfileResult:
    """Result of :func:`profile_likelihood`.

    Attributes
    ----------
    profiled : str
        Name of the profiled quantity (a parameter or a species).
    grid : np.ndarray
        The fixed values, shape ``(n_grid,)``.
    loss : np.ndarray
        Best attainable objective at each grid value (``nan`` where the inner
        fit failed).
    delta_loss : np.ndarray
        ``loss - min(loss)``: the profile relative to its minimum, which is what
        the likelihood-ratio threshold applies to.
    mle : float
        Grid value at the profile minimum (the maximum-likelihood estimate, to
        grid resolution).
    ci : tuple[float | None, float | None]
        ``(lo, hi)`` confidence bounds, the interpolated points where
        ``delta_loss`` crosses ``delta`` either side of the minimum. ``None`` on
        a side means the profile never crosses the threshold there (the bound is
        open / the parameter is not identified on that side).
    fits : list[CalibrationResult | None]
        The re-optimised fit at each grid point, for extracting RMSEs or the
        re-optimised parameters. ``None`` where the inner fit failed.
    delta : float
        The likelihood-ratio threshold used.
    """

    profiled: str
    grid: np.ndarray
    loss: np.ndarray
    delta_loss: np.ndarray
    mle: float
    ci: tuple[Optional[float], Optional[float]]
    fits: list
    delta: float


def _lin_cross(x0, y0, x1, y1, yt):
    """Linear interpolation: the ``x`` where the segment crosses level ``yt``."""
    if y1 == y0:
        return float(x0)
    return float(x0 + (yt - y0) * (x1 - x0) / (y1 - y0))


def _interp_ci(grid, delta_loss, delta):
    """Confidence bounds = interpolated crossings of ``delta`` either side of the
    profile minimum.

    A side that runs to the grid edge with every point below ``delta`` is
    genuinely open and returns ``None``. A side whose outward scan hits a NaN
    (a failed inner fit) *before* a crossing is **indeterminate** --- the
    crossing may be hidden in the failed region --- so it also returns ``None``
    but emits a warning, so a numerical failure is not silently reported as
    non-identifiability.
    """
    if not np.any(np.isfinite(delta_loss)):
        return (None, None)
    imin = int(np.nanargmin(delta_loss))

    def _scan(indices):
        # Walk outward from the minimum over adjacent (inner, outer) pairs.
        # Stop at the first crossing, or at the first NaN (which blocks the
        # side). Returns (bound_or_None, blocked_by_nan).
        for inner, outer in indices:
            y_in, y_out = delta_loss[inner], delta_loss[outer]
            if not (np.isfinite(y_in) and np.isfinite(y_out)):
                return None, True
            if y_in < delta <= y_out:
                return _lin_cross(grid[outer], y_out, grid[inner], y_in, delta), False
        return None, False

    lo, lo_blocked = _scan([(i, i - 1) for i in range(imin, 0, -1)])
    hi, hi_blocked = _scan([(i, i + 1) for i in range(imin, len(grid) - 1)])
    if lo is None and lo_blocked:
        warnings.warn(
            "profile lower confidence bound is indeterminate: a failed inner "
            "fit (NaN loss) lies between the minimum and the lower grid edge, "
            "so the open bound may reflect a solver failure rather than "
            "non-identifiability.",
            stacklevel=2,
        )
    if hi is None and hi_blocked:
        warnings.warn(
            "profile upper confidence bound is indeterminate: a failed inner "
            "fit (NaN loss) lies between the minimum and the upper grid edge, "
            "so the open bound may reflect a solver failure rather than "
            "non-identifiability.",
            stacklevel=2,
        )
    return (lo, hi)


def profile_likelihood(
    reactor: Reactor,
    C0: jnp.ndarray,
    observations: jnp.ndarray,
    t_obs: jnp.ndarray,
    free_params: list[str],
    *,
    grid,
    profile_param: Optional[str] = None,
    profile_ic: Optional[str] = None,
    delta: float = 1.92,
    warm_start: bool = True,
    polish: bool = True,
    polish_passes: int = 2,
    polish_tol: float = 0.05,
    anchor: Optional[float] = None,
    transforms: Optional[dict[str, str]] = None,
    initial_params: Optional[jnp.ndarray] = None,
    observed_species: Optional[list[str]] = None,
    loss: str = "nll",
    sigma=None,
    priors: Optional[dict[str, tuple[float, float]]] = None,
    use_priors: bool = True,
    free_ic: Optional[FreeICConfig] = None,
    optimizer: OptimizerConfig = OptimizerConfig(n_starts=8),
) -> ProfileResult:
    """Profile-likelihood analysis of one parameter or initial condition.

    Fixes the profiled quantity at each value in ``grid``, re-optimises every
    other free quantity with :func:`aquakin.calibrate`, and returns the profile
    of best-attainable objective plus the likelihood-ratio confidence interval.

    Exactly one of ``profile_param`` or ``profile_ic`` must be given. The
    profiled quantity is removed from the corresponding free set automatically if
    present. Single batch only: ``C0`` / ``observations`` / ``t_obs`` are single
    arrays, not lists.

    Parameters
    ----------
    reactor, C0, observations, t_obs, free_params
        As for :func:`aquakin.calibrate` (single batch).
    grid : array-like
        Values at which to fix the profiled quantity.
    profile_param : str, optional
        Name of a rate parameter to profile.
    profile_ic : str, optional
        Name of a species whose initial concentration to profile.
    delta : float, optional
        Likelihood-ratio threshold for the confidence interval. Default ``1.92``
        = ``0.5 * chi2_{1, 0.95}`` (the one-degree-of-freedom 95% level). Use
        with ``loss="nll"`` and a calibrated ``sigma`` so the objective is a
        proper negative log-likelihood and the threshold is meaningful.
    warm_start : bool, optional
        If ``True`` (default), run a cold multistart only at the ``anchor`` grid
        point and warm-start each subsequent point from its neighbour's fit, so
        consecutive points stay in one local minimum (a continuation sweep). If
        ``False``, run an independent multistart at every grid point.
    polish : bool, optional
        If ``True`` (default), after the sweep re-fit any grid point whose loss
        exceeds a neighbour's (by more than ``polish_tol``), warm-started from
        that neighbour, for up to ``polish_passes`` passes. Removes points
        stranded in a worse local minimum than the continuation found nearby.
    anchor : float, optional
        Grid value to start the continuation sweep from. Defaults to the grid
        midpoint. Ignored when ``warm_start=False``.
    initial_params, transforms, observed_species, loss, sigma, priors,
    use_priors, free_ic, optimizer
        Forwarded to each inner :func:`aquakin.calibrate` call (``free_ic`` is a
        :class:`~aquakin.FreeICConfig`, ``optimizer`` an
        :class:`~aquakin.OptimizerConfig`; the profiled parameter/species is
        removed from ``free_ic`` for the inner fits). ``optimizer.n_starts``
        applies to the cold anchor (and to every point when
        ``warm_start=False``); warm-started points use a single start. The inner
        fits force ``laplace=False`` (the profile, not the Laplace posterior, is
        the identifiability estimate here).

    Returns
    -------
    ProfileResult
    """
    if (profile_param is None) == (profile_ic is None):
        raise ValueError("Pass exactly one of profile_param or profile_ic.")
    if isinstance(C0, (list, tuple)):
        raise NotImplementedError(
            "profile_likelihood supports a single batch; C0 must be one vector."
        )
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("grid must be a non-empty 1-D array.")
    if delta <= 0:
        raise ValueError(f"delta must be > 0; got {delta}.")

    model = reactor.model
    C0 = jnp.asarray(C0)
    base_params = (
        jnp.asarray(initial_params) if initial_params is not None else model.default_parameters()
    )

    # Resolve the profiled quantity and strip it from the relevant free set.
    inner_free = list(free_params)
    inner_free_ic = list(free_ic.species) if free_ic is not None else []
    if profile_param is not None:
        if profile_param not in model.param_index:
            raise KeyError(
                f"Unknown profile_param '{profile_param}'. Available: {model.parameters}"
            )
        inner_free = [p for p in inner_free if p != profile_param]
        p_idx = model.param_index[profile_param]
        profiled = profile_param
    else:
        if profile_ic not in model.species_index:
            raise KeyError(f"Unknown profile_ic '{profile_ic}'. Available: {model.species}")
        inner_free_ic = [s for s in inner_free_ic if s != profile_ic]
        s_idx = model.species_index[profile_ic]
        profiled = profile_ic
    if not inner_free:
        raise ValueError(
            "After removing the profiled quantity, free_params is empty. Each "
            "grid point is a calibrate() fit, which needs at least one free rate "
            "parameter to re-optimise: free at least one other parameter (when "
            "profiling a parameter), or keep the rate(s) in free_params (when "
            "profiling an initial condition)."
        )

    # One compiled-objective cache shared across every grid point: the inner
    # fits are structurally identical (same reactor, observations, free set,
    # transforms, loss) and differ only in the pinned value / warm start, which
    # calibrate threads as runtime arguments -- so they reuse one compiled
    # program instead of recompiling the stiff objective + Jacobian per point.
    # Rebuild the free-IC config for the inner fits with the profiled species (if
    # any) removed; carry the caller's bounds / prior across.
    inner_free_ic_cfg = (
        replace(free_ic, species=inner_free_ic) if (free_ic is not None and inner_free_ic) else None
    )
    inner_kw = dict(
        transforms=transforms,
        observed_species=observed_species,
        loss=loss,
        sigma=sigma,
        priors=priors,
        use_priors=use_priors,
        free_ic=inner_free_ic_cfg,
        laplace=False,
        _compiled_cache={},
    )

    def _start_state(warm):
        """initial_params and C0 to seed an inner fit (warm or cold)."""
        if warm is not None:
            init_p = warm.params
            base_C0 = warm.C0_fitted[0] if warm.C0_fitted is not None else C0
        else:
            init_p, base_C0 = base_params, C0
        return init_p, base_C0

    def _fit_point(value, warm):
        init_p, base_C0 = _start_state(warm)
        if profile_param is not None:
            init_p = init_p.at[p_idx].set(value)
            C0_pt = base_C0
        else:
            C0_pt = base_C0.at[s_idx].set(value)
        # Cold (anchor) points use the full multistart; warm-continued points
        # start from the neighbour's optimum, so one start suffices.
        n = optimizer.n_starts if warm is None else 1
        try:
            return calibrate(
                reactor,
                C0_pt,
                observations,
                t_obs,
                inner_free,
                initial_params=init_p,
                optimizer=replace(optimizer, n_starts=n),
                **inner_kw,
            )
        except Exception as exc:
            # Record this point as a gap (NaN) but surface the failure: a real
            # bug (bad parameter/species name, shape mismatch) fails identically
            # at every grid point and would otherwise masquerade as a string of
            # legitimate solver failures. The exception type makes that visible.
            warnings.warn(
                f"profile inner fit failed at {profiled}={float(value):g} "
                f"({type(exc).__name__}: {exc}); recorded as a gap. A failure "
                f"at every grid point usually indicates a configuration error, "
                f"not a numerical limit.",
                stacklevel=2,
            )
            return None

    n = len(grid)
    fits: list = [None] * n

    if warm_start:
        a_idx = n // 2 if anchor is None else int(np.argmin(np.abs(grid - float(anchor))))
        fits[a_idx] = _fit_point(grid[a_idx], None)
        last_good = fits[a_idx]
        for i in range(a_idx + 1, n):  # sweep up
            fits[i] = _fit_point(grid[i], last_good)
            if fits[i] is not None:
                last_good = fits[i]
        last_good = fits[a_idx]
        for i in range(a_idx - 1, -1, -1):  # sweep down
            fits[i] = _fit_point(grid[i], last_good)
            if fits[i] is not None:
                last_good = fits[i]
    else:
        for i in range(n):
            fits[i] = _fit_point(grid[i], None)

    if polish:
        for _ in range(polish_passes):
            improved = False
            for i in range(n):
                cur = fits[i]
                for j in (i - 1, i + 1):
                    if not (0 <= j < n) or fits[j] is None:
                        continue
                    cur_loss = np.inf if cur is None else cur.loss
                    if fits[j].loss < cur_loss - polish_tol:
                        cand = _fit_point(grid[i], fits[j])
                        if cand is not None and cand.loss < cur_loss - 1e-9:
                            fits[i] = cand
                            cur = cand
                            improved = True
            if not improved:
                break

    loss = np.array([f.loss if f is not None else np.nan for f in fits])
    if not np.any(np.isfinite(loss)):
        # Every inner fit failed: there is no profile minimum to anchor on, so
        # report a clean 'unidentified' result rather than letting nanmin /
        # nanargmin raise on the all-NaN array.
        return ProfileResult(
            profiled=profiled,
            grid=grid,
            loss=loss,
            delta_loss=loss.copy(),
            mle=float("nan"),
            ci=(None, None),
            fits=fits,
            delta=delta,
        )
    delta_loss = loss - np.nanmin(loss)
    mle = float(grid[int(np.nanargmin(loss))])
    ci = _interp_ci(grid, delta_loss, delta)
    return ProfileResult(
        profiled=profiled,
        grid=grid,
        loss=loss,
        delta_loss=delta_loss,
        mle=mle,
        ci=ci,
        fits=fits,
        delta=delta,
    )
