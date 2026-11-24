"""The fit run shared by :func:`aquakin.calibrate` and :meth:`Plant.calibrate`:
the multistart checks, the starting point, the run itself and the result."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquakin.integrate.calibration.coords import _ic_theta_from_pools
from aquakin.integrate.calibration.criteria import _fit_statistics
from aquakin.integrate.calibration.laplace import (
    _at_bound_named,
    _at_bound_sides,
    _laplace_posterior,
    _LaplaceOut,
)
from aquakin.integrate.calibration.multistart import (
    _clip_start,
    _optimizer_bounds,
    _resolve_starts,
    _run_multistart,
)
from aquakin.integrate.calibration.objective import _build_objective, _check_start_gradient
from aquakin.integrate.calibration.problem import _noise_estimate
from aquakin.integrate.calibration.result import CalibrationResult

if TYPE_CHECKING:
    from aquakin.integrate.calibration.config import OptimizerConfig
    from aquakin.integrate.calibration.objective import _FitConfig
    from aquakin.integrate.calibration.problem import _CalibrationProblem


def _check_start_args(optimizer: OptimizerConfig) -> None:
    """Check the multistart settings of an :class:`OptimizerConfig`.

    Shared by :func:`calibrate` and :meth:`Plant.calibrate`.

    Raises
    ------
    ValueError
        If ``n_starts < 1``, or an explicit ``starts`` list is combined with
        ``n_starts > 1``.
    """
    if optimizer.n_starts < 1:
        raise ValueError(f"optimizer.n_starts must be >= 1; got {optimizer.n_starts}.")
    if optimizer.starts is not None and optimizer.n_starts > 1:
        raise ValueError(
            "Pass optimizer.starts (an explicit start list) or optimizer.n_starts > 1 "
            "(jittered restarts), not both."
        )


def _run_fit(
    problem: _CalibrationProblem,
    fm,
    cfg: _FitConfig,
    optimizer: OptimizerConfig,
    ic_start,
    progress=None,
    *,
    build_objective=None,
    run_multistart=None,
) -> CalibrationResult:
    """Run a resolved fit and assemble its result.

    The part of a fit that does not depend on what is being fitted, shared by
    :func:`calibrate` and :meth:`Plant.calibrate`: build the compiled objective
    on the forward model, place the start (the fitted pools start from their
    prior centre unless ``ic_start`` gives them), check the start-point
    gradient, run the multistart optimiser, and assemble the
    :class:`CalibrationResult`.

    Parameters
    ----------
    problem : _CalibrationProblem
        The resolved problem.
    fm : forward model
        The forward-solve seam (``_ReactorForwardModel`` or the plant's).
    cfg : _FitConfig
        The fit configuration.
    optimizer : OptimizerConfig
        The optimiser settings; only its explicit ``starts`` list is read here
        (the rest is already in ``cfg``).
    ic_start : list, optional
        Starting pools, one per dataset, in place of the prior centre (a
        profile sweep warm-starts them from a neighbouring fit).
    progress : callable, optional
        Passed to the multistart runner when given.
    build_objective, run_multistart : callable, optional
        The objective builder and the multistart runner. ``None`` uses
        :func:`_build_objective` and :func:`_run_multistart`. A caller in
        another module passes the names it imported, so they stay the place its
        tests stub the solve out.

    Returns
    -------
    CalibrationResult
    """
    build_objective = _build_objective if build_objective is None else build_objective
    run_multistart = _run_multistart if run_multistart is None else run_multistart
    bundle = build_objective(problem, fm, cfg)

    ic_theta0 = problem.ic_center_full
    if ic_start is not None and problem.m_ic:
        ic_theta0 = _ic_theta_from_pools(problem, ic_start)
    theta0, opt_bounds, starts = _initial_point(problem, optimizer.starts, ic_theta0)

    if cfg.check_finite:
        _check_start_gradient(cfg, bundle, theta0)

    extra = {} if progress is None else {"progress": progress}
    result = run_multistart(cfg, bundle, theta0, opt_bounds, starts=starts, **extra)
    return _fit_result(problem, fm, cfg, bundle, result, opt_bounds)


def _initial_point(problem: _CalibrationProblem, starts, ic_theta0):
    """The optimiser's starting point, bounds and explicit start list.

    Returns ``(theta0, opt_bounds, starts)``. ``theta0`` is the rate start
    (from ``p0_full``) followed by ``ic_theta0``, moved inside the bounds; with
    an explicit ``OptimizerConfig.starts`` list, ``starts`` holds each entry
    (moved inside the bounds the same way) and ``theta0`` is its first one.
    Shared by :func:`calibrate` and :meth:`Plant.calibrate`."""
    rate_theta0 = problem.rate_theta0()
    opt_bounds = _optimizer_bounds(problem, rate_theta0)
    if starts is None:
        theta0 = jnp.concatenate([rate_theta0, ic_theta0]) if problem.m_ic else rate_theta0
        return _clip_start(problem, theta0, opt_bounds), opt_bounds, None
    start_thetas = [
        _clip_start(
            problem,
            jnp.concatenate([th, ic_theta0]) if problem.m_ic else th,
            opt_bounds,
        )
        for th in _resolve_starts(problem, starts)
    ]
    return start_thetas[0], opt_bounds, start_thetas


def _fit_result(problem: _CalibrationProblem, fm, cfg: _FitConfig, bundle, result, opt_bounds):
    """Assemble the :class:`CalibrationResult` of a finished fit: the fitted
    parameters and initial pools, the quantities that ended on a bound, the
    noise estimate of the concentrated likelihood, and the Laplace posterior.
    Shared by :func:`calibrate` and :meth:`Plant.calibrate`."""
    theta_opt = jnp.asarray(result.x)
    rate_theta_opt = theta_opt[: problem.n_rate]
    ic_opt = theta_opt[problem.n_rate :]
    physical_opt = problem.physical_from_theta(rate_theta_opt)
    full_params = problem.p0_full.at[problem.free_indices].set(physical_opt)

    # Fitted per-dataset initial states (when free_ic is active).
    C0_fitted = None
    ic_named = None
    ic_fractions = None
    if problem.m_ic:
        C0_fitted = []
        ic_named = []
        mt = problem.m_ic_theta
        for k, (C0_i, *_rest) in enumerate(problem.datasets):
            vals = np.asarray(problem.ic_pools(ic_opt[k * mt : (k + 1) * mt], C0_i))
            C0_fitted.append(C0_i.at[problem.ic_species_idx].set(jnp.asarray(vals)))
            ic_named.append({s: float(v) for s, v in zip(problem.free_ic, vals)})
        if problem.ic_fixed_total:
            ic_fractions = [
                {s: v / sum(pools.values()) for s, v in pools.items()} for pools in ic_named
            ]

    # --- Quantities that ended on an active bound ---
    coord_transforms = list(problem.transforms) + ["positive_log"] * (
        theta_opt.shape[0] - problem.n_rate
    )
    sides = _at_bound_sides(theta_opt, opt_bounds, coord_transforms)
    at_bound = _at_bound_named(problem, sides)
    if at_bound:
        what = (
            "their Laplace standard deviation is reported as nan and they are held "
            "on the bound in the posterior"
            if cfg.laplace
            else "a local (Laplace) uncertainty is not valid for them"
        )
        # stacklevel 3: past this helper and the calibrate entry point.
        warnings.warn(
            f"calibrate: {at_bound} ended on an optimiser bound, so {what}. Check "
            "the bound, or use profile_likelihood for a one-sided interval.",
            RuntimeWarning,
            stacklevel=3,
        )

    # Concentrated likelihood: the noise level of each output at the MAP.
    sigma_hat = None
    if problem.sigma_estimate:
        sigma_hat = np.asarray(_noise_estimate(problem, fm, theta_opt))

    # --- Laplace posterior ---
    # Rate-only by default (pools held at the MAP); joint rate + pool with
    # LaplaceConfig(include_ic=True).
    laplace_out = (
        _laplace_posterior(
            problem,
            fm,
            cfg,
            rate_theta_opt,
            ic_opt,
            sigma_hat_at_map=sigma_hat,
            at_bound_mask=sides != "",
        )
        if cfg.laplace
        else _LaplaceOut(*[None] * len(_LaplaceOut._fields))
    )

    # Report the loss as the full scalar objective at the optimum, identical
    # across optimizers (the Gauss-Newton path's ``r.cost`` drops the nll
    # sum(log(sigma)) normaliser; evaluating the objective puts both on scale),
    # and in the loss's own units (the optimiser saw it divided by loss_scale).
    reported_loss = float(bundle.value_and_grad(theta_opt)[0]) * bundle.loss_scale
    stats = _fit_statistics(problem, physical_opt, ic_opt, reported_loss)

    return CalibrationResult(
        params=full_params,
        params_named={name: float(physical_opt[i]) for i, name in enumerate(problem.free_params)},
        loss=reported_loss,
        converged=bool(result.success),
        message=str(result.message),
        n_iter=int(result.nit),
        parameter_names=list(problem.free_params),
        transforms=list(problem.transforms),
        **laplace_out._asdict(),
        priors_applied=dict(problem.active_priors),
        C0_fitted=C0_fitted,
        ic_named=ic_named,
        sigma_estimate=sigma_hat,
        ic_fractions=ic_fractions,
        at_bound=at_bound,
        **stats,
    )
