"""MAP calibration with parameter transforms and Laplace-approximate posterior.

This is the richer companion to :func:`aquakin.fit`. Differences from
:func:`fit`:

- Parameters can declare a transform (``positive_log`` for ``p > 0``,
  ``logit`` for ``0 < p < 1``, ``none`` for identity). Optimisation runs in
  unconstrained space — more robust convergence than box constraints.
- Loss is configurable: ``"mse"`` (default), ``"wmse"`` (weighted MSE),
  ``"nll"`` (Gaussian negative log-likelihood). The last two accept a
  per-observation ``sigma``; ``"nll"`` also accepts ``sigma="estimate"``, the
  concentrated likelihood with the noise level of each output estimated from
  the residuals.
- Optional Laplace covariance — a Gaussian approximation of the posterior
  around the MAP, from a Hessian of the loss in unconstrained space.
  Marginal standard deviations are then propagated back to physical space
  via the transform's first derivative (delta method).

The optimiser is SciPy L-BFGS-B with the gradient supplied by ``jax.grad``
through Diffrax's adjoint, same as :func:`fit`. The Laplace Hessian is
available two ways (``laplace_method``):

- ``"fd"`` (default): finite-difference the gradient. General, but carries a
  step-size choice and FD noise.
- ``"gauss_newton"``: ``H = J^T J`` with ``J`` the residual Jacobian by
  reverse-mode AD. Exact, PSD by construction, and for ``loss="nll"`` the
  Fisher information. It needs only first-order AD, so it works with the
  default ``RecursiveCheckpointAdjoint``. The *full* AD Hessian is avoided
  on purpose: ``jax.hessian`` (forward-over-reverse) hits the adjoint's
  ``custom_vjp`` (reverse-only), and computing second derivatives through the
  stiff implicit solve is unreliable even with ``DirectAdjoint``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from aquakin.integrate._common import (
    DifferentiationConfig,
    Reactor,
    check_finite_gradient,
    forward_adjoint,
    native_time_factor,
    with_adjoint,
)
from aquakin.integrate._transforms import (
    dphysical_dunconstrained,
    from_unconstrained,
    to_unconstrained,
)
from aquakin.integrate.calibration.config import (
    _VALID_LOSSES,
    _VALID_OPTIMIZERS,
    FreeICConfig,
    LaplaceConfig,
    OptimizerConfig,
    _resolve_laplace,
)
from aquakin.integrate.calibration.coords import (
    _DEFAULT_IC_BOUNDS,
    _free_ic_fields,
    _from_unconstrained,
    _ic_coordinate_names,
    _ic_theta_from_pools,
    _jacobian_physical_wrt_theta,
    _joint_names,
    _per_species_ic_bounds,
    _resolve_ic_coords,
    _simplex_fraction_cov,
    _simplex_pools,
    _to_unconstrained,
)
from aquakin.integrate.calibration.laplace import (
    _AT_BOUND_ATOL,
    _AT_BOUND_RTOL,
    _at_bound_named,
    _at_bound_sides,
    _laplace_covariance,
    _laplace_posterior,
    _LaplaceOut,
)
from aquakin.integrate.calibration.loss import (
    _SSE_FLOOR,
    _build_loss,
    _build_masked_loss,
    _build_masked_residual,
    _build_residual,
    _check_sigma_arg,
    _coerce_sigma,
    _convert_observations,
    _data_term,
    _fill_missing,
    _is_sigma_estimate,
    _masked_parts,
    _missing_mask,
    _n_obs_per_output,
    _observation_unit_factors,
    _observed_counts,
    _pooled_sse,
    _sigma_per_dataset,
)
from aquakin.integrate.calibration.multistart import (
    _IRLS_MAX_ITER,
    _SOLVE_FAILURE_HINT,
    _clip_start,
    _is_solve_failure,
    _optimizer_bounds,
    _OptOut,
    _resolve_starts,
    _run_multistart,
)
from aquakin.integrate.calibration.objective import (
    _build_objective,
    _check_start_gradient,
    _FitConfig,
    _forward_jac,
    _ObjectiveBundle,
    _residual_parts,
)
from aquakin.integrate.calibration.observation import (
    observation_size,
    observation_takes_time,
    observe,
)
from aquakin.integrate.calibration.problem import (
    _array_digest,
    _bound_to_unconstrained,
    _CalibrationProblem,
    _noise_estimate,
    _predict,
    _ReactorForwardModel,
    _resolve_problem,
    _resolve_rate_bounds,
)
from aquakin.integrate.calibration.result import (
    CalibrationResult,
    PredictiveBand,
)
from aquakin.integrate.identifiability import _NULL_LOADING_TOL, _NULL_RTOL

# Every name this module has exported. The implementation lives in the
# ``aquakin.integrate.calibration`` subpackage; the names are re-exported here so
# existing imports (tests, ``fisher.py``, ``profile.py``, the plant layer) keep
# working. ``tests/unit/test_calibrate_exports.py`` pins the list.
__all__ = [
    "_AT_BOUND_ATOL",
    "_AT_BOUND_RTOL",
    "_DEFAULT_IC_BOUNDS",
    "_IRLS_MAX_ITER",
    "_NULL_LOADING_TOL",
    "_NULL_RTOL",
    "_SOLVE_FAILURE_HINT",
    "_SSE_FLOOR",
    "_VALID_LOSSES",
    "_VALID_OPTIMIZERS",
    "CalibrationResult",
    "DifferentiationConfig",
    "FreeICConfig",
    "LaplaceConfig",
    "OptimizerConfig",
    "PredictiveBand",
    "Reactor",
    "_CalibrationProblem",
    "_FitConfig",
    "_LaplaceOut",
    "_ObjectiveBundle",
    "_OptOut",
    "_ReactorForwardModel",
    "_array_digest",
    "_at_bound_named",
    "_at_bound_sides",
    "_bound_to_unconstrained",
    "_build_loss",
    "_build_masked_loss",
    "_build_masked_residual",
    "_build_objective",
    "_build_residual",
    "_check_sigma_arg",
    "_check_start_args",
    "_check_start_gradient",
    "_clip_start",
    "_coerce_sigma",
    "_convert_observations",
    "_data_term",
    "_fill_missing",
    "_fit_result",
    "_forward_jac",
    "_free_ic_fields",
    "_from_unconstrained",
    "_ic_coordinate_names",
    "_ic_theta_from_pools",
    "_initial_point",
    "_is_sigma_estimate",
    "_is_solve_failure",
    "_jacobian_physical_wrt_theta",
    "_joint_names",
    "_laplace_covariance",
    "_laplace_posterior",
    "_masked_parts",
    "_missing_mask",
    "_n_obs_per_output",
    "_noise_estimate",
    "_observation_unit_factors",
    "_observed_counts",
    "_optimizer_bounds",
    "_per_species_ic_bounds",
    "_pooled_sse",
    "_predict",
    "_residual_parts",
    "_resolve_ic_coords",
    "_resolve_laplace",
    "_resolve_problem",
    "_resolve_rate_bounds",
    "_resolve_starts",
    "_run_fit",
    "_run_multistart",
    "_sigma_per_dataset",
    "_simplex_fraction_cov",
    "_simplex_pools",
    "_to_unconstrained",
    "calibrate",
    "check_finite_gradient",
    "dphysical_dunconstrained",
    "forward_adjoint",
    "from_unconstrained",
    "native_time_factor",
    "observation_size",
    "observation_takes_time",
    "observe",
    "to_unconstrained",
    "with_adjoint",
]


# --- Main entry point --------------------------------------------------


def calibrate(
    reactor: Reactor,
    C0: jnp.ndarray,
    observations: jnp.ndarray,
    t_obs: jnp.ndarray,
    free_params: list[str],
    *,
    transforms: dict[str, str] | None = None,
    initial_params: jnp.ndarray | None = None,
    observed_species: list[str] | None = None,
    observation_fn=None,
    observation_units: dict[str, str] | None = None,
    time_unit: str | None = None,
    loss: str = "mse",
    sigma: jnp.ndarray | None = None,
    priors: dict[str, tuple[float, float]] | None = None,
    use_priors: bool = True,
    diff: DifferentiationConfig = DifferentiationConfig(method="through_solve"),
    optimizer: OptimizerConfig = OptimizerConfig(),
    laplace: bool | LaplaceConfig = True,
    free_ic: FreeICConfig | None = None,
    _compiled_cache: dict | None = None,
    _ic_start: list | None = None,
) -> CalibrationResult:
    """MAP fit with optional Laplace posterior approximation.

    Parameters
    ----------
    reactor : Reactor
        Batch reactor (or anything with a compatible ``solve``). Same usage
        contract as :func:`fit`.
    C0 : jnp.ndarray or list of jnp.ndarray
        Initial concentration vector. Pass a *list* of vectors for a joint
        multi-batch fit: each entry is one dataset's initial state, the
        batches share the parameter vector and prior, and their data terms are
        summed. ``observations`` and ``t_obs`` must then be matching lists.
    observations : jnp.ndarray or list of jnp.ndarray
        Observed values, shape ``(n_t,)`` for a single species or
        ``(n_t, n_observed)``. In multi-batch mode, a list of such arrays (one
        per dataset; the datasets may have different ``n_t``). A ``NaN`` entry
        is a missing observation: it is left out of the loss and the Laplace
        Hessian, so outputs measured at different times can share one
        ``t_obs`` grid (``"mse"`` / ``"wmse"`` then average over the observed
        entries only).
    t_obs : jnp.ndarray or list of jnp.ndarray
        Observation times, shape ``(n_t,)``. ``C0`` is taken at ``t=0``;
        the solver integrates from ``0`` to ``t_obs[-1]``. In multi-batch mode,
        a list of time grids, one per dataset. In the model's native time unit
        unless ``time_unit`` is given.
    free_params : list[str]
        Namespaced parameter names to calibrate. Others held fixed.
    transforms : dict[str, str], optional
        Per-parameter transform. Keys may be any subset of ``free_params``;
        unspecified entries fall back to the parameter's declared
        ``transform`` on the model (default ``"none"``).
    initial_params : jnp.ndarray, optional
        Starting parameter vector. Defaults to
        ``reactor.model.default_parameters()``.
    observed_species : list[str], optional
        Species names corresponding to columns of ``observations``. If
        ``None``, every model species is taken to be observed.
    observation_fn : callable, optional
        A measurement that is a function of several states, used in place of
        ``observed_species``: ``observation_fn(C, params) -> y`` maps one time
        point's concentration vector ``(n_species,)`` and the full parameter
        vector to the observed outputs, a scalar or ``(n_outputs,)``. It must be
        written with ``jax.numpy`` so it can be differentiated. The columns of
        ``observations`` (and a per-output ``sigma``) follow its outputs. A
        function with three required positional arguments,
        ``observation_fn(t, C, params)``, also receives the sample time ``t``
        (a scalar, in the unit of ``t_obs``), for a measurement whose mapping
        changes with time, such as a sensor drift or a time-varying dilution.
        ``t`` is the entry of ``t_obs`` itself, exactly as given. A function
        with exactly three required positional arguments is the time-aware
        form; one that can be called with two (``f(C, params=None)`` or
        ``f(C, params, scale=1.0)``, say) is the ``(C, params)`` form.
    observation_units : dict[str, str], optional
        The units of observed species measured in a unit other than the one
        the model declares for them, e.g. ``{"SNH": "mg/L", "XS":
        "kgCOD/m3"}``. Those columns of ``observations``, and of an absolute
        ``sigma``, are converted to the model's units before the fit, so the
        loss, residuals, ``sigma_estimate`` and the posterior are all in the
        model's units. Only the scale changes: a unit that measures another
        quantity (moles where the model has grams) raises. Not available with
        ``observation_fn``, whose outputs have no declared unit. ``None``
        (default) takes the observations to be in the model's units.
    time_unit : str, optional
        The time unit ``t_obs`` is expressed in (``"s"``, ``"min"``, ``"h"``,
        ``"d"``), matching :meth:`BatchReactor.solve`. Every dataset's ``t_obs``
        is converted into the model's native (rate-constant) time unit before
        the solve, so an hour-valued ``t_obs`` carried over from a
        ``solve(time_unit="h")`` run is interpreted correctly rather than as
        native-unit days (the silent 24x time-axis compression this guards
        against). Default ``None`` interprets ``t_obs`` in the native unit. The
        fitted rate constants are always in native units.
    loss : {"mse", "wmse", "nll"}, optional
        Loss function.
    sigma : jnp.ndarray or "estimate", optional
        Per-observation standard deviation for ``"wmse"`` / ``"nll"``.
        Scalar, ``(n_observed,)``, or ``(n_t, n_observed)``. In multi-batch
        mode, either a single value/array shared across datasets or a list with
        one entry per dataset.

        With ``loss="nll"``, ``sigma="estimate"`` estimates an unknown noise
        level instead: the concentrated (profile) likelihood. The noise std of
        each observed output is profiled out at its maximum-likelihood value
        ``sigma_j^2 = SSE_j / N_j`` (``SSE_j`` the sum of squared residuals of
        output ``j``, pooled over the datasets, ``N_j`` its number of observed,
        non-``NaN`` entries), which leaves the objective
        ``sum_j (N_j / 2) log(SSE_j / N_j) + N / 2`` -- the ``"nll"`` loss at the
        estimated noise. The estimate is returned as
        ``CalibrationResult.sigma_estimate``. It covers every dataset, so
        pass it as ``sigma="estimate"``, not inside a per-dataset list. The
        Gauss-Newton Laplace
        covariance is the Fisher information with the estimate plugged in (the
        usual ``sigma^2 (J^T J)^{-1}``); the finite-difference one is the
        curvature of the concentrated objective. For a profile likelihood of
        the concentrated objective the likelihood-ratio threshold is unchanged
        (1.92 for 95 %); with one output the profile rise is
        ``(N / 2) log(SSE / SSE_min)``. With ``optimizer="gauss_newton"`` the
        fit alternates least-squares solves at a fixed noise level with
        re-estimating it, to a stationary point of the concentrated objective.
    priors : dict[str, tuple[float, float]], optional
        Gaussian priors as ``name -> (mean, std)`` in physical space, added to
        the objective as ``0.5 * sum(((p - mean) / std) ** 2)``. Overrides any
        prior declared on the model for the same parameter. Only entries whose
        name is in ``free_params`` are used.
    use_priors : bool, optional
        If ``True`` (default), parameters whose model declaration carries a
        ``prior:`` block contribute their Gaussian prior to the objective (for
        the free parameters), in addition to any passed via ``priors``. Set
        ``False`` to ignore the model-declared priors. Priors regularise
        otherwise non-identifiable parameter combinations toward literature
        values; for a proper Bayesian MAP / posterior, combine them with
        ``loss="nll"`` and a measurement ``sigma`` so the data term is a true
        negative log-likelihood (the prior curvature then enters the Laplace
        covariance automatically).
    optimizer : OptimizerConfig, optional
        Optimiser backend and multistart settings; see :class:`OptimizerConfig`
        for the fields (``method`` {"lbfgsb", "gauss_newton"}, ``n_starts``,
        ``starts``, ``jitter`` / ``jitter_schedule``, ``seed``, ``max_iter``,
        ``tol``, ``bounds``, ``use_model_bounds``, ``param_halfwidth``). The free
        rate parameters stay inside their model-declared ``bounds:`` by default,
        mapped into the unconstrained space the optimiser works in; pass
        ``bounds={name: (lo, hi)}`` to set or override absolute bounds, and
        ``param_halfwidth`` for an extra box around the start. A starting value
        outside the bounds is moved to the nearest bound with a
        ``RuntimeWarning``. ``starts=[...]`` runs an explicit list of starts (a
        grid, say) through one compiled objective and keeps the best; each start
        is moved inside the bounds the same way, and the ``param_halfwidth``
        box, if any, stays centred on ``initial_params`` for every start.
        ``"gauss_newton"`` minimises the residual *vector*
        with SciPy ``least_squares`` (its Jacobian by AD, direction from
        ``diff.mode``) and is markedly more robust on the multimodal landscapes
        of stiff reaction-model fits; ``n_starts > 1`` is a deterministic,
        ``seed``-reproducible multistart that keeps the lowest-loss optimum.
        Default ``OptimizerConfig()`` (a single L-BFGS-B start).
    laplace : bool or LaplaceConfig, optional
        Laplace covariance approximation at the MAP. ``True`` (default) computes
        it with default settings, ``False`` disables it, and a
        :class:`LaplaceConfig` computes it with tuning (``method`` {"fd",
        "gauss_newton"}, ``ridge``, ``fd_step``, ``dtmax``). It is
        interpretable as a Bayesian posterior only when ``loss="nll"`` with a
        calibrated ``sigma`` (the loss then IS a proper Gaussian negative
        log-likelihood); for ``"mse"`` / ``"wmse"`` it is the inverse loss
        curvature -- the right shape but not the right absolute scale. See
        :func:`fit` if you only need point estimates.
    free_ic : FreeICConfig, optional
        If given, also fit the initial concentration of the named
        ``species`` (per dataset, in log space, box-bounded by
        :attr:`FreeICConfig.bounds` -- one pair, or per species -- with an
        optional log-space prior via ``prior_log_std``) while the rate
        parameters stay shared -- useful when an unmeasured initial state (e.g.
        a biofilm reservoir) is not known. With ``fixed_total=True`` the pools'
        sum is held at its value in ``C0`` and only their split is fitted. The
        fitted pools are returned on ``CalibrationResult.C0_fitted`` /
        ``ic_named``; the Laplace posterior is taken over the rate parameters
        with the pools held at their optimum. Default ``None`` (no free ICs).
    diff : DifferentiationConfig, optional
        How the data-term gradient / residual Jacobian is formed.
        ``mode`` ({"reverse", "forward"}) is the autodiff direction;
        ``method`` ({"stable", "through_solve"}) is how the reverse adjoint is
        formed. ``mode="reverse", method="through_solve"`` (the calibrate default)
        differentiates *through* the diffrax solve (``RecursiveCheckpointAdjoint``);
        for a stiff model this reverse pass goes non-finite above a step-size
        threshold, so the reactor must carry a ``dtmax`` cap.
        ``mode="reverse", method="stable"`` replaces the integrator's adjoint with
        an explicit per-step transposed-stage solve
        (:func:`~aquakin.esdirk_adjoint_solve`) -- a robust adaptive ESDIRK forward
        whose backward is finite with no cap (batch reactors only; its gradient
        agrees with the capped ``through_solve`` path to the optimiser tolerance).
        ``mode="forward"`` uses ``jacfwd`` and builds a forward-capable reactor
        adjoint internally (forward-mode AD stays finite at any step -- the fix for
        a stiff model whose reverse adjoint overflows); pair it with
        ``optimizer="gauss_newton"`` and ``method="through_solve"`` (forward +
        ``method="stable"`` is rejected, the stable adjoint being reverse-only).
        ``check_finite`` (default ``True``) raises a friendly error if the
        start-point gradient is non-finite. ``adjoint_max_steps`` is the
        (allocated) forward step count for the ``method="stable"`` solve -- the
        backward scan walks this whole saved-trajectory buffer, so set it to a
        tight upper bound on the step count (ignored for ``through_solve``).
        ``adjoint_low_memory`` recomputes each step's stages in the backward pass
        instead of saving the ``~n_stages`` × dense-stage buffer -- memory for
        compute, with the gradient unchanged.

    Returns
    -------
    CalibrationResult
    """
    # Resolve the public DifferentiationConfig into the internal AD selectors
    # (the config owns the vocabulary: diff.mode is the AD direction, and
    # diff.gradient_backend() maps method -> "stable_adjoint" / "jax_adjoint").
    diff.validated()
    ad_mode = diff.mode
    gradient = diff.gradient_backend()

    if not free_params:
        raise ValueError("free_params must be non-empty.")
    if loss not in _VALID_LOSSES:
        raise ValueError(f"loss must be one of {_VALID_LOSSES}; got {loss!r}.")
    _check_sigma_arg(sigma, loss)
    _check_start_args(optimizer)
    if ad_mode == "forward" and gradient == "stable_adjoint":
        raise ValueError(
            "diff=DifferentiationConfig(mode='forward', method='stable') is "
            "incompatible: the stable discrete adjoint is reverse-only. Use "
            "mode='forward', method='through_solve' (forward-mode through the "
            "diffrax solve) OR mode='reverse', method='stable' (cap-free reverse)."
        )
    # ad_mode='forward' needs a forward-capable adjoint; build it internally so
    # diffrax never appears in user code.
    if ad_mode == "forward":
        reactor = with_adjoint(reactor, forward_adjoint())
    if gradient == "stable_adjoint" and not hasattr(reactor, "conditions"):
        raise ValueError(
            "gradient='stable_adjoint' is implemented for batch reactors "
            "(those exposing a single-location `conditions`); got a reactor "
            f"without one ({type(reactor).__name__})."
        )

    # Unpack the config objects into the internal knobs.
    cfg = _FitConfig.from_configs(
        optimizer,
        laplace,
        gradient=gradient,
        ad_mode=ad_mode,
        check_finite=diff.check_finite,
        compiled_cache=_compiled_cache,
        stable_adjoint_max_steps=diff.adjoint_max_steps,
        stable_adjoint_low_memory=diff.adjoint_low_memory,
    )
    ic_species, ic_bounds, ic_prior_log_std = _free_ic_fields(free_ic)

    if observation_units:
        if observation_fn is not None:
            raise ValueError(
                "observation_units converts observed species; an observation_fn's "
                "outputs have no declared unit, so convert its observations yourself."
            )
        names = observed_species if observed_species is not None else reactor.model.species
        columns = [(sp, reactor.model.species_units.get(sp, "")) for sp in names]
        observations, sigma = _convert_observations(
            observations, sigma, _observation_unit_factors(columns, observation_units)
        )

    # Resolve the arguments once into the static problem, then run the fit on the
    # reactor forward-model seam.
    problem = _resolve_problem(
        reactor.model,
        C0,
        observations,
        t_obs,
        free_params,
        transforms=transforms,
        initial_params=initial_params,
        observed_species=observed_species,
        time_unit=time_unit,
        loss=loss,
        sigma=sigma,
        priors=priors,
        use_priors=use_priors,
        free_ic=ic_species,
        ic_bounds=ic_bounds,
        ic_prior_log_std=ic_prior_log_std,
        param_halfwidth=optimizer.param_halfwidth,
        observation_fn=observation_fn,
        ic_fixed_total=bool(free_ic is not None and free_ic.fixed_total),
        bounds=optimizer.bounds,
        use_model_bounds=optimizer.use_model_bounds,
    )
    fm = _ReactorForwardModel(
        reactor, gradient, cfg.stable_adjoint_max_steps, cfg.stable_adjoint_low_memory
    )
    return _run_fit(problem, fm, cfg, optimizer, _ic_start)


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
    # sum(log(sigma)) normaliser; evaluating the objective puts both on scale).
    reported_loss = float(bundle.value_and_grad(theta_opt)[0])

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
    )
