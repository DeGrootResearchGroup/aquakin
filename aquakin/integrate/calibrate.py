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
from collections import namedtuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

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


# Uniform optimiser output so the multistart loop and downstream code are
# agnostic to which backend (L-BFGS-B or Gauss-Newton least-squares) ran.
_OptOut = namedtuple("_OptOut", "x fun success message nit")


# Outer iterations of the Gauss-Newton noise re-estimation (``sigma="estimate"``);
# each is a warm-started least-squares solve, and the noise level typically
# settles in a handful.
_IRLS_MAX_ITER = 50


# --- Laplace covariance -----------------------------------------------


def _laplace_covariance(H, ridge: float):
    """Laplace covariance and marginal std devs from an unconstrained-space Hessian.

    Symmetrise ``H``, eigen-decompose it, and invert it in full: every
    direction with information keeps its variance ``1 / (w_k + ridge)``, however
    weakly identified. Only an exactly null direction (eigenvalue at or below
    ``_NULL_RTOL`` (1e-10) times the largest, which includes a negative-curvature
    direction of a finite-difference Hessian) cannot be inverted. It is left out
    of ``cov``, and every parameter that loads on it gets an infinite std, so a
    non-identifiable parameter is reported as such rather than as perfectly
    known. This is the convention of :func:`aquakin.identifiability`.

    The null test is relative to the largest eigenvalue, so it does not depend
    on the Hessian's absolute scale (an observable on a tiny scale, such as a
    trace species in molar units, gives a small but well-structured ``JᵀJ``).

    This one covariance backs ``calibrate``'s ``posterior_cov`` /
    ``params_named_std`` and :meth:`CalibrationResult.posterior_draws`, so the
    reported std devs and the posterior draws agree.

    Parameters
    ----------
    H : array-like, shape (d, d)
        The (Gauss-Newton or finite-difference) Hessian of the loss.
    ridge : float
        Tikhonov regulariser added to each eigenvalue before inversion.

    Returns
    -------
    cov : np.ndarray, shape (d, d)
        ``sum_k v_k v_k^T / (w_k + ridge)`` over the non-null directions; this
        is ``inv(H + ridge I)`` when no direction is null.
    std : np.ndarray, shape (d,)
        ``sqrt(diag(cov))``, set to ``inf`` for each parameter that loads on a
        null direction.
    """
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    w_max = float(w.max())
    if not np.isfinite(w_max) or w_max <= 0.0:
        raise ValueError(
            "Laplace Hessian is not finite / positive-definite; cannot form a "
            "posterior covariance (check the fit converged and the model output "
            "is finite)."
        )
    null = w <= _NULL_RTOL * w_max
    Vk = V[:, ~null]
    cov = (Vk / (w[~null] + ridge)) @ Vk.T
    std = np.sqrt(np.diag(cov))
    involved = (np.abs(V[:, null]) > _NULL_LOADING_TOL).any(axis=1)
    return cov, np.where(involved, np.inf, std)


@dataclass
class _FitConfig:
    """The AD / optimiser / Laplace knobs, resolved from the public arguments
    (and the :class:`DifferentiationConfig`), passed as one object to the build /
    optimise / posterior stages instead of a dozen scalars."""

    gradient: str
    ad_mode: str
    check_finite: bool
    stable_adjoint_max_steps: int
    stable_adjoint_low_memory: bool
    optimizer: str
    max_iter: int
    tol: float
    n_starts: int
    jitter: float
    jitter_schedule: tuple | None
    seed: int
    laplace: bool
    laplace_method: str
    laplace_ridge: float
    laplace_fd_step: float
    laplace_dtmax: float | None
    compiled_cache: dict | None
    laplace_include_ic: bool = False

    @classmethod
    def from_configs(
        cls,
        optimizer: OptimizerConfig,
        laplace,
        *,
        gradient: str,
        ad_mode: str,
        check_finite: bool,
        compiled_cache: dict | None,
        stable_adjoint_max_steps: int = 0,
        stable_adjoint_low_memory: bool = False,
    ) -> _FitConfig:
        """Build the fit configuration from the public config objects.

        Parameters
        ----------
        optimizer : OptimizerConfig
            Optimiser backend and multistart settings.
        laplace : bool or LaplaceConfig
            The ``laplace=`` argument, normalized by :func:`_resolve_laplace`.
        gradient : {"jax_adjoint", "stable_adjoint"}
            The gradient backend, from ``DifferentiationConfig.gradient_backend``.
        ad_mode : {"auto", "forward", "reverse"}
            The AD direction of the residual Jacobian.
        check_finite : bool
            Whether to check the start-point gradient for non-finite values.
        compiled_cache : dict, optional
            The shared compiled-objective cache (a profile sweep passes one).
        stable_adjoint_max_steps : int
            Saved-trajectory length of a reactor's stable adjoint. A plant
            manages its own adjoint through its forward model, so it leaves 0.
        stable_adjoint_low_memory : bool
            Whether a reactor's stable adjoint recomputes its stages.

        Returns
        -------
        _FitConfig
        """
        laplace_on, lap = _resolve_laplace(laplace)
        return cls(
            gradient=gradient,
            ad_mode=ad_mode,
            check_finite=check_finite,
            stable_adjoint_max_steps=int(stable_adjoint_max_steps),
            stable_adjoint_low_memory=bool(stable_adjoint_low_memory),
            optimizer=optimizer.method,
            max_iter=optimizer.max_iter,
            tol=optimizer.tol,
            n_starts=optimizer.n_starts,
            jitter=optimizer.jitter,
            jitter_schedule=optimizer.jitter_schedule,
            seed=optimizer.seed,
            laplace=laplace_on,
            laplace_method=lap.method,
            laplace_ridge=lap.ridge,
            laplace_fd_step=lap.fd_step,
            laplace_dtmax=lap.dtmax,
            compiled_cache=compiled_cache,
            laplace_include_ic=lap.include_ic,
        )


def _forward_jac(cfg: _FitConfig, fm: _ReactorForwardModel) -> bool:
    """Whether to form the residual Jacobian by forward-mode AD (jacfwd).

    ``forward`` / ``reverse`` force the direction; ``auto`` reproduces the legacy
    inference -- forward iff the reactor is forward-capable and we are not on the
    reverse-only stable-adjoint backend."""
    if cfg.ad_mode == "forward":
        return True
    if cfg.ad_mode == "reverse":
        return False
    return cfg.gradient != "stable_adjoint" and fm.forward_capable()


def _residual_parts(
    problem: _CalibrationProblem,
    fm: _ReactorForwardModel,
    rate_thetas,
    ic_thetas,
    p0_full_arg,
    C0s,
    ic_center,
    *,
    include_ic_prior: bool,
    sigma_hat=None,
):
    """The stacked residual vector whose ``0.5*||.||^2`` is the objective's
    theta-dependent part. For the concentrated likelihood
    (``problem.sigma_estimate``) the raw residuals are divided by
    ``sigma_hat`` (the current noise estimate per output), giving the
    ``"nll"`` residual at that noise level. Shared by the Gauss-Newton fit (full theta,
    ``include_ic_prior=True``) and the Gauss-Newton Laplace Hessian (rate thetas
    only, pools fixed at the MAP, ``include_ic_prior=False`` -- the ic-prior block
    has zero Jacobian w.r.t. the rates). One source of truth keeps the two in
    sync."""
    physical = problem.physical_from_theta(rate_thetas)
    p = p0_full_arg.at[problem.free_indices].set(physical)
    parts = []
    for (_t, _ts, _loss_i, resid_fn_i), pred in zip(
        problem.dataset_static, _predict(problem, fm, p, ic_thetas, C0s)
    ):
        r = resid_fn_i(pred)
        if problem.sigma_estimate:
            r = (r / sigma_hat).reshape(-1)
        parts.append(r)
    if problem.has_priors:
        parts.append(problem.prior_mask * (physical - problem.prior_mean) / problem.prior_std)
    if include_ic_prior and problem.m_ic and problem.ic_prior_log_std:
        parts.append((ic_thetas - ic_center) / problem.ic_prior_log_std)
    return jnp.concatenate(parts)


@dataclass
class _ObjectiveBundle:
    """The compiled callables the optimiser drives: the scalar
    value-and-gradient (L-BFGS-B) and the residual / Jacobian (Gauss-Newton).
    ``res_j`` / ``jac_j`` are ``None`` for the L-BFGS-B optimiser."""

    value_and_grad: object
    res_j: object
    jac_j: object
    use_forward_jac: bool
    # Concentrated likelihood only: ``sigma_hat_of(theta)`` is the noise
    # estimate per output at ``theta``, and ``set_sigma_hat`` sets the noise
    # level the Gauss-Newton residual / Jacobian are scaled by.
    sigma_hat_of: object = None
    set_sigma_hat: object = None


def _build_objective(
    problem: _CalibrationProblem, fm: _ReactorForwardModel, cfg: _FitConfig
) -> _ObjectiveBundle:
    """Build the (optionally cache-shared) compiled objective / residual / Jacobian.

    The per-call-varying data (``p0_full`` / per-dataset initial states / ic-prior
    centre) is threaded into the compiled programs as *arguments* -- so a sequence
    of structurally-identical fits (a ``profile_likelihood`` sweep) reuses one
    compiled program via a shared ``cfg.compiled_cache``."""
    struct_key = problem.struct_key(cfg.gradient)

    def _cached_jit(key, build):
        if cfg.compiled_cache is None:
            return build()
        fn = cfg.compiled_cache.get(key)
        if fn is None:
            fn = build()
            cfg.compiled_cache[key] = fn
        return fn

    def objective(theta, p0_full_arg, C0s, ic_center):
        rate_thetas = theta[: problem.n_rate]
        ic_thetas = theta[problem.n_rate :]
        physical = problem.physical_from_theta(rate_thetas)
        p = p0_full_arg.at[problem.free_indices].set(physical)
        # Sum the data terms over every dataset (the batches share ``p``).
        data_term = _data_term(problem, _predict(problem, fm, p, ic_thetas, C0s))
        if problem.has_priors:
            data_term = data_term + 0.5 * jnp.sum(
                problem.prior_mask * ((physical - problem.prior_mean) / problem.prior_std) ** 2
            )
        if problem.m_ic and problem.ic_prior_log_std:
            data_term = data_term + 0.5 * jnp.sum(
                ((ic_thetas - ic_center) / problem.ic_prior_log_std) ** 2
            )
        return data_term

    _obj_vg_jit = _cached_jit(
        ("obj_vg",) + struct_key, lambda: jax.jit(jax.value_and_grad(objective))
    )
    data = problem.data

    def obj_value_and_grad(theta):
        return _obj_vg_jit(theta, *data)

    res_j = None
    jac_j = None
    use_forward = False
    sigma_hat_of = set_sigma_hat = None
    # The noise level the Gauss-Newton residual is scaled by under the
    # concentrated likelihood: a runtime argument, so re-estimating it does not
    # recompile. Unused (ones) otherwise.
    sigma_box = [jnp.ones(problem.n_observed)]
    if problem.sigma_estimate:

        def _sigma_hat(theta, p0_full_arg, C0s, ic_center):
            physical = problem.physical_from_theta(theta[: problem.n_rate])
            p = p0_full_arg.at[problem.free_indices].set(physical)
            preds = _predict(problem, fm, p, theta[problem.n_rate :], C0s)
            return jnp.sqrt(_pooled_sse(problem, preds) / problem.n_obs_per_output)

        _sig_core = _cached_jit(("sigma_hat",) + struct_key, lambda: jax.jit(_sigma_hat))

        def sigma_hat_of(theta):
            return _sig_core(jnp.asarray(theta), *data)

        def set_sigma_hat(value):
            sigma_box[0] = jnp.asarray(value)

    if cfg.optimizer == "gauss_newton":
        # Minimise the residual vector (0.5||r||^2 == the scalar objective) with
        # trust-region least-squares. The residual Jacobian is by forward-mode AD
        # if the reactor is forward-capable (DirectAdjoint), else reverse-mode.
        def _full_residual(theta, p0_full_arg, C0s, ic_center, sigma_hat):
            return _residual_parts(
                problem,
                fm,
                theta[: problem.n_rate],
                theta[problem.n_rate :],
                p0_full_arg,
                C0s,
                ic_center,
                include_ic_prior=True,
                sigma_hat=sigma_hat,
            )

        # The stable adjoint is a reverse-only custom_vjp, so its residual
        # Jacobian must be reverse-mode regardless of the reactor's adjoint.
        use_forward = _forward_jac(cfg, fm)
        _jac = (jax.jacfwd if use_forward else jax.jacrev)(_full_residual, argnums=0)
        _res_core = _cached_jit(("gn_res",) + struct_key, lambda: jax.jit(_full_residual))
        _jac_core = _cached_jit(("gn_jac", use_forward) + struct_key, lambda: jax.jit(_jac))

        def _res_call(theta):
            return _res_core(theta, *data, sigma_box[0])

        def _jac_call(theta):
            return _jac_core(theta, *data, sigma_box[0])

        res_j = _res_call
        jac_j = _jac_call

    return _ObjectiveBundle(
        obj_value_and_grad, res_j, jac_j, use_forward, sigma_hat_of, set_sigma_hat
    )


def _optimizer_bounds(problem: _CalibrationProblem, rate_theta0):
    """Box bounds in unconstrained space.

    Rate dims carry the absolute bounds ``problem.rate_bounds`` (model-declared
    and/or explicit, already mapped through each transform), intersected with
    ``theta0 +/- param_halfwidth`` when that is given; the halfwidth box is
    centred on the start moved inside the absolute bounds, so the intersection
    is never empty. Free-IC dims are log-bounded by ``ic_bounds`` (one pair, or
    one per pool), and unbounded for a ``fixed_total`` split.
    Returns ``(bounds, lb, ub, ls_bounds, has_bounds)`` -- ``bounds`` the
    L-BFGS-B list (``None`` if unbounded), ``ls_bounds`` the least-squares
    ``(lb, ub)`` pair."""
    rate_th0 = np.asarray(rate_theta0, dtype=float)
    if problem.rate_bounds is not None:
        rate_lb = np.asarray(problem.rate_bounds[0], dtype=float)
        rate_ub = np.asarray(problem.rate_bounds[1], dtype=float)
    else:
        rate_lb = np.full(problem.n_rate, -np.inf)
        rate_ub = np.full(problem.n_rate, np.inf)
    if problem.param_halfwidth is not None:
        centre = np.clip(rate_th0, rate_lb, rate_ub)
        rate_lb = np.maximum(rate_lb, centre - problem.param_halfwidth)
        rate_ub = np.minimum(rate_ub, centre + problem.param_halfwidth)
    if problem.ic_fixed_total:
        # The log-ratio coordinates of a split are unbounded (softmax keeps every
        # pool positive and the total fixed).
        n_ic = problem.m_ic_theta * problem.n_datasets
        ic_lb = np.full(n_ic, -np.inf)
        ic_ub = np.full(n_ic, np.inf)
    else:
        # ``ic_bounds`` is one (lo, hi) pair or per-pool (lo, hi) arrays.
        lo = np.broadcast_to(np.asarray(problem.ic_bounds[0], dtype=float), (problem.m_ic,))
        hi = np.broadcast_to(np.asarray(problem.ic_bounds[1], dtype=float), (problem.m_ic,))
        ic_lb = np.tile(np.log(lo), problem.n_datasets)
        ic_ub = np.tile(np.log(hi), problem.n_datasets)
    lb = np.concatenate([rate_lb, ic_lb])
    ub = np.concatenate([rate_ub, ic_ub])
    has_bounds = bool(np.isfinite(lb).any() or np.isfinite(ub).any())
    if has_bounds:
        bounds = [
            (None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
            for lo, hi in zip(lb, ub)
        ]
    else:
        bounds = None
    return bounds, lb, ub, (lb, ub), has_bounds


def _clip_start(problem: _CalibrationProblem, theta0, opt_bounds):
    """Move the start inside the optimiser bounds, warning (by name) for any
    free rate parameter whose starting value lay outside its bounds."""
    _bounds, lb, ub, _ls, has_bounds = opt_bounds
    if not has_bounds:
        return theta0
    th = np.asarray(theta0, dtype=float)
    clipped = np.clip(th, lb, ub)
    outside = [problem.free_params[i] for i in range(problem.n_rate) if clipped[i] != th[i]]
    if outside:
        warnings.warn(
            f"calibrate: the starting value of {outside} lies outside its bounds; "
            "the optimiser starts from the nearest bound instead.",
            RuntimeWarning,
            stacklevel=4,  # past _initial_point and the calibrate entry point
        )
    return jnp.asarray(clipped)


def _check_start_gradient(cfg: _FitConfig, bundle: _ObjectiveBundle, theta0):
    """Evaluate the gradient / Jacobian once at the start point and fail loudly
    (with the remedy) if it is non-finite -- a stiff reverse-mode adjoint
    overflowing -- rather than letting the optimizer wander on NaNs."""
    try:
        if cfg.optimizer == "gauss_newton":
            probe = bundle.jac_j(jnp.asarray(theta0))
        else:
            _v0, probe = bundle.value_and_grad(jnp.asarray(theta0))
    except Exception as exc:
        if not _is_solve_failure(exc):
            raise
        raise RuntimeError(
            "Calibration failed: the forward solve failed at the starting guess, "
            "before the optimiser took a step. Check the initial parameters and "
            "initial state, or start from a better guess. The solve error is the "
            "direct cause above."
        ) from exc
    # Forward-mode only enters through the Gauss-Newton Jacobian; L-BFGS-B always
    # uses a reverse scalar gradient, so its cap-free fix is gradient='stable_adjoint'.
    on_finite_path = cfg.gradient == "stable_adjoint" or (
        cfg.ad_mode == "forward" and cfg.optimizer == "gauss_newton"
    )
    if on_finite_path:
        remedy = (
            "It was already on a finite-by-construction path "
            f"(ad_mode={cfg.ad_mode!r}, gradient={cfg.gradient!r}, "
            f"optimizer={cfg.optimizer!r}); check the model, data, and "
            "parameter ranges."
        )
    elif cfg.optimizer == "gauss_newton":
        remedy = (
            "Pass ad_mode='forward' (forward-mode, finite through a stiff "
            "solve) or gradient='stable_adjoint' (cap-free reverse-mode); "
            "either is handled internally with no diffrax or dtmax in your "
            "code."
        )
    else:
        remedy = (
            "Pass gradient='stable_adjoint' (cap-free reverse-mode, handled "
            "internally), or switch to optimizer='gauss_newton' with "
            "ad_mode='forward'."
        )
    check_finite_gradient(probe, what="calibration gradient", remedy=remedy)


_SOLVE_FAILURE_HINT = (
    "This usually means the optimiser stepped into a stiff or infeasible region "
    "of parameter space. Bound the rate constants with "
    "OptimizerConfig(bounds={name: (lo, hi)}) or "
    "OptimizerConfig(param_halfwidth=...) (e.g. np.log(100) for two decades "
    "either side of the start), start from a better initial guess, or tighten "
    "the free_ic bounds."
)


def _is_solve_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is a runtime failure of the forward solve (as opposed to
    a programming error).

    A solve that goes bad inside compiled code (a non-finite linear solve, a
    step-count overrun) is reported through ``equinox.error_if``, which raises
    from a host callback. It reaches Python either as equinox's own runtime
    error or, under a plain ``jax.jit``, wrapped in a JAX runtime error (or a
    ``ValueError`` on older JAX) whose message carries the callback failure.
    """
    if type(exc).__name__ in ("EquinoxRuntimeError", "_EquinoxRuntimeError"):
        return True
    return isinstance(exc, (RuntimeError, ValueError)) and "Callback error" in str(exc)


def _resolve_starts(problem: _CalibrationProblem, starts) -> list:
    """The explicit ``OptimizerConfig.starts`` as unconstrained rate-theta
    vectors. Each entry is a full physical parameter vector (only the free
    entries are read) or a ``{name: value}`` dict over free parameters overlaid
    on ``p0_full``."""
    if isinstance(starts, (np.ndarray, jnp.ndarray)) and np.ndim(starts) == 1:
        raise ValueError(
            "optimizer.starts must be a list of starts (vectors or dicts), not a "
            "single vector; pass initial_params for one start."
        )
    starts = list(starts)
    if not starts:
        raise ValueError("optimizer.starts must contain at least one start.")
    model = problem.model
    n_params = int(problem.p0_full.shape[0])
    out = []
    for s_i, start in enumerate(starts):
        if isinstance(start, dict):
            p = problem.p0_full
            for name, value in start.items():
                if name not in problem.free_params:
                    raise ValueError(
                        f"optimizer.starts[{s_i}]: '{name}' is not a free parameter; "
                        f"a start sets free parameters only ({problem.free_params})."
                    )
                p = p.at[model.param_index[name]].set(float(value))
        else:
            p = jnp.asarray(start, dtype=float)
            if p.shape != (n_params,):
                raise ValueError(
                    f"optimizer.starts[{s_i}] has shape {p.shape}; a vector start must "
                    f"be a full parameter vector of shape ({n_params},), or pass a "
                    "{name: value} dict."
                )
        for name, t in zip(problem.free_params, problem.transforms):
            v = float(p[model.param_index[name]])
            if (t == "positive_log" and v <= 0.0) or (t == "logit" and not 0.0 < v < 1.0):
                raise ValueError(
                    f"optimizer.starts[{s_i}]: '{name}' = {v} is outside the domain "
                    f"of its transform {t!r}."
                )
        out.append(
            jnp.stack(
                [
                    _to_unconstrained(p[model.param_index[name]], t)
                    for name, t in zip(problem.free_params, problem.transforms)
                ]
            )
        )
    return out


def _run_multistart(
    cfg: _FitConfig, bundle: _ObjectiveBundle, theta0, opt_bounds, starts=None, progress=None
) -> _OptOut:
    """Run the optimiser from the start point, then from ``n_starts - 1``
    deterministic jittered restarts, keeping the lowest finite loss. With an
    explicit ``starts`` list (full unconstrained theta vectors), run from each
    of those instead (``theta0`` is then unused).

    ``progress`` (see :mod:`aquakin.progress`) is told of every objective
    evaluation, so a caller can follow the fit and cancel it between two.

    A start whose forward solve fails part-way (see :func:`_is_solve_failure`)
    is skipped with a ``RuntimeWarning``; if every start fails, a
    ``RuntimeError`` with remedies is raised, chained to the first failure.
    """
    from aquakin.progress import report

    bounds, lb, ub, ls_bounds, _has = opt_bounds
    n_planned = len(starts) if starts is not None else cfg.n_starts
    evaluations = [0]
    current = [0]

    def _tick():
        evaluations[0] += 1
        report(
            progress,
            "fit",
            evaluations[0],
            None,
            f"start {current[0] + 1} of {n_planned}, evaluation {evaluations[0]}",
        )

    def _np_loss_and_grad(x_np):
        _tick()
        x = jnp.asarray(x_np)
        val, grad = bundle.value_and_grad(x)
        return float(val), np.asarray(grad)

    def _run_from(x_start):
        if cfg.optimizer == "lbfgsb":
            r = minimize(
                _np_loss_and_grad,
                np.asarray(x_start),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": cfg.max_iter, "gtol": cfg.tol},
            )
            return _OptOut(
                np.asarray(r.x), float(r.fun), bool(r.success), str(r.message), int(r.nit)
            )

        def _lsq(x0):
            def _residual(x):
                _tick()
                return np.asarray(bundle.res_j(jnp.asarray(x)), dtype=float)

            r = least_squares(
                _residual,
                np.asarray(x0),
                jac=lambda x: np.asarray(bundle.jac_j(jnp.asarray(x)), dtype=float),
                method="trf",
                bounds=ls_bounds,
                max_nfev=cfg.max_iter,
                xtol=cfg.tol,
                ftol=cfg.tol,
            )
            # ``n_iter`` reports optimiser ITERATIONS, consistent with L-BFGS-B's
            # ``nit``. scipy.least_squares exposes no iteration count, so use the
            # Jacobian-evaluation count ``njev`` (~one per trust-region iteration).
            return r, int(r.njev) if r.njev is not None else int(r.nfev)

        if bundle.sigma_hat_of is None:
            r, n_iter_ls = _lsq(x_start)
            return _OptOut(
                np.asarray(r.x), float(r.cost), bool(r.success), str(r.message), n_iter_ls
            )
        # Concentrated likelihood: alternate a least-squares solve at a fixed
        # noise level with re-estimating it from the residuals. At the fixed
        # point the weighted-least-squares gradient equals the gradient of the
        # concentrated objective, so this is a stationary point of it.
        x = np.asarray(x_start, dtype=float)
        sig = np.asarray(bundle.sigma_hat_of(x))
        n_iter_total, converged = 0, False
        for _ in range(_IRLS_MAX_ITER):
            bundle.set_sigma_hat(sig)
            r, n_it = _lsq(x)
            n_iter_total += n_it
            x = np.asarray(r.x)
            sig_new = np.asarray(bundle.sigma_hat_of(x))
            converged = bool(np.max(np.abs(sig_new / sig - 1.0)) <= max(cfg.tol, 1e-12))
            sig = sig_new
            if converged:
                break
        bundle.set_sigma_hat(sig)
        message = str(r.message) if converged else "noise re-estimation did not converge"
        fun = float(bundle.value_and_grad(jnp.asarray(x))[0])
        return _OptOut(x, fun, bool(r.success) and converged, message, n_iter_total)

    failures = []  # (start index, exception) of starts whose solve failed

    def _try_from(s, x_start):
        current[0] = s
        try:
            return _run_from(x_start)
        except Exception as exc:
            if not _is_solve_failure(exc):
                raise
            failures.append((s, exc))
            return None

    def _better(cand, best):
        return best is None or (np.isfinite(cand.fun) and cand.fun < best.fun)

    if starts is not None:
        result = None
        for s, x_start in enumerate(starts):
            x_np = np.asarray(x_start, dtype=float)
            if bounds is not None:
                x_np = np.clip(x_np, lb, ub)
            cand = _try_from(s, x_np)
            if cand is not None and _better(cand, result):
                result = cand
        n_run = len(starts)
    else:
        result = _try_from(0, theta0)
        n_run = cfg.n_starts
    if starts is None and cfg.n_starts > 1:
        theta0_np = np.asarray(theta0)
        # Default: a single Gaussian stream of scale ``jitter`` seeded once.
        # ``jitter_schedule`` (a tuple of scales): start s uses scale
        # schedule[(s-1) % len] with its own generator (seed + s).
        seq_rng = None if cfg.jitter_schedule else np.random.default_rng(cfg.seed)
        for s in range(1, cfg.n_starts):
            if cfg.jitter_schedule:
                jit = cfg.jitter_schedule[(s - 1) % len(cfg.jitter_schedule)]
                noise = np.random.default_rng(cfg.seed + s).normal(0.0, jit, size=theta0_np.shape)
            else:
                noise = seq_rng.normal(0.0, cfg.jitter, size=theta0_np.shape)
            perturbed = theta0_np + noise
            if bounds is not None:
                perturbed = np.clip(perturbed, lb, ub)
            cand = _try_from(s, perturbed)
            if cand is None:
                continue
            if _better(cand, result):
                result = cand
    if result is None:
        what = "the optimiser run" if n_run == 1 else f"all {n_run} optimiser starts"
        raise RuntimeError(
            f"Calibration failed: the forward solve failed in {what}, at a parameter "
            f"point the optimiser reached. {_SOLVE_FAILURE_HINT} The first solve "
            "error is the direct cause above."
        ) from failures[0][1]
    if failures:
        idx = ", ".join(str(s) for s, _exc in failures)
        which = "indexing optimizer.starts" if starts is not None else "0 is the unperturbed start"
        warnings.warn(
            f"calibrate: the forward solve failed in {len(failures)} of {n_run} "
            f"optimiser starts (start index {idx}; {which}); "
            f"they were skipped and the best remaining start was kept. {_SOLVE_FAILURE_HINT}",
            RuntimeWarning,
            stacklevel=3,
        )
    return result


class _LaplaceOut(
    namedtuple(
        "_LaplaceOut",
        [
            "posterior_cov",
            "posterior_std_unconstrained",
            "params_named_std",
            "hessian_unconstrained",
            "joint_cov",
            "joint_hessian",
            "joint_names",
            "ic_named_std",
            "ic_fractions_std",
        ],
        defaults=(None,),
    )
):
    """What :func:`_laplace_posterior` returns. The ``joint_*`` / ``ic_named_std``
    fields are ``None`` unless the joint rate + initial-pool posterior was
    requested (``LaplaceConfig(include_ic=True)`` with fitted pools)."""


# A fitted coordinate within a tolerance of a finite optimiser bound (in
# unconstrained space) counts as on the bound. The tolerance depends on the
# coordinate's transform:
#
# - ``positive_log`` / ``logit`` (and the log-space initial pools): an absolute
#   ``_AT_BOUND_RTOL`` in the unconstrained coordinate, which is already a
#   relative tolerance on the physical value. It does not shrink when the bound
#   maps to zero (a physical bound of 1 under ``positive_log``, or 0.5 under
#   ``logit``).
# - ``none`` (the coordinate is the physical value): ``_AT_BOUND_RTOL *
#   max(|bound|, |theta|)`` plus a tiny floor, ``_AT_BOUND_ATOL`` times the box
#   width (or times 1 when the other side is open). It follows the parameter's
#   own size, so a small parameter well inside its box (k = 5e-6 in (1e-6, 1),
#   say) is not mistaken for one on the bound.
#
# L-BFGS-B ends exactly on an active bound; the trust-region least-squares
# solver stays strictly inside, within round-off of the bound.
_AT_BOUND_RTOL = 1e-5
_AT_BOUND_ATOL = 1e-12


def _at_bound_sides(theta_opt, opt_bounds, transforms=None) -> np.ndarray:
    """Per coordinate of ``theta_opt``: ``"lower"`` / ``"upper"`` if it ended on
    that finite optimiser bound, else ``""``. ``transforms`` names each
    coordinate's transform; ``None`` treats every coordinate as ``"none"``."""
    _bounds, lb, ub, _ls, has_bounds = opt_bounds
    th = np.asarray(theta_opt, dtype=float)
    sides = np.full(th.shape, "", dtype=object)
    if not has_bounds:
        return sides
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    physical = (
        np.ones(th.shape, dtype=bool)
        if transforms is None
        else np.asarray([t == "none" for t in transforms], dtype=bool)
    )
    both = np.isfinite(lb) & np.isfinite(ub)
    with np.errstate(invalid="ignore"):
        atol = _AT_BOUND_ATOL * np.where(both, ub - lb, 1.0)
        tol_lo = np.where(
            physical, _AT_BOUND_RTOL * np.maximum(np.abs(lb), np.abs(th)) + atol, _AT_BOUND_RTOL
        )
        tol_hi = np.where(
            physical, _AT_BOUND_RTOL * np.maximum(np.abs(ub), np.abs(th)) + atol, _AT_BOUND_RTOL
        )
        on_lo = np.isfinite(lb) & (np.abs(th - lb) <= tol_lo)
        on_hi = np.isfinite(ub) & (np.abs(ub - th) <= tol_hi)
    sides[on_lo] = "lower"
    sides[on_hi & ~on_lo] = "upper"
    return sides


def _at_bound_named(problem: _CalibrationProblem, sides) -> dict[str, str]:
    """``{name: side}`` for the coordinates on a bound, named as
    :func:`_joint_names` names them."""
    return {n: str(sd) for n, sd in zip(_joint_names(problem), sides) if sd}


def _laplace_posterior(
    problem: _CalibrationProblem,
    fm: _ReactorForwardModel,
    cfg: _FitConfig,
    rate_theta_opt,
    ic_opt,
    sigma_hat_at_map=None,
    at_bound_mask=None,
) -> _LaplaceOut:
    """Laplace covariance at the MAP.

    For the concentrated likelihood (``loss="nll", sigma="estimate"``) the
    Gauss-Newton Hessian is the Fisher information with the noise estimate
    ``sigma_hat_at_map`` plugged in, ``sum_j J_j^T J_j / sigma_j^2``; the FD
    Hessian is that of the concentrated objective itself, which also carries
    the (asymptotically small) curvature from estimating the noise.

    By default the Hessian is over the rate parameters only, with the fitted
    initial pools held at their MAP. With ``cfg.laplace_include_ic`` (and fitted
    pools) it is over the rates *and* the log-space pools jointly; the rate
    covariance is then the rate block of the joint covariance, i.e. the marginal
    over the pools. ``hessian_unconstrained`` is always the rate-rate block of
    the Hessian, so it means the same thing in both modes. The Hessian is more
    step-sensitive than the fit, so it can use a tighter integrator cap
    (``laplace_dtmax``) via the forward model's ``with_dtmax``.

    ``at_bound_mask`` (a boolean per fitted coordinate, rates then pools) marks
    coordinates that ended on an active optimiser bound. They are held fixed:
    the covariance is the inverse of the Hessian over the remaining
    coordinates, embedded with zero rows / columns for the fixed ones, whose
    std is reported as ``nan``."""
    d = problem.n_rate
    joint = bool(cfg.laplace_include_ic and problem.m_ic)
    fm_lap = fm.with_dtmax(cfg.laplace_dtmax)
    theta_at = jnp.concatenate([rate_theta_opt, ic_opt]) if joint else rate_theta_opt

    def _split(theta):
        # Joint mode differentiates the pools too; otherwise they stay at the MAP.
        return (theta[:d], theta[d:]) if joint else (theta, ic_opt)

    if cfg.laplace_method == "gauss_newton":
        # Gauss-Newton / Fisher Hessian H = J^T J, with J the Jacobian of the
        # scaled residuals. Only FIRST-order AD through the solve; for
        # loss='nll' this is the exact Fisher information (PSD by construction).
        # The ic-prior residual depends on the pools only, so it enters the
        # joint Hessian and is omitted (zero Jacobian) in the rate-only one.
        sigma_hat = sigma_hat_at_map
        if problem.sigma_estimate and sigma_hat is None:
            raise ValueError("The concentrated-likelihood Laplace needs the noise estimate.")

        def _residual_vec(theta):
            rate_thetas, ic_thetas = _split(theta)
            return _residual_parts(
                problem,
                fm_lap,
                rate_thetas,
                ic_thetas,
                problem.p0_full,
                problem.C0_base,
                problem.ic_center_full,
                include_ic_prior=joint,
                sigma_hat=sigma_hat,
            )

        use_fwd = _forward_jac(cfg, fm_lap)
        J = (jax.jacfwd if use_fwd else jax.jacrev)(_residual_vec)(theta_at)
        H = J.T @ J
    elif cfg.laplace_method == "fd":
        # FD Hessian of the loss (pools fixed at the MAP unless joint).
        def _objective(theta):
            rate_thetas, ic_thetas = _split(theta)
            physical = problem.physical_from_theta(rate_thetas)
            p = problem.p0_full.at[problem.free_indices].set(physical)
            total = _data_term(problem, _predict(problem, fm_lap, p, ic_thetas, problem.C0_base))
            if problem.has_priors:
                total = total + 0.5 * jnp.sum(
                    problem.prior_mask * ((physical - problem.prior_mean) / problem.prior_std) ** 2
                )
            if joint and problem.ic_prior_log_std:
                total = total + 0.5 * jnp.sum(
                    ((ic_thetas - problem.ic_center_full) / problem.ic_prior_log_std) ** 2
                )
            return total

        grad_fn = jax.jit(jax.grad(_objective))
        n_theta = theta_at.shape[0]
        H_rows = []
        for i in range(n_theta):
            step = max(abs(float(theta_at[i])), 1.0) * cfg.laplace_fd_step
            e_i = jnp.zeros(n_theta).at[i].set(step)
            g_plus = grad_fn(theta_at + e_i)
            g_minus = grad_fn(theta_at - e_i)
            H_rows.append((g_plus - g_minus) / (2.0 * step))
        H = jnp.stack(H_rows)
    else:
        raise ValueError(
            f"laplace_method must be 'fd' or 'gauss_newton'; got {cfg.laplace_method!r}."
        )
    H = 0.5 * (H + H.T)  # symmetrise away asymmetry / FD noise

    n_h = int(H.shape[0])
    fixed = np.zeros(n_h, dtype=bool)
    if at_bound_mask is not None:
        fixed = np.asarray(at_bound_mask, dtype=bool)[:n_h]
    H_np = np.asarray(H)
    cov_np = np.zeros((n_h, n_h))
    std_np = np.full(n_h, np.nan)
    if not fixed.all():
        keep = ~fixed
        cov_k, std_k = _laplace_covariance(H_np[np.ix_(keep, keep)], cfg.laplace_ridge)
        cov_np[np.ix_(keep, keep)] = cov_k
        std_np[keep] = std_k
    names = _joint_names(problem) if joint else list(problem.free_params)
    unidentified = [n for n, sd, fx in zip(names, std_np, fixed) if not fx and not np.isfinite(sd)]
    if unidentified:
        warnings.warn(
            "The Laplace Hessian has an exactly null direction, so the data do "
            f"not identify {unidentified}; their standard deviation is inf.",
            RuntimeWarning,
            stacklevel=3,
        )
    cov = jnp.asarray(cov_np)
    std_unconstrained = jnp.asarray(std_np)

    # Delta-method projection to physical space.
    jac = jnp.stack(
        [_jacobian_physical_wrt_theta(rate_theta_opt[i], problem.transforms[i]) for i in range(d)]
    )
    std_physical = jnp.abs(jac) * std_unconstrained[:d]
    params_named_std = {name: float(std_physical[i]) for i, name in enumerate(problem.free_params)}

    if not joint:
        return _LaplaceOut(
            cov, std_unconstrained, params_named_std, H, None, None, None, None, None
        )

    m = problem.m_ic
    ic_fractions_std = None
    if problem.ic_fixed_total:
        # Delta method from the log-ratio block of each dataset to the fractions
        # (and pools = total * fractions).
        mt = problem.m_ic_theta
        cov_np_full = np.asarray(cov)
        ic_named_std, ic_fractions_std = [], []
        for k in range(problem.n_datasets):
            sl = slice(d + k * mt, d + (k + 1) * mt)
            z_k = np.asarray(ic_opt)[k * mt : (k + 1) * mt]
            sd_f = np.sqrt(np.diag(_simplex_fraction_cov(z_k, cov_np_full[sl, sl])))
            total = float(
                np.sum(np.asarray(problem.C0_base[k])[np.asarray(problem.ic_species_idx)])
            )
            ic_fractions_std.append({s: float(v) for s, v in zip(problem.free_ic, sd_f)})
            ic_named_std.append({s: float(total * v) for s, v in zip(problem.free_ic, sd_f)})
    else:
        # Pools are fitted as log(value), so d(value)/d(theta) = value.
        ic_std = np.exp(np.asarray(ic_opt)) * np.asarray(std_unconstrained[d:])
        ic_named_std = [
            {s: float(v) for s, v in zip(problem.free_ic, ic_std[k * m : (k + 1) * m])}
            for k in range(problem.n_datasets)
        ]
    return _LaplaceOut(
        posterior_cov=cov[:d, :d],
        posterior_std_unconstrained=std_unconstrained[:d],
        params_named_std=params_named_std,
        hessian_unconstrained=H[:d, :d],
        joint_cov=cov,
        joint_hessian=H,
        joint_names=_joint_names(problem),
        ic_named_std=ic_named_std,
        ic_fractions_std=ic_fractions_std,
    )


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
