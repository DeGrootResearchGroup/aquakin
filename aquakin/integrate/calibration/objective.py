"""The compiled objective: the resolved fit settings (:class:`_FitConfig`), the
jit-cached value-and-gradient, residual and Jacobian, and the start-gradient check."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from aquakin.integrate._common import check_finite_gradient
from aquakin.integrate._config import FORWARD_THROUGH_SOLVE_DIFF, STABLE_REVERSE_DIFF
from aquakin.integrate.calibration.config import _resolve_laplace
from aquakin.integrate.calibration.loss import _data_term, _pooled_sse
from aquakin.integrate.calibration.multistart import _is_solve_failure
from aquakin.integrate.calibration.problem import _predict

if TYPE_CHECKING:
    from aquakin.integrate.calibration.config import OptimizerConfig
    from aquakin.integrate.calibration.problem import _CalibrationProblem, _ReactorForwardModel


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


def _add_prior_term(problem: _CalibrationProblem, value, physical, ic_thetas, ic_center):
    """``value`` plus the prior penalty the objective adds to its data term.

    Parameters
    ----------
    problem : _CalibrationProblem
        The resolved problem (its priors and free-IC settings).
    value : jnp.ndarray or float
        What the penalty is added to: the data term in the objective, or 0 for
        the penalty alone.
    physical : jnp.ndarray
        Free rate parameters in physical space, shape ``(n_rate,)``.
    ic_thetas : jnp.ndarray
        Fitted initial-pool coordinates of every dataset.
    ic_center : jnp.ndarray
        The prior centre of ``ic_thetas``.

    Returns
    -------
    jnp.ndarray or float
        ``value``, plus ``0.5 * sum(((physical - mean) / std)**2)`` over the
        rates with a prior, plus ``0.5 * sum(((ic_thetas - ic_center) /
        prior_log_std)**2)`` when the pools have a prior; ``value`` unchanged
        when there is no prior.
    """
    if problem.has_priors:
        value = value + 0.5 * jnp.sum(
            problem.prior_mask * ((physical - problem.prior_mean) / problem.prior_std) ** 2
        )
    if problem.m_ic and problem.ic_prior_log_std:
        value = value + 0.5 * jnp.sum(((ic_thetas - ic_center) / problem.ic_prior_log_std) ** 2)
    return value


@dataclass
class _ObjectiveBundle:
    """The compiled callables the optimiser drives: the scalar
    value-and-gradient (L-BFGS-B) and the residual / Jacobian (Gauss-Newton).
    ``res_j`` / ``jac_j`` are ``None`` for the L-BFGS-B optimiser.

    All of them are of the objective divided by :attr:`loss_scale` (1 until
    the multistart runner sets it from each start's loss, see
    :func:`_scale_objective`),
    so SciPy's absolute tolerances act relative to the starting misfit;
    multiply a value by ``loss_scale`` to get the loss itself."""

    value_and_grad: object
    res_j: object
    jac_j: object
    use_forward_jac: bool
    # Concentrated likelihood only: ``sigma_hat_of(theta)`` is the noise
    # estimate per output at ``theta``, and ``set_sigma_hat`` sets the noise
    # level the Gauss-Newton residual / Jacobian are scaled by.
    sigma_hat_of: object = None
    set_sigma_hat: object = None
    # One-element holder of the scale, shared with the callables above.
    scale_box: list = field(default_factory=lambda: [1.0])
    # The problem's data mean square for loss="mse" (None: never scaled), from
    # which each start's scale is set (see :func:`_scale_objective`).
    data_mean_square: float | None = None

    @property
    def loss_scale(self) -> float:
        """The factor the optimiser's objective is divided by."""
        return self.scale_box[0]


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
        return _add_prior_term(problem, data_term, physical, ic_thetas, ic_center)

    _obj_vg_jit = _cached_jit(
        ("obj_vg",) + struct_key, lambda: jax.jit(jax.value_and_grad(objective))
    )
    data = problem.data
    # The optimiser's scale of the objective (see _ObjectiveBundle): applied
    # outside the compiled programs, so it is not part of their cache key.
    scale_box = [1.0]

    def obj_value_and_grad(theta):
        value, grad = _obj_vg_jit(theta, *data)
        return value / scale_box[0], grad / scale_box[0]

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
            return _res_core(theta, *data, sigma_box[0]) / jnp.sqrt(scale_box[0])

        def _jac_call(theta):
            return _jac_core(theta, *data, sigma_box[0]) / jnp.sqrt(scale_box[0])

        res_j = _res_call
        jac_j = _jac_call

    return _ObjectiveBundle(
        obj_value_and_grad,
        res_j,
        jac_j,
        use_forward,
        sigma_hat_of,
        set_sigma_hat,
        scale_box,
        data_mean_square=problem.data_mean_square,
    )


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
    # uses a reverse scalar gradient, so its cap-free fix is the stable adjoint.
    on_finite_path = cfg.gradient == "stable_adjoint" or (
        cfg.ad_mode == "forward" and cfg.optimizer == "gauss_newton"
    )
    if on_finite_path:
        method = "stable" if cfg.gradient == "stable_adjoint" else "through_solve"
        remedy = (
            "It was already on a finite-by-construction path "
            f"(diff=DifferentiationConfig(mode={cfg.ad_mode!r}, method={method!r}), "
            f"OptimizerConfig(method={cfg.optimizer!r})); check the model, data, "
            "and parameter ranges."
        )
    else:
        # The stable adjoint serves a reactor and a plant fit; the forward-mode
        # Gauss-Newton Jacobian is a reactor-fit option only.
        remedy = (
            f"Pass {STABLE_REVERSE_DIFF} (cap-free reverse-mode, handled "
            "internally), or, for a reactor fit, "
            'optimizer=aquakin.OptimizerConfig(method="gauss_newton") with '
            f"{FORWARD_THROUGH_SOLVE_DIFF} (forward-mode, finite through a stiff "
            "solve). Neither needs diffrax or a dtmax cap in your code."
        )
    check_finite_gradient(probe, what="calibration gradient", remedy=remedy)
