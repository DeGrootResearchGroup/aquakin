"""MAP calibration with parameter transforms and Laplace-approximate posterior.

This is the richer companion to :func:`aquakin.fit`. Differences from
:func:`fit`:

- Parameters can declare a transform (``positive_log`` for ``p > 0``,
  ``logit`` for ``0 < p < 1``, ``none`` for identity). Optimisation runs in
  unconstrained space — more robust convergence than box constraints.
- Loss is configurable: ``"mse"`` (default), ``"wmse"`` (weighted MSE),
  ``"nll"`` (Gaussian negative log-likelihood). The last two accept a
  per-observation ``sigma``.
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from aquakin.integrate._common import Reactor

# --- Parameter transforms ----------------------------------------------

_VALID_LOSSES = ("mse", "wmse", "nll")


def _to_unconstrained(value: jnp.ndarray, transform: str) -> jnp.ndarray:
    if transform == "none":
        return value
    if transform == "positive_log":
        return jnp.log(value)
    if transform == "logit":
        return jnp.log(value / (1.0 - value))
    raise ValueError(f"Unknown transform {transform!r}")


def _from_unconstrained(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    if transform == "none":
        return theta
    if transform == "positive_log":
        return jnp.exp(theta)
    if transform == "logit":
        return jax.nn.sigmoid(theta)
    raise ValueError(f"Unknown transform {transform!r}")


def _jacobian_physical_wrt_theta(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    """``dp/dtheta`` at the given ``theta``, used for the delta-method std."""
    if transform == "none":
        return jnp.ones_like(theta)
    if transform == "positive_log":
        return jnp.exp(theta)
    if transform == "logit":
        s = jax.nn.sigmoid(theta)
        return s * (1.0 - s)
    raise ValueError(f"Unknown transform {transform!r}")


# --- Loss factory ------------------------------------------------------


def _build_loss(
    loss_type: str,
    observations: jnp.ndarray,
    sigma: Optional[jnp.ndarray],
):
    """Return ``loss(pred) -> scalar``.

    ``pred`` and ``observations`` have shape ``(n_t, n_observed)``.
    """
    if loss_type == "mse":
        def _loss(pred):
            return jnp.mean((pred - observations) ** 2)
        return _loss
    if loss_type == "wmse":
        if sigma is None:
            raise ValueError("loss='wmse' requires a sigma argument.")
        sig = sigma
        def _loss(pred):
            return jnp.mean(((pred - observations) / sig) ** 2)
        return _loss
    if loss_type == "nll":
        if sigma is None:
            raise ValueError("loss='nll' requires a sigma argument.")
        sig = sigma
        def _loss(pred):
            return jnp.sum(
                jnp.log(sig) + (pred - observations) ** 2 / (2.0 * sig ** 2)
            )
        return _loss
    raise ValueError(
        f"Unknown loss {loss_type!r}; choose one of {_VALID_LOSSES}."
    )


def _build_residual(
    loss_type: str,
    observations: jnp.ndarray,
    sigma: Optional[jnp.ndarray],
):
    """Return ``residual(pred) -> 1-D array`` whose half-sum-of-squares equals
    the (theta-dependent part of the) scalar loss from :func:`_build_loss`.

    This lets the Laplace covariance use the Gauss-Newton / Fisher Hessian
    ``H = J^T J`` (``J`` the residual Jacobian), which needs only first-order
    AD through the solve --- unlike the full Hessian, whose forward-over-reverse
    pass hits the adjoint's ``custom_vjp`` and whose second-order solve is
    unreliable. For ``nll`` this ``H`` is exactly the Fisher information.
    """
    obs = observations
    if loss_type == "mse":
        scale = jnp.sqrt(2.0 / obs.size)
        def _resid(pred):
            return (scale * (pred - obs)).reshape(-1)
        return _resid
    if loss_type == "wmse":
        if sigma is None:
            raise ValueError("loss='wmse' requires a sigma argument.")
        scale = jnp.sqrt(2.0 / obs.size)
        def _resid(pred):
            return (scale * (pred - obs) / sigma).reshape(-1)
        return _resid
    if loss_type == "nll":
        if sigma is None:
            raise ValueError("loss='nll' requires a sigma argument.")
        def _resid(pred):
            return ((pred - obs) / sigma).reshape(-1)
        return _resid
    raise ValueError(
        f"Unknown loss {loss_type!r}; choose one of {_VALID_LOSSES}."
    )


# --- Result dataclass --------------------------------------------------


@dataclass
class CalibrationResult:
    """Result of :func:`calibrate`.

    Attributes
    ----------
    params : jnp.ndarray
        Full parameter vector after optimisation (fixed entries unchanged).
    params_named : dict[str, float]
        Free parameters by namespaced name, in physical space.
    loss : float
        Final loss (in physical-parameter space at the MAP).
    converged : bool
        Whether SciPy declared convergence.
    message : str
        Optimiser status message.
    n_iter : int
    parameter_names : list[str]
        Ordered free-parameter names. Matches the rows / cols of
        ``posterior_cov`` and ``hessian_unconstrained`` when present.
    transforms : list[str]
        Transform used for each free parameter, in the same order.
    posterior_cov : jnp.ndarray, optional
        ``(d, d)`` covariance matrix in *unconstrained* space (Laplace).
        ``None`` if ``laplace=False`` was passed.
    posterior_std_unconstrained : jnp.ndarray, optional
        Marginal std devs in unconstrained space.
    params_named_std : dict[str, float], optional
        Marginal std devs propagated back to physical space via the
        delta method (``|dp/dtheta| * std_unconstrained``).
    hessian_unconstrained : jnp.ndarray, optional
        Raw FD Hessian of the loss at the MAP, unconstrained-space.
    priors_applied : dict[str, tuple[float, float]], optional
        Gaussian priors (physical-space ``(mean, std)``) that were added to the
        objective, by free-parameter name. Empty if no priors were active.
    """

    params: jnp.ndarray
    params_named: dict[str, float]
    loss: float
    converged: bool
    message: str
    n_iter: int
    parameter_names: list[str]
    transforms: list[str]
    posterior_cov: Optional[jnp.ndarray] = None
    posterior_std_unconstrained: Optional[jnp.ndarray] = None
    params_named_std: Optional[dict[str, float]] = field(default=None)
    hessian_unconstrained: Optional[jnp.ndarray] = None
    priors_applied: dict[str, tuple[float, float]] = field(default_factory=dict)


# --- Main entry point --------------------------------------------------


def calibrate(
    reactor: Reactor,
    C0: jnp.ndarray,
    observations: jnp.ndarray,
    t_obs: jnp.ndarray,
    free_params: list[str],
    *,
    transforms: Optional[dict[str, str]] = None,
    initial_params: Optional[jnp.ndarray] = None,
    observed_species: Optional[list[str]] = None,
    loss: str = "mse",
    sigma: Optional[jnp.ndarray] = None,
    priors: Optional[dict[str, tuple[float, float]]] = None,
    use_priors: bool = True,
    laplace: bool = True,
    laplace_method: str = "fd",
    laplace_ridge: float = 1e-6,
    laplace_fd_step: float = 1e-3,
    n_starts: int = 1,
    jitter: float = 0.5,
    seed: int = 0,
    max_iter: int = 500,
    tol: float = 1e-6,
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
        per dataset; the datasets may have different ``n_t``).
    t_obs : jnp.ndarray or list of jnp.ndarray
        Observation times, shape ``(n_t,)``. ``C0`` is taken at ``t=0``;
        the solver integrates from ``0`` to ``t_obs[-1]``. In multi-batch mode,
        a list of time grids, one per dataset.
    free_params : list[str]
        Namespaced parameter names to calibrate. Others held fixed.
    transforms : dict[str, str], optional
        Per-parameter transform. Keys may be any subset of ``free_params``;
        unspecified entries fall back to the parameter's declared
        ``transform`` on the network (default ``"none"``).
    initial_params : jnp.ndarray, optional
        Starting parameter vector. Defaults to
        ``reactor.network.default_parameters()``.
    observed_species : list[str], optional
        Species names corresponding to columns of ``observations``. If
        ``None``, every network species is taken to be observed.
    loss : {"mse", "wmse", "nll"}, optional
        Loss function.
    sigma : jnp.ndarray, optional
        Per-observation standard deviation for ``"wmse"`` / ``"nll"``.
        Scalar, ``(n_observed,)``, or ``(n_t, n_observed)``. In multi-batch
        mode, either a single value/array shared across datasets or a list with
        one entry per dataset.
    priors : dict[str, tuple[float, float]], optional
        Gaussian priors as ``name -> (mean, std)`` in physical space, added to
        the objective as ``0.5 * sum(((p - mean) / std) ** 2)``. Overrides any
        prior declared on the network for the same parameter. Only entries whose
        name is in ``free_params`` are used.
    use_priors : bool, optional
        If ``True`` (default), parameters whose network declaration carries a
        ``prior:`` block contribute their Gaussian prior to the objective (for
        the free parameters), in addition to any passed via ``priors``. Set
        ``False`` to ignore the network-declared priors. Priors regularise
        otherwise non-identifiable parameter combinations toward literature
        values; for a proper Bayesian MAP / posterior, combine them with
        ``loss="nll"`` and a measurement ``sigma`` so the data term is a true
        negative log-likelihood (the prior curvature then enters the Laplace
        covariance automatically).
    laplace : bool, optional
        If ``True``, compute the Laplace covariance approximation at the
        MAP. The result is interpretable as a Bayesian posterior only when
        ``loss="nll"`` with a calibrated ``sigma`` (i.e. the loss IS a
        proper Gaussian negative log-likelihood); for ``"mse"`` /
        ``"wmse"`` the covariance is the inverse loss curvature, which has
        the right shape but not the right absolute scale for posterior
        inference. See :func:`fit` if you only need point estimates.
    laplace_method : {"fd", "gauss_newton"}, optional
        How to form the Laplace Hessian. ``"fd"`` (default) finite-differences
        the gradient. ``"gauss_newton"`` uses ``H = J^T J`` with ``J`` the
        residual Jacobian by reverse-mode AD (``jax.jacrev``) -- exact (no FD
        step), PSD by construction, and for ``loss="nll"`` the Fisher
        information. It needs only first-order AD through the solve, so it works
        with the default ``RecursiveCheckpointAdjoint``; the full Hessian does
        not (see module docstring).
    laplace_ridge : float
        Diagonal ridge added to the Hessian for positive-definiteness.
    laplace_fd_step : float
        Relative finite-difference step for the Hessian rows (``"fd"`` only).
    n_starts : int, optional
        Number of optimiser starts (default ``1``). With ``n_starts > 1`` the
        calibration is run from several starting points and the lowest-loss
        result is kept --- a deterministic multistart that escapes local minima
        on the multimodal landscapes typical of stiff reaction-network fits.
        Start 0 is the supplied / default ``initial_params`` (unperturbed); the
        remaining starts perturb the unconstrained start vector by Gaussian noise
        of scale ``jitter``. The Laplace posterior is computed once, at the
        winning optimum. Fully reproducible given ``seed``.
    jitter : float, optional
        Standard deviation (in unconstrained / transformed space) of the
        Gaussian perturbation applied to each multistart start after the first.
        Ignored when ``n_starts == 1``.
    seed : int, optional
        Seed for the multistart perturbations, so a re-run reproduces the same
        starts and therefore the same optimum.
    max_iter, tol : passed through to SciPy ``L-BFGS-B``.

    Returns
    -------
    CalibrationResult
    """
    if not free_params:
        raise ValueError("free_params must be non-empty.")
    if loss not in _VALID_LOSSES:
        raise ValueError(
            f"loss must be one of {_VALID_LOSSES}; got {loss!r}."
        )
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1; got {n_starts}.")

    network = reactor.network
    for name in free_params:
        if name not in network.param_index:
            raise KeyError(
                f"Unknown parameter '{name}'. Available: {network.parameters}"
            )

    # Resolve transforms per free param.
    transforms = dict(transforms or {})
    resolved_transforms: list[str] = []
    for name in free_params:
        t = transforms.get(name)
        if t is None:
            t = network.parameter_transforms.get(name, "none")
        resolved_transforms.append(t)

    # Initial params (physical space).
    p0_full = (
        jnp.asarray(initial_params)
        if initial_params is not None
        else network.default_parameters()
    )
    free_indices = jnp.asarray([network.param_index[n] for n in free_params])

    # Validate initial physical values against their transforms.
    for name, t in zip(free_params, resolved_transforms):
        v = float(p0_full[network.param_index[name]])
        if t == "positive_log" and v <= 0.0:
            raise ValueError(
                f"Parameter '{name}' has transform 'positive_log' but initial "
                f"value {v} <= 0."
            )
        if t == "logit" and not (0.0 < v < 1.0):
            raise ValueError(
                f"Parameter '{name}' has transform 'logit' but initial value "
                f"{v} is not in (0, 1)."
            )

    # --- Datasets (one or several batches sharing the parameter vector) ---
    # A single-batch call passes plain arrays. A multi-batch call passes lists
    # of arrays for C0 / observations / t_obs (and optionally sigma); the
    # batches share one parameter vector and one prior, and their data terms
    # are summed -- a joint maximum-a-posteriori fit. Multi-batch mode is a
    # list/tuple whose elements are themselves vectors.
    def _is_multi(x) -> bool:
        return (
            isinstance(x, (list, tuple))
            and len(x) > 0
            and isinstance(x[0], (list, tuple, np.ndarray, jnp.ndarray))
        )

    multi = _is_multi(C0)
    C0_list = list(C0) if multi else [C0]
    obs_list = list(observations) if multi else [observations]
    tobs_list = list(t_obs) if multi else [t_obs]
    n_datasets = len(C0_list)
    if not (len(obs_list) == len(tobs_list) == n_datasets):
        raise ValueError(
            "In multi-dataset mode, C0, observations and t_obs must be lists of "
            f"equal length; got {n_datasets}, {len(obs_list)}, {len(tobs_list)}."
        )
    if isinstance(sigma, (list, tuple)):
        sigma_list = list(sigma)
        if len(sigma_list) != n_datasets:
            raise ValueError(
                f"sigma list has {len(sigma_list)} entries but there are "
                f"{n_datasets} datasets."
            )
    else:
        sigma_list = [sigma] * n_datasets

    if observed_species is None:
        obs_species_indices = jnp.arange(network.n_species)
        n_observed = network.n_species
    else:
        obs_species_indices = jnp.asarray(
            [network.species_index[s] for s in observed_species]
        )
        n_observed = len(observed_species)

    # Validate each dataset and build its (C0, t_eval, t_span, loss) tuple.
    datasets = []
    for ds, (C0_i, obs_i, tobs_i, sig_i) in enumerate(
        zip(C0_list, obs_list, tobs_list, sigma_list)
    ):
        C0_i = jnp.asarray(C0_i)
        tobs_i = jnp.asarray(tobs_i)
        if tobs_i.ndim != 1 or tobs_i.shape[0] < 1:
            raise ValueError(
                f"dataset {ds}: t_obs must be a non-empty 1-D array, got shape "
                f"{tobs_i.shape}."
            )
        if float(tobs_i[0]) < 0.0:
            raise ValueError(
                f"dataset {ds}: t_obs must be non-negative; got {float(tobs_i[0])}."
            )
        if tobs_i.shape[0] > 1 and not bool(jnp.all(jnp.diff(tobs_i) > 0)):
            raise ValueError(f"dataset {ds}: t_obs must be strictly ascending.")
        obs_i = jnp.asarray(obs_i)
        if obs_i.ndim == 1:
            obs_i = obs_i[:, None]
        if obs_i.shape[0] != tobs_i.shape[0]:
            raise ValueError(
                f"dataset {ds}: observations has {obs_i.shape[0]} rows but t_obs "
                f"has {tobs_i.shape[0]} entries."
            )
        if obs_i.shape[1] != n_observed:
            raise ValueError(
                f"dataset {ds}: observations has {obs_i.shape[1]} columns but "
                f"{n_observed} species were specified."
            )
        sig_arr = jnp.asarray(sig_i) if sig_i is not None else None
        datasets.append(
            (C0_i, tobs_i, (0.0, float(tobs_i[-1])),
             _build_loss(loss, obs_i, sig_arr),
             _build_residual(loss, obs_i, sig_arr))
        )

    # Resolve Gaussian priors for the free parameters. Network-declared priors
    # apply by default (use_priors); the explicit ``priors`` argument overrides
    # per parameter. Build aligned (mean, std, mask) arrays in free-param order.
    active_priors: dict[str, tuple[float, float]] = {}
    if use_priors:
        net_priors = getattr(network, "parameter_priors", {})
        for name in free_params:
            if name in net_priors:
                active_priors[name] = net_priors[name]
    if priors:
        for name, ms in priors.items():
            if name in free_params:
                active_priors[name] = (float(ms[0]), float(ms[1]))
    prior_mean = jnp.asarray(
        [active_priors.get(n, (0.0, 1.0))[0] for n in free_params]
    )
    prior_std = jnp.asarray(
        [active_priors.get(n, (0.0, 1.0))[1] for n in free_params]
    )
    prior_mask = jnp.asarray(
        [1.0 if n in active_priors else 0.0 for n in free_params]
    )
    has_priors = bool(active_priors)

    transform_array = resolved_transforms  # captured as Python list (static)

    # --- The objective in unconstrained space --------------------------

    def physical_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [_from_unconstrained(theta[i], transform_array[i]) for i in range(theta.shape[0])]
        )

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        physical = physical_from_theta(theta)
        p = p0_full.at[free_indices].set(physical)
        # Sum the data terms over every dataset (the batches share ``p``).
        data_term = 0.0
        for C0_i, tobs_i, tspan_i, loss_fn_i, _resid_fn_i in datasets:
            sol = reactor.solve(C0_i, p, t_span=tspan_i, t_eval=tobs_i)
            data_term = data_term + loss_fn_i(sol.C[:, obs_species_indices])
        if has_priors:
            data_term = data_term + 0.5 * jnp.sum(
                prior_mask * ((physical - prior_mean) / prior_std) ** 2
            )
        return data_term

    obj_value_and_grad = jax.jit(jax.value_and_grad(objective))

    # --- Run SciPy L-BFGS-B in unconstrained space ---------------------

    theta0 = jnp.stack(
        [
            _to_unconstrained(p0_full[network.param_index[name]], t)
            for name, t in zip(free_params, resolved_transforms)
        ]
    )

    def _np_loss_and_grad(x_np):
        x = jnp.asarray(x_np)
        val, grad = obj_value_and_grad(x)
        return float(val), np.asarray(grad)

    def _run_from(x_start):
        return minimize(
            _np_loss_and_grad,
            np.asarray(x_start),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": tol},
        )

    # Start 0 is the supplied/default initial point; the rest are deterministic
    # jittered restarts. Keep the lowest finite loss (multimodal landscapes).
    result = _run_from(theta0)
    if n_starts > 1:
        rng = np.random.RandomState(seed)
        theta0_np = np.asarray(theta0)
        for _ in range(1, n_starts):
            perturbed = theta0_np + rng.normal(0.0, jitter, size=theta0_np.shape)
            cand = _run_from(perturbed)
            if np.isfinite(cand.fun) and cand.fun < result.fun:
                result = cand
    theta_opt = jnp.asarray(result.x)
    physical_opt = physical_from_theta(theta_opt)
    full_params = p0_full.at[free_indices].set(physical_opt)

    # --- Laplace posterior --------------------------------------------

    posterior_cov = None
    posterior_std_unconstrained = None
    params_named_std = None
    hessian_unconstrained = None
    if laplace:
        d = int(theta_opt.shape[0])
        if laplace_method == "gauss_newton":
            # Gauss-Newton / Fisher Hessian H = J^T J, with J the Jacobian of the
            # scaled residuals (0.5||r||^2 == the loss). Only FIRST-order AD
            # through the solve (jax.jacrev), which works with the default
            # reverse-mode adjoint; the full Hessian does not (its
            # forward-over-reverse pass hits the adjoint's custom_vjp, and the
            # second-order solve is unreliable). For loss='nll' this is the
            # exact Fisher information; it is PSD by construction.
            def _residual_vec(theta):
                physical = physical_from_theta(theta)
                p = p0_full.at[free_indices].set(physical)
                parts = []
                for C0_i, tobs_i, tspan_i, _loss_i, resid_fn_i in datasets:
                    sol = reactor.solve(C0_i, p, t_span=tspan_i, t_eval=tobs_i)
                    parts.append(resid_fn_i(sol.C[:, obs_species_indices]))
                if has_priors:
                    parts.append(prior_mask * (physical - prior_mean) / prior_std)
                return jnp.concatenate(parts)

            J = jax.jacrev(_residual_vec)(theta_opt)
            H = J.T @ J
        elif laplace_method == "fd":
            # FD Hessian of the loss at theta_opt (finite-difference the gradient).
            grad_fn = jax.jit(jax.grad(objective))
            H_rows = []
            for i in range(d):
                step = max(abs(float(theta_opt[i])), 1.0) * laplace_fd_step
                e_i = jnp.zeros(d).at[i].set(step)
                g_plus = grad_fn(theta_opt + e_i)
                g_minus = grad_fn(theta_opt - e_i)
                H_rows.append((g_plus - g_minus) / (2.0 * step))
            H = jnp.stack(H_rows)
        else:
            raise ValueError(
                f"laplace_method must be 'fd' or 'gauss_newton'; got "
                f"{laplace_method!r}."
            )
        H = 0.5 * (H + H.T)  # symmetrise away asymmetry / FD noise

        H_ridge = H + laplace_ridge * jnp.eye(d)
        posterior_cov = jnp.linalg.inv(H_ridge)
        posterior_std_unconstrained = jnp.sqrt(jnp.diag(posterior_cov))
        hessian_unconstrained = H

        # Delta-method projection to physical space.
        jac = jnp.stack(
            [
                _jacobian_physical_wrt_theta(theta_opt[i], resolved_transforms[i])
                for i in range(d)
            ]
        )
        std_physical = jnp.abs(jac) * posterior_std_unconstrained
        params_named_std = {
            name: float(std_physical[i]) for i, name in enumerate(free_params)
        }

    return CalibrationResult(
        params=full_params,
        params_named={name: float(physical_opt[i]) for i, name in enumerate(free_params)},
        loss=float(result.fun),
        converged=bool(result.success),
        message=str(result.message),
        n_iter=int(result.nit),
        parameter_names=list(free_params),
        transforms=list(resolved_transforms),
        posterior_cov=posterior_cov,
        posterior_std_unconstrained=posterior_std_unconstrained,
        params_named_std=params_named_std,
        hessian_unconstrained=hessian_unconstrained,
        priors_applied=dict(active_priors),
    )
