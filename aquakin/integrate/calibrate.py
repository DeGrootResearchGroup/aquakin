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
  around the MAP, via a finite-difference Hessian of the loss in
  unconstrained space. Marginal standard deviations are then propagated
  back to physical space via the transform's first derivative (delta
  method).

The optimiser is SciPy L-BFGS-B with the gradient supplied by ``jax.grad``
through Diffrax's adjoint, same as :func:`fit`. The Hessian is FD because
``jax.hessian`` is incompatible with Diffrax's ``while_loop``-based
adjoint (this is documented in the WastewaterAD prior art that this code
ports from).
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
    laplace_ridge: float = 1e-6,
    laplace_fd_step: float = 1e-3,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> CalibrationResult:
    """MAP fit with optional Laplace posterior approximation.

    Parameters
    ----------
    reactor : Reactor
        Batch reactor (or anything with a compatible ``solve``). Same usage
        contract as :func:`fit`.
    C0 : jnp.ndarray
        Initial concentration vector.
    observations : jnp.ndarray
        Observed values, shape ``(n_t,)`` for a single species or
        ``(n_t, n_observed)``.
    t_obs : jnp.ndarray
        Observation times, shape ``(n_t,)``. ``C0`` is taken at ``t=0``;
        the solver integrates from ``0`` to ``t_obs[-1]``.
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
        Scalar, ``(n_observed,)``, or ``(n_t, n_observed)``.
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
    laplace_ridge : float
        Diagonal ridge added to the Hessian for positive-definiteness.
    laplace_fd_step : float
        Relative finite-difference step for the Hessian rows.
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

    # Observation prep.
    t_obs = jnp.asarray(t_obs)
    if t_obs.ndim != 1 or t_obs.shape[0] < 1:
        raise ValueError(f"t_obs must be a non-empty 1-D array, got shape {t_obs.shape}.")
    if float(t_obs[0]) < 0.0:
        raise ValueError(f"t_obs must be non-negative; got t_obs[0] = {float(t_obs[0])}.")
    if t_obs.shape[0] > 1 and not bool(jnp.all(jnp.diff(t_obs) > 0)):
        raise ValueError("t_obs must be strictly ascending.")

    observations = jnp.asarray(observations)
    if observations.ndim == 1:
        observations = observations[:, None]
    if observations.shape[0] != t_obs.shape[0]:
        raise ValueError(
            f"observations has {observations.shape[0]} rows but t_obs has "
            f"{t_obs.shape[0]} entries."
        )
    if observed_species is None:
        obs_species_indices = jnp.arange(network.n_species)
        n_observed = network.n_species
    else:
        obs_species_indices = jnp.asarray(
            [network.species_index[s] for s in observed_species]
        )
        n_observed = len(observed_species)
    if observations.shape[1] != n_observed:
        raise ValueError(
            f"observations has {observations.shape[1]} columns but {n_observed} "
            f"species were specified."
        )

    if sigma is not None:
        sigma = jnp.asarray(sigma)

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

    t_span = (0.0, float(t_obs[-1]))
    loss_fn_pred = _build_loss(loss, observations, sigma)
    transform_array = resolved_transforms  # captured as Python list (static)

    # --- The objective in unconstrained space --------------------------

    def physical_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [_from_unconstrained(theta[i], transform_array[i]) for i in range(theta.shape[0])]
        )

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        physical = physical_from_theta(theta)
        p = p0_full.at[free_indices].set(physical)
        sol = reactor.solve(C0, p, t_span=t_span, t_eval=t_obs)
        pred = sol.C[:, obs_species_indices]
        data_term = loss_fn_pred(pred)
        if has_priors:
            prior_term = 0.5 * jnp.sum(
                prior_mask * ((physical - prior_mean) / prior_std) ** 2
            )
            return data_term + prior_term
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

    result = minimize(
        _np_loss_and_grad,
        np.asarray(theta0),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol},
    )
    theta_opt = jnp.asarray(result.x)
    physical_opt = physical_from_theta(theta_opt)
    full_params = p0_full.at[free_indices].set(physical_opt)

    # --- Laplace posterior --------------------------------------------

    posterior_cov = None
    posterior_std_unconstrained = None
    params_named_std = None
    hessian_unconstrained = None
    if laplace:
        # FD Hessian of the loss at theta_opt.
        grad_fn = jax.jit(jax.grad(objective))

        d = int(theta_opt.shape[0])
        H_rows = []
        for i in range(d):
            step = max(abs(float(theta_opt[i])), 1.0) * laplace_fd_step
            e_i = jnp.zeros(d).at[i].set(step)
            g_plus = grad_fn(theta_opt + e_i)
            g_minus = grad_fn(theta_opt - e_i)
            H_rows.append((g_plus - g_minus) / (2.0 * step))
        H = jnp.stack(H_rows)
        H = 0.5 * (H + H.T)  # symmetrise away FD noise

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
