"""The Laplace posterior around the MAP, its covariance (with the null-direction
convention shared with :mod:`aquakin.integrate.identifiability`), and the
at-bound checks."""

from __future__ import annotations

import warnings
from collections import namedtuple
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._transforms import dphysical_dunconstrained, from_unconstrained
from aquakin.integrate.calibration.coords import (
    _joint_names,
    _simplex_fraction_cov,
)
from aquakin.integrate.calibration.loss import _data_term
from aquakin.integrate.calibration.objective import _forward_jac, _residual_parts
from aquakin.integrate.calibration.problem import _predict
from aquakin.integrate.identifiability import _NULL_LOADING_TOL, _NULL_RTOL

if TYPE_CHECKING:
    from aquakin.integrate.calibration.objective import _FitConfig
    from aquakin.integrate.calibration.problem import _CalibrationProblem, _ReactorForwardModel


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
        [
            dphysical_dunconstrained(
                from_unconstrained(rate_theta_opt[i], problem.transforms[i]), problem.transforms[i]
            )
            for i in range(d)
        ]
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
