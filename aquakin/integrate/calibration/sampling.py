"""Sampling the posterior of a calibration: :func:`sample_posterior`.

A random-walk Metropolis sampler over the unconstrained coordinates the fit ran
in (the rate parameters through their transforms, then the logs of the fitted
initial pools). The target is built from the objective the MAP fit minimized
(:func:`_objective_fn`), so the data term, the priors and the transforms are one
function shared with the fit. Which density ``exp(-objective)`` is taken to be
is the ``prior_space`` choice:

- ``"transformed"`` (the default): ``log p(theta) = -objective(theta)``, a
  density in the transformed coordinates. This is the density the fit's mode
  and its Laplace approximation describe, so the draws are directly comparable
  with the Laplace posterior and the profile likelihood. Beyond any stated
  rate prior, the prior is flat in ``theta`` (in ``log k`` for a
  ``positive_log`` rate).
- ``"physical"``: ``log p(theta) = -objective(theta) + sum_i log |d p_i / d
  theta_i|``, a density in the physical values carried over to ``theta`` by
  the Jacobian of the rate transforms. Beyond any stated rate prior, the prior
  is flat in the physical value ``k``.

The pools carry no Jacobian in either mode: their prior is defined on their
log coordinates already.

Each chain proposes ``theta + s L z`` with ``L L^T`` the Laplace covariance of
the fit, ``z`` standard normal and ``s`` starting at ``2.38 / sqrt(d)``, the
optimal scale for a Gaussian target. During the warm-up each chain adapts
``log s`` towards a target acceptance rate by a Robbins-Monro step; the scale is
then frozen, so the kept draws come from a fixed Metropolis kernel. A proposal
outside the bounds the fit used is rejected, and so is one whose forward solve
fails or gives a non-finite density.

The diagnostics are those of Vehtari, Gelman, Simpson, Carpenter and Buerkner
(2021), "Rank-normalization, folding, and localization: an improved R-hat for
assessing convergence of MCMC", Bayesian Analysis 16(2): the rank-normalized
split R-hat (the larger of its bulk and folded values) and the bulk effective
sample size, with Geyer's initial monotone sequence for the autocorrelation
sum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from aquakin.integrate._transforms import dphysical_dunconstrained, from_unconstrained
from aquakin.integrate.calibration.config import SamplerConfig
from aquakin.integrate.calibration.coords import _joint_names
from aquakin.integrate.calibration.multistart import _is_solve_failure, _optimizer_bounds
from aquakin.integrate.calibration.objective import _objective_fn
from aquakin.integrate.calibration.problem import _ReactorForwardModel
from aquakin.integrate.calibration.result import PosteriorSamples
from aquakin.progress import report

if TYPE_CHECKING:
    from aquakin.integrate._common import Reactor
    from aquakin.integrate.calibration.problem import _CalibrationProblem
    from aquakin.integrate.calibration.result import CalibrationResult

__all__ = ["sample_posterior"]

# Robbins-Monro exponent of the warm-up scale adaptation: steps (t + 1)**-0.6
# shrink fast enough to settle and slowly enough to keep adapting.
_ADAPT_EXPONENT = 0.6
# A Laplace eigenvalue below this fraction of the largest is a direction the
# Hessian does not see (null, or held on a bound); see _proposal_factor.
_NULL_EIG_RTOL = 1e-12
# The spaces the sampled density can be a density in; see the module docstring.
_PRIOR_SPACES = ("transformed", "physical")


def _target_acceptance(d: int) -> float:
    """Acceptance rate the warm-up aims for in ``d`` dimensions: 0.44, the
    optimum of a one-dimensional random walk, or 0.234, the optimum as the
    dimension grows (Roberts, Gelman and Gilks 1997), whose efficiency is
    within a few percent of the best from two dimensions on."""
    return 0.44 if d == 1 else 0.234


# --- The target density ------------------------------------------------


def _log_jacobian(problem: _CalibrationProblem, rate_thetas):
    """``sum_i log |d p_i / d theta_i|`` over the rate transforms.

    Parameters
    ----------
    problem : _CalibrationProblem
        The resolved problem (its transforms).
    rate_thetas : jnp.ndarray
        The rate coordinates, shape ``(n_rate,)``.

    Returns
    -------
    jnp.ndarray
        The log-Jacobian, a scalar: ``theta_i`` for ``positive_log``,
        ``log(p (1 - p))`` for ``logit``, 0 for ``none``.
    """
    terms = [
        jnp.log(
            jnp.abs(
                dphysical_dunconstrained(from_unconstrained(rate_thetas[i], t), t),
            )
        )
        for i, t in enumerate(problem.transforms)
    ]
    return jnp.sum(jnp.stack(terms))


def _check_prior_space(prior_space: str) -> None:
    """Refuse a ``prior_space`` that is not one of :data:`_PRIOR_SPACES`.

    Raises
    ------
    ValueError
        If ``prior_space`` is not ``"transformed"`` or ``"physical"``.
    """
    if prior_space not in _PRIOR_SPACES:
        raise ValueError(
            f"prior_space must be 'transformed' (a prior flat in the coordinates "
            f"the fit ran in, as Laplace assumes) or 'physical' (flat in the "
            f"physical rate values); got {prior_space!r}."
        )


def _log_density(problem: _CalibrationProblem, fm, prior_space: str):
    """The log posterior density of the unconstrained coordinates, up to a
    constant: ``-objective(theta)``, plus ``log |J(theta)|`` of the rate
    transforms when ``prior_space="physical"``.

    Parameters
    ----------
    problem : _CalibrationProblem
        The resolved problem, with ``loss="nll"``.
    fm : forward model
        The forward-solve seam.
    prior_space : {"transformed", "physical"}
        The space the density ``exp(-objective)`` is a density in; see the
        module docstring.

    Returns
    -------
    callable
        ``log_p(theta) -> scalar``, a pure JAX function of the coordinates
        (rates, then fitted pools).
    """
    objective = _objective_fn(problem, fm)
    data = problem.data
    if prior_space == "transformed":

        def log_p(theta):
            return -objective(theta, *data)

    else:

        def log_p(theta):
            return -objective(theta, *data) + _log_jacobian(problem, theta[: problem.n_rate])

    return log_p


class _BatchDensity:
    """The log density at every chain's point.

    Each point is one call of the compiled density: on a CPU that is cheaper
    than one ``vmap`` over the chains, whose batched adaptive solve steps every
    chain as finely as the stiffest one needs. A point whose forward solve
    fails at runtime, or whose density is not finite, gets ``-inf``, so such a
    proposal is rejected. A point outside the support ``[lb, ub]`` gets
    ``-inf`` without being solved at.
    """

    def __init__(self, log_p, lb: np.ndarray, ub: np.ndarray):
        self._log_p = jax.jit(log_p)
        self.lb = lb
        self.ub = ub

    def __call__(self, thetas: np.ndarray) -> np.ndarray:
        inside = np.all((thetas >= self.lb) & (thetas <= self.ub), axis=1)
        values = np.array([self._point(th) if ok else -np.inf for th, ok in zip(thetas, inside)])
        return np.where(np.isfinite(values), values, -np.inf)

    def _point(self, theta: np.ndarray) -> float:
        try:
            return float(self._log_p(jnp.asarray(theta)))
        except Exception as exc:
            if not _is_solve_failure(exc):
                raise
            return -np.inf


# --- The sampler -------------------------------------------------------


def _proposal_factor(cov: np.ndarray) -> np.ndarray:
    """A factor ``L`` with ``L L^T`` the proposal covariance.

    The proposal is the Laplace covariance, except along a direction the
    Laplace Hessian does not see (an exactly null direction, or a parameter
    held on a bound, both of which have zero variance there): such a direction
    gets the largest Laplace variance instead, so the chains still move along
    it and the bounds and the data decide how far.

    Parameters
    ----------
    cov : np.ndarray
        The Laplace covariance in the unconstrained coordinates, ``(d, d)``.

    Returns
    -------
    np.ndarray
        ``Q diag(sqrt(s))``, shape ``(d, d)``, from the eigendecomposition
        ``Q diag(s) Q^T`` of ``cov`` with its small eigenvalues raised.
    """
    cov = 0.5 * (cov + cov.T)
    s, Q = np.linalg.eigh(cov)
    s_max = float(s.max())
    s = np.where(s > _NULL_EIG_RTOL * s_max, s, s_max)
    return Q * np.sqrt(s)[None, :]


def _start_points(density, theta_map, L, config: SamplerConfig, rng):
    """Each chain's starting point and its log density: a draw of the Laplace
    posterior around the MAP, moved inside the support, or the MAP itself when
    the density is not finite there."""
    d = theta_map.shape[0]
    starts = theta_map[None, :] + rng.standard_normal((config.n_chains, d)) @ L.T
    starts = np.clip(starts, density.lb, density.ub)
    log_p = density(starts)
    bad = ~np.isfinite(log_p)
    if np.any(bad):
        starts[bad] = theta_map
        log_p[bad] = density(theta_map[None, :])[0]
    return starts, log_p


def _metropolis(density, theta_map, L, config: SamplerConfig, progress=None):
    """Run the chains.

    Parameters
    ----------
    density : _BatchDensity
        The log density of a batch of points, one per chain.
    theta_map : np.ndarray
        The MAP, shape ``(d,)``.
    L : np.ndarray
        The proposal factor from :func:`_proposal_factor`, ``(d, d)``.
    config : SamplerConfig
        Chains, draws, warm-up and seed.
    progress : callable, optional
        Called with an :class:`aquakin.Progress` after each step (stage
        ``"sample"``); raising :class:`aquakin.Cancelled` from it stops the run.

    Returns
    -------
    tuple
        ``(chains, acceptance)``: the kept draws, ``(n_chains, n_draws, d)``,
        and each chain's acceptance rate after the warm-up, ``(n_chains,)``.
    """
    rng = np.random.default_rng(config.seed)
    d = theta_map.shape[0]
    target = _target_acceptance(d)
    current, current_lp = _start_points(density, theta_map, L, config, rng)
    log_scale = np.full(config.n_chains, np.log(2.38 / np.sqrt(d)))
    chains = np.empty((config.n_chains, config.n_draws, d))
    accepted = np.zeros(config.n_chains)
    for step in range(config.n_warmup + config.n_draws):
        z = rng.standard_normal((config.n_chains, d)) @ L.T
        proposal = current + np.exp(log_scale)[:, None] * z
        proposal_lp = density(proposal)
        with np.errstate(invalid="ignore"):
            log_alpha = np.minimum(0.0, np.nan_to_num(proposal_lp - current_lp, nan=-np.inf))
        accept = np.log(rng.uniform(size=config.n_chains)) < log_alpha
        current[accept] = proposal[accept]
        current_lp[accept] = proposal_lp[accept]
        if step < config.n_warmup:
            gain = (step + 1.0) ** -_ADAPT_EXPONENT
            log_scale += gain * (np.exp(log_alpha) - target)
        else:
            chains[:, step - config.n_warmup] = current
            accepted += accept
        phase = "warm-up" if step < config.n_warmup else "drawing"
        report(progress, "sample", step + 1, config.n_warmup + config.n_draws, phase)
    return chains, accepted / config.n_draws


# --- Diagnostics -------------------------------------------------------


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Each chain's first and second halves as separate chains, ``(2m, n//2)``
    (the middle draw of an odd-length chain is dropped)."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half :]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Normal scores of the ranks over every chain: ``Phi^-1((r - 3/8) / (S +
    1/4))`` with ``r`` the average rank of each draw among the ``S`` draws."""
    ranks = rankdata(x, method="average").reshape(x.shape)
    return ndtri((ranks - 0.375) / (x.size + 0.25))


def _rhat_basic(x: np.ndarray) -> float:
    """The potential scale reduction ``sqrt(var_plus / W)`` of chains ``x``,
    ``(m, n)``: ``W`` the mean within-chain variance, ``var_plus`` the pooled
    estimate ``(n - 1) / n W + B / n`` with ``B / n`` the variance of the chain
    means. ``nan`` when every chain is constant."""
    n = x.shape[1]
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    if within == 0.0:
        return float("nan")
    between_over_n = float(np.var(np.mean(x, axis=1), ddof=1))
    return float(np.sqrt(((n - 1) / n * within + between_over_n) / within))


def _split_rhat(x: np.ndarray) -> float:
    """Rank-normalized split R-hat of chains ``x``, ``(m, n)``: the larger of
    the bulk value (on the normal scores of the split chains) and the folded
    value (on the normal scores of their distance from the median), which
    catches chains that agree in location but not in scale."""
    x = np.asarray(x, dtype=float)
    bulk = _rhat_basic(_rank_normalize(_split_chains(x)))
    folded = _rhat_basic(_rank_normalize(_split_chains(np.abs(x - np.median(x)))))
    return max(bulk, folded)


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Each chain's autocovariance at lags ``0 .. n-1`` (normalized by ``n``),
    by FFT; shape ``(m, n)``."""
    n = x.shape[1]
    centred = x - x.mean(axis=1, keepdims=True)
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centred, n=n_fft, axis=1)
    return np.fft.irfft(f * np.conj(f), n=n_fft, axis=1)[:, :n] / n


def _ess(x: np.ndarray) -> float:
    """Effective sample size of chains ``x``, ``(m, n)``.

    The combined autocorrelation ``rho_t = 1 - (W - mean_c acov_c(t)) /
    var_plus`` is summed in pairs of lags while the pair sum is positive
    (Geyer's initial positive sequence), the pair sums are made non-increasing
    (initial monotone sequence), and ``ESS = m n / tau`` with ``tau = -1 + 2
    sum rho_t``. ``tau`` is floored at ``1 / log10(m n)``, as in Stan, so a
    strongly antithetic chain cannot report an unbounded ESS.
    """
    m, n = x.shape
    acov = _autocovariance(x)
    mean_var = float(np.mean(acov[:, 0])) * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n + float(np.var(np.mean(x, axis=1), ddof=1))
    rho = np.zeros(n)
    rho[0] = 1.0
    rho_even = 1.0
    rho_odd = 1.0 - (mean_var - float(np.mean(acov[:, 1]))) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0.0:
        rho_even = 1.0 - (mean_var - float(np.mean(acov[:, t + 1]))) / var_plus
        rho_odd = 1.0 - (mean_var - float(np.mean(acov[:, t + 2]))) / var_plus
        if rho_even + rho_odd >= 0.0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0.0:
        rho[max_t + 1] = rho_even
    # Initial monotone sequence: no pair sum may exceed the one before it.
    for k in range(1, max_t - 1, 2):
        if rho[k + 1] + rho[k + 2] > rho[k - 1] + rho[k]:
            rho[k + 1] = rho[k + 2] = 0.5 * (rho[k - 1] + rho[k])
    tau = -1.0 + 2.0 * float(np.sum(rho[: max_t + 1])) + rho[max_t + 1]
    tau = max(tau, 1.0 / np.log10(m * n))
    return m * n / tau


def _ess_bulk(x: np.ndarray) -> float:
    """Bulk effective sample size: :func:`_ess` of the rank-normalized split
    chains ``x``, ``(m, n)``."""
    return _ess(_rank_normalize(_split_chains(np.asarray(x, dtype=float))))


# --- Entry points --------------------------------------------------------


def _check_sampling_result(result: CalibrationResult) -> None:
    """Refuse a result the sampler cannot start from.

    Raises
    ------
    ValueError
        If the result was not returned by a fit, its loss is not a likelihood,
        or it has no Laplace covariance over every sampled coordinate.
    """
    record = result._fit_record
    if record is None:
        raise ValueError(
            "sample_posterior needs the CalibrationResult a fit returned "
            "(aquakin.calibrate or Plant.calibrate); this one carries no fit."
        )
    if not record.problem.is_likelihood:
        raise ValueError(
            "sample_posterior samples a posterior, which needs loss='nll' (with a "
            "known sigma, or sigma='estimate'); 'mse' and 'wmse' are not "
            "likelihoods. Refit with loss='nll'."
        )
    if result.posterior_cov is None:
        raise ValueError(
            "sample_posterior starts its proposal from the Laplace covariance; "
            "calibrate with laplace=True."
        )
    if record.problem.m_ic and result.joint_cov is None:
        raise ValueError(
            "sample_posterior samples the fitted initial pools with the rates, so "
            "it needs their joint Laplace covariance; calibrate with "
            "laplace=aquakin.LaplaceConfig(include_ic=True)."
        )


def _samples_result(problem, params, chains, acceptance) -> PosteriorSamples:
    """Assemble :class:`PosteriorSamples` from the chains in unconstrained
    coordinates, ``(n_chains, n_draws, d)``, and the calibrated full parameter
    vector ``params``."""
    names = _joint_names(problem)
    flat = chains.reshape(-1, chains.shape[2])
    physical = np.asarray(jax.vmap(problem.physical_from_theta)(flat[:, : problem.n_rate]))
    draws = np.tile(np.asarray(params, dtype=float), (flat.shape[0], 1))
    draws[:, np.asarray(problem.free_indices)] = physical
    ic_draws = None
    if problem.m_ic:
        mt = problem.m_ic_theta
        ic_draws = []
        for k, (C0_k, *_rest) in enumerate(problem.datasets):
            block = flat[:, problem.n_rate + k * mt : problem.n_rate + (k + 1) * mt]
            pools = np.asarray(jax.vmap(lambda c, C0_k=C0_k: problem.ic_pools(c, C0_k))(block))
            ic_draws.append({s: pools[:, j] for j, s in enumerate(problem.free_ic)})
    return PosteriorSamples(
        draws=draws,
        parameter_names=list(problem.free_params),
        params_named={n: physical[:, i] for i, n in enumerate(problem.free_params)},
        ic_draws=ic_draws,
        chains=chains,
        coordinate_names=names,
        r_hat={n: _split_rhat(chains[:, :, j]) for j, n in enumerate(names)},
        ess_bulk={n: _ess_bulk(chains[:, :, j]) for j, n in enumerate(names)},
        acceptance_rate=acceptance,
    )


def _run_sampler(
    result: CalibrationResult, fm, config: SamplerConfig, prior_space: str, progress=None
) -> PosteriorSamples:
    """Sample the posterior of a fitted result through the forward model ``fm``.

    Shared by :func:`sample_posterior` and :meth:`Plant.sample_posterior`,
    which differ only in the forward model they pass. The problem, the MAP and
    the Laplace covariance come from ``result``, which
    :func:`_check_sampling_result` has accepted.

    Parameters
    ----------
    result : CalibrationResult
        The fit to sample.
    fm : forward model
        The forward-solve seam the density solves through.
    config : SamplerConfig
        Chains, draws, warm-up and seed.
    prior_space : {"transformed", "physical"}
        The space the sampled density is a density in (see
        :func:`sample_posterior`).
    progress : callable, optional
        As for :func:`sample_posterior`.

    Returns
    -------
    PosteriorSamples

    Raises
    ------
    ValueError
        If ``prior_space`` is not ``"transformed"`` or ``"physical"``.
    """
    _check_prior_space(prior_space)
    problem = result._fit_record.problem
    theta_map = np.asarray(result._fit_record.theta_map, dtype=float)
    cov = result.joint_cov if problem.m_ic else result.posterior_cov
    _bounds, lb, ub, _ls, _has = _optimizer_bounds(problem, problem.rate_theta0())
    density = _BatchDensity(_log_density(problem, fm, prior_space), lb, ub)
    L = _proposal_factor(np.asarray(cov, dtype=float))
    chains, acceptance = _metropolis(density, theta_map, L, config, progress)
    return _samples_result(problem, result.params, chains, acceptance)


def sample_posterior(
    result: CalibrationResult,
    reactor: Reactor,
    *,
    config: SamplerConfig = SamplerConfig(),
    prior_space: str = "transformed",
    progress=None,
) -> PosteriorSamples:
    """Draw from the posterior of a calibration by Markov chain Monte Carlo.

    The Laplace posterior of :func:`aquakin.calibrate` is a Gaussian at the
    MAP; this samples the posterior itself, so a curved, skewed or bounded
    posterior is represented as it is. The density is built from the objective
    the fit minimized (the same likelihood, priors and transforms), over the
    coordinates the fit ran in. Fitted initial pools are sampled jointly with
    the rates.

    A transformed rate (``positive_log`` or ``logit``) needs one more choice:
    the space in which the posterior is a density, which fixes the prior the
    data are combined with. ``prior_space="transformed"`` (the default) takes
    the prior to be flat in the coordinates the fit ran in (in ``log k`` for a
    ``positive_log`` rate), times any rate prior given to the fit. This is the
    density the MAP and its Laplace approximation describe, so the draws, the
    Laplace posterior and :func:`aquakin.profile_likelihood` answer the same
    question: where the posterior is near Gaussian in those coordinates the
    three intervals agree, and a difference shows a genuinely non-Gaussian
    posterior. ``prior_space="physical"`` takes the prior to be flat in the
    physical value ``k`` instead, times any rate prior: the sampled density
    gains the Jacobian ``|dk / d log k| = k`` of the transform, which moves
    the draws of a poorly determined rate upwards (by about the variance of
    ``log k`` on the log scale). Choose it when the prior you mean is uniform
    in ``k``; a Gaussian rate prior (``priors=``) is stated on the physical
    value in both modes. Untransformed rates and the fitted initial pools (whose
    prior is defined on their logarithms) are sampled the same way in both.

    The sampler is a random-walk Metropolis with several chains started near
    the MAP. Proposals are Gaussian with the Laplace covariance, scaled by
    ``2.38 / sqrt(d)`` in ``d`` dimensions and adapted during the warm-up
    towards an acceptance rate of 0.234 (0.44 in one dimension). Draws stay
    inside the bounds the fit used. Each step costs one forward solve per
    chain, and no gradient.

    With ``sigma="estimate"`` the density is the concentrated likelihood the
    fit maximized, which is the posterior with each output's noise level
    integrated out under the Jeffreys prior ``p(sigma) ∝ 1 / sigma``.

    Parameters
    ----------
    result : CalibrationResult
        A fit from :func:`aquakin.calibrate` with ``loss="nll"`` and a Laplace
        posterior (``LaplaceConfig(include_ic=True)`` when initial pools were
        fitted). It supplies the problem (data, priors, transforms, bounds),
        the starting point and the proposal covariance.
    reactor : Reactor
        The reactor the forward solves run on: the one calibrated, or one on
        the same model with other solver settings.
    config : SamplerConfig, optional
        Chains, draws, warm-up and seed. Default ``SamplerConfig()``.
    prior_space : {"transformed", "physical"}, optional
        The space in which the posterior is a density: ``"transformed"`` for a
        prior flat in the transformed coordinates of the rates (the density
        the fit and its Laplace approximation use), ``"physical"`` for a prior
        flat in the physical rate values (the behaviour of aquakin 0.3.0).
        Either is multiplied by any rate prior of the fit. Default
        ``"transformed"``.
    progress : callable, optional
        Called with an :class:`aquakin.Progress` after each step of the chains
        (stage ``"sample"``, ``total`` the warm-up plus the draws); raising
        :class:`aquakin.Cancelled` from it stops the sampling.

    Returns
    -------
    PosteriorSamples
        The draws (laid out as :meth:`CalibrationResult.posterior_draws`), the
        chains, and the R-hat, effective sample size and acceptance rate.
        Check that every ``r_hat`` is below about 1.01 and that ``ess_bulk``
        is large enough for the precision you need before using the draws.

    Raises
    ------
    ValueError
        If ``result`` did not come from a reactor fit with ``loss="nll"`` and a
        Laplace posterior, ``reactor`` is on another model, or ``prior_space``
        is not ``"transformed"`` or ``"physical"``.
    """
    _check_sampling_result(result)
    problem = result._fit_record.problem
    if not isinstance(result._fit_record.forward_model, _ReactorForwardModel):
        raise ValueError(
            "This result comes from Plant.calibrate; sample it with plant.sample_posterior(result)."
        )
    if reactor.model is not problem.model:
        raise ValueError(
            "reactor is on a different model from the one the result was calibrated on."
        )
    fm = _ReactorForwardModel(reactor, "jax_adjoint", 0, False)
    return _run_sampler(result, fm, config, prior_space, progress)
