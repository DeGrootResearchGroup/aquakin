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

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from aquakin.integrate._common import Reactor

# --- Parameter transforms ----------------------------------------------

_VALID_LOSSES = ("mse", "wmse", "nll")
_VALID_OPTIMIZERS = ("lbfgsb", "gauss_newton")
# Gradient backend. Both compute a discrete adjoint and both use JAX autodiff
# for the model derivatives (d f/d y, d f/d theta); they differ in how the
# integrator's adjoint is formed:
#   "jax_adjoint"    -- JAX/diffrax differentiate the whole solve
#                       (RecursiveCheckpointAdjoint). Needs a dtmax cap for stiff
#                       networks (reverse-mode overflows above a step threshold).
#   "stable_adjoint" -- AD for the model, plus an explicit per-step transposed
#                       solve for the integrator's adjoint
#                       (aquakin.implicit_euler_adjoint_solve). Cap-free and
#                       numerically stable for stiff networks.
_VALID_GRADIENTS = ("jax_adjoint", "stable_adjoint")

# Uniform optimiser output so the multistart loop and downstream code are
# agnostic to which backend (L-BFGS-B or Gauss-Newton least-squares) ran.
_OptOut = namedtuple("_OptOut", "x fun success message nit")


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


# --- Result dataclasses ------------------------------------------------


@dataclass
class PredictiveBand:
    """Posterior-predictive band from :meth:`CalibrationResult.predictive_band`.

    Attributes
    ----------
    t : np.ndarray
        Output time grid, shape ``(n_t,)`` (the ``t_eval`` passed in).
    median : np.ndarray
        Pointwise median over the posterior draws, shape ``(n_t, n_species)``
        (or ``(n_t, n_observed)`` if ``observed_species`` was given).
    lo, hi : np.ndarray
        Lower / upper percentile envelopes, same shape as ``median``.
    percentiles : tuple[float, float]
        The ``(lo, hi)`` percentiles used.
    n_valid : int
        Number of posterior draws that solved to a finite trajectory and were
        included in the percentiles.
    species : list[str] or None
        Observed-species labels for the columns, or ``None`` for all species.
    """

    t: np.ndarray
    median: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    percentiles: tuple[float, float]
    n_valid: int
    species: Optional[list[str]] = None


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
        The full scalar objective at the MAP: the summed data loss plus any
        prior penalty. It is the same value regardless of ``optimizer`` --- for
        ``loss="nll"`` it is the complete Gaussian negative log-likelihood
        (including the ``sum(log(sigma))`` normaliser), not the Gauss-Newton
        ``0.5*||residual||^2`` (which drops that constant).
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
    C0_fitted : list[jnp.ndarray], optional
        When ``free_ic`` is used, the fitted initial-state vector for each
        dataset (full species vectors, free pools set to their fitted values).
        ``None`` if no initial conditions were fit.
    ic_named : list[dict[str, float]], optional
        When ``free_ic`` is used, the fitted free initial pools per dataset, by
        species name. ``None`` if no initial conditions were fit.
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
    C0_fitted: Optional[list] = None
    ic_named: Optional[list] = None

    def predictive_band(
        self,
        reactor: Reactor,
        C0: jnp.ndarray,
        t_eval: jnp.ndarray,
        *,
        n_draw: int = 200,
        percentiles: tuple[float, float] = (2.5, 97.5),
        seed: int = 0,
        eig_keep: float = 1e-2,
        observed_species: Optional[list[str]] = None,
    ) -> PredictiveBand:
        """Posterior-predictive band by propagating Laplace draws through a solve.

        Samples the calibrated parameters from the Laplace posterior
        ``N(MAP, H^{-1})`` (``H`` the Hessian stored on this result), inverse-
        transforms each draw to physical space, sets it into the fitted parameter
        vector, integrates ``reactor`` from ``C0`` over ``t_eval``, and returns
        the pointwise percentile envelope across draws. Near-null Hessian
        directions (non-identifiable parameter combinations) are dropped so the
        draws stay finite: only eigenvectors with eigenvalue above
        ``eig_keep * max_eigenvalue`` are sampled.

        Requires that the calibration was run with ``laplace=True`` (so a Hessian
        is available). The supplied ``C0`` may differ from the calibration
        initial state --- e.g. propagate the calibrated-rate uncertainty through a
        held-out / validation batch.

        Parameters
        ----------
        reactor : Reactor
            Reactor to integrate (forward solves only; any adjoint is fine).
        C0 : jnp.ndarray
            Initial state to propagate.
        t_eval : jnp.ndarray
            Output times; the band is reported at these points.
        n_draw : int
            Number of posterior draws.
        percentiles : tuple[float, float]
            Lower / upper band percentiles (default the central 95%).
        seed : int
            Seed for the draws (reproducible).
        eig_keep : float
            Relative eigenvalue threshold for the identifiable subspace.
        observed_species : list[str], optional
            Restrict the returned band to these species. ``None`` returns all.

        Returns
        -------
        PredictiveBand
        """
        if self.hessian_unconstrained is None:
            raise ValueError(
                "predictive_band requires a Laplace posterior; call "
                "calibrate(..., laplace=True)."
            )
        network = reactor.network
        names = self.parameter_names
        free_idx = jnp.asarray([network.param_index[n] for n in names])
        theta_map = np.array([
            float(_to_unconstrained(jnp.asarray(self.params_named[n]), t))
            for n, t in zip(names, self.transforms)
        ])

        # Identifiable-subspace draws: eigen-decompose the Hessian, keep
        # high-curvature directions, sample with std = 1/sqrt(eigenvalue) along
        # each. Use the *ridged* Hessian (recovered as inv(posterior_cov)) that
        # ``posterior_cov`` and ``params_named_std`` are built from, so the band
        # is drawn from exactly the reported posterior rather than the un-ridged
        # raw Hessian (which differs along near-null directions).
        H = np.linalg.inv(np.asarray(self.posterior_cov))
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        keep = w > eig_keep * max(w.max(), 1.0)
        if not np.any(keep):
            raise ValueError(
                "No identifiable directions above eig_keep; posterior is "
                "degenerate (try a smaller eig_keep or more informative data)."
            )
        std_k = 1.0 / np.sqrt(w[keep])
        rng = np.random.RandomState(seed)
        draws = theta_map[None, :] + (
            rng.standard_normal((n_draw, int(keep.sum()))) * std_k[None, :]
        ) @ V[:, keep].T

        base = self.params
        C0_j = jnp.asarray(C0)
        t_eval_j = jnp.asarray(t_eval)
        t_end = float(np.asarray(t_eval)[-1])
        transforms = self.transforms
        curves = []
        for theta in draws:
            physical = jnp.stack([
                _from_unconstrained(jnp.asarray(theta[i]), transforms[i])
                for i in range(len(names))
            ])
            p = base.at[free_idx].set(physical)
            try:
                cc = np.asarray(
                    reactor.solve(C0_j, p, t_span=(0.0, t_end), t_eval=t_eval_j).C
                )
            except Exception:
                continue
            if np.all(np.isfinite(cc)):
                curves.append(cc)
        if not curves:
            raise RuntimeError("All posterior draws failed to solve.")
        curves = np.array(curves)
        lo = np.percentile(curves, percentiles[0], axis=0)
        hi = np.percentile(curves, percentiles[1], axis=0)
        median = np.percentile(curves, 50.0, axis=0)
        if observed_species is not None:
            sp_idx = [network.species_index[s] for s in observed_species]
            lo, hi, median = lo[:, sp_idx], hi[:, sp_idx], median[:, sp_idx]
        return PredictiveBand(
            t=np.asarray(t_eval), median=median, lo=lo, hi=hi,
            percentiles=tuple(percentiles), n_valid=len(curves),
            species=list(observed_species) if observed_species is not None else None,
        )


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
    laplace_dtmax: Optional[float] = None,
    free_ic: Optional[list[str]] = None,
    ic_bounds: tuple[float, float] = (1e-3, 1e4),
    ic_prior_log_std: Optional[float] = None,
    param_halfwidth: Optional[float] = None,
    optimizer: str = "lbfgsb",
    gradient: str = "jax_adjoint",
    stable_adjoint_max_steps: int = 100_000,
    n_starts: int = 1,
    jitter: float = 0.5,
    jitter_schedule: Optional[tuple] = None,
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
    laplace_dtmax : float, optional
        Integrator-step cap used only for the Laplace Hessian. The Hessian (a
        Jacobian/gradient through the solve) is more step-sensitive than the fit
        itself, so for very stiff networks it can need a tighter cap than the fit
        reactor uses -- pass the fit reactor at a loose cap (fast) and set
        ``laplace_dtmax`` to a tighter one. ``None`` (default) reuses the fit
        reactor. Requires a ``BatchReactor``-style reactor (it is reconstructed
        with the new cap from ``network``/``conditions``/``rtol``/``atol``/
        ``adjoint``).
    free_ic : list[str], optional
        Species whose *initial* concentration is fitted in addition to the rate
        parameters. The pools are fitted per dataset (each batch gets its own
        initial values, in log space, box-bounded by ``ic_bounds``) while the
        rate parameters remain shared. Useful when an unmeasured initial state
        (e.g. a biofilm reservoir) is not known. The Laplace posterior is taken
        over the rate parameters with the fitted pools held at their optimum.
        The fitted pools are returned on ``CalibrationResult.C0_fitted`` /
        ``ic_named``.
    ic_bounds : tuple[float, float], optional
        ``(lo, hi)`` box bounds (physical concentration) for every free initial
        pool. Default ``(1e-3, 1e4)``.
    ic_prior_log_std : float, optional
        If given, a weak Gaussian prior in log space tethering each fitted pool
        to its supplied starting value, with this standard deviation. ``None``
        (default) leaves the pools governed only by the data and ``ic_bounds``.
    param_halfwidth : float, optional
        If given, box-bound each free *rate* parameter to ``theta0 +/-
        param_halfwidth`` in unconstrained (transformed) space --- a symmetric
        log-space box around the starting value that keeps the optimiser from
        wandering to extreme values. ``None`` (default) leaves rates unbounded
        (positivity is still enforced by the transform). Free initial pools are
        always bounded by ``ic_bounds`` regardless.
    gradient : {"jax_adjoint", "stable_adjoint"}, optional
        How parameter gradients of the data term are taken. Both compute a
        discrete adjoint and both use JAX autodiff for the model derivatives
        (``df/dy``, ``df/dtheta``); they differ only in how the *integrator's*
        adjoint is formed. ``"jax_adjoint"`` (default) lets JAX/diffrax
        differentiate the whole solve (``RecursiveCheckpointAdjoint``); for stiff
        networks this reverse-mode pass goes non-finite above a step-size
        threshold, so the reactor must carry a ``dtmax`` cap. ``"stable_adjoint"``
        keeps the autodiff model derivatives but replaces the integrator's
        adjoint with an explicit per-step transposed-stage solve
        (:func:`~aquakin.esdirk_adjoint_solve`) -- a robust adaptive ESDIRK
        forward (Kvaerno5, the same high-order method the reactors use) whose
        backward is finite with no cap. It is reverse-mode only, so it forces a
        reverse-mode residual Jacobian under ``optimizer="gauss_newton"``. Built
        from the reactor's network and (single-location) ``conditions`` at the
        reactor's ``rtol``/``atol``; supported for batch reactors. Because it
        matches the reactor's forward solver, its gradients agree with the
        capped ``"jax_adjoint"`` path to the optimiser's tolerance.
    stable_adjoint_max_steps : int, optional
        Maximum (and allocated) number of forward steps for the
        ``gradient="stable_adjoint"`` solve. The backward scan walks this whole
        saved-trajectory buffer, so its cost scales with this value -- set it to
        a tight upper bound on the actual step count, not far above it. Ignored
        for ``gradient="jax_adjoint"``.
    optimizer : {"lbfgsb", "gauss_newton"}, optional
        Optimisation backend. ``"lbfgsb"`` (default) minimises the scalar loss
        with SciPy L-BFGS-B (reverse-mode gradient). ``"gauss_newton"`` minimises
        the residual *vector* with SciPy ``least_squares`` (trust-region
        reflective) -- a Gauss-Newton method that exploits the least-squares
        structure and is markedly more robust on the multimodal landscapes of
        stiff reaction-network fits. The residual Jacobian is formed by AD:
        forward mode (``jacfwd``) when ``reactor`` was built with
        ``adjoint=diffrax.DirectAdjoint()``, otherwise reverse mode (``jacrev``).
        Forward mode stays finite at any integrator step, so for very stiff
        networks whose reverse-mode adjoint is non-finite, pass a
        ``DirectAdjoint`` reactor and use ``"gauss_newton"``.
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
        Ignored when ``n_starts == 1`` or when ``jitter_schedule`` is given.
    jitter_schedule : tuple of float, optional
        Cyclic per-start jitter scales. When given, start ``s`` (>= 1) uses scale
        ``jitter_schedule[(s-1) % len]`` and its own ``RandomState(seed + s)``,
        instead of the single ``jitter`` scale with one shared stream. A wider
        schedule explores farther from the start (useful when the global basin is
        only reached by larger perturbations). Default ``None`` (use ``jitter``).
    seed : int, optional
        Seed for the multistart perturbations, so a re-run reproduces the same
        starts and therefore the same optimum.
    max_iter, tol : passed through to the optimiser (L-BFGS-B ``maxiter``/``gtol``,
        or least-squares ``max_nfev``/``xtol``/``ftol``).

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
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {_VALID_OPTIMIZERS}; got {optimizer!r}."
        )
    if gradient not in _VALID_GRADIENTS:
        raise ValueError(
            f"gradient must be one of {_VALID_GRADIENTS}; got {gradient!r}."
        )
    if gradient == "stable_adjoint" and not hasattr(reactor, "conditions"):
        raise ValueError(
            "gradient='stable_adjoint' is implemented for batch reactors "
            "(those exposing a single-location `conditions`); got a reactor "
            f"without one ({type(reactor).__name__})."
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
    n_rate = len(free_params)

    # --- Free initial conditions (optional) ----------------------------
    # When free_ic is given, each dataset additionally fits the initial
    # concentration of those species (log space, box-bounded by ic_bounds),
    # appended to the optimisation vector after the rate block. The rate
    # parameters stay shared across datasets; the initial pools are per-dataset.
    free_ic_list = list(free_ic or [])
    for s in free_ic_list:
        if s not in network.species_index:
            raise KeyError(
                f"Unknown free_ic species '{s}'. Available: {network.species}"
            )
    m_ic = len(free_ic_list)
    if m_ic and not (0.0 < ic_bounds[0] < ic_bounds[1]):
        raise ValueError(f"ic_bounds must satisfy 0 < lo < hi; got {ic_bounds}.")
    ic_species_idx = jnp.asarray(
        [network.species_index[s] for s in free_ic_list], dtype=int
    )
    ic_center_blocks = []
    if m_ic:
        ic_np_idx = [network.species_index[s] for s in free_ic_list]
        for C0_i, *_rest in datasets:
            vals = np.clip(np.asarray(C0_i)[ic_np_idx], ic_bounds[0], ic_bounds[1])
            ic_center_blocks.append(np.log(vals))
    ic_center_full = (
        jnp.asarray(np.concatenate(ic_center_blocks)) if m_ic else jnp.zeros(0)
    )

    # --- The objective in unconstrained space --------------------------

    def physical_from_theta(rate_thetas: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [_from_unconstrained(rate_thetas[i], transform_array[i])
             for i in range(n_rate)]
        )

    # Stable-adjoint backend: predictions come from the cap-free ESDIRK
    # (Kvaerno5, matching the reactor's forward solver) solve whose integrator
    # adjoint is an explicit per-step transposed-stage solve (model derivatives
    # still by autodiff), instead of differentiating the reactor's diffrax solve.
    # Built from the reactor's network + (single-location) conditions, matching
    # the reactor's tolerances.
    if gradient == "stable_adjoint":
        from aquakin.integrate.discrete_adjoint import esdirk_adjoint_solve

        _da_fields = reactor.conditions.fields
        _da_rhs = lambda t, y, p: network.dCdt(y, p, _da_fields, 0)

    def _predict(p, ic_thetas, rctr=None):
        """Predicted observed-species trajectory per dataset, applying the
        per-dataset free initial pools (if any). ``rctr`` defaults to the fit
        reactor; the Laplace pass can supply a tighter-capped one (ignored by
        the stable-adjoint backend, which needs no cap)."""
        rctr = reactor if rctr is None else rctr
        preds = []
        for k, (C0_i, tobs_i, tspan_i, _loss_i, _resid_i) in enumerate(datasets):
            C0_k = C0_i
            if m_ic:
                C0_k = C0_i.at[ic_species_idx].set(
                    jnp.exp(ic_thetas[k * m_ic:(k + 1) * m_ic])
                )
            if gradient == "stable_adjoint":
                ys = esdirk_adjoint_solve(
                    _da_rhs, C0_k, p, tspan_i, tobs_i,
                    rtol=reactor.rtol, atol=reactor.atol,
                    max_steps=stable_adjoint_max_steps,
                )
                preds.append(ys[:, obs_species_indices])
            else:
                sol = rctr.solve(C0_k, p, t_span=tspan_i, t_eval=tobs_i)
                preds.append(sol.C[:, obs_species_indices])
        return preds

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        rate_thetas = theta[:n_rate]
        ic_thetas = theta[n_rate:]
        physical = physical_from_theta(rate_thetas)
        p = p0_full.at[free_indices].set(physical)
        # Sum the data terms over every dataset (the batches share ``p``).
        data_term = 0.0
        for (_C0, _t, _ts, loss_fn_i, _r), pred in zip(datasets, _predict(p, ic_thetas)):
            data_term = data_term + loss_fn_i(pred)
        if has_priors:
            data_term = data_term + 0.5 * jnp.sum(
                prior_mask * ((physical - prior_mean) / prior_std) ** 2
            )
        if m_ic and ic_prior_log_std:
            data_term = data_term + 0.5 * jnp.sum(
                ((ic_thetas - ic_center_full) / ic_prior_log_std) ** 2
            )
        return data_term

    obj_value_and_grad = jax.jit(jax.value_and_grad(objective))

    # --- Run SciPy L-BFGS-B in unconstrained space ---------------------

    rate_theta0 = jnp.stack(
        [
            _to_unconstrained(p0_full[network.param_index[name]], t)
            for name, t in zip(free_params, resolved_transforms)
        ]
    )
    theta0 = jnp.concatenate([rate_theta0, ic_center_full]) if m_ic else rate_theta0

    # Box bounds (unconstrained space). Rate dims are bounded to
    # theta0 +/- param_halfwidth when param_halfwidth is given (a symmetric box
    # in transformed space around the start), else unbounded; free-IC dims are
    # always bounded in log space by ic_bounds.
    rate_th0 = np.asarray(rate_theta0, dtype=float)
    if param_halfwidth is not None:
        rate_lb = list(rate_th0 - param_halfwidth)
        rate_ub = list(rate_th0 + param_halfwidth)
    else:
        rate_lb = [-np.inf] * n_rate
        rate_ub = [np.inf] * n_rate
    n_ic = m_ic * n_datasets
    ic_lb = [float(np.log(ic_bounds[0]))] * n_ic
    ic_ub = [float(np.log(ic_bounds[1]))] * n_ic
    _lb = np.array(rate_lb + ic_lb)
    _ub = np.array(rate_ub + ic_ub)
    _has_bounds = (param_halfwidth is not None) or m_ic
    if _has_bounds:
        bounds = [
            (None if not np.isfinite(lo) else float(lo),
             None if not np.isfinite(hi) else float(hi))
            for lo, hi in zip(_lb, _ub)
        ]
    else:
        bounds = None

    def _np_loss_and_grad(x_np):
        x = jnp.asarray(x_np)
        val, grad = obj_value_and_grad(x)
        return float(val), np.asarray(grad)

    if optimizer == "gauss_newton":
        # Minimise the residual vector (0.5||r||^2 == the scalar objective) with
        # trust-region least-squares. The residual Jacobian is by forward-mode AD
        # if the reactor is forward-capable (DirectAdjoint), else reverse-mode.
        def _full_residual(theta):
            rate_thetas = theta[:n_rate]
            ic_thetas = theta[n_rate:]
            physical = physical_from_theta(rate_thetas)
            p = p0_full.at[free_indices].set(physical)
            parts = []
            for (_C0, _t, _ts, _loss_i, resid_fn_i), pred in zip(
                datasets, _predict(p, ic_thetas)
            ):
                parts.append(resid_fn_i(pred))
            if has_priors:
                parts.append(prior_mask * (physical - prior_mean) / prior_std)
            if m_ic and ic_prior_log_std:
                parts.append((ic_thetas - ic_center_full) / ic_prior_log_std)
            return jnp.concatenate(parts)

        # The stable adjoint is a reverse-only custom_vjp, so its residual
        # Jacobian must be reverse-mode regardless of the reactor's adjoint.
        _use_forward = gradient != "stable_adjoint" and isinstance(
            getattr(reactor, "adjoint", None), diffrax.DirectAdjoint
        )
        _jac = jax.jacfwd(_full_residual) if _use_forward else jax.jacrev(_full_residual)
        _res_j = jax.jit(_full_residual)
        _jac_j = jax.jit(_jac)
        # trust-region least-squares accepts +/-inf bounds (= unbounded dims).
        _ls_bounds = (_lb, _ub)

    def _run_from(x_start):
        if optimizer == "lbfgsb":
            r = minimize(
                _np_loss_and_grad,
                np.asarray(x_start),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "gtol": tol},
            )
            return _OptOut(np.asarray(r.x), float(r.fun), bool(r.success),
                           str(r.message), int(r.nit))
        r = least_squares(
            lambda x: np.asarray(_res_j(jnp.asarray(x)), dtype=float),
            np.asarray(x_start),
            jac=lambda x: np.asarray(_jac_j(jnp.asarray(x)), dtype=float),
            method="trf", bounds=_ls_bounds, max_nfev=max_iter,
            xtol=tol, ftol=tol,
        )
        return _OptOut(np.asarray(r.x), float(r.cost), bool(r.success),
                       str(r.message), int(r.nfev))

    # Start 0 is the supplied/default initial point; the rest are deterministic
    # jittered restarts. Keep the lowest finite loss (multimodal landscapes).
    result = _run_from(theta0)
    if n_starts > 1:
        theta0_np = np.asarray(theta0)
        # Two jitter schemes. Default: a single Gaussian stream of scale
        # ``jitter`` seeded once. ``jitter_schedule`` (a tuple of scales):
        # start s uses scale schedule[(s-1) % len] with its own RandomState
        # (seed + s), so the per-start perturbations match a per-start-seeded
        # cyclic-jitter multistart (and a wider schedule explores farther).
        seq_rng = None if jitter_schedule else np.random.RandomState(seed)
        for s in range(1, n_starts):
            if jitter_schedule:
                jit = jitter_schedule[(s - 1) % len(jitter_schedule)]
                noise = np.random.RandomState(seed + s).normal(
                    0.0, jit, size=theta0_np.shape)
            else:
                noise = seq_rng.normal(0.0, jitter, size=theta0_np.shape)
            perturbed = theta0_np + noise
            if bounds is not None:
                perturbed = np.clip(perturbed, _lb, _ub)
            cand = _run_from(perturbed)
            if np.isfinite(cand.fun) and cand.fun < result.fun:
                result = cand
    theta_opt = jnp.asarray(result.x)
    rate_theta_opt = theta_opt[:n_rate]
    ic_opt = theta_opt[n_rate:]
    physical_opt = physical_from_theta(rate_theta_opt)
    full_params = p0_full.at[free_indices].set(physical_opt)

    # Fitted per-dataset initial states (when free_ic is active).
    C0_fitted = None
    ic_named = None
    if m_ic:
        C0_fitted = []
        ic_named = []
        for k, (C0_i, *_rest) in enumerate(datasets):
            vals = np.exp(np.asarray(ic_opt)[k * m_ic:(k + 1) * m_ic])
            C0_fitted.append(C0_i.at[ic_species_idx].set(jnp.asarray(vals)))
            ic_named.append({s: float(v) for s, v in zip(free_ic_list, vals)})

    # --- Laplace posterior (over the rate parameters; pools held at MAP) ---
    # The free initial pools are fixed at their fitted values and the Laplace
    # covariance is taken over the rate constants, so the posterior and
    # predictive_band describe rate uncertainty (matching the established
    # practice for these fits).

    posterior_cov = None
    posterior_std_unconstrained = None
    params_named_std = None
    hessian_unconstrained = None
    if laplace:
        d = n_rate
        # The Laplace Hessian (a Jacobian/gradient through the solve) is more
        # step-sensitive than the fit, so it can use a tighter integrator cap.
        # Reconstruct the reactor at laplace_dtmax when given; else reuse the fit
        # reactor.
        if laplace_dtmax is not None and laplace_dtmax != getattr(reactor, "dtmax", None):
            lap_reactor = type(reactor)(
                reactor.network, reactor.conditions, rtol=reactor.rtol,
                atol=reactor.atol, adjoint=reactor.adjoint, dtmax=laplace_dtmax)
        else:
            lap_reactor = reactor
        if laplace_method == "gauss_newton":
            # Gauss-Newton / Fisher Hessian H = J^T J, with J the Jacobian of the
            # scaled residuals (0.5||r||^2 == the loss). Only FIRST-order AD
            # through the solve (jax.jacrev), which works with the default
            # reverse-mode adjoint; the full Hessian does not (its
            # forward-over-reverse pass hits the adjoint's custom_vjp, and the
            # second-order solve is unreliable). For loss='nll' this is the
            # exact Fisher information; it is PSD by construction.
            def _residual_vec(rate_thetas):
                physical = physical_from_theta(rate_thetas)
                p = p0_full.at[free_indices].set(physical)
                parts = []
                for (_C0, _t, _ts, _loss_i, resid_fn_i), pred in zip(
                    datasets, _predict(p, ic_opt, lap_reactor)
                ):
                    parts.append(resid_fn_i(pred))
                if has_priors:
                    parts.append(prior_mask * (physical - prior_mean) / prior_std)
                return jnp.concatenate(parts)

            _use_fwd_lap = gradient != "stable_adjoint" and isinstance(
                getattr(lap_reactor, "adjoint", None), diffrax.DirectAdjoint)
            J = (jax.jacfwd if _use_fwd_lap else jax.jacrev)(_residual_vec)(rate_theta_opt)
            H = J.T @ J
        elif laplace_method == "fd":
            # FD Hessian of the loss over the rates (pools fixed at MAP), using
            # the (possibly tighter-capped) Laplace reactor.
            def _objective_rate(rate_thetas):
                physical = physical_from_theta(rate_thetas)
                p = p0_full.at[free_indices].set(physical)
                total = 0.0
                for (_C0, _t, _ts, loss_fn_i, _r), pred in zip(
                    datasets, _predict(p, ic_opt, lap_reactor)
                ):
                    total = total + loss_fn_i(pred)
                if has_priors:
                    total = total + 0.5 * jnp.sum(
                        prior_mask * ((physical - prior_mean) / prior_std) ** 2)
                return total

            grad_fn = jax.jit(jax.grad(_objective_rate))
            H_rows = []
            for i in range(d):
                step = max(abs(float(rate_theta_opt[i])), 1.0) * laplace_fd_step
                e_i = jnp.zeros(d).at[i].set(step)
                g_plus = grad_fn(rate_theta_opt + e_i)
                g_minus = grad_fn(rate_theta_opt - e_i)
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
                _jacobian_physical_wrt_theta(rate_theta_opt[i], resolved_transforms[i])
                for i in range(d)
            ]
        )
        std_physical = jnp.abs(jac) * posterior_std_unconstrained
        params_named_std = {
            name: float(std_physical[i]) for i, name in enumerate(free_params)
        }

    # Report the loss as the full scalar objective at the optimum, identical
    # across optimizers. The Gauss-Newton path's ``r.cost`` is 0.5*||residual||^2,
    # which for ``loss="nll"`` omits the sum(log(sigma)) normaliser that the
    # L-BFGS-B scalar objective includes; evaluating ``objective`` directly puts
    # both on the same scale (and is a no-op for the L-BFGS-B path, whose
    # ``r.fun`` already is this objective).
    reported_loss = float(obj_value_and_grad(theta_opt)[0])

    return CalibrationResult(
        params=full_params,
        params_named={name: float(physical_opt[i]) for i, name in enumerate(free_params)},
        loss=reported_loss,
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
        C0_fitted=C0_fitted,
        ic_named=ic_named,
    )
