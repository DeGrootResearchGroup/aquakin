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

import warnings
from collections import namedtuple
from dataclasses import dataclass, field, replace

import diffrax
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

# --- Parameter transforms ----------------------------------------------

_VALID_LOSSES = ("mse", "wmse", "nll")
_VALID_OPTIMIZERS = ("lbfgsb", "gauss_newton")

# Uniform optimiser output so the multistart loop and downstream code are
# agnostic to which backend (L-BFGS-B or Gauss-Newton least-squares) ran.
_OptOut = namedtuple("_OptOut", "x fun success message nit")


# --- Public configuration objects --------------------------------------
#
# The optimiser, Laplace-posterior and free-initial-condition knobs are grouped
# into dataclasses (like :class:`DifferentiationConfig`) so the ``calibrate`` /
# ``Plant.calibrate`` / ``profile_likelihood`` signatures stay short and the
# clusters are opt-in with sensible defaults.


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimiser and multistart settings for calibration.

    Parameters
    ----------
    method : {"lbfgsb", "gauss_newton"}
        Optimisation backend. ``"lbfgsb"`` (default) is SciPy L-BFGS-B on the
        scalar loss; ``"gauss_newton"`` is a least-squares solve on the residual.
    n_starts : int
        Number of optimiser starts. ``1`` (default) is a single start from the
        initial parameters; ``>1`` adds jittered restarts and keeps the best.
    jitter : float
        Std dev of the Gaussian restart perturbation in unconstrained space
        (ignored when ``jitter_schedule`` is given).
    jitter_schedule : tuple of float, optional
        Explicit per-restart jitter scales; overrides ``jitter`` and fixes the
        restart sequence for reproducibility.
    seed : int
        Seed for the restart perturbations.
    max_iter : int
        Maximum optimiser iterations per start.
    tol : float
        Optimiser convergence tolerance.
    param_halfwidth : float, optional
        If given, box-bound each free rate parameter to ``theta0 +/-
        param_halfwidth`` in unconstrained space; otherwise the rate parameters
        are unbounded.
    """

    method: str = "lbfgsb"
    n_starts: int = 1
    jitter: float = 0.5
    jitter_schedule: tuple | None = None
    seed: int = 0
    max_iter: int = 500
    tol: float = 1e-6
    param_halfwidth: float | None = None


@dataclass(frozen=True)
class LaplaceConfig:
    """Laplace-posterior settings.

    Pass an instance to ``laplace=`` to enable the posterior with tuning;
    ``laplace=True`` uses these defaults and ``laplace=False`` disables it.

    Parameters
    ----------
    method : {"fd", "gauss_newton"}
        How the unconstrained-space Hessian is formed: ``"fd"`` (default)
        finite-differences the gradient; ``"gauss_newton"`` uses ``J^T J`` with
        the residual Jacobian by AD (PSD by construction, Fisher for ``nll``).
    ridge : float
        Ridge added to the Hessian before inversion.
    eig_keep : float
        Relative eigenvalue floor for the eigen-truncated covariance.
    fd_step : float
        Relative step for the finite-difference Hessian (``method="fd"``).
    dtmax : float, optional
        If given, rebuild the reactor with this ``dtmax`` cap for the Hessian
        pass only -- a tighter solve for the second-order-sensitive covariance.
    """

    method: str = "fd"
    ridge: float = 1e-6
    eig_keep: float = 1e-2
    fd_step: float = 1e-3
    dtmax: float | None = None


@dataclass(frozen=True)
class FreeICConfig:
    """Free-initial-condition settings for calibration.

    Parameters
    ----------
    species : list of str
        Species whose initial concentration is fitted alongside the parameters.
        For a plant, ``"unit.species"`` names.
    bounds : tuple of float
        ``(lo, hi)`` box bounds on each fitted initial pool, in physical units
        (the fit runs in log space). Must satisfy ``0 < lo < hi``.
    prior_log_std : float, optional
        If given, a Gaussian prior of this log-space std pulls each fitted pool
        toward its supplied initial value; ``None`` leaves it governed only by
        the data and ``bounds``.
    """

    species: list
    bounds: tuple = (1e-3, 1e4)
    prior_log_std: float | None = None


def _resolve_laplace(laplace) -> tuple[bool, LaplaceConfig]:
    """Normalise the ``laplace=`` argument to ``(enabled, LaplaceConfig)``.

    Accepts ``True`` (enable with defaults), ``False`` (disable), or a
    :class:`LaplaceConfig` (enable with tuning).
    """
    if laplace is True:
        return True, LaplaceConfig()
    if laplace is False:
        return False, LaplaceConfig()
    if isinstance(laplace, LaplaceConfig):
        return True, laplace
    raise TypeError(f"laplace must be a bool or LaplaceConfig; got {type(laplace).__name__}.")


def _free_ic_fields(free_ic) -> tuple[list | None, tuple, float | None]:
    """Unpack an optional :class:`FreeICConfig` into ``(species, bounds,
    prior_log_std)`` for the internal problem resolver."""
    if free_ic is None:
        return None, (1e-3, 1e4), None
    if isinstance(free_ic, FreeICConfig):
        return free_ic.species, free_ic.bounds, free_ic.prior_log_std
    raise TypeError(f"free_ic must be a FreeICConfig or None; got {type(free_ic).__name__}.")


def _to_unconstrained(value: jnp.ndarray, transform: str) -> jnp.ndarray:
    return to_unconstrained(value, transform)


def _from_unconstrained(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    return from_unconstrained(theta, transform)


def _jacobian_physical_wrt_theta(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    """``dp/dtheta`` at the given ``theta``, used for the delta-method std."""
    return dphysical_dunconstrained(from_unconstrained(theta, transform), transform)


# --- Loss factory ------------------------------------------------------


def _build_loss(
    loss_type: str,
    observations: jnp.ndarray,
    sigma: jnp.ndarray | None,
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
            return jnp.sum(jnp.log(sig) + (pred - observations) ** 2 / (2.0 * sig**2))

        return _loss
    raise ValueError(f"Unknown loss {loss_type!r}; choose one of {_VALID_LOSSES}.")


def _build_residual(
    loss_type: str,
    observations: jnp.ndarray,
    sigma: jnp.ndarray | None,
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
    raise ValueError(f"Unknown loss {loss_type!r}; choose one of {_VALID_LOSSES}.")


# --- Laplace covariance -----------------------------------------------


def _laplace_covariance(H, ridge: float, eig_keep: float):
    """Eigen-truncated Laplace covariance from an unconstrained-space Hessian.

    Symmetrise ``H``, add the ``ridge`` regulariser, eigen-decompose, and
    **drop** the near-null directions (ridged eigenvalue at or below
    ``eig_keep * largest_eigenvalue``) -- the non-identifiable parameter
    combinations the data do not constrain. The covariance is built from the
    surviving directions only, so it describes the identifiable subspace.

    The threshold is **purely relative** to the largest eigenvalue, so the
    best-identified direction is always kept regardless of the Hessian's
    absolute scale. (An earlier absolute floor wrongly discarded every
    direction of a uniformly small-scale but well-structured Hessian -- e.g.
    a trace-species observable in molar units, where ``JᵀJ`` is ~1e-2 yet
    spans several orders of magnitude.)

    This is the single regulariser shared by ``calibrate``'s
    ``posterior_cov`` / ``params_named_std`` and
    :meth:`CalibrationResult.predictive_band`, so the reported marginal std
    devs and the predictive draws can never disagree about which directions
    are identified (the inconsistency the two used to have).

    Parameters
    ----------
    H : array-like, shape (d, d)
        The (Gauss-Newton or finite-difference) Hessian of the loss.
    ridge : float
        Tikhonov regulariser added to the diagonal before inversion.
    eig_keep : float
        Relative eigenvalue floor for the identifiable subspace.

    Returns
    -------
    cov : np.ndarray, shape (d, d)
        The eigen-truncated covariance ``sum_k (1/w_k) v_k v_k^T`` over the
        kept directions (rank = number of kept directions).
    kept_eigvals : np.ndarray, shape (m,)
        The kept (ridged) eigenvalues ``w_k``.
    kept_eigvecs : np.ndarray, shape (d, m)
        The corresponding eigenvectors (columns).
    """
    # The threshold is a *fraction* of the largest eigenvalue, so it only keeps
    # the best-identified direction when eig_keep < 1 (then w_max > thr). At
    # eig_keep >= 1 every direction would be dropped, leaving an all-zero
    # covariance -- reject it with a clear message rather than returning that
    # silently.
    if not (0.0 <= eig_keep < 1.0):
        raise ValueError(
            f"eig_keep must be in [0, 1) (it is the relative eigenvalue floor, "
            f"a fraction of the largest eigenvalue); got {eig_keep}. A value "
            f">= 1 would drop every direction and give a degenerate covariance."
        )
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H + ridge * np.eye(H.shape[0]))
    w_max = float(w.max())
    if not np.isfinite(w_max) or w_max <= 0.0:
        raise ValueError(
            "Laplace Hessian is not finite / positive-definite after ridging; "
            "cannot form a posterior covariance (check the fit converged and "
            "the model output is finite)."
        )
    # Relative eigen-truncation: keep directions whose ridged eigenvalue exceeds
    # eig_keep * the largest. With eig_keep < 1 (enforced above) and w_max > 0,
    # the best-identified direction is always kept and `keep` is never empty.
    thr = eig_keep * w_max
    keep = w > thr
    wk = w[keep]
    Vk = V[:, keep]
    cov = (Vk / wk) @ Vk.T
    return cov, wk, Vk


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
    species: list[str] | None = None


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
        Optimiser iteration count: ``nit`` for the L-BFGS-B optimiser, and the
        Jacobian-evaluation count (``njev``, ~one per trust-region iteration) for
        the Gauss-Newton least-squares optimiser, which exposes no direct
        iteration count. Either way it is an iteration-scale figure, not the raw
        residual/function-evaluation count.
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
    posterior_cov: jnp.ndarray | None = None
    posterior_std_unconstrained: jnp.ndarray | None = None
    params_named_std: dict[str, float] | None = field(default=None)
    hessian_unconstrained: jnp.ndarray | None = None
    priors_applied: dict[str, tuple[float, float]] = field(default_factory=dict)
    C0_fitted: list | None = None
    ic_named: list | None = None

    def predictive_band(
        self,
        reactor: Reactor,
        C0: jnp.ndarray,
        t_eval: jnp.ndarray,
        *,
        n_draw: int = 200,
        percentiles: tuple[float, float] = (2.5, 97.5),
        seed: int = 0,
        eig_keep: float | None = None,
        observed_species: list[str] | None = None,
    ) -> PredictiveBand:
        """Posterior-predictive band by propagating Laplace draws through a solve.

        Samples the calibrated parameters from the Laplace posterior
        ``N(MAP, posterior_cov)``, inverse-transforms each draw to physical
        space, sets it into the fitted parameter vector, integrates ``reactor``
        from ``C0`` over ``t_eval``, and returns the pointwise percentile
        envelope across draws. The draws are taken from ``self.posterior_cov``
        --- the same eigen-truncated covariance behind ``params_named_std`` ---
        so the band and the reported marginal std devs regularise identically.
        The non-identifiable directions were already dropped (at calibrate time,
        via ``laplace_eig_keep``), so they carry zero variance and the draws
        stay finite.

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
        eig_keep : float, optional
            Deprecated and ignored. The identifiable-subspace truncation is now
            applied once, at calibrate time, via
            ``calibrate(laplace=LaplaceConfig(eig_keep=...))``, so the band and
            ``params_named_std`` share one regulariser. Passing a value here
            emits a ``DeprecationWarning``.
        observed_species : list[str], optional
            Restrict the returned band to these species. ``None`` returns all.

        Returns
        -------
        PredictiveBand
        """
        if self.posterior_cov is None:
            raise ValueError(
                "predictive_band requires a Laplace posterior; call calibrate(..., laplace=True)."
            )
        if eig_keep is not None:
            warnings.warn(
                "predictive_band(eig_keep=...) is deprecated and ignored; the "
                "identifiable-subspace truncation is set once at calibrate time "
                "via calibrate(laplace=LaplaceConfig(eig_keep=...)).",
                DeprecationWarning,
                stacklevel=2,
            )
        model = reactor.model
        names = self.parameter_names
        free_idx = jnp.asarray([model.param_index[n] for n in names])
        theta_map = np.array(
            [
                float(_to_unconstrained(jnp.asarray(self.params_named[n]), t))
                for n, t in zip(names, self.transforms)
            ]
        )

        # Draw from N(theta_map, posterior_cov). posterior_cov is the
        # eigen-truncated covariance that also backs params_named_std, so the
        # band and the marginal std devs regularise identically. Eigen-decompose
        # to sample; the truncated (non-identifiable) directions carry zero
        # variance and so add no spread (a tiny relative floor drops them and any
        # round-off negatives).
        cov = np.asarray(self.posterior_cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        s, Q = np.linalg.eigh(cov)
        pos = s > 1e-12 * max(float(s.max()), 0.0)
        if not np.any(pos):
            raise ValueError(
                "Posterior covariance has no positive-variance directions; the "
                "Laplace posterior is degenerate."
            )
        std_k = np.sqrt(s[pos])
        rng = np.random.default_rng(seed)
        draws = (
            theta_map[None, :]
            + (rng.standard_normal((n_draw, int(pos.sum()))) * std_k[None, :]) @ Q[:, pos].T
        )

        base = self.params
        C0_j = jnp.asarray(C0)
        t_eval_j = jnp.asarray(t_eval)
        t_end = float(np.asarray(t_eval)[-1])
        transforms = self.transforms
        curves = []
        first_error = None
        for theta in draws:
            physical = jnp.stack(
                [
                    _from_unconstrained(jnp.asarray(theta[i]), transforms[i])
                    for i in range(len(names))
                ]
            )
            p = base.at[free_idx].set(physical)
            try:
                cc = np.asarray(
                    reactor.solve(C0_j, params=p, t_span=(0.0, t_end), t_eval=t_eval_j).C
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc  # keep the first cause, to report if all fail
                continue
            if np.all(np.isfinite(cc)):
                curves.append(cc)
        if not curves:
            # Don't swallow the cause: surface the first solve failure (or note
            # that every draw merely went non-finite) instead of a bare message.
            if first_error is not None:
                raise RuntimeError(
                    "All posterior draws failed to solve; the first error is the "
                    "direct cause above."
                ) from first_error
            raise RuntimeError(
                "All posterior draws produced a non-finite trajectory (no "
                "exception raised) -- the Laplace draws may leave the feasible "
                "region; check the fit or tighten laplace_eig_keep."
            )
        curves = np.array(curves)
        lo = np.percentile(curves, percentiles[0], axis=0)
        hi = np.percentile(curves, percentiles[1], axis=0)
        median = np.percentile(curves, 50.0, axis=0)
        if observed_species is not None:
            sp_idx = [model.species_index[s] for s in observed_species]
            lo, hi, median = lo[:, sp_idx], hi[:, sp_idx], median[:, sp_idx]
        return PredictiveBand(
            t=np.asarray(t_eval),
            median=median,
            lo=lo,
            hi=hi,
            percentiles=tuple(percentiles),
            n_valid=len(curves),
            species=list(observed_species) if observed_species is not None else None,
        )


# --- Forward-model seam ------------------------------------------------
# The single point where calibration touches the object being fitted. The rest
# of the machinery (transforms, priors, free-IC, objective assembly, multistart,
# Laplace) is generic over "a thing that turns a parameter vector + initial state
# into an observed-species trajectory". Today the only implementation is the
# batch-reactor one below; a plant-backed model implementing the same two-method
# contract (``solve_trajectory`` + ``with_dtmax``) could be dropped in without
# touching the generic layer.


@dataclass
class _ReactorForwardModel:
    """Batch-reactor forward solve for :func:`calibrate`.

    ``solve_trajectory`` returns the full ``(n_t, n_species)`` trajectory (the
    generic layer slices out the observed columns). On the ``stable_adjoint``
    backend it integrates the cap-free ESDIRK discrete-adjoint solve built from
    the reactor's model + single-location conditions; otherwise it calls the
    reactor's own diffrax ``solve``.
    """

    reactor: Reactor
    gradient: str
    stable_adjoint_max_steps: int
    stable_adjoint_low_memory: bool

    def __post_init__(self):
        self._da_rhs = None
        if self.gradient == "stable_adjoint":
            from aquakin.integrate.discrete_adjoint import esdirk_adjoint_solve

            model = self.reactor.model
            fields = self.reactor.conditions.fields
            self._esdirk = esdirk_adjoint_solve
            self._da_rhs = lambda t, y, p: model.dCdt(y, p, fields, 0)

    @property
    def model(self):
        return self.reactor.model

    def solve_trajectory(self, p, C0_k, tspan, tobs):
        """Full ``(n_t, n_species)`` trajectory for one dataset."""
        if self.gradient == "stable_adjoint":
            return self._esdirk(
                self._da_rhs,
                C0_k,
                p,
                tspan,
                tobs,
                rtol=self.reactor.rtol,
                atol=self.reactor.atol,
                max_steps=self.stable_adjoint_max_steps,
                low_memory=self.stable_adjoint_low_memory,
            )
        return self.reactor.solve(C0_k, params=p, t_span=tspan, t_eval=tobs).C

    def forward_capable(self) -> bool:
        """Whether the reactor's adjoint supports forward-mode AD (jacfwd)."""
        return isinstance(getattr(self.reactor, "adjoint", None), diffrax.DirectAdjoint)

    def with_dtmax(self, dtmax) -> _ReactorForwardModel:
        """A clone whose reactor caps the integrator step at ``dtmax`` -- the
        (possibly tighter) reactor the Laplace Hessian is formed on. The
        ``stable_adjoint`` backend needs no cap, so it reuses ``self`` unchanged.
        """
        if dtmax is None or dtmax == getattr(self.reactor, "dtmax", None):
            return self
        rctr = type(self.reactor)(
            self.reactor.model,
            self.reactor.conditions,
            rtol=self.reactor.rtol,
            atol=self.reactor.atol,
            integrator=replace(self.reactor.integrator, dtmax=dtmax),
            diff=self.reactor.diff,
        )
        return _ReactorForwardModel(
            rctr, self.gradient, self.stable_adjoint_max_steps, self.stable_adjoint_low_memory
        )


# --- Resolved calibration problem --------------------------------------


@dataclass
class _CalibrationProblem:
    """The static, resolved calibration problem: everything derived from the
    ``calibrate`` arguments *once*, up front, and then held constant through the
    objective build, multistart and Laplace passes.

    Split out so each downstream stage (:func:`_build_objective`,
    :func:`_run_multistart`, :func:`_laplace_posterior`) takes one ``problem``
    argument instead of a dozen closed-over locals, and so the coercion can be
    unit-tested on its own.
    """

    model: object
    # Free rate parameters.
    free_params: list
    free_indices: jnp.ndarray
    transforms: list  # resolved transform per free param, same order
    n_rate: int
    p0_full: jnp.ndarray
    param_halfwidth: float | None
    # Datasets (one or several batches sharing the parameter vector).
    datasets: list  # full (C0, tobs, tspan, loss_fn, resid_fn) tuples
    dataset_static: list  # (tobs, tspan, loss_fn, resid_fn) per dataset
    C0_base: tuple  # per-dataset initial states
    n_datasets: int
    obs_species_indices: jnp.ndarray
    n_observed: int
    # Priors (aligned to free_params order).
    active_priors: dict
    prior_mean: jnp.ndarray
    prior_std: jnp.ndarray
    prior_mask: jnp.ndarray
    has_priors: bool
    # Free initial conditions (optional).
    free_ic: list
    m_ic: int
    ic_species_idx: jnp.ndarray
    ic_center_full: jnp.ndarray
    ic_prior_log_std: float | None
    ic_bounds: tuple

    def physical_from_theta(self, rate_thetas: jnp.ndarray) -> jnp.ndarray:
        """Free rate parameters in physical space from their unconstrained
        ``theta`` block (applying each parameter's transform)."""
        return jnp.stack(
            [_from_unconstrained(rate_thetas[i], self.transforms[i]) for i in range(self.n_rate)]
        )

    def rate_theta0(self) -> jnp.ndarray:
        """The unconstrained start vector for the rate block (from ``p0_full``)."""
        return jnp.stack(
            [
                _to_unconstrained(self.p0_full[self.model.param_index[name]], t)
                for name, t in zip(self.free_params, self.transforms)
            ]
        )

    @property
    def data(self) -> tuple:
        """The per-call-varying data threaded into the compiled objective as
        *arguments* (so a profile-likelihood sweep reuses one compiled program)."""
        return (self.p0_full, self.C0_base, self.ic_center_full)

    def struct_key(self, gradient: str) -> tuple:
        """Structural cache key: everything that changes the compiled program's
        shape. A ``_compiled_cache`` shared across differently-shaped fits keys
        on this so it rebuilds rather than mis-hitting."""
        return (
            self.n_rate,
            self.m_ic,
            len(self.dataset_static),
            gradient,
            self.n_observed,
            tuple(int(ds[0].shape[0]) for ds in self.dataset_static),
        )


def _predict(problem: _CalibrationProblem, fm: _ReactorForwardModel, p, ic_thetas, C0s):
    """Predicted observed-species trajectory per dataset, applying the
    per-dataset free initial pools (if any). ``C0s`` is the per-dataset initial
    states (a runtime argument, so a pinned-IC sweep reuses the compiled
    program)."""
    preds = []
    for k, ((tobs_i, tspan_i, _loss_i, _resid_i), C0_i) in enumerate(
        zip(problem.dataset_static, C0s)
    ):
        C0_k = C0_i
        if problem.m_ic:
            C0_k = C0_i.at[problem.ic_species_idx].set(
                jnp.exp(ic_thetas[k * problem.m_ic : (k + 1) * problem.m_ic])
            )
        ys = fm.solve_trajectory(p, C0_k, tspan_i, tobs_i)
        preds.append(ys[:, problem.obs_species_indices])
    return preds


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
    laplace_eig_keep: float
    laplace_fd_step: float
    laplace_dtmax: float | None
    compiled_cache: dict | None


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
):
    """The stacked residual vector whose ``0.5*||.||^2`` is the objective's
    theta-dependent part. Shared by the Gauss-Newton fit (full theta,
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
        parts.append(resid_fn_i(pred))
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
        data_term = 0.0
        for (_t, _ts, loss_fn_i, _r), pred in zip(
            problem.dataset_static, _predict(problem, fm, p, ic_thetas, C0s)
        ):
            data_term = data_term + loss_fn_i(pred)
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
    if cfg.optimizer == "gauss_newton":
        # Minimise the residual vector (0.5||r||^2 == the scalar objective) with
        # trust-region least-squares. The residual Jacobian is by forward-mode AD
        # if the reactor is forward-capable (DirectAdjoint), else reverse-mode.
        def _full_residual(theta, p0_full_arg, C0s, ic_center):
            return _residual_parts(
                problem,
                fm,
                theta[: problem.n_rate],
                theta[problem.n_rate :],
                p0_full_arg,
                C0s,
                ic_center,
                include_ic_prior=True,
            )

        # The stable adjoint is a reverse-only custom_vjp, so its residual
        # Jacobian must be reverse-mode regardless of the reactor's adjoint.
        use_forward = _forward_jac(cfg, fm)
        _jac = (jax.jacfwd if use_forward else jax.jacrev)(_full_residual, argnums=0)
        _res_core = _cached_jit(("gn_res",) + struct_key, lambda: jax.jit(_full_residual))
        _jac_core = _cached_jit(("gn_jac", use_forward) + struct_key, lambda: jax.jit(_jac))

        def _res_call(theta):
            return _res_core(theta, *data)

        def _jac_call(theta):
            return _jac_core(theta, *data)

        res_j = _res_call
        jac_j = _jac_call

    return _ObjectiveBundle(obj_value_and_grad, res_j, jac_j, use_forward)


def _optimizer_bounds(problem: _CalibrationProblem, rate_theta0):
    """Box bounds in unconstrained space. Rate dims are bounded to
    ``theta0 +/- param_halfwidth`` when given (else unbounded); free-IC dims are
    always log-bounded by ``ic_bounds``. Returns
    ``(bounds, lb, ub, ls_bounds, has_bounds)`` -- ``bounds`` the L-BFGS-B list
    (``None`` if unbounded), ``ls_bounds`` the least-squares ``(lb, ub)`` pair."""
    rate_th0 = np.asarray(rate_theta0, dtype=float)
    if problem.param_halfwidth is not None:
        rate_lb = list(rate_th0 - problem.param_halfwidth)
        rate_ub = list(rate_th0 + problem.param_halfwidth)
    else:
        rate_lb = [-np.inf] * problem.n_rate
        rate_ub = [np.inf] * problem.n_rate
    n_ic = problem.m_ic * problem.n_datasets
    ic_lb = [float(np.log(problem.ic_bounds[0]))] * n_ic
    ic_ub = [float(np.log(problem.ic_bounds[1]))] * n_ic
    lb = np.array(rate_lb + ic_lb)
    ub = np.array(rate_ub + ic_ub)
    has_bounds = (problem.param_halfwidth is not None) or problem.m_ic
    if has_bounds:
        bounds = [
            (None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
            for lo, hi in zip(lb, ub)
        ]
    else:
        bounds = None
    return bounds, lb, ub, (lb, ub), has_bounds


def _check_start_gradient(cfg: _FitConfig, bundle: _ObjectiveBundle, theta0):
    """Evaluate the gradient / Jacobian once at the start point and fail loudly
    (with the remedy) if it is non-finite -- a stiff reverse-mode adjoint
    overflowing -- rather than letting the optimizer wander on NaNs."""
    if cfg.optimizer == "gauss_newton":
        probe = bundle.jac_j(jnp.asarray(theta0))
    else:
        _v0, probe = bundle.value_and_grad(jnp.asarray(theta0))
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


def _run_multistart(cfg: _FitConfig, bundle: _ObjectiveBundle, theta0, opt_bounds) -> _OptOut:
    """Run the optimiser from the start point, then from ``n_starts - 1``
    deterministic jittered restarts, keeping the lowest finite loss."""
    bounds, lb, ub, ls_bounds, _has = opt_bounds

    def _np_loss_and_grad(x_np):
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
        r = least_squares(
            lambda x: np.asarray(bundle.res_j(jnp.asarray(x)), dtype=float),
            np.asarray(x_start),
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
        n_iter_ls = int(r.njev) if r.njev is not None else int(r.nfev)
        return _OptOut(np.asarray(r.x), float(r.cost), bool(r.success), str(r.message), n_iter_ls)

    result = _run_from(theta0)
    if cfg.n_starts > 1:
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
            cand = _run_from(perturbed)
            if np.isfinite(cand.fun) and cand.fun < result.fun:
                result = cand
    return result


def _laplace_posterior(
    problem: _CalibrationProblem,
    fm: _ReactorForwardModel,
    cfg: _FitConfig,
    rate_theta_opt,
    ic_opt,
):
    """Laplace covariance over the rate parameters (pools held at the MAP).

    Returns ``(posterior_cov, posterior_std_unconstrained, params_named_std,
    hessian_unconstrained)``. The Hessian is more step-sensitive than the fit, so
    it can use a tighter integrator cap (``laplace_dtmax``) via the forward
    model's ``with_dtmax``."""
    d = problem.n_rate
    fm_lap = fm.with_dtmax(cfg.laplace_dtmax)
    if cfg.laplace_method == "gauss_newton":
        # Gauss-Newton / Fisher Hessian H = J^T J, with J the Jacobian of the
        # scaled residuals. Only FIRST-order AD through the solve; for
        # loss='nll' this is the exact Fisher information (PSD by construction).
        # Rate-only residual (pools fixed at the MAP), ic-prior block omitted.
        def _residual_vec(rate_thetas):
            return _residual_parts(
                problem,
                fm_lap,
                rate_thetas,
                ic_opt,
                problem.p0_full,
                problem.C0_base,
                problem.ic_center_full,
                include_ic_prior=False,
            )

        use_fwd = _forward_jac(cfg, fm_lap)
        J = (jax.jacfwd if use_fwd else jax.jacrev)(_residual_vec)(rate_theta_opt)
        H = J.T @ J
    elif cfg.laplace_method == "fd":
        # FD Hessian of the loss over the rates (pools fixed at MAP).
        def _objective_rate(rate_thetas):
            physical = problem.physical_from_theta(rate_thetas)
            p = problem.p0_full.at[problem.free_indices].set(physical)
            total = 0.0
            for (_t, _ts, loss_fn_i, _r), pred in zip(
                problem.dataset_static, _predict(problem, fm_lap, p, ic_opt, problem.C0_base)
            ):
                total = total + loss_fn_i(pred)
            if problem.has_priors:
                total = total + 0.5 * jnp.sum(
                    problem.prior_mask * ((physical - problem.prior_mean) / problem.prior_std) ** 2
                )
            return total

        grad_fn = jax.jit(jax.grad(_objective_rate))
        H_rows = []
        for i in range(d):
            step = max(abs(float(rate_theta_opt[i])), 1.0) * cfg.laplace_fd_step
            e_i = jnp.zeros(d).at[i].set(step)
            g_plus = grad_fn(rate_theta_opt + e_i)
            g_minus = grad_fn(rate_theta_opt - e_i)
            H_rows.append((g_plus - g_minus) / (2.0 * step))
        H = jnp.stack(H_rows)
    else:
        raise ValueError(
            f"laplace_method must be 'fd' or 'gauss_newton'; got {cfg.laplace_method!r}."
        )
    H = 0.5 * (H + H.T)  # symmetrise away asymmetry / FD noise

    cov_np, _, _ = _laplace_covariance(np.asarray(H), cfg.laplace_ridge, cfg.laplace_eig_keep)
    posterior_cov = jnp.asarray(cov_np)
    posterior_std_unconstrained = jnp.sqrt(jnp.diag(posterior_cov))

    # Delta-method projection to physical space.
    jac = jnp.stack(
        [_jacobian_physical_wrt_theta(rate_theta_opt[i], problem.transforms[i]) for i in range(d)]
    )
    std_physical = jnp.abs(jac) * posterior_std_unconstrained
    params_named_std = {name: float(std_physical[i]) for i, name in enumerate(problem.free_params)}
    return posterior_cov, posterior_std_unconstrained, params_named_std, H


def _resolve_problem(
    model,
    C0,
    observations,
    t_obs,
    free_params,
    *,
    transforms,
    initial_params,
    observed_species,
    time_unit,
    loss,
    sigma,
    priors,
    use_priors,
    free_ic,
    ic_bounds,
    ic_prior_log_std,
    param_halfwidth,
) -> _CalibrationProblem:
    """Validate and coerce the ``calibrate`` arguments into a
    :class:`_CalibrationProblem`. Everything here is done once, up front; the
    result is held constant through the objective / multistart / Laplace passes."""
    for name in free_params:
        if name not in model.param_index:
            raise KeyError(f"Unknown parameter '{name}'. Available: {model.parameters}")

    # Resolve transforms per free param.
    transforms = dict(transforms or {})
    resolved_transforms: list[str] = []
    for name in free_params:
        t = transforms.get(name)
        if t is None:
            t = model.parameter_transforms.get(name, "none")
        resolved_transforms.append(t)

    # Initial params (physical space).
    p0_full = (
        jnp.asarray(initial_params) if initial_params is not None else model.default_parameters()
    )
    free_indices = jnp.asarray([model.param_index[n] for n in free_params])

    # Validate initial physical values against their transforms.
    for name, t in zip(free_params, resolved_transforms):
        v = float(p0_full[model.param_index[name]])
        if t == "positive_log" and v <= 0.0:
            raise ValueError(
                f"Parameter '{name}' has transform 'positive_log' but initial value {v} <= 0."
            )
        if t == "logit" and not (0.0 < v < 1.0):
            raise ValueError(
                f"Parameter '{name}' has transform 'logit' but initial value {v} is not in (0, 1)."
            )

    # --- Datasets (one or several batches sharing the parameter vector) ---
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
                f"sigma list has {len(sigma_list)} entries but there are {n_datasets} datasets."
            )
    else:
        sigma_list = [sigma] * n_datasets

    if observed_species is None:
        obs_species_indices = jnp.arange(model.n_species)
        n_observed = model.n_species
    else:
        obs_species_indices = jnp.asarray([model.species_index[s] for s in observed_species])
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
                f"dataset {ds}: t_obs must be a non-empty 1-D array, got shape {tobs_i.shape}."
            )
        if float(tobs_i[0]) < 0.0:
            raise ValueError(f"dataset {ds}: t_obs must be non-negative; got {float(tobs_i[0])}.")
        if tobs_i.shape[0] > 1 and not bool(jnp.all(jnp.diff(tobs_i) > 0)):
            raise ValueError(f"dataset {ds}: t_obs must be strictly ascending.")
        # Convert this dataset's t_obs into the model's native (rate-constant)
        # time unit, the same way reactor.solve(time_unit=...) does.
        tobs_i = tobs_i * native_time_factor(model.time_unit, time_unit)
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
            (
                C0_i,
                tobs_i,
                (0.0, float(tobs_i[-1])),
                _build_loss(loss, obs_i, sig_arr),
                _build_residual(loss, obs_i, sig_arr),
            )
        )

    # Resolve Gaussian priors for the free parameters. Model-declared priors
    # apply by default (use_priors); the explicit ``priors`` overrides per param.
    active_priors: dict[str, tuple[float, float]] = {}
    if use_priors:
        net_priors = getattr(model, "parameter_priors", {})
        for name in free_params:
            if name in net_priors:
                active_priors[name] = net_priors[name]
    if priors:
        for name, ms in priors.items():
            if name in free_params:
                active_priors[name] = (float(ms[0]), float(ms[1]))
    prior_mean = jnp.asarray([active_priors.get(n, (0.0, 1.0))[0] for n in free_params])
    prior_std = jnp.asarray([active_priors.get(n, (0.0, 1.0))[1] for n in free_params])
    prior_mask = jnp.asarray([1.0 if n in active_priors else 0.0 for n in free_params])
    has_priors = bool(active_priors)
    n_rate = len(free_params)

    # --- Free initial conditions (optional) ----------------------------
    free_ic_list = list(free_ic or [])
    for s in free_ic_list:
        if s not in model.species_index:
            raise KeyError(f"Unknown free_ic species '{s}'. Available: {model.species}")
    m_ic = len(free_ic_list)
    if m_ic and not (0.0 < ic_bounds[0] < ic_bounds[1]):
        raise ValueError(f"ic_bounds must satisfy 0 < lo < hi; got {ic_bounds}.")
    ic_species_idx = jnp.asarray([model.species_index[s] for s in free_ic_list], dtype=int)
    ic_center_blocks = []
    if m_ic:
        ic_np_idx = [model.species_index[s] for s in free_ic_list]
        for C0_i, *_rest in datasets:
            vals = np.clip(np.asarray(C0_i)[ic_np_idx], ic_bounds[0], ic_bounds[1])
            ic_center_blocks.append(np.log(vals))
    ic_center_full = jnp.asarray(np.concatenate(ic_center_blocks)) if m_ic else jnp.zeros(0)

    C0_base = tuple(C0_i for (C0_i, *_rest) in datasets)
    dataset_static = [
        (tobs_i, tspan_i, loss_fn_i, resid_fn_i)
        for (_C0, tobs_i, tspan_i, loss_fn_i, resid_fn_i) in datasets
    ]

    return _CalibrationProblem(
        model=model,
        free_params=list(free_params),
        free_indices=free_indices,
        transforms=resolved_transforms,
        n_rate=n_rate,
        p0_full=p0_full,
        param_halfwidth=param_halfwidth,
        datasets=datasets,
        dataset_static=dataset_static,
        C0_base=C0_base,
        n_datasets=n_datasets,
        obs_species_indices=obs_species_indices,
        n_observed=n_observed,
        active_priors=active_priors,
        prior_mean=prior_mean,
        prior_std=prior_std,
        prior_mask=prior_mask,
        has_priors=has_priors,
        free_ic=free_ic_list,
        m_ic=m_ic,
        ic_species_idx=ic_species_idx,
        ic_center_full=ic_center_full,
        ic_prior_log_std=ic_prior_log_std,
        ic_bounds=ic_bounds,
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
    sigma : jnp.ndarray, optional
        Per-observation standard deviation for ``"wmse"`` / ``"nll"``.
        Scalar, ``(n_observed,)``, or ``(n_t, n_observed)``. In multi-batch
        mode, either a single value/array shared across datasets or a list with
        one entry per dataset.
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
        ``jitter`` / ``jitter_schedule``, ``seed``, ``max_iter``, ``tol``,
        ``param_halfwidth``). ``"gauss_newton"`` minimises the residual *vector*
        with SciPy ``least_squares`` (its Jacobian by AD, direction from
        ``diff.mode``) and is markedly more robust on the multimodal landscapes
        of stiff reaction-model fits; ``n_starts > 1`` is a deterministic,
        ``seed``-reproducible multistart that keeps the lowest-loss optimum.
        Default ``OptimizerConfig()`` (a single L-BFGS-B start).
    laplace : bool or LaplaceConfig, optional
        Laplace covariance approximation at the MAP. ``True`` (default) computes
        it with default settings, ``False`` disables it, and a
        :class:`LaplaceConfig` computes it with tuning (``method`` {"fd",
        "gauss_newton"}, ``ridge``, ``eig_keep``, ``fd_step``, ``dtmax``). It is
        interpretable as a Bayesian posterior only when ``loss="nll"`` with a
        calibrated ``sigma`` (the loss then IS a proper Gaussian negative
        log-likelihood); for ``"mse"`` / ``"wmse"`` it is the inverse loss
        curvature -- the right shape but not the right absolute scale. See
        :func:`fit` if you only need point estimates.
    free_ic : FreeICConfig, optional
        If given, also fit the initial concentration of the named
        ``species`` (per dataset, in log space, box-bounded by
        :attr:`FreeICConfig.bounds`, with an optional log-space prior via
        ``prior_log_std``) while the rate parameters stay shared -- useful when
        an unmeasured initial state (e.g. a biofilm reservoir) is not known. The
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
        instead of saving the ``~n_stages``x dense-stage buffer -- memory for
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
    if optimizer.n_starts < 1:
        raise ValueError(f"optimizer.n_starts must be >= 1; got {optimizer.n_starts}.")
    if optimizer.method not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"optimizer.method must be one of {_VALID_OPTIMIZERS}; got {optimizer.method!r}."
        )
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

    # Unpack the config objects into the internal scalar knobs.
    laplace_on, lap = _resolve_laplace(laplace)
    ic_species, ic_bounds, ic_prior_log_std = _free_ic_fields(free_ic)

    # Resolve the arguments once into the static problem + the fit config, then
    # build the compiled objective on the reactor forward-model seam.
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
    )
    cfg = _FitConfig(
        gradient=gradient,
        ad_mode=ad_mode,
        check_finite=diff.check_finite,
        stable_adjoint_max_steps=int(diff.adjoint_max_steps),
        stable_adjoint_low_memory=bool(diff.adjoint_low_memory),
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
        laplace_eig_keep=lap.eig_keep,
        laplace_fd_step=lap.fd_step,
        laplace_dtmax=lap.dtmax,
        compiled_cache=_compiled_cache,
    )
    fm = _ReactorForwardModel(
        reactor, gradient, cfg.stable_adjoint_max_steps, cfg.stable_adjoint_low_memory
    )
    bundle = _build_objective(problem, fm, cfg)

    # --- Run the optimiser in unconstrained space ----------------------
    rate_theta0 = problem.rate_theta0()
    theta0 = jnp.concatenate([rate_theta0, problem.ic_center_full]) if problem.m_ic else rate_theta0
    opt_bounds = _optimizer_bounds(problem, rate_theta0)

    if cfg.check_finite:
        _check_start_gradient(cfg, bundle, theta0)

    result = _run_multistart(cfg, bundle, theta0, opt_bounds)

    theta_opt = jnp.asarray(result.x)
    rate_theta_opt = theta_opt[: problem.n_rate]
    ic_opt = theta_opt[problem.n_rate :]
    physical_opt = problem.physical_from_theta(rate_theta_opt)
    full_params = problem.p0_full.at[problem.free_indices].set(physical_opt)

    # Fitted per-dataset initial states (when free_ic is active).
    C0_fitted = None
    ic_named = None
    if problem.m_ic:
        C0_fitted = []
        ic_named = []
        for k, (C0_i, *_rest) in enumerate(problem.datasets):
            vals = np.exp(np.asarray(ic_opt)[k * problem.m_ic : (k + 1) * problem.m_ic])
            C0_fitted.append(C0_i.at[problem.ic_species_idx].set(jnp.asarray(vals)))
            ic_named.append({s: float(v) for s, v in zip(problem.free_ic, vals)})

    # --- Laplace posterior (over the rate parameters; pools held at MAP) ---
    posterior_cov = None
    posterior_std_unconstrained = None
    params_named_std = None
    hessian_unconstrained = None
    if cfg.laplace:
        (
            posterior_cov,
            posterior_std_unconstrained,
            params_named_std,
            hessian_unconstrained,
        ) = _laplace_posterior(problem, fm, cfg, rate_theta_opt, ic_opt)

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
        posterior_cov=posterior_cov,
        posterior_std_unconstrained=posterior_std_unconstrained,
        params_named_std=params_named_std,
        hessian_unconstrained=hessian_unconstrained,
        priors_applied=dict(problem.active_priors),
        C0_fitted=C0_fitted,
        ic_named=ic_named,
    )
