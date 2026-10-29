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

import hashlib
import inspect
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
from aquakin.integrate.identifiability import _NULL_LOADING_TOL, _NULL_RTOL

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
    starts : sequence, optional
        An explicit list of optimiser starts, in physical units, in place of
        the single start from ``initial_params``. Each entry is either a full
        parameter vector (``(n_params,)``; only the free entries are read) or a
        ``{name: value}`` dict over free parameters, overlaid on
        ``initial_params`` (parameters a dict leaves out start from there).
        Every start is run and the lowest-loss optimum is kept. All starts go
        through the one compiled objective, so a grid of starts costs one
        compilation, not one per start. Fitted initial pools start from ``C0``
        for every start. A start outside ``bounds`` is moved to the nearest
        bound with a ``RuntimeWarning``, as ``initial_params`` is. Cannot be
        combined with ``n_starts > 1``.
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
    bounds : dict[str, tuple[float | None, float | None]], optional
        Absolute bounds on free rate parameters in physical units, as
        ``{name: (lo, hi)}`` (``None`` on a side leaves it open). An entry
        replaces the parameter's model-declared ``bounds:`` for this fit.
        Entries for parameters that are not in ``free_params`` are ignored, so
        one mapping can be reused across fits (for example a
        :func:`~aquakin.profile_likelihood` sweep). Like the declared bounds,
        each is mapped into the unconstrained space the optimiser works in
        (``log`` for ``positive_log``, ``logit`` for ``logit``), so a bound at
        or beyond the edge of the transform's domain (``lo <= 0`` for
        ``positive_log``; ``lo <= 0`` or ``hi >= 1`` for ``logit``) is open.
    use_model_bounds : bool
        If ``True`` (default), each free rate parameter is kept inside the
        ``bounds:`` declared on it in the model, as :func:`aquakin.fit` does.
        The bounds are absolute: every multistart start searches the same
        region. Set ``False`` to ignore the declared bounds (the behaviour
        before 0.2.0); explicit ``bounds`` still apply.
    param_halfwidth : float, optional
        If given, also box-bound each free rate parameter to ``theta0 +/-
        param_halfwidth`` in unconstrained space around the starting value
        (``initial_params``, moved inside the absolute bounds if needed). The
        box is intersected with the absolute bounds above. ``None`` (default)
        adds no such box.
    """

    method: str = "lbfgsb"
    n_starts: int = 1
    starts: tuple | list | None = None
    jitter: float = 0.5
    jitter_schedule: tuple | None = None
    seed: int = 0
    max_iter: int = 500
    tol: float = 1e-6
    param_halfwidth: float | None = None
    bounds: dict | None = None
    use_model_bounds: bool = True


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
    eig_keep : float, optional
        Deprecated and ignored. The covariance is now the full inverse of the
        Hessian, so a weakly identified direction keeps its (large) variance
        instead of being dropped with zero variance, which understated
        ``params_named_std``. A parameter that the data do not identify at all
        (an exactly null Hessian direction) gets an infinite std and a
        ``RuntimeWarning``. Use :func:`aquakin.identifiability` to flag weakly
        identified directions. Passing a value emits a ``DeprecationWarning``.
    fd_step : float
        Relative step for the finite-difference Hessian (``method="fd"``).
    dtmax : float, optional
        If given, rebuild the reactor with this ``dtmax`` cap for the Hessian
        pass only -- a tighter solve for the second-order-sensitive covariance.
    include_ic : bool
        If ``True`` and initial pools are fitted (``free_ic``), form the
        Hessian over the rate parameters *and* the fitted pools jointly. The
        joint covariance is returned as ``CalibrationResult.joint_cov`` and the
        rate ``posterior_cov`` / ``params_named_std`` become its rate block, the
        marginal over the pools. This accounts for rate-pool correlation (e.g.
        ``k`` against ``[X]0`` when only their product is seen in the data),
        which the default ``False`` -- pools held fixed at their MAP -- ignores,
        understating the rate uncertainty. Ignored when no pools are fitted.
    """

    method: str = "fd"
    ridge: float = 1e-6
    eig_keep: float | None = None
    fd_step: float = 1e-3
    dtmax: float | None = None
    include_ic: bool = False

    def __post_init__(self):
        if self.eig_keep is not None:
            warnings.warn(
                "LaplaceConfig(eig_keep=...) is deprecated and ignored; the Laplace "
                "covariance is the full inverse of the Hessian, with an infinite "
                "std for a parameter on an exactly null direction. Use "
                "aquakin.identifiability(..., eig_keep=...) to flag weak directions.",
                DeprecationWarning,
                stacklevel=3,
            )


@dataclass(frozen=True)
class FreeICConfig:
    """Free-initial-condition settings for calibration.

    Parameters
    ----------
    species : list of str
        Species whose initial concentration is fitted alongside the parameters.
        For a plant, ``"unit.species"`` names.
    bounds : tuple of float, or dict
        Box bounds on the fitted initial pools, in physical units (the fit runs
        in log space). Either one ``(lo, hi)`` pair shared by every pool, or a
        ``{species: (lo, hi)}`` dict for pools of very different size; a
        species the dict leaves out gets the default ``(1e-3, 1e4)``. Each
        pair must satisfy ``0 < lo < hi``. For a plant the dict keys are
        ``"unit.species"`` names. Not used with ``fixed_total``.
    prior_log_std : float, optional
        If given, a Gaussian prior of this log-space std pulls each fitted pool
        toward its supplied initial value; ``None`` leaves it governed only by
        the data and ``bounds``. With ``fixed_total`` it acts on the log-ratio
        coordinates of the split instead (see below).
    fixed_total : bool
        If ``True``, hold the sum of the fitted pools at its value in each
        dataset's ``C0`` and fit only how it is split between them: for a
        known total COD ``S_S0 + X_S0``, say, put the total into ``C0`` with
        any split as the starting guess. The split is fitted on the simplex,
        through the ``m - 1`` log-ratio coordinates
        ``z_j = log(c_j / c_0)`` (``c_0`` is the first listed species), so the
        pools stay positive and sum to the total exactly. The fitted split is
        returned as ``CalibrationResult.ic_fractions`` and, with
        ``LaplaceConfig(include_ic=True)``, its standard deviation as
        ``ic_fractions_std``. In ``joint_names`` / ``joint_cov`` and in
        :func:`aquakin.fisher_information` the coordinates are named
        ``"ic[k].<c_j>:<c_0>"`` (the ratio ``c_j / c_0``, in log space).
        Every listed pool must be positive in ``C0`` (for a plant, in ``y0``).
        Default ``False``.
    """

    species: list
    bounds: tuple | dict = (1e-3, 1e4)
    prior_log_std: float | None = None
    fixed_total: bool = False


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


_DEFAULT_IC_BOUNDS = (1e-3, 1e4)


def _free_ic_fields(free_ic) -> tuple[list | None, tuple | dict, float | None]:
    """Unpack an optional :class:`FreeICConfig` into ``(species, bounds,
    prior_log_std)`` for the internal problem resolver."""
    if free_ic is None:
        return None, _DEFAULT_IC_BOUNDS, None
    if isinstance(free_ic, FreeICConfig):
        return free_ic.species, free_ic.bounds, free_ic.prior_log_std
    raise TypeError(f"free_ic must be a FreeICConfig or None; got {type(free_ic).__name__}.")


def _per_species_ic_bounds(species, bounds) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` arrays, one entry per fitted pool, from a shared ``(lo, hi)``
    pair or a ``{species: (lo, hi)}`` dict (missing species get the default).
    Each pair must satisfy ``0 < lo < hi``."""
    species = list(species)
    if isinstance(bounds, dict):
        unknown = [s for s in bounds if s not in species]
        if unknown:
            raise KeyError(
                f"FreeICConfig.bounds names species that are not fitted: {unknown}; "
                f"fitted species are {species}."
            )
        pairs = [bounds.get(s, _DEFAULT_IC_BOUNDS) for s in species]
    else:
        pairs = [bounds] * len(species)
    lo, hi = [], []
    for s, pair in zip(species, pairs):
        if np.ndim(pair) != 1 or len(pair) != 2:
            raise ValueError(f"ic_bounds must be a (lo, hi) pair; got {pair!r} for '{s}'.")
        if not (0.0 < float(pair[0]) < float(pair[1])):
            raise ValueError(f"ic_bounds must satisfy 0 < lo < hi; got {pair} for '{s}'.")
        lo.append(float(pair[0]))
        hi.append(float(pair[1]))
    return np.asarray(lo), np.asarray(hi)


def _resolve_ic_coords(species, ic_bounds, fixed_total, pools_per_dataset):
    """The free-IC coordinate setup shared by the reactor and plant resolvers.

    ``species`` names the fitted pools and ``pools_per_dataset`` holds their
    starting values in each dataset's initial state. Returns ``(ic_bounds,
    ic_center_full, fixed_total)``: the per-pool ``(lo, hi)`` arrays (the
    bounds as given when nothing is fitted or for a split, which uses none),
    the unconstrained start of every pool coordinate (log pools, or the
    log-ratios of a ``fixed_total`` split, which is what the ic prior pulls
    toward), and ``fixed_total`` switched off when nothing is fitted."""
    species = list(species)
    if len(set(species)) != len(species):
        raise ValueError(f"free_ic species must be distinct; got {species}.")
    m_ic = len(species)
    fixed_total = bool(fixed_total and m_ic)
    if not m_ic:
        return ic_bounds, jnp.zeros(0), False
    if fixed_total:
        if isinstance(ic_bounds, dict) or tuple(ic_bounds) != _DEFAULT_IC_BOUNDS:
            raise ValueError(
                "FreeICConfig(fixed_total=True) fits the split of a fixed total on the "
                "simplex, so it takes no bounds; leave bounds at its default."
            )
    else:
        ic_bounds = _per_species_ic_bounds(species, ic_bounds)
    blocks = []
    for ds, pools in enumerate(pools_per_dataset):
        pools = np.asarray(pools, dtype=float)
        if fixed_total:
            if np.any(pools <= 0.0):
                raise ValueError(
                    f"dataset {ds}: with FreeICConfig(fixed_total=True) every pool in "
                    f"{species} must be positive in C0 (their sum is the fixed "
                    f"total, their split the starting guess); got {pools.tolist()}."
                )
            blocks.append(np.log(pools[1:]) - np.log(pools[0]))
        else:
            blocks.append(np.log(np.clip(pools, ic_bounds[0], ic_bounds[1])))
    return ic_bounds, jnp.asarray(np.concatenate(blocks)), fixed_total


def _simplex_pools(z, total):
    """Pools ``total * softmax([0, z])``: positive, summing to ``total``, from the
    ``m - 1`` log-ratio coordinates ``z_j = log(c_j / c_0)``."""
    logits = jnp.concatenate([jnp.zeros(1, dtype=jnp.result_type(z, float)), z])
    return total * jax.nn.softmax(logits)


def _simplex_fraction_cov(z, cov_z) -> np.ndarray:
    """Delta-method covariance of the fractions ``softmax([0, z])`` from the
    covariance of the log-ratio coordinates ``z``."""
    f = np.asarray(jax.nn.softmax(jnp.concatenate([jnp.zeros(1), jnp.asarray(z)])))
    J = (np.diag(f) - np.outer(f, f))[:, 1:]  # d f / d z
    return J @ np.asarray(cov_z) @ J.T


def _to_unconstrained(value: jnp.ndarray, transform: str) -> jnp.ndarray:
    return to_unconstrained(value, transform)


def _from_unconstrained(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    return from_unconstrained(theta, transform)


def _jacobian_physical_wrt_theta(theta: jnp.ndarray, transform: str) -> jnp.ndarray:
    """``dp/dtheta`` at the given ``theta``, used for the delta-method std."""
    return dphysical_dunconstrained(from_unconstrained(theta, transform), transform)


# --- Loss factory ------------------------------------------------------


# Floor on a per-output sum of squared residuals in the concentrated likelihood,
# so an exact fit (SSE = 0, synthetic noise-free data) gives a finite objective.
_SSE_FLOOR = 1e-300
# Outer iterations of the Gauss-Newton noise re-estimation (``sigma="estimate"``);
# each is a warm-started least-squares solve, and the noise level typically
# settles in a handful.
_IRLS_MAX_ITER = 50


def _build_loss(
    loss_type: str,
    observations: jnp.ndarray,
    sigma: jnp.ndarray | None,
):
    """Return ``loss(pred) -> scalar``.

    ``pred`` and ``observations`` have shape ``(n_t, n_observed)``. ``NaN``
    entries of ``observations`` are missing: they are masked out of the loss
    (``mse`` / ``wmse`` average over the observed entries only). For the
    concentrated likelihood (``loss_type="nll"`` with ``sigma="estimate"``) it
    returns the per-output sum of squared residuals over the observed entries,
    shape ``(n_observed,)``, which :func:`_data_term` sums over the datasets
    before taking the log.
    """
    missing = _missing_mask(observations)
    if _is_sigma_estimate(sigma):
        if loss_type != "nll":
            raise ValueError('sigma="estimate" requires loss="nll".')
        _obs, gap = _fill_missing(observations, missing)

        def _sse(pred):
            return jnp.sum(gap(pred) ** 2, axis=0)

        return _sse
    if missing is not None:
        return _build_masked_loss(loss_type, observations, sigma, missing)
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
    missing = _missing_mask(observations)
    if _is_sigma_estimate(sigma):
        if loss_type != "nll":
            raise ValueError('sigma="estimate" requires loss="nll".')
        # Raw (n_t, n_observed) residuals, zero at missing entries: the caller
        # divides each column by the current noise estimate.
        _obs, gap = _fill_missing(observations, missing)
        return gap
    if missing is not None:
        return _build_masked_residual(loss_type, observations, sigma, missing)
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


def _missing_mask(observations):
    """Boolean ``NaN`` mask of ``observations`` (missing entries), or ``None``
    when every entry is present (the unmasked loss is then used unchanged)."""
    obs = np.asarray(observations)
    if not np.issubdtype(obs.dtype, np.floating):
        return None
    nan = np.isnan(obs)
    return jnp.asarray(nan) if nan.any() else None


def _fill_missing(observations, missing):
    """``(obs_filled, gap)``: the observations with missing entries set to 0,
    and ``gap(pred)``, the difference ``pred - observations`` with exact zeros
    at the missing entries.

    A missing entry must add nothing to the loss *or* to its gradient, even
    when the prediction there is ``NaN`` or ``inf`` (an ``observation_fn`` such
    as ``log(C)`` evaluated where ``C = 0``). Multiplying by a 0/1 weight does
    not achieve that (``0 * NaN`` is ``NaN``), and neither does masking the
    output alone, since the derivative of the unselected branch of
    ``jnp.where`` still enters the gradient as ``0 * NaN``. So the prediction
    itself is replaced by the filled observation at a missing entry before the
    difference is formed: neither branch is then non-finite, and the
    derivative with respect to ``pred`` there is exactly 0. With no missing
    entries, the observations are returned unchanged and ``gap`` is the plain
    difference.
    """
    if missing is None:
        return observations, lambda pred: pred - observations
    obs_filled = jnp.where(missing, 0.0, observations)

    def _gap(pred):
        safe = jnp.where(missing, obs_filled, pred)
        return jnp.where(missing, 0.0, safe - obs_filled)

    return obs_filled, _gap


def _masked_parts(loss_type, observations, sigma, missing):
    """``(gap, sig, n_kept)`` for a loss with missing entries: the masked
    difference of :func:`_fill_missing`, ``sigma`` broadcast to the data shape
    with 1 at the missing entries (so a ``NaN``, zero or infinite noise level
    given for an entry that was not measured cannot reach the loss), and the
    number of observed entries."""
    if loss_type in ("wmse", "nll") and sigma is None:
        raise ValueError(f"loss={loss_type!r} requires a sigma argument.")
    if loss_type not in _VALID_LOSSES:
        raise ValueError(f"Unknown loss {loss_type!r}; choose one of {_VALID_LOSSES}.")
    _obs, gap = _fill_missing(observations, missing)
    sig = None
    if sigma is not None:
        sig = jnp.where(missing, 1.0, jnp.broadcast_to(sigma, observations.shape))
    return gap, sig, float(np.sum(~np.asarray(missing)))


def _build_masked_loss(loss_type, observations, sigma, missing):
    """:func:`_build_loss` with the ``NaN`` (missing) entries left out: each
    contributes exactly 0 to the value and to the gradient."""
    gap, sig, n_kept = _masked_parts(loss_type, observations, sigma, missing)
    if loss_type == "mse":
        return lambda pred: jnp.sum(gap(pred) ** 2) / n_kept
    if loss_type == "wmse":
        return lambda pred: jnp.sum((gap(pred) / sig) ** 2) / n_kept
    # sig is 1 at a missing entry, so its log-normalizer term is 0 there.
    return lambda pred: jnp.sum(jnp.log(sig) + gap(pred) ** 2 / (2.0 * sig**2))


def _build_masked_residual(loss_type, observations, sigma, missing):
    """:func:`_build_residual` with zero residuals at the missing entries (so
    they add nothing to the Gauss-Newton / Fisher Hessian)."""
    gap, sig, n_kept = _masked_parts(loss_type, observations, sigma, missing)
    if loss_type == "mse":
        scale = jnp.sqrt(2.0 / n_kept)
        return lambda pred: (scale * gap(pred)).reshape(-1)
    if loss_type == "wmse":
        scale = jnp.sqrt(2.0 / n_kept)
        return lambda pred: (scale * gap(pred) / sig).reshape(-1)
    return lambda pred: (gap(pred) / sig).reshape(-1)


def _is_sigma_estimate(sigma) -> bool:
    """Whether ``sigma`` asks for the concentrated likelihood."""
    return isinstance(sigma, str) and sigma == "estimate"


def _check_sigma_arg(sigma, loss) -> None:
    """Validate a string ``sigma``: only ``"estimate"``, and only with
    ``loss="nll"`` (the concentrated likelihood)."""
    if isinstance(sigma, str):
        if not _is_sigma_estimate(sigma):
            raise ValueError(f'sigma must be an array, None or "estimate"; got {sigma!r}.')
        if loss != "nll":
            raise ValueError(
                'sigma="estimate" (the concentrated likelihood) requires loss="nll"; '
                f"got loss={loss!r}."
            )


def _sigma_per_dataset(sigma, n_datasets: int) -> list:
    """One ``sigma`` entry per dataset: a list is checked against the dataset
    count, anything else is broadcast."""
    if not isinstance(sigma, (list, tuple)):
        return [sigma] * n_datasets
    sigma_list = list(sigma)
    if len(sigma_list) != n_datasets:
        raise ValueError(
            f"sigma list has {len(sigma_list)} entries but there are {n_datasets} datasets."
        )
    if any(isinstance(s, str) for s in sigma_list):
        # The concentrated likelihood pools the residuals of every dataset,
        # so it cannot be switched on for some datasets only.
        raise ValueError(
            'sigma="estimate" applies to every dataset; pass it as sigma="estimate", '
            "not as an entry of a per-dataset sigma list."
        )
    return sigma_list


def _coerce_sigma(sig_i):
    """One dataset's ``sigma`` as the loss builders take it: ``"estimate"``,
    an array, or ``None``."""
    if _is_sigma_estimate(sig_i):
        return "estimate"
    if isinstance(sig_i, str):
        raise ValueError(f'sigma must be an array, None or "estimate"; got {sig_i!r}.')
    return jnp.asarray(sig_i) if sig_i is not None else None


def _observed_counts(ds: int, obs_i) -> np.ndarray:
    """Observed (non-NaN) entries per output of one dataset's observations,
    shape ``(n_t, n_out)``. An ``inf`` or an all-missing dataset is rejected."""
    obs_np = np.asarray(obs_i, dtype=float)
    if np.isinf(obs_np).any():
        raise ValueError(f"dataset {ds}: observations contain inf; mark a missing value with NaN.")
    if np.isnan(obs_np).all():
        raise ValueError(f"dataset {ds}: every observation is missing (NaN).")
    return np.sum(~np.isnan(obs_np), axis=0)


def _n_obs_per_output(sigma_estimate: bool, n_seen_per_output):
    """The observation count of each output for the concentrated likelihood
    (``None`` when the noise is not estimated). Every output needs at least
    one observation to estimate its noise."""
    if not sigma_estimate:
        return None
    unseen = np.flatnonzero(np.asarray(n_seen_per_output) == 0)
    if unseen.size:
        raise ValueError(
            f'sigma="estimate" needs at least one observation of every output; '
            f"output(s) {unseen.tolist()} are missing (NaN) in every dataset."
        )
    return jnp.asarray(n_seen_per_output, dtype=float)


def _data_term(problem, preds):
    """The summed data term of the objective over the datasets.

    With a known (or no) ``sigma`` it is the sum of the per-dataset losses. For
    the concentrated likelihood (``sigma="estimate"``) the per-output sums of
    squares are pooled over the datasets and the noise level of output ``j`` is
    profiled out at its maximum-likelihood value ``sigma_j^2 = SSE_j / N_j``,
    which leaves ``sum_j (N_j / 2) log(SSE_j / N_j) + N / 2``: the ``"nll"``
    loss evaluated at the estimated noise.
    """
    if not problem.sigma_estimate:
        total = 0.0
        for (_t, _ts, loss_fn_i, _r), pred in zip(problem.dataset_static, preds):
            total = total + loss_fn_i(pred)
        return total
    sse = _pooled_sse(problem, preds)
    n = problem.n_obs_per_output
    return 0.5 * jnp.sum(n * jnp.log(sse / n)) + 0.5 * jnp.sum(n)


def _pooled_sse(problem, preds):
    """Per-output sum of squared residuals over all datasets (floored)."""
    sse = 0.0
    for (_t, _ts, sse_fn_i, _r), pred in zip(problem.dataset_static, preds):
        sse = sse + sse_fn_i(pred)
    return jnp.maximum(sse, _SSE_FLOOR)


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
        ``(d, d)`` covariance matrix of the rate parameters in *unconstrained*
        space (Laplace). With ``LaplaceConfig(include_ic=True)`` it is the rate
        block of ``joint_cov`` (marginal over the fitted pools); otherwise it is
        conditional on the pools held at their MAP. ``None`` if
        ``laplace=False`` was passed.
    posterior_std_unconstrained : jnp.ndarray, optional
        Marginal std devs in unconstrained space.
    params_named_std : dict[str, float], optional
        Marginal std devs propagated back to physical space via the
        delta method (``|dp/dtheta| * std_unconstrained``).
    hessian_unconstrained : jnp.ndarray, optional
        Rate-rate block of the loss Hessian at the MAP, unconstrained space:
        finite-difference for ``LaplaceConfig(method="fd")``, Gauss-Newton
        ``J^T J`` (the Fisher information for ``loss="nll"``) for
        ``method="gauss_newton"``.
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
    joint_cov : jnp.ndarray, optional
        With ``LaplaceConfig(include_ic=True)`` and fitted pools, the joint
        covariance of the rate parameters and the fitted pools in unconstrained
        space (pools in log space), ordered as ``joint_names``. ``None``
        otherwise.
    joint_hessian : jnp.ndarray, optional
        The joint Hessian ``joint_cov`` was built from, same ordering.
    joint_names : list[str], optional
        Row/column names of ``joint_cov``: the rate names, then
        ``"ic[k].<species>"`` for each dataset ``k``'s fitted pools.
    ic_named_std : list[dict[str, float]], optional
        Marginal std devs of the fitted pools in physical units, per dataset
        and by species (same layout as ``ic_named``). Only with
        ``include_ic=True``.
    ic_fractions : list[dict[str, float]], optional
        With ``FreeICConfig(fixed_total=True)``, the fitted split of each
        dataset's total, as fractions summing to 1, by species. ``None``
        otherwise.
    ic_fractions_std : list[dict[str, float]], optional
        Std devs of ``ic_fractions`` (delta method from the joint covariance of
        the log-ratio coordinates). Only with ``fixed_total`` and
        ``include_ic=True``.
    at_bound : dict[str, str]
        The fitted quantities that ended on an optimiser bound, mapped to
        ``"lower"`` or ``"upper"``: free rate parameters by name and fitted
        initial pools as ``"ic[k].<species>"``. Empty when nothing is on a
        bound. The bound is an active constraint there, so the local curvature
        does not describe their uncertainty: their ``params_named_std`` /
        ``ic_named_std`` and ``posterior_std_unconstrained`` entries are
        ``nan``, they have zero variance in ``posterior_cov`` / ``joint_cov``
        (posterior draws keep them on the bound), and the other parameters'
        covariance is conditional on them staying on the bound. A
        ``RuntimeWarning`` names them. Use :func:`aquakin.profile_likelihood`
        for a one-sided interval, or revisit the bound.
    sigma_estimate : np.ndarray, optional
        With ``loss="nll", sigma="estimate"``, the estimated noise standard
        deviation of each observed output, ``sqrt(SSE_j / N_j)`` at the MAP
        (pooled over the datasets), shape ``(n_observed,)``. ``None``
        otherwise.
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
    joint_cov: jnp.ndarray | None = None
    joint_hessian: jnp.ndarray | None = None
    joint_names: list[str] | None = None
    ic_named_std: list | None = None
    ic_fractions: list | None = None
    ic_fractions_std: list | None = None
    at_bound: dict[str, str] = field(default_factory=dict)
    sigma_estimate: np.ndarray | None = None

    def posterior_draws(self, model, n_draw: int = 200, *, seed: int = 0) -> np.ndarray:
        """Full parameter vectors drawn from the Laplace posterior.

        Draws the free rate parameters from ``N(MAP, posterior_cov)`` in
        unconstrained space, maps each draw to physical values, and sets it into
        the fitted parameter vector ``params``; fixed parameters keep their
        values. An exactly null (non-identifiable) direction of the Hessian is
        not in ``posterior_cov`` and is not sampled; such parameters have an
        infinite ``params_named_std``. A parameter that ended on an optimiser
        bound (``at_bound``) has zero variance and stays on the bound in every
        draw. Initial pools are not drawn. The
        rows can be passed as ``param_draws`` to the experiment-design
        functions, and :meth:`predictive_band` propagates the same draws.

        Parameters
        ----------
        model : CompiledModel
            The calibrated model (``reactor.model``), to place the free
            parameters in the full vector.
        n_draw : int
            Number of draws.
        seed : int
            Seed for the draws (reproducible).

        Returns
        -------
        np.ndarray
            Shape ``(n_draw, n_params)``.

        Raises
        ------
        ValueError
            If there is no Laplace posterior, or it has no positive-variance
            direction.
        """
        if self.posterior_cov is None:
            raise ValueError(
                "posterior_draws requires a Laplace posterior; call calibrate(..., laplace=True)."
            )
        names = self.parameter_names
        free_idx = np.asarray([model.param_index[n] for n in names])
        theta_map = np.array(
            [
                float(_to_unconstrained(jnp.asarray(self.params_named[n]), t))
                for n, t in zip(names, self.transforms)
            ]
        )

        # Draw from N(theta_map, posterior_cov), the covariance that also backs
        # params_named_std. Eigen-decompose to sample; an exactly null Hessian
        # direction has zero variance in posterior_cov and is not sampled (a
        # tiny relative floor drops it and any round-off negatives).
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
        thetas = (
            theta_map[None, :]
            + (rng.standard_normal((n_draw, int(pos.sum()))) * std_k[None, :]) @ Q[:, pos].T
        )
        out = np.tile(np.asarray(self.params, dtype=float), (n_draw, 1))
        for i, t in enumerate(self.transforms):
            out[:, free_idx[i]] = np.asarray(_from_unconstrained(jnp.asarray(thetas[:, i]), t))
        return out

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

        Samples the calibrated parameters from the Laplace posterior with
        :meth:`posterior_draws`, integrates ``reactor``
        from ``C0`` over ``t_eval``, and returns the pointwise percentile
        envelope across draws. The draws are taken from ``self.posterior_cov``,
        the covariance behind ``params_named_std``. A parameter on an exactly
        null Hessian direction (infinite std) is not sampled, so the draws stay
        finite.

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
            Deprecated and ignored. The band uses ``self.posterior_cov``, the
            covariance behind ``params_named_std``. Passing a value here emits
            a ``DeprecationWarning``.
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
                "predictive_band(eig_keep=...) is deprecated and ignored; the band "
                "is drawn from the calibration's posterior_cov.",
                DeprecationWarning,
                stacklevel=2,
            )
        model = reactor.model
        draws = self.posterior_draws(model, n_draw, seed=seed)
        C0_j = jnp.asarray(C0)
        t_eval_j = jnp.asarray(t_eval)
        t_end = float(np.asarray(t_eval)[-1])
        curves = []
        first_error = None
        for p in draws:
            p = jnp.asarray(p)
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
                "region; check the fit and params_named_std for poorly identified "
                "parameters."
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
    # Optional observation function (C, params) -> y, or (t, C, params) -> y;
    # replaces the species columns ``obs_species_indices`` when set.
    observation_fn: object = None
    # Per-dataset observation times exactly as given, in ``t_obs``'s unit
    # (``dataset_static`` holds them converted to the model's native unit): a
    # time-aware ``observation_fn`` receives these, so ``t == t_obs[i]`` holds
    # exactly whatever the unit conversion. Empty when not applicable.
    obs_times: tuple = ()
    # Concentrated likelihood (loss="nll", sigma="estimate"): the per-dataset
    # loss returns per-output sums of squares, the residual raw residuals, and
    # ``n_obs_per_output`` holds the observation count of each output.
    sigma_estimate: bool = False
    n_obs_per_output: object = None
    # Fingerprint of what the per-dataset loss / residual closures bake in (the
    # loss type, observations, sigma and observation times in both the native
    # unit and ``t_obs``'s unit), so a shared
    # ``_compiled_cache`` cannot reuse an objective built for other data.
    data_key: tuple = ()
    # Absolute bounds on the rate block in unconstrained space, as ``(lb, ub)``
    # arrays of length ``n_rate`` (+/-inf where open), or ``None`` for none.
    # Only the optimiser sees them (not the compiled objective), so they are
    # not part of :meth:`struct_key`.
    rate_bounds: tuple | None = None
    # FreeICConfig(fixed_total=True): the pools of each dataset sum to their
    # total in its C0, fitted as m_ic - 1 log-ratio coordinates per dataset.
    ic_fixed_total: bool = False

    @property
    def m_ic_theta(self) -> int:
        """Number of fitted initial-pool coordinates per dataset."""
        if not self.m_ic:
            return 0
        return self.m_ic - 1 if self.ic_fixed_total else self.m_ic

    def ic_pools(self, ic_thetas_k, C0_k):
        """One dataset's fitted pool values from its coordinates: ``exp`` of the
        log-space pools, or (``ic_fixed_total``) the split of the total of the
        pools in ``C0_k`` given by the log-ratio coordinates."""
        if self.ic_fixed_total:
            total = jnp.sum(C0_k[self.ic_species_idx])
            return _simplex_pools(ic_thetas_k, total)
        return jnp.exp(ic_thetas_k)

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
        """Cache key for the compiled objective: everything the compiled program
        bakes in as a constant. The per-call data threaded in as *arguments*
        (``p0_full``, the initial states, the ic-prior centre; see :attr:`data`)
        are left out, so a ``profile_likelihood`` sweep, which changes only
        those, reuses one program. Everything else is in the key: the shapes,
        the gradient path, the free set and transforms, the observed columns /
        ``observation_fn``, the priors, the free-IC set, and a fingerprint of
        the loss type, observations, ``sigma`` and observation times
        (``data_key``). A ``_compiled_cache`` shared across fits on different
        data therefore rebuilds instead of silently reusing the wrong
        objective. The forward model (reactor) is not in the key: a cache must
        not be shared across different reactors."""
        return (
            self.n_rate,
            self.m_ic,
            self.ic_fixed_total,
            len(self.dataset_static),
            gradient,
            self.n_observed,
            tuple(int(ds[0].shape[0]) for ds in self.dataset_static),
            self.observation_fn,
            self.sigma_estimate,
            tuple(self.free_params),
            tuple(self.transforms),
            _array_digest(self.free_indices, self.obs_species_indices, self.ic_species_idx),
            _array_digest(self.prior_mean, self.prior_std, self.prior_mask)
            if self.has_priors
            else None,
            self.ic_prior_log_std,
            self.data_key,
        )


def _array_digest(*arrays) -> str:
    """A short content hash of some arrays (``None`` entries allowed), for the
    compiled-objective cache key."""
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b"None|")
            continue
        arr = np.ascontiguousarray(np.asarray(a))
        h.update(f"{arr.dtype}{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


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
            mt = problem.m_ic_theta
            C0_k = C0_i.at[problem.ic_species_idx].set(
                problem.ic_pools(ic_thetas[k * mt : (k + 1) * mt], C0_i)
            )
        ys = fm.solve_trajectory(p, C0_k, tspan_i, tobs_i)
        t_user = problem.obs_times[k] if problem.obs_times else None
        preds.append(observe(ys, p, problem.obs_species_indices, problem.observation_fn, t=t_user))
    return preds


def observation_takes_time(observation_fn) -> bool:
    """Whether ``observation_fn`` has the time-aware signature
    ``(t, C, params)`` rather than ``(C, params)``.

    The rule, on the positional parameters of the signature (those that can be
    filled by position), is:

    - exactly three of them *required* (no default): the time-aware form
      ``observation_fn(t, C, params)``. A time-aware function must therefore
      require all of ``t``, ``C`` and ``params``; further parameters may have
      defaults.
    - at most two required and at least two in all: the ``(C, params)`` form,
      called as ``observation_fn(C, params)`` (so ``f(C, p=None)`` and
      ``f(C, p, scale=2.0)`` are this form).
    - a ``*args`` parameter, or a signature that cannot be inspected: the
      ``(C, params)`` form.

    Raises
    ------
    TypeError
        If ``observation_fn`` is not callable, or its signature fits neither
        form (fewer than two positional parameters, or more than three
        required).
    """
    if not callable(observation_fn):
        raise TypeError(
            "observation_fn must be a callable (C, params) -> y or (t, C, params) -> y."
        )
    try:
        sig = inspect.signature(observation_fn)
    except (TypeError, ValueError):
        return False  # no introspectable signature: the (C, params) form
    params = list(sig.parameters.values())
    if any(q.kind is inspect.Parameter.VAR_POSITIONAL for q in params):
        return False
    positional = [
        q
        for q in params
        if q.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    n_req = sum(1 for q in positional if q.default is inspect.Parameter.empty)
    if n_req == 3:
        return True
    if n_req <= 2 and len(positional) >= 2:
        return False
    raise TypeError(
        "observation_fn must take (C, params), or require (t, C, params), as "
        f"positional arguments; got a signature {sig}."
    )


def observe(ys, p, species_idx, observation_fn=None, t=None):
    """Observed outputs of a ``(n_t, n_species)`` trajectory: the
    ``species_idx`` columns, or ``observation_fn`` applied at each time --
    ``observation_fn(C, p)``, or ``observation_fn(t, C, p)`` with the sample
    times ``t`` ``(n_t,)`` for a time-aware function."""
    if observation_fn is None:
        return ys[:, species_idx]
    if observation_takes_time(observation_fn):
        if t is None:
            raise ValueError("A time-aware observation_fn(t, C, params) needs the sample times.")
        return jax.vmap(lambda tt, c: jnp.atleast_1d(observation_fn(tt, c, p)))(
            jnp.asarray(t, dtype=float), ys
        )
    return jax.vmap(lambda c: jnp.atleast_1d(observation_fn(c, p)))(ys)


def observation_size(observation_fn, C0, p, t0: float = 0.0) -> int:
    """Number of outputs of ``observation_fn``, checked to be a 1-D vector
    (or a scalar) on the model's state and parameter vectors (and the time
    ``t0`` for a time-aware function)."""
    C0 = jnp.asarray(C0, dtype=float)
    if observation_takes_time(observation_fn):
        y = jnp.asarray(observation_fn(jnp.asarray(float(t0)), C0, jnp.asarray(p)))
    else:
        y = jnp.asarray(observation_fn(C0, jnp.asarray(p)))
    if y.ndim > 1:
        raise ValueError(
            f"observation_fn must return a scalar or 1-D vector per time point; got shape {y.shape}."
        )
    return int(y.size)


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
    cfg: _FitConfig, bundle: _ObjectiveBundle, theta0, opt_bounds, starts=None
) -> _OptOut:
    """Run the optimiser from the start point, then from ``n_starts - 1``
    deterministic jittered restarts, keeping the lowest finite loss. With an
    explicit ``starts`` list (full unconstrained theta vectors), run from each
    of those instead (``theta0`` is then unused).

    A start whose forward solve fails part-way (see :func:`_is_solve_failure`)
    is skipped with a ``RuntimeWarning``; if every start fails, a
    ``RuntimeError`` with remedies is raised, chained to the first failure.
    """
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

        def _lsq(x0):
            r = least_squares(
                lambda x: np.asarray(bundle.res_j(jnp.asarray(x)), dtype=float),
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


def _joint_names(problem: _CalibrationProblem) -> list[str]:
    """Row/column names of the joint rate + initial-pool Hessian: the rate
    names, then ``"ic[k].<species>"`` for each dataset ``k``'s fitted pools, or
    ``"ic[k].<c_j>:<c_0>"`` for the log-ratio coordinates of a fixed-total
    split."""
    return list(problem.free_params) + _ic_coordinate_names(
        problem.free_ic, problem.n_datasets, problem.ic_fixed_total
    )


def _ic_coordinate_names(species, n_datasets: int, fixed_total: bool) -> list[str]:
    """Names of the fitted initial-pool coordinates, dataset by dataset."""
    species = list(species)
    per = [f"{s}:{species[0]}" for s in species[1:]] if fixed_total else species
    return [f"ic[{k}].{c}" for k in range(n_datasets) for c in per]


def _ic_theta_from_pools(problem: _CalibrationProblem, C0s) -> jnp.ndarray:
    """Unconstrained free-IC coordinates of the pools in ``C0s`` (one initial
    state per dataset), laid out as ``problem.ic_center_full`` is."""
    idx = np.asarray(problem.ic_species_idx)
    blocks = []
    for C0_i in C0s:
        pools = np.asarray(C0_i, dtype=float)[idx]
        if problem.ic_fixed_total:
            blocks.append(np.log(pools[1:]) - np.log(pools[0]))
        else:
            lo, hi = problem.ic_bounds
            blocks.append(np.log(np.clip(pools, lo, hi)))
    return jnp.asarray(np.concatenate(blocks))


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
    observation_fn=None,
    ic_fixed_total=False,
    bounds=None,
    use_model_bounds=True,
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
    sigma_list = _sigma_per_dataset(sigma, n_datasets)

    if observation_fn is not None:
        if observed_species is not None:
            raise ValueError("Pass observed_species or observation_fn, not both.")
        obs_species_indices = jnp.arange(model.n_species)
        n_observed = observation_size(
            observation_fn, C0_list[0], p0_full, t0=float(np.asarray(tobs_list[0]).reshape(-1)[0])
        )
        what = "outputs were returned by observation_fn"
    elif observed_species is None:
        obs_species_indices = jnp.arange(model.n_species)
        n_observed = model.n_species
        what = "species were specified"
    else:
        obs_species_indices = jnp.asarray([model.species_index[s] for s in observed_species])
        n_observed = len(observed_species)
        what = "species were specified"

    # Validate each dataset and build its (C0, t_eval, t_span, loss) tuple.
    datasets = []
    tobs_user = []
    digest_parts = []
    n_seen_per_output = np.zeros(n_observed, dtype=int)  # observed (non-NaN) entries
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
        # time unit, the same way reactor.solve(time_unit=...) does. The times
        # as given are kept for a time-aware observation_fn: dividing the
        # native times by the factor again need not return them exactly.
        tobs_user.append(tobs_i)
        tobs_i = tobs_i * native_time_factor(model.time_unit, time_unit)
        obs_i = jnp.asarray(obs_i)
        if obs_i.ndim == 1:
            obs_i = obs_i[:, None]
        seen_i = _observed_counts(ds, obs_i)
        if obs_i.shape[0] != tobs_i.shape[0]:
            raise ValueError(
                f"dataset {ds}: observations has {obs_i.shape[0]} rows but t_obs "
                f"has {tobs_i.shape[0]} entries."
            )
        if obs_i.shape[1] != n_observed:
            raise ValueError(
                f"dataset {ds}: observations has {obs_i.shape[1]} columns but {n_observed} {what}."
            )
        n_seen_per_output += seen_i
        sig_arr = _coerce_sigma(sig_i)
        digest_parts += [obs_i, sig_arr, tobs_i, tobs_user[-1]]
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
    ic_np_idx = [model.species_index[s] for s in free_ic_list]
    ic_species_idx = jnp.asarray(ic_np_idx, dtype=int)
    ic_bounds, ic_center_full, ic_fixed_total = _resolve_ic_coords(
        free_ic_list,
        ic_bounds,
        ic_fixed_total,
        [np.asarray(C0_i, dtype=float)[ic_np_idx] for (C0_i, *_rest) in datasets],
    )

    sigma_estimate = _is_sigma_estimate(sigma)
    # Each output's noise is estimated from its observed (non-NaN) entries.
    n_obs_per_output = _n_obs_per_output(sigma_estimate, n_seen_per_output)
    rate_bounds = _resolve_rate_bounds(
        model, free_params, resolved_transforms, bounds, use_model_bounds
    )

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
        observation_fn=observation_fn,
        obs_times=tuple(tobs_user),
        sigma_estimate=sigma_estimate,
        n_obs_per_output=n_obs_per_output,
        ic_fixed_total=ic_fixed_total,
        data_key=(loss, _array_digest(*digest_parts)),
        rate_bounds=rate_bounds,
    )


def _noise_estimate(problem: _CalibrationProblem, fm, theta):
    """``sqrt(SSE_j / N_j)`` per output at ``theta`` (concentrated likelihood)."""
    theta = jnp.asarray(theta)
    physical = problem.physical_from_theta(theta[: problem.n_rate])
    p = problem.p0_full.at[problem.free_indices].set(physical)
    preds = _predict(problem, fm, p, theta[problem.n_rate :], problem.C0_base)
    return jnp.sqrt(_pooled_sse(problem, preds) / problem.n_obs_per_output)


def _bound_to_unconstrained(value, transform: str, side: str) -> float:
    """One physical bound mapped into unconstrained space. A side that is open
    (``None``) or at / beyond the edge of the transform's domain maps to
    ``-inf`` (lower) or ``+inf`` (upper)."""
    open_value = -np.inf if side == "lower" else np.inf
    if value is None:
        return open_value
    v = float(value)
    if not np.isfinite(v):
        return open_value
    if transform == "positive_log" and v <= 0.0:
        return open_value
    if transform == "logit" and not (0.0 < v < 1.0):
        return open_value
    return float(to_unconstrained(v, transform, xp=np))


def _resolve_rate_bounds(model, free_params, transforms, bounds, use_model_bounds):
    """Absolute bounds on the free rate parameters in unconstrained space.

    Each parameter takes its explicit ``bounds`` entry if there is one, else its
    model-declared ``bounds:`` (when ``use_model_bounds``), else none. Returns
    ``(lb, ub)`` arrays (+/-inf where open) or ``None`` if no side is bounded.
    """
    explicit = dict(bounds or {})
    for name in explicit:
        if name not in model.param_index:
            raise KeyError(f"Unknown parameter '{name}' in bounds. Available: {model.parameters}")
    declared = getattr(model, "parameter_bounds", {}) if use_model_bounds else {}
    lb, ub = [], []
    for name, t in zip(free_params, transforms):
        b = explicit.get(name, declared.get(name))
        if b is None:
            lb.append(-np.inf)
            ub.append(np.inf)
            continue
        if len(b) != 2:
            raise ValueError(f"bounds for '{name}' must be a (lo, hi) pair; got {b!r}.")
        lo, hi = b
        if lo is not None and hi is not None and not float(lo) < float(hi):
            raise ValueError(
                f"bounds for '{name}' must satisfy lo < hi; got {(lo, hi)}. To hold "
                "a parameter fixed, leave it out of free_params."
            )
        lb.append(_bound_to_unconstrained(lo, t, "lower"))
        ub.append(_bound_to_unconstrained(hi, t, "upper"))
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    if not (np.isfinite(lb).any() or np.isfinite(ub).any()):
        return None
    return lb, ub


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
    if optimizer.n_starts < 1:
        raise ValueError(f"optimizer.n_starts must be >= 1; got {optimizer.n_starts}.")
    if optimizer.starts is not None and optimizer.n_starts > 1:
        raise ValueError(
            "Pass optimizer.starts (an explicit start list) or optimizer.n_starts > 1 "
            "(jittered restarts), not both."
        )
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
        observation_fn=observation_fn,
        ic_fixed_total=bool(free_ic is not None and free_ic.fixed_total),
        bounds=optimizer.bounds,
        use_model_bounds=optimizer.use_model_bounds,
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
        laplace_fd_step=lap.fd_step,
        laplace_dtmax=lap.dtmax,
        compiled_cache=_compiled_cache,
        laplace_include_ic=lap.include_ic,
    )
    fm = _ReactorForwardModel(
        reactor, gradient, cfg.stable_adjoint_max_steps, cfg.stable_adjoint_low_memory
    )
    bundle = _build_objective(problem, fm, cfg)

    # --- Run the optimiser in unconstrained space ----------------------
    # The pools start at C0 (the prior centre) unless a separate start is given.
    ic_theta0 = problem.ic_center_full
    if _ic_start is not None and problem.m_ic:
        ic_theta0 = _ic_theta_from_pools(problem, _ic_start)
    theta0, opt_bounds, starts = _initial_point(problem, optimizer.starts, ic_theta0)

    if cfg.check_finite:
        _check_start_gradient(cfg, bundle, theta0)

    result = _run_multistart(cfg, bundle, theta0, opt_bounds, starts=starts)
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
