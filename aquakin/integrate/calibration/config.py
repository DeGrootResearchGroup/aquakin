"""The public configuration objects of :func:`aquakin.calibrate`."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

_VALID_LOSSES = ("mse", "wmse", "nll")
_VALID_OPTIMIZERS = ("lbfgsb", "gauss_newton")


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
        Optimiser convergence tolerance. With ``loss="mse"`` it is relative to
        the loss at the start when that is below 1, so it does not depend on
        the unit of the observations.
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

    def __post_init__(self):
        # Validated here, not by each entry point, so no caller (calibrate,
        # Plant.calibrate, a profile sweep) can reach the optimiser dispatch with
        # a method it does not know.
        if self.method not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer.method must be one of {_VALID_OPTIMIZERS}; got {self.method!r}."
            )


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
