"""Information criteria for comparing calibrated model structures: the maximized
log-likelihood of a fit, its parameter and observation counts, AIC, AICc, BIC,
and a comparison table over fits of several models to the same data."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aquakin.integrate.calibration.objective import _add_prior_term

if TYPE_CHECKING:
    from aquakin.integrate.calibration.problem import _CalibrationProblem
    from aquakin.integrate.calibration.result import CalibrationResult

__all__ = ["ModelComparison", "aic", "aicc", "bic", "compare_models"]

_CRITERIA = ("aic", "aicc", "bic")


def _fit_statistics(problem: _CalibrationProblem, physical, ic_thetas, map_loss: float) -> dict:
    """The fit statistics a :class:`CalibrationResult` carries for the criteria.

    The objective at the optimum is the data term plus the prior penalty, and
    for ``loss="nll"`` the data term is the Gaussian negative log-likelihood
    less its ``(N/2) log(2 pi)`` constant: with a known ``sigma`` it is
    ``sum(log(sigma) + r**2 / (2 sigma**2))``, and with ``sigma="estimate"``
    the same expression at each output's profiled noise level. So the maximized
    log-likelihood is ``-(map_loss - prior penalty) - (N/2) log(2 pi)``.

    Parameters
    ----------
    problem : _CalibrationProblem
        The resolved problem.
    physical : jnp.ndarray
        The fitted rate parameters in physical space, shape ``(n_rate,)``.
    ic_thetas : jnp.ndarray
        The fitted initial-pool coordinates of every dataset.
    map_loss : float
        The objective at the optimum (``CalibrationResult.loss``).

    Returns
    -------
    dict
        ``log_likelihood`` (``None`` unless ``loss="nll"``), ``n_observations``
        (observed entries over every dataset), ``n_estimated`` (free rates, plus
        fitted initial-pool coordinates, plus one noise level per output under
        ``sigma="estimate"``), and the private ``_observations_key`` and
        ``_prior_in_loss`` the criteria check.
    """
    prior_in_loss = problem.has_priors or bool(problem.m_ic and problem.ic_prior_log_std)
    log_likelihood = None
    if problem.is_likelihood:
        penalty = float(_add_prior_term(problem, 0.0, physical, ic_thetas, problem.ic_center_full))
        n = problem.n_observations
        log_likelihood = -(map_loss - penalty) - 0.5 * n * math.log(2.0 * math.pi)
    n_noise = problem.n_observed if problem.sigma_estimate else 0
    return {
        "log_likelihood": log_likelihood,
        "n_observations": problem.n_observations,
        "n_estimated": problem.n_rate + int(ic_thetas.shape[0]) + n_noise,
        "_observations_key": problem.observations_key,
        "_prior_in_loss": prior_in_loss,
    }


def _counts(result: CalibrationResult) -> tuple[float, int, int]:
    """``(log_likelihood, k, n)`` of a fit, or a ``ValueError`` saying why the
    fit has no likelihood; warns when the fit's objective carried a prior."""
    if result.log_likelihood is None:
        raise ValueError(
            "This fit has no log-likelihood: information criteria need loss='nll'. "
            "Refit with loss='nll' and a known sigma, or with sigma='estimate' "
            "(the least-squares form, which estimates one noise level per output)."
        )
    if result._prior_in_loss:
        warnings.warn(
            "This fit's objective included a prior, so its parameters are the MAP "
            "estimate, not the maximum-likelihood one, and the log-likelihood there "
            "is below its maximum. For an information criterion, refit with "
            "use_priors=False (and no FreeICConfig prior_log_std).",
            UserWarning,
            stacklevel=3,
        )
    return result.log_likelihood, result.n_estimated, result.n_observations


def aic(result: CalibrationResult) -> float:
    """Akaike's information criterion of a fit, ``2 k - 2 log L``.

    Parameters
    ----------
    result : CalibrationResult
        A fit by :func:`aquakin.calibrate` or :meth:`Plant.calibrate` with
        ``loss="nll"``. ``k`` is ``result.n_estimated`` (free rates, fitted
        initial pools and estimated noise levels) and ``log L`` is
        ``result.log_likelihood``.

    Returns
    -------
    float
        The AIC; lower is better supported, and only differences between fits
        to the same observations mean anything.

    Raises
    ------
    ValueError
        If the fit was not by ``loss="nll"`` (see :func:`compare_models`).

    Warns
    -----
    UserWarning
        If the objective included a prior, so the fit is not the
        maximum-likelihood point.
    """
    log_l, k, _n = _counts(result)
    return 2.0 * k - 2.0 * log_l


def aicc(result: CalibrationResult) -> float:
    """The small-sample corrected AIC, ``AIC + 2 k (k + 1) / (n - k - 1)``.

    Burnham and Anderson recommend it over :func:`aic` unless ``n / k`` is
    large (above about 40); the two agree as ``n`` grows.

    Parameters
    ----------
    result : CalibrationResult
        A fit with ``loss="nll"``; ``n`` is ``result.n_observations``.

    Returns
    -------
    float
        The AICc.

    Raises
    ------
    ValueError
        If the fit was not by ``loss="nll"``, or ``n <= k + 1``, where the
        correction is undefined.

    Warns
    -----
    UserWarning
        If the objective included a prior.
    """
    log_l, k, n = _counts(result)
    if n - k - 1 <= 0:
        raise ValueError(
            f"AICc is undefined for {n} observations and {k} estimated quantities "
            "(it needs n > k + 1); use aic() or collect more data."
        )
    return 2.0 * k - 2.0 * log_l + 2.0 * k * (k + 1) / (n - k - 1)


def bic(result: CalibrationResult) -> float:
    """Schwarz's Bayesian information criterion, ``k log(n) - 2 log L``.

    Parameters
    ----------
    result : CalibrationResult
        A fit with ``loss="nll"``.

    Returns
    -------
    float
        The BIC; it penalizes parameters more than AIC once ``n >= 8``.

    Raises
    ------
    ValueError
        If the fit was not by ``loss="nll"``.

    Warns
    -----
    UserWarning
        If the objective included a prior.
    """
    log_l, k, n = _counts(result)
    return k * math.log(n) - 2.0 * log_l


@dataclass
class ModelComparison:
    """Fits of several models to the same observations, ranked by a criterion.

    Attributes
    ----------
    criterion : str
        ``"aic"``, ``"aicc"`` or ``"bic"``.
    names : list[str]
        The model names, best (lowest criterion) first.
    values : dict[str, float]
        The criterion of each model.
    delta : dict[str, float]
        Each model's criterion minus the best one's (0 for the best).
    weights : dict[str, float]
        ``exp(-delta / 2)`` normalized to sum to 1: the Akaike weights for AIC
        and AICc, and the approximate posterior model probabilities under equal
        prior model probabilities for BIC.
    log_likelihood : dict[str, float]
        The maximized log-likelihood of each model.
    n_estimated : dict[str, int]
        The number of estimated quantities ``k`` of each model.
    n_observations : int
        The number of observations, shared by every fit.
    """

    criterion: str
    names: list[str]
    values: dict[str, float]
    delta: dict[str, float]
    weights: dict[str, float]
    log_likelihood: dict[str, float]
    n_estimated: dict[str, int]
    n_observations: int

    @property
    def best(self) -> str:
        """The name of the model with the lowest criterion."""
        return self.names[0]


def compare_models(
    results: Mapping[str, CalibrationResult], criterion: str = "aic"
) -> ModelComparison:
    """Rank fits of competing models to the same observations.

    Parameters
    ----------
    results : Mapping[str, CalibrationResult]
        At least two fits by name, each with ``loss="nll"`` (a known ``sigma``
        or ``sigma="estimate"``). Least-squares fits compare through
        ``sigma="estimate"``, whose likelihood is the least-squares AIC of
        Burnham and Anderson with one noise level per output; an ``"mse"`` or
        ``"wmse"`` fit has no likelihood and is refused.
    criterion : {"aic", "aicc", "bic"}
        The criterion to rank by.

    Returns
    -------
    ModelComparison

    Raises
    ------
    ValueError
        For an unknown ``criterion``, fewer than two fits, a fit without a
        likelihood, or fits whose observations or observation times differ
        (their likelihoods are of different data, so their criteria cannot be
        compared).

    Warns
    -----
    UserWarning
        For each fit whose objective included a prior.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}; got {criterion!r}.")
    if len(results) < 2:
        raise ValueError(f"compare_models needs at least two fits; got {len(results)}.")
    keys = {name: r._observations_key for name, r in results.items()}
    unrecorded = [name for name, key in keys.items() if not key]
    if unrecorded:
        raise ValueError(
            f"The fits {unrecorded} carry no record of their observations (they were "
            "not returned by calibrate or Plant.calibrate), so they cannot be checked "
            "to be of the same data."
        )
    first = next(iter(keys))
    differ = [name for name, key in keys.items() if key != keys[first]]
    if differ:
        raise ValueError(
            f"The fits {differ} are not of the same observations as '{first}': "
            "information criteria compare models only on the same data (the same "
            "observed values, missing entries and observation times)."
        )
    score = {"aic": aic, "aicc": aicc, "bic": bic}[criterion]
    values = {name: score(r) for name, r in results.items()}
    names = sorted(values, key=values.__getitem__)
    delta = {name: values[name] - values[names[0]] for name in names}
    raw = {name: math.exp(-0.5 * d) for name, d in delta.items()}
    total = sum(raw.values())
    return ModelComparison(
        criterion=criterion,
        names=names,
        values=values,
        delta=delta,
        weights={name: w / total for name, w in raw.items()},
        log_likelihood={name: results[name].log_likelihood for name in names},
        n_estimated={name: results[name].n_estimated for name in names},
        n_observations=results[first].n_observations,
    )
