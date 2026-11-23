"""The loss and residual builders: missing-data masking, ``sigma`` handling and
observation-unit conversion."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aquakin.integrate.calibration.config import _VALID_LOSSES

# --- Loss factory ------------------------------------------------------


# Floor on a per-output sum of squared residuals in the concentrated likelihood,
# so an exact fit (SSE = 0, synthetic noise-free data) gives a finite objective.
_SSE_FLOOR = 1e-300


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


# A start whose ``"mse"`` misfit is below this fraction of the data's mean
# square (an RMS misfit 1e-8 of the data's RMS, below any integrator tolerance)
# already fits the data to the solver's precision: the objective is not scaled
# up past it, so round-off is not mistaken for misfit.
_SCALE_FLOOR = 1e-16


def _data_mean_square(loss_type: str, observations) -> float | None:
    """The mean square of the observed entries over every dataset, for the
    ``"mse"`` loss (``None`` for another loss); see :func:`_objective_scale`.

    Parameters
    ----------
    loss_type : str
        The loss.
    observations : list of array-like
        The observations of each dataset (``NaN`` marks a missing entry).

    Returns
    -------
    float or None
    """
    if loss_type != "mse":
        return None
    values = np.concatenate([np.asarray(o, dtype=float).reshape(-1) for o in observations])
    values = values[~np.isnan(values)]
    return float(np.mean(values**2)) if values.size else 0.0


def _objective_scale(data_mean_square: float | None, start_loss: float) -> float:
    """The fixed factor the optimizer's objective is divided by.

    SciPy's stopping tests are absolute once the objective is below 1: L-BFGS-B
    stops when a step lowers it by less than ``ftol * max(|f|, 1)`` or when the
    largest projected gradient entry is below ``tol``, and least squares has an
    absolute gradient test too. The ``"mse"`` loss carries the square of the
    data's unit, so on data of order 1e-6 (concentrations in mol/L) the loss and
    its gradient are below those thresholds at any start, and the optimizer
    returns the start unchanged while reporting convergence. Dividing the
    objective by its value at the start makes the tests relative to the
    starting misfit, whatever the unit. A loss of 1 or more is already in
    SciPy's relative regime and is left unscaled, and the scale is never below
    ``_SCALE_FLOOR`` times the data's mean square, so a start that already fits
    to round-off is not scaled up to it.

    A constant factor does not move the minimum. It is applied only to what the
    optimizer sees: the reported loss and the Laplace posterior use the
    unscaled objective. ``"wmse"`` and ``"nll"`` are in units of ``sigma``
    (and the concentrated likelihood in log units), so they are not scaled.

    Parameters
    ----------
    data_mean_square : float or None
        :func:`_data_mean_square` of the problem (``None``: do not scale).
    start_loss : float
        The objective at the optimizer's (first) start.

    Returns
    -------
    float
        ``min(1, max(start_loss, _SCALE_FLOOR * data_mean_square))``, or 1 when
        the loss is not scaled or that is not positive and finite.
    """
    if data_mean_square is None:
        return 1.0
    scale = min(1.0, max(float(start_loss), _SCALE_FLOOR * data_mean_square))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _is_sigma_estimate(sigma) -> bool:
    """Whether ``sigma`` asks for the concentrated likelihood."""
    return isinstance(sigma, str) and sigma == "estimate"


def _observation_unit_factors(columns, units) -> np.ndarray | None:
    """Per-column factors from declared observation units to the model's.

    Parameters
    ----------
    columns : list of (str, str)
        One ``(name, model unit)`` pair per observed column, in column order.
    units : dict[str, str] or None
        ``name -> declared unit`` for the columns given in another unit. A name
        covers every column of that name.

    Returns
    -------
    np.ndarray or None
        ``x_model = factor * x_declared`` per column, or ``None`` when nothing
        is declared or every factor is exactly 1.

    Raises
    ------
    ValueError
        For a name that is not an observed column, a unit that is not a
        string or cannot be read, or one that cannot be converted to the
        model's (another quantity, or a model unit that is blank or
        unreadable).
    """
    from aquakin.utils.units import conversion_factor

    if not units:
        return None
    names = [name for name, _ in columns]
    unknown = sorted(set(units) - set(names))
    if unknown:
        raise ValueError(
            f"observation_units names {unknown}, which are not observed columns; "
            f"the observed columns are {sorted(set(names))}."
        )
    factors = np.ones(len(columns))
    for i, (name, model_unit) in enumerate(columns):
        if name not in units:
            continue
        unit = units[name]
        if not isinstance(unit, str):
            raise ValueError(f"observation_units['{name}'] must be a string; got {unit!r}.")
        try:
            factor = conversion_factor(unit, model_unit)
        except ValueError as exc:
            raise ValueError(f"observation_units['{name}']: {exc}") from None
        if factor is None:
            raise ValueError(
                f"observation_units['{name}'] is '{unit}', and the model declares "
                f"'{model_unit}'; one of the two cannot be read, so the "
                f"observations cannot be converted."
            )
        factors[i] = factor
    return None if np.all(factors == 1.0) else factors


def _convert_observations(observations, sigma, factors):
    """Observations (and an absolute ``sigma``) multiplied into model units.

    ``factors`` is the per-column array of :func:`_observation_unit_factors`.
    Works on one dataset or a list of them; a 1-D dataset is a single observed
    column. ``sigma`` keeps its form: a scalar becomes per-column when the
    factors differ, an array or a per-dataset list is scaled along its last
    axis, and ``"estimate"`` / ``None`` are left alone (an estimated noise level
    is reported in model units).
    """
    if factors is None:
        return observations, sigma
    f = jnp.asarray(factors)

    def scale(arr):
        arr = jnp.asarray(arr)
        if arr.ndim == 1 and f.shape[0] == 1:
            return arr * f[0]
        return arr * f

    multi = isinstance(observations, (list, tuple))
    if multi:
        observations = [scale(o) for o in observations]
    else:
        observations = scale(observations)

    def scale_sigma(sig):
        if sig is None or _is_sigma_estimate(sig):
            return sig
        arr = jnp.asarray(sig)
        if arr.ndim == 0:
            return arr * f if f.shape[0] > 1 else arr * f[0]
        return scale(arr)

    # A list of sigmas is one per dataset only in multi-batch mode; for a
    # single dataset it is a per-column (or per-entry) array.
    if multi and isinstance(sigma, (list, tuple)):
        sigma = [scale_sigma(s) for s in sigma]
    else:
        sigma = scale_sigma(sigma)
    return observations, sigma


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
