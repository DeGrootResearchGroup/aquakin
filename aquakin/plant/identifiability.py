"""Identifiability of plant parameters: Fisher information and profile likelihood.

The plant analogues of :func:`aquakin.fisher_information` and
:func:`aquakin.profile_likelihood`. Both read a plant's outputs the way
:meth:`Plant.calibrate` does -- as channels of reconstructed streams, named by
:class:`~aquakin.plant.PlantObservable` -- and work in its coordinates: plant
parameters by ``"<model>.<param>"`` name, each through its transform.

:func:`plant_fisher_information` evaluates

    F = sum_r  S_r^T  Sigma_r^{-1}  S_r

over the runs ``r`` of a design, where ``S_r`` is the sensitivity of run
``r``'s observed channels to the unconstrained free parameters and
``Sigma_r = diag(sigma_r^2)`` its noise covariance. It needs assumed parameter
values and the design (initial state, sampling times, observed channels, noise),
never data. At a :meth:`Plant.calibrate` MAP with ``loss="nll"`` and the
Gauss-Newton Laplace Hessian it equals that Hessian.

The sensitivity comes from one of two solves:

- ``mode="forward"`` -- :meth:`Plant.solve_sensitivity`, the variational
  ``[y; S]`` solve, gives the state sensitivity ``dy/dtheta`` at every sampling
  time, and the observed channels' sensitivity is its image under the stream
  reconstruction (a Jacobian-vector product per parameter). Its memory does not
  grow with the number of observations, so it suits long horizons.
- ``mode="reverse"`` -- the stable discrete adjoint through ``plant.solve`` (the
  solve :meth:`Plant.calibrate` differentiates), one pullback per observation
  row. It suits a design with few observations and many parameters.

:func:`plant_profile_likelihood` fixes one parameter at each value on a grid,
re-fits the others with :meth:`Plant.calibrate`, and reads the likelihood-ratio
interval off the profile -- the same warm-started continuation, polish pass and
interval interpolation as the reactor profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.core.hints import did_you_mean
from aquakin.integrate._common import DifferentiationConfig, check_finite_gradient, to_native_time
from aquakin.integrate._transforms import dphysical_dunconstrained
from aquakin.integrate.calibrate import FreeICConfig, OptimizerConfig, _sigma_per_dataset
from aquakin.integrate.fisher import (
    FisherInformation,
    _active_priors,
    _observation_noise,
    _resolve_transforms,
)
from aquakin.integrate.profile import (
    ProfileResult,
    _check_profile_args,
    _cold_starts,
    _guarded,
    _point_optimizer,
    _profile_result,
    _profile_sweep,
)
from aquakin.plant.calibrate import (
    _check_free_params,
    _normalize_observables,
    _PlantForwardModel,
    _PlantParamNamespace,
    _resolve_observable_columns,
    calibrate_plant,
)

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.plant.plant import Plant

_MODES = ("forward", "reverse")


# --- Fisher information --------------------------------------------------


def _is_run_list(t_obs) -> bool:
    """A list of sampling-time arrays (one per run) rather than one array."""
    return (
        isinstance(t_obs, (list, tuple))
        and len(t_obs) > 0
        and isinstance(t_obs[0], (list, tuple, np.ndarray, jnp.ndarray))
    )


def _per_run(value, n_runs: int, name: str) -> list:
    """One entry per run: a list (or tuple) is checked against the run count,
    anything else is broadcast."""
    if n_runs > 1 and isinstance(value, (list, tuple)):
        if len(value) != n_runs:
            raise ValueError(f"{name} has {len(value)} entries but there are {n_runs} runs.")
        return list(value)
    return [value] * n_runs


def _per_run_span(t_span, n_runs: int) -> list:
    """One ``(t0, t1)`` window (or ``None``) per run; a single window
    broadcasts."""
    if t_span is not None and n_runs > 1 and isinstance(t_span[0], (list, tuple)):
        return _per_run(list(t_span), n_runs, "t_span")
    return [t_span] * n_runs


def _observed_channels(plant: Plant, resolved_observables, ts_native, states, p):
    """The observed channels at each saved time, shape ``(n_t, n_observed)``:
    the plant's outputs reconstructed from ``states`` (``(n_t, ndof)``) at the
    native times ``ts_native`` and parameters ``p``, in observable order.
    Differentiable in ``states`` and ``p``."""
    keys = [plant._parse_endpoint(endpoint, role="source") for endpoint, _ in resolved_observables]

    def _at(t, y):
        outputs = plant.outputs_at(t, y, p)
        return jnp.concatenate(
            [outputs[key].C[ch] for key, (_, ch) in zip(keys, resolved_observables)]
        )

    return jax.vmap(_at)(ts_native, states)


def _forward_sensitivity(plant, run, resolved_observables, p, free_idx, wrt, options):
    """``(y_obs, dy_obs/dphysical)`` of one run from the variational solve,
    shapes ``(n_t, n_obs)`` and ``(n_t, n_obs, k)``.

    :meth:`Plant.solve_sensitivity` gives ``S = dy/dtheta`` at the sampling
    times; the channels' sensitivity is ``dg/dy . S + dg/dtheta`` for the
    reconstruction ``g``, one Jacobian-vector product per parameter."""
    span, t_eval, _ = to_native_time(plant.time_unit, run["time_unit"], run["t_span"], run["t_obs"])
    ts, ys, S = plant.solve_sensitivity(
        p, wrt, t_span=span, t_eval=jnp.asarray(t_eval), y0=run["y0"], **options
    )

    def g(states, free_values):
        return _observed_channels(
            plant, resolved_observables, ts, states, p.at[free_idx].set(free_values)
        )

    y_obs, jvp = jax.linearize(g, ys, p[free_idx])
    eye = jnp.eye(free_idx.shape[0])
    dy = jax.vmap(jvp, in_axes=(2, 0), out_axes=2)(S, eye)
    return y_obs, dy


def _reverse_sensitivity(plant, run, resolved_observables, p, free_idx, integrator, diff):
    """``(y_obs, dy_obs/dphysical)`` of one run from reverse-mode AD through
    ``plant.solve`` -- the forward model :meth:`Plant.calibrate` uses, so the
    two differentiate the same discrete solve."""
    fm = _PlantForwardModel(
        plant=plant,
        observables=resolved_observables,
        y0=run["y0"],
        integrator=integrator,
        diff=diff,
        time_unit=run["time_unit"],
    )
    span = (float(run["t_span"][0]), float(run["t_span"][1]))
    t_obs = jnp.asarray(run["t_obs"])

    def h(free_values):
        return fm.solve_trajectory(p.at[free_idx].set(free_values), None, span, t_obs)

    y_obs = h(p[free_idx])
    dy = jax.jacrev(h)(p[free_idx])  # one pullback per observation entry
    return y_obs, dy


def plant_fisher_information(
    plant,
    observables,
    t_obs,
    free_params: list[str],
    *,
    sigma,
    relative_sigma: bool = False,
    sigma_floor: float = 0.0,
    mask=None,
    t_span=None,
    y0=None,
    params: jnp.ndarray | None = None,
    transforms: dict[str, str] | None = None,
    priors: dict[str, tuple[float, float]] | None = None,
    use_priors: bool = False,
    mode: str = "forward",
    time_unit: str | None = None,
    integrator=None,
    diff: DifferentiationConfig = DifferentiationConfig(),
    sensitivity_options: dict | None = None,
) -> FisherInformation:
    """Fisher information of a plant design at assumed parameter values.

    Parameters
    ----------
    plant : Plant or list of Plant
        The plant, with its influent(s) added. A list gives one plant per run,
        for runs that differ in their influent; they must share the parameter
        names.
    observables : list
        The measured channels, in the forms :meth:`Plant.calibrate` accepts: a
        :class:`PlantObservable`, a ``(stream_or_unit, channels)`` pair, a
        ``{"stream": ..., "channels": ...}`` dict, or a bare stream name. A unit
        name reads the unit's outlet, so a concentration unit's states are
        observed through it (``("digester", ["S_IN", "S_gas_ch4"])``). The
        columns run in observable order, channels within a stream first.
    t_obs : array-like or list of array-like
        Sampling times, strictly ascending, shape ``(n_t,)``, in the plant's
        time unit (or ``time_unit``). A list of arrays describes a multi-run
        design (one entry per run); the runs' information adds.
    free_params : list of str
        Plant parameters to estimate (``"<model>.<param>"``).
    sigma : float, array-like, or list
        Measurement standard deviation: a scalar, one value per observed
        channel ``(n_observed,)``, or one per observation ``(n_t, n_observed)``.
        In a multi-run design a list gives one such entry per run.
    relative_sigma : bool, optional
        Read ``sigma`` as relative to the signal: the standard deviation of an
        observation is ``max(sigma * |y|, sigma_floor)``, with ``y`` the
        predicted channel at the assumed parameters. Default ``False``
        (``sigma`` is absolute).
    sigma_floor : float, optional
        Lower bound on a relative standard deviation, so a channel near zero
        does not carry unbounded information. Used only with
        ``relative_sigma``.
    mask : array-like of bool or list, optional
        Which observations are taken: ``True`` where a channel is measured at a
        sampling time, broadcasting to ``(n_t, n_observed)``. An unmeasured
        entry carries no information. One per run in a multi-run design.
        ``None`` measures everything.
    t_span : tuple or list of tuple, optional
        Integration window ``(t0, t1)``; defaults to ``(t_obs[0], t_obs[-1])``,
        as in :meth:`Plant.calibrate`. One per run, or one shared by every run.
    y0 : jnp.ndarray or list, optional
        Initial plant state (e.g. ``bsm2_warm_start(plant)``); defaults to the
        plant's initial state. One per run, or one shared by every run.
    params : jnp.ndarray, optional
        Full parameter vector the information is evaluated at. Defaults to
        :meth:`Plant.default_parameters`.
    transforms : dict, optional
        Per-parameter transform override (``"positive_log"``, ``"logit"``,
        ``"none"``); otherwise each parameter's model-declared transform.
    priors : dict, optional
        Gaussian priors ``{name: (mean, std)}`` in physical space, added with
        ``use_priors``.
    use_priors : bool, optional
        Add the prior precision (model-declared priors overridden by
        ``priors``) to ``F``. Default ``False``: the information from the data
        alone.
    mode : {"forward", "reverse"}, optional
        ``"forward"`` (default) takes the sensitivity from
        :meth:`Plant.solve_sensitivity`, whose memory does not grow with the
        number of observations. ``"reverse"`` differentiates ``plant.solve``
        with ``diff`` (the stable discrete adjoint by default), one pullback per
        observation; it is the solve :meth:`Plant.calibrate` differentiates.
    time_unit : str, optional
        Unit of ``t_obs`` and ``t_span``.
    integrator : IntegratorConfig, optional
        Plant integrator for ``mode="reverse"``.
    diff : DifferentiationConfig, optional
        How the gradient flows through ``plant.solve`` in ``mode="reverse"``.
    sensitivity_options : dict, optional
        Keyword arguments for :meth:`Plant.solve_sensitivity` in
        ``mode="forward"`` (``rtol``, ``atol``, ``sens_rtol``, ``factormax``,
        ``dtmax``, ``max_steps``).

    Returns
    -------
    FisherInformation
        The matrix over ``free_params`` in unconstrained coordinates, with the
        physical-space matrix and the Cramer-Rao standard deviations as methods.
        ``jacobian`` stacks the noise-scaled sensitivity rows run by run, each
        ``(n_t, n_observed)`` block flattened time-major.

    Raises
    ------
    KeyError
        If a parameter, stream or channel name is unknown.
    ValueError
        If ``mode`` is unknown, the runs are inconsistent, or a noise level is
        not positive.

    Notes
    -----
    In the forward mode a channel's sensitivity is ``dg/dy . S + dg/dtheta``:
    the state sensitivity carried through the stream reconstruction ``g``, plus
    the reconstruction's own dependence on the parameters (a translator that
    reads a model parameter). Both modes therefore differentiate the same
    observed quantity and agree to the solver tolerance.

    Examples
    --------
    >>> fim = plant_fisher_information(
    ...     plant, [("effluent", ["SNH", "SNO"])], jnp.linspace(1.0, 7.0, 25),
    ...     ["asm1.muH", "asm1.KS"], sigma=[0.2, 0.5], y0=bsm1_warm_start(plant))
    >>> aquakin.identifiability(fim.matrix, names=fim.names,
    ...                         values=fim.values, transforms=fim.transforms)
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}.")
    if not free_params:
        raise ValueError("free_params must be non-empty.")
    if relative_sigma and sigma_floor < 0.0:
        raise ValueError(f"sigma_floor must be >= 0; got {sigma_floor}.")
    if integrator is None:
        from aquakin.plant.plant import IntegratorConfig

        integrator = IntegratorConfig()

    multi = _is_run_list(t_obs)
    tobs_list = list(t_obs) if multi else [t_obs]
    n_runs = len(tobs_list)
    if isinstance(plant, (list, tuple)):
        plants = list(plant)
        if len(plants) != n_runs:
            raise ValueError(f"plant has {len(plants)} entries but there are {n_runs} runs.")
    else:
        plants = [plant] * n_runs
    sigma_list = _sigma_per_dataset(sigma, n_runs) if multi else [sigma]
    mask_list = _per_run(mask, n_runs, "mask") if multi else [mask]
    y0_list = _per_run(y0, n_runs, "y0") if multi else [y0]
    span_list = _per_run_span(t_span, n_runs)

    ns = _PlantParamNamespace(plants[0])
    _check_free_params(ns, free_params)
    names = list(ns.param_index)
    for r, pl in enumerate(plants[1:], start=1):
        if list(pl.parameter_names()) != names:
            raise ValueError(f"run {r}: its plant has different parameters from run 0's.")
    rate_transforms = _resolve_transforms(ns, free_params, transforms)
    p = jnp.asarray(params) if params is not None else plants[0].default_parameters()
    free_idx = jnp.asarray([ns.param_index[n] for n in free_params], dtype=int)
    for name, tr, value in zip(free_params, rate_transforms, np.asarray(p)[np.asarray(free_idx)]):
        if tr == "positive_log" and value <= 0.0:
            raise ValueError(f"Parameter '{name}' has transform 'positive_log' but value {value}.")
        if tr == "logit" and not 0.0 < value < 1.0:
            raise ValueError(f"Parameter '{name}' has transform 'logit' but value {value}.")
    # d physical / d unconstrained, per free parameter (the chain to the fitted
    # coordinates).
    dphys = np.asarray(
        [
            float(dphysical_dunconstrained(float(p[i]), tr, xp=np))
            for i, tr in zip(np.asarray(free_idx), rate_transforms)
        ]
    )

    observable_specs = _normalize_observables(observables, None, None)
    options = dict(sensitivity_options or {})
    blocks = []
    for r in range(n_runs):
        pl = plants[r]
        resolved_observables, n_observed = _resolve_observable_columns(pl, observable_specs)
        tobs = jnp.asarray(tobs_list[r], dtype=float)
        if tobs.ndim != 1 or tobs.shape[0] < 1:
            raise ValueError(f"run {r}: t_obs must be a non-empty 1-D array.")
        if not bool(jnp.all(jnp.diff(tobs) > 0)):
            raise ValueError(f"run {r}: t_obs must be strictly ascending.")
        span = span_list[r]
        if span is None:
            span = (float(tobs[0]), float(tobs[-1]))
        run = {
            "t_obs": tobs,
            "t_span": (float(span[0]), float(span[1])),
            "y0": None if y0_list[r] is None else jnp.asarray(y0_list[r]),
            "time_unit": time_unit,
        }
        if mode == "forward":
            y_obs, dy = _forward_sensitivity(
                pl, run, resolved_observables, p, free_idx, list(free_params), options
            )
        else:
            y_obs, dy = _reverse_sensitivity(
                pl, run, resolved_observables, p, free_idx, integrator, diff
            )
        n_t = int(tobs.shape[0])
        if relative_sigma:
            sig_r = jnp.maximum(
                jnp.broadcast_to(jnp.asarray(sigma_list[r], dtype=float), (n_t, n_observed))
                * jnp.abs(y_obs),
                sigma_floor,
            )
        else:
            sig_r = sigma_list[r]
        _, keep, sigma_div = _observation_noise(f"run {r}", sig_r, mask_list[r], n_t, n_observed)
        scaled = dy * dphys / sigma_div[..., None]
        if keep is not None:
            scaled = jnp.where(keep[..., None], scaled, 0.0)
        blocks.append(scaled.reshape(-1, len(free_params)))

    active = _active_priors(ns, free_params, priors, use_priors)
    if active:
        std = np.asarray([active.get(n, (0.0, 1.0))[1] for n in free_params])
        on = np.asarray([n in active for n in free_params], dtype=float)
        blocks.append(jnp.asarray(np.diag(on * dphys / std)))

    J = jnp.concatenate(blocks)
    check_finite_gradient(
        J,
        what="the plant Fisher-information sensitivity",
        remedy=(
            'try mode="reverse" (the stable discrete adjoint), or tighten the '
            "variational solve through sensitivity_options"
        ),
    )
    J = np.asarray(J)
    F = J.T @ J
    values = {n: float(p[i]) for n, i in zip(free_params, np.asarray(free_idx))}
    return FisherInformation(
        matrix=0.5 * (F + F.T),
        names=list(free_params),
        transforms=rate_transforms,
        values=values,
        jacobian=J,
    )


# --- Profile likelihood --------------------------------------------------


def plant_profile_likelihood(
    plant: Plant,
    observations,
    t_obs,
    free_params: list[str],
    *,
    grid,
    profile_param: str,
    delta: float = 1.92,
    warm_start: bool = True,
    polish: bool = True,
    polish_passes: int = 2,
    polish_tol: float = 0.05,
    anchor: float | None = None,
    params: jnp.ndarray | None = None,
    loss: str = "nll",
    sigma=None,
    free_ic: FreeICConfig | None = None,
    optimizer: OptimizerConfig = OptimizerConfig(),
    **calibrate_kwargs,
) -> ProfileResult:
    """Profile-likelihood analysis of one plant parameter.

    Fixes ``profile_param`` at each value in ``grid``, re-fits every other
    free parameter with :meth:`Plant.calibrate`, and returns the profile of the
    best attainable objective with its likelihood-ratio confidence interval.
    The sweep is the one :func:`aquakin.profile_likelihood` runs: a cold fit at
    the anchor, a warm-started continuation outward, and a polish pass that
    re-fits any point a neighbour beats. The inner fits share one compiled
    objective, so the plant's objective compiles once for the whole sweep.

    Parameters
    ----------
    plant, observations, t_obs, free_params
        As for :meth:`Plant.calibrate`. ``profile_param`` is removed from
        ``free_params`` if present; at least one other parameter must remain.
    grid : array-like
        Values at which to fix the profiled parameter.
    profile_param : str
        The plant parameter to profile (``"<model>.<param>"``).
    delta : float, optional
        Likelihood-ratio threshold. Default ``1.92`` = ``0.5 * chi2_{1, 0.95}``;
        meaningful with ``loss="nll"`` and a calibrated ``sigma``, or with
        ``sigma="estimate"`` (the concentrated likelihood).
    warm_start, polish, polish_passes, polish_tol, anchor
        As for :func:`aquakin.profile_likelihood`.
    params : jnp.ndarray, optional
        Starting parameter vector of the cold fit (the profiled entry is
        overwritten at each grid point). Defaults to
        :meth:`Plant.default_parameters`.
    loss, sigma : optional
        The objective, as for :meth:`Plant.calibrate`. Default ``loss="nll"``.
    free_ic : FreeICConfig, optional
        Assembled-state initial conditions fitted alongside the rates at every
        grid point; a warm-continued point starts its pools from the
        neighbour's fit.
    optimizer : OptimizerConfig, optional
        For the cold anchor fit (and every fit with ``warm_start=False``); a
        warm-continued point runs one start from its neighbour. An explicit
        ``starts`` list drops the profiled parameter from dict starts. Default
        one start, as for :meth:`Plant.calibrate`.
    **calibrate_kwargs
        Forwarded to every :meth:`Plant.calibrate` call: ``target``,
        ``observed_channels``, ``observables``, ``t_span``, ``y0``,
        ``transforms``, ``time_unit``, ``priors``, ``use_priors``,
        ``check_finite``, ``integrator``, ``diff``. The inner fits force
        ``laplace=False``.

    Returns
    -------
    ProfileResult
        ``fits`` holds each grid point's :class:`~aquakin.CalibrationResult`.

    Raises
    ------
    KeyError
        If ``profile_param`` is not a plant parameter.
    ValueError
        If no free parameter remains, or ``grid`` / ``delta`` / ``optimizer``
        is invalid.
    """
    grid = _check_profile_args(grid, delta, optimizer)
    if "laplace" in calibrate_kwargs:
        raise TypeError("plant_profile_likelihood runs its inner fits with laplace=False.")
    ns = _PlantParamNamespace(plant)
    if profile_param not in ns.param_index:
        suffix = did_you_mean(profile_param, list(ns.param_index))
        raise KeyError(f"Unknown profile_param '{profile_param}'.{suffix}")
    inner_free = [n for n in free_params if n != profile_param]
    if not inner_free:
        raise ValueError(
            "After removing the profiled parameter, free_params is empty. Each grid "
            "point is a Plant.calibrate fit, which needs at least one free parameter "
            "to re-optimise."
        )
    p_idx = ns.param_index[profile_param]
    base_params = jnp.asarray(params) if params is not None else plant.default_parameters()
    cold_starts = _cold_starts(optimizer, profile_param)
    # One compiled objective for every grid point: the fits differ only in the
    # pinned value and the start, which reach it as runtime arguments.
    cache: dict = {}

    def _fit_point(value, warm):
        init_p = base_params if warm is None else warm.params
        ic_start = None if warm is None or not free_ic else warm.C0_fitted
        return calibrate_plant(
            plant,
            observations,
            t_obs,
            inner_free,
            params=init_p.at[p_idx].set(value),
            loss=loss,
            sigma=sigma,
            free_ic=free_ic,
            optimizer=_point_optimizer(optimizer, warm, cold_starts),
            laplace=False,
            _compiled_cache=cache,
            _ic_start=ic_start,
            **calibrate_kwargs,
        )

    fits = _profile_sweep(
        grid,
        _guarded(_fit_point, profile_param),
        warm_start=warm_start,
        anchor=anchor,
        polish=polish,
        polish_passes=polish_passes,
        polish_tol=polish_tol,
    )
    return _profile_result(profile_param, grid, fits, delta)


__all__ = ["plant_fisher_information", "plant_profile_likelihood"]
