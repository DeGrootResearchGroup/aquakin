"""Plant calibration through the reactor-calibration forward-model seam.

:func:`aquakin.calibrate` fits a *reactor* -- a single kinetic model with a
species-vector state. A :class:`~aquakin.plant.Plant` has a different forward
contract (a flat assembled state across units, ``plant.solve(t_span, ...)``,
parameters concatenated across the unit models, and observations read back as a
*reconstructed stream* rather than the raw state), so it cannot be passed to
``calibrate`` directly.

The calibration machinery, however, was factored (see
``aquakin/integrate/calibrate.py``) so that everything except the forward solve
-- transforms, priors, the objective / residual assembly, multistart, and the
Laplace posterior -- is generic over "a thing that turns a parameter vector into
an observed-quantity trajectory". This module supplies the *plant* end of that
contract:

- :class:`_PlantForwardModel` -- the plant analogue of ``_ReactorForwardModel``:
  ``solve_trajectory`` runs ``plant.solve`` (the cap-free stable adjoint by
  default, so a stiff-plant reverse gradient is finite) and reads back the target
  stream's concentrations.
- :class:`_PlantParamNamespace` -- adapts the plant's by-name parameter surface
  (``parameter_index`` / ``default_parameters`` and the per-model transforms /
  priors) to the small interface ``_CalibrationProblem`` expects of a ``model``.

The generic ``_build_objective`` / ``_run_multistart`` / ``_laplace_posterior``
are then reused unchanged. This version fits **kinetic parameters against one or
more output streams' channels** (via :class:`PlantObservable`); per-dataset free
initial conditions, multi-batch joint fits and the reactor ``predictive_band``
are not yet wired for plants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

from aquakin.core.hints import did_you_mean
from aquakin.integrate._common import DifferentiationConfig
from aquakin.integrate.calibrate import (
    CalibrationResult,
    _build_loss,
    _build_objective,
    _build_residual,
    _CalibrationProblem,
    _check_start_gradient,
    _FitConfig,
    _laplace_posterior,
    _optimizer_bounds,
    _run_multistart,
)

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.plant.plant import Plant

# ``IntegratorConfig`` lives in ``plant.plant``, which imports *this* module at
# class-definition time -- so it is imported lazily (at call time) to avoid the
# import cycle.


# --- Public observable specification -----------------------------------


@dataclass(frozen=True)
class PlantObservable:
    """One calibration observable: some channels of a plant output stream.

    ``stream`` is a registered stream name (``"effluent"``, ``"ras"``, ...; see
    :meth:`Plant.list_streams`) or a ``"unit.port"`` / unit name. ``channels`` are
    the species of that stream's model to compare against data, in order; ``None``
    observes every stream species. Pass a list of these as ``observables=`` to
    :meth:`Plant.calibrate` to fit against several streams at once (e.g. the
    effluent ammonia *and* nitrate); the observation columns then run in the order
    given, channels within a stream first."""

    stream: str
    channels: Optional[tuple] = None


def _normalize_observables(observables, target, observed_channels) -> list[PlantObservable]:
    """Coerce the observable specification to a list of :class:`PlantObservable`.

    ``observables`` (when given) wins and accepts a :class:`PlantObservable`, a
    ``{"stream": ..., "channels": ...}`` dict, a ``(stream, channels)`` tuple, or
    a bare stream-name string. Otherwise the single-stream ``target`` /
    ``observed_channels`` form is used."""
    if observables is None:
        return [PlantObservable(target, observed_channels)]
    out: list[PlantObservable] = []
    for obs in observables:
        if isinstance(obs, PlantObservable):
            out.append(obs)
        elif isinstance(obs, str):
            out.append(PlantObservable(obs))
        elif isinstance(obs, dict):
            out.append(PlantObservable(obs["stream"], obs.get("channels")))
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            out.append(PlantObservable(obs[0], obs[1]))
        else:
            raise TypeError(
                "each observable must be a PlantObservable, a stream name, a "
                "{'stream', 'channels'} dict, or a (stream, channels) pair; got "
                f"{obs!r}."
            )
    if not out:
        raise ValueError("observables must be non-empty.")
    return out


# --- Forward-model seam (the plant end of the calibrate contract) ------


@dataclass
class _PlantForwardModel:
    """Plant forward solve for calibration.

    ``solve_trajectory`` integrates the plant once and reconstructs each
    observable's stream, slicing its observed channels and concatenating them into
    the ``(n_t, n_observed)`` matrix the loss compares against data. Owning the
    extraction here (rather than a single-stream slice in the generic layer) is
    what lets one fit target several streams at once. The reverse gradient flows
    from the streams, back through the reconstructed states, through
    ``plant.solve``'s discrete adjoint, to the parameters.
    """

    plant: "Plant"
    observables: tuple  # ((endpoint, channel_index_array), ...)
    y0: Optional[jnp.ndarray]
    integrator: object
    diff: DifferentiationConfig
    time_unit: Optional[str]

    def solve_trajectory(self, p, C0_k, tspan, tobs):
        # ``C0_k`` (the reactor species-IC hook) is unused: a plant is warm-started
        # from its own assembled ``y0``, not a per-species initial vector.
        sol = self.plant.solve(
            tspan,
            t_eval=tobs,
            params=p,
            y0=self.y0,
            integrator=self.integrator,
            diff=self.diff,
            time_unit=self.time_unit,
        )
        cols = [
            self.plant.stream(sol, endpoint, p).C[:, ch_idx]
            for endpoint, ch_idx in self.observables
        ]
        return jnp.concatenate(cols, axis=1)  # (n_t, n_observed)

    def forward_capable(self) -> bool:
        # Forward-mode AD through the whole plant solve is not wired for the
        # Gauss-Newton Jacobian yet; the reverse stable adjoint is the finite path.
        return False

    def with_dtmax(self, dtmax) -> "_PlantForwardModel":
        """A clone whose integrator caps the step at ``dtmax`` (the tighter solve
        the Laplace Hessian may want). ``None`` reuses ``self``."""
        if dtmax is None or dtmax == getattr(self.integrator, "dtmax", None):
            return self
        return replace(self, integrator=replace(self.integrator, dtmax=dtmax))


# --- Parameter-namespace adapter ---------------------------------------


@dataclass
class _PlantParamNamespace:
    """Adapt a plant's by-name parameter surface to the small ``model`` interface
    ``_CalibrationProblem`` needs (``param_index`` for the free-index lookup, plus
    the per-model transforms / priors so a rate constant is fit in log space and a
    model-declared prior flows through, exactly as for a reactor's model)."""

    plant: "Plant"

    def __post_init__(self):
        self.param_index = {
            name: self.plant.parameter_index(name) for name in self.plant.parameter_names()
        }
        # Pull each parameter's declared transform / prior from its owning model.
        # A plant parameter is addressed ``"<model>.<param>"``; the per-model
        # tables are keyed by the model's own (namespaced) ``<param>``.
        self.parameter_transforms: dict[str, str] = {}
        self.parameter_priors: dict[str, tuple[float, float]] = {}
        for net in self.plant._ordered_models():
            for pname in net.param_index:
                key = f"{net.name}.{pname}"
                if pname in net.parameter_transforms:
                    self.parameter_transforms[key] = net.parameter_transforms[pname]
                if pname in net.parameter_priors:
                    self.parameter_priors[key] = net.parameter_priors[pname]

    @property
    def parameters(self) -> list[str]:
        return list(self.param_index)

    def default_parameters(self) -> jnp.ndarray:
        return self.plant.default_parameters()


# --- Problem resolution ------------------------------------------------


def _resolve_endpoint_species(plant: "Plant", target: str):
    """Resolve a stream ``target`` (semantic name or ``"unit.port"``) to
    ``(endpoint, stream_model)`` -- the string ``plant.stream`` accepts and the
    model whose species label the reconstructed concentration columns."""
    resolved = plant.named_streams.get(target)
    if resolved is None and "." not in target and target not in plant.units:
        suffix = did_you_mean(target, list(plant.named_streams) + list(plant.units))
        raise KeyError(
            f"Unknown calibration target '{target}'. Pass a registered stream "
            f"name (plant.list_streams()): {sorted(plant.named_streams)}, or a "
            f"'unit.port' / unit name (plant.list_units()).{suffix}"
        )
    endpoint = resolved if resolved is not None else target
    unit = endpoint.split(".")[0]
    if unit not in plant.units:
        suffix = did_you_mean(unit, list(plant.units))
        raise KeyError(f"Unknown unit '{unit}' in target '{target}'.{suffix}")
    return endpoint, plant.units[unit].model


def _resolve_plant_problem(
    plant: "Plant",
    observations,
    t_obs,
    free_params,
    *,
    observables,
    t_span,
    params,
    transforms,
    use_priors,
    priors,
    loss,
    sigma,
) -> tuple[_CalibrationProblem, tuple]:
    """Validate + coerce the plant-calibration arguments into a
    ``_CalibrationProblem`` (single dataset, no free ICs). Returns the problem and
    the resolved observables ``((endpoint, channel_index_array), ...)`` for the
    forward model."""
    ns = _PlantParamNamespace(plant)
    for name in free_params:
        if name not in ns.param_index:
            suffix = did_you_mean(name, list(ns.param_index))
            raise KeyError(
                f"Unknown plant parameter '{name}'. Keys are '<model>.<param>' "
                f"(plant.parameter_names()).{suffix}"
            )

    # Resolve each observable -> (endpoint, channel indices). The observation
    # columns run in observable order, channels within a stream first; the forward
    # model reconstructs and concatenates them in the same order.
    resolved_observables: list[tuple] = []
    n_observed = 0
    for observable in observables:
        endpoint, stream_model = _resolve_endpoint_species(plant, observable.stream)
        if observable.channels is None:
            ch_idx = jnp.arange(stream_model.n_species)
            n_observed += int(stream_model.n_species)
        else:
            for s in observable.channels:
                if s not in stream_model.species_index:
                    suffix = did_you_mean(s, list(stream_model.species))
                    raise KeyError(
                        f"Unknown observed channel '{s}' in stream "
                        f"'{observable.stream}' (model '{stream_model.name}').{suffix}"
                    )
            ch_idx = jnp.asarray([stream_model.species_index[s] for s in observable.channels])
            n_observed += len(observable.channels)
        resolved_observables.append((endpoint, ch_idx))
    resolved_observables = tuple(resolved_observables)
    # The forward model returns exactly the observed columns, so the generic
    # per-dataset slice is the identity.
    obs_species_indices = jnp.arange(n_observed)

    # Resolve transforms per free param (explicit override wins over the
    # parameter's model-declared transform, else "none").
    transforms = dict(transforms or {})
    resolved_transforms: list[str] = []
    for name in free_params:
        t = transforms.get(name)
        if t is None:
            t = ns.parameter_transforms.get(name, "none")
        resolved_transforms.append(t)

    p0_full = jnp.asarray(params) if params is not None else plant.default_parameters()
    free_indices = jnp.asarray([ns.param_index[n] for n in free_params])

    for name, t in zip(free_params, resolved_transforms):
        v = float(p0_full[ns.param_index[name]])
        if t == "positive_log" and v <= 0.0:
            raise ValueError(
                f"Parameter '{name}' has transform 'positive_log' but initial value {v} <= 0."
            )
        if t == "logit" and not (0.0 < v < 1.0):
            raise ValueError(
                f"Parameter '{name}' has transform 'logit' but initial value {v} is not in (0, 1)."
            )

    # Single dataset (multi-batch plant fits are a later extension).
    tobs = jnp.asarray(t_obs)
    if tobs.ndim != 1 or tobs.shape[0] < 1:
        raise ValueError(f"t_obs must be a non-empty 1-D array, got shape {tobs.shape}.")
    if tobs.shape[0] > 1 and not bool(jnp.all(jnp.diff(tobs) > 0)):
        raise ValueError("t_obs must be strictly ascending.")
    obs = jnp.asarray(observations)
    if obs.ndim == 1:
        obs = obs[:, None]
    if obs.shape[0] != tobs.shape[0]:
        raise ValueError(
            f"observations has {obs.shape[0]} rows but t_obs has {tobs.shape[0]} entries."
        )
    if obs.shape[1] != n_observed:
        raise ValueError(
            f"observations has {obs.shape[1]} columns but {n_observed} channels "
            f"were specified across the observable(s)."
        )
    if t_span is None:
        t_span = (float(tobs[0]), float(tobs[-1]))
    tspan = (float(t_span[0]), float(t_span[1]))
    sig_arr = jnp.asarray(sigma) if sigma is not None else None

    # ``C0_base`` is a placeholder: the plant is warm-started from ``y0``, so the
    # reactor species-IC hook is unused (and ``m_ic == 0`` disables IC fitting).
    C0_placeholder = jnp.zeros(1)
    datasets = [
        (
            C0_placeholder,
            tobs,
            tspan,
            _build_loss(loss, obs, sig_arr),
            _build_residual(loss, obs, sig_arr),
        )
    ]

    # Priors: model-declared (use_priors) then explicit overrides.
    active_priors: dict[str, tuple[float, float]] = {}
    if use_priors:
        for name in free_params:
            if name in ns.parameter_priors:
                active_priors[name] = ns.parameter_priors[name]
    if priors:
        for name, ms in priors.items():
            if name in free_params:
                active_priors[name] = (float(ms[0]), float(ms[1]))
    prior_mean = jnp.asarray([active_priors.get(n, (0.0, 1.0))[0] for n in free_params])
    prior_std = jnp.asarray([active_priors.get(n, (0.0, 1.0))[1] for n in free_params])
    prior_mask = jnp.asarray([1.0 if n in active_priors else 0.0 for n in free_params])

    problem = _CalibrationProblem(
        model=ns,
        free_params=list(free_params),
        free_indices=free_indices,
        transforms=resolved_transforms,
        n_rate=len(free_params),
        p0_full=p0_full,
        param_halfwidth=None,
        datasets=datasets,
        dataset_static=[(tobs, tspan, datasets[0][3], datasets[0][4])],
        C0_base=(C0_placeholder,),
        n_datasets=1,
        obs_species_indices=obs_species_indices,
        n_observed=n_observed,
        active_priors=active_priors,
        prior_mean=prior_mean,
        prior_std=prior_std,
        prior_mask=prior_mask,
        has_priors=bool(active_priors),
        free_ic=[],
        m_ic=0,
        ic_species_idx=jnp.asarray([], dtype=int),
        ic_center_full=jnp.zeros(0),
        ic_prior_log_std=None,
        ic_bounds=(1e-3, 1e4),
    )
    return problem, resolved_observables


# --- Public entry point ------------------------------------------------


def calibrate_plant(
    plant: "Plant",
    observations,
    t_obs,
    free_params: list,
    *,
    target: str = "effluent",
    observed_channels: Optional[list] = None,
    observables: Optional[list] = None,
    t_span: Optional[tuple] = None,
    y0: Optional[jnp.ndarray] = None,
    params: Optional[jnp.ndarray] = None,
    transforms: Optional[dict] = None,
    time_unit: Optional[str] = None,
    loss: str = "mse",
    sigma: Optional[jnp.ndarray] = None,
    priors: Optional[dict] = None,
    use_priors: bool = True,
    optimizer: str = "lbfgsb",
    n_starts: int = 1,
    jitter: float = 0.5,
    jitter_schedule: Optional[tuple] = None,
    seed: int = 0,
    max_iter: int = 500,
    tol: float = 1e-6,
    check_finite: bool = True,
    laplace: bool = False,
    laplace_method: str = "fd",
    laplace_ridge: float = 1e-6,
    laplace_eig_keep: float = 1e-2,
    laplace_fd_step: float = 1e-3,
    laplace_dtmax: Optional[float] = None,
    integrator=None,
    diff: DifferentiationConfig = DifferentiationConfig(),
) -> CalibrationResult:
    """MAP-calibrate a plant's parameters against an output stream.

    Bound onto :class:`~aquakin.plant.Plant` as ``plant.calibrate(...)``. The
    plant analogue of :func:`aquakin.calibrate`: it fits plant parameters (by
    ``"<model>.<param>"`` name -- see :meth:`Plant.parameter_names`) so a target
    stream's channels match ``observations``. The forward solve is the cap-free
    stable adjoint by default, so a reverse-mode gradient through a stiff plant is
    finite with no ``dtmax`` to tune.

    Parameters
    ----------
    plant : Plant
        The plant to calibrate. It must already have its influent(s) added.
    observations : array-like
        Observed values, shape ``(n_t,)`` for a single channel or
        ``(n_t, n_channels)``.
    t_obs : array-like
        Observation times, shape ``(n_t,)``, in the plant's time unit (or
        ``time_unit`` if given). The solve integrates over ``t_span`` and reports
        at ``t_obs``.
    free_params : list of str
        Plant parameter names to calibrate (``"<model>.<param>"``). Others fixed.
    target : str, optional
        The single output stream to compare against -- a registered stream name
        (``"effluent"``, ``"ras"``, ...; see :meth:`Plant.list_streams`) or a
        ``"unit.port"`` / unit name. Default ``"effluent"``. Ignored when
        ``observables`` is given.
    observed_channels : list of str, optional
        Species of the ``target`` stream's model that ``observations`` columns
        correspond to. ``None`` observes every stream species. Ignored when
        ``observables`` is given.
    observables : list, optional
        Fit against **several streams at once**. Each entry is a
        :class:`PlantObservable` (``stream`` + ``channels``), a ``{"stream": ...,
        "channels": ...}`` dict, a ``(stream, channels)`` pair, or a bare stream
        name. The ``observations`` columns then run in observable order, channels
        within a stream first -- e.g.
        ``observables=[PlantObservable("effluent", ["SNH", "SNO"]),
        PlantObservable("wastage", ["XS"])]`` expects 3 columns. Overrides
        ``target`` / ``observed_channels``.
    t_span : tuple, optional
        ``(t0, t1)`` integration window. Defaults to ``(t_obs[0], t_obs[-1])``.
    y0 : jnp.ndarray, optional
        Warm-start plant state (e.g. ``bsm2_warm_start(plant)`` or a saved steady
        state). Strongly recommended for a stiff plant.
    params : jnp.ndarray, optional
        Starting parameter vector. Defaults to :meth:`Plant.default_parameters`.
    transforms : dict, optional
        Per-parameter transform override (``"positive_log"`` / ``"logit"`` /
        ``"none"``). Unspecified free params fall back to the parameter's
        model-declared transform.
    time_unit : str, optional
        Unit ``t_obs`` / ``t_span`` are expressed in; passed to ``plant.solve``.
    loss, sigma, priors, use_priors, optimizer, n_starts, jitter,
    jitter_schedule, seed, max_iter, tol, check_finite, laplace,
    laplace_method, laplace_ridge, laplace_eig_keep, laplace_fd_step,
    laplace_dtmax :
        As in :func:`aquakin.calibrate` (the shared machinery). ``laplace``
        defaults to ``False`` here (a plant Hessian is expensive).
    integrator : IntegratorConfig, optional
        Plant integrator configuration passed to ``plant.solve``.
    diff : DifferentiationConfig, optional
        How the gradient flows through ``plant.solve``. Default
        ``mode='reverse', method='stable'`` -- the cap-free discrete adjoint.

    Returns
    -------
    CalibrationResult
        Same result type as :func:`aquakin.calibrate`. ``predictive_band`` (which
        takes a reactor) does not apply to a plant fit.

    Notes
    -----
    Fits kinetic parameters against one or more output streams. Per-dataset free
    initial conditions and multi-batch joint fits are not yet supported.
    """
    if integrator is None:
        from aquakin.plant.plant import IntegratorConfig

        integrator = IntegratorConfig()

    if not free_params:
        raise ValueError("free_params must be non-empty.")

    observable_specs = _normalize_observables(observables, target, observed_channels)
    problem, resolved_observables = _resolve_plant_problem(
        plant,
        observations,
        t_obs,
        free_params,
        observables=observable_specs,
        t_span=t_span,
        params=params,
        transforms=transforms,
        use_priors=use_priors,
        priors=priors,
        loss=loss,
        sigma=sigma,
    )

    # Label the (fixed, single-solve) problem's gradient path from the plant
    # DifferentiationConfig so the generic layer's finite-path reasoning is right.
    gradient = "stable_adjoint" if diff.method == "stable" else "jax_adjoint"
    cfg = _FitConfig(
        gradient=gradient,
        ad_mode="reverse",
        check_finite=check_finite,
        stable_adjoint_max_steps=0,
        stable_adjoint_low_memory=False,
        optimizer=optimizer,
        max_iter=max_iter,
        tol=tol,
        n_starts=n_starts,
        jitter=jitter,
        jitter_schedule=jitter_schedule,
        seed=seed,
        laplace=laplace,
        laplace_method=laplace_method,
        laplace_ridge=laplace_ridge,
        laplace_eig_keep=laplace_eig_keep,
        laplace_fd_step=laplace_fd_step,
        laplace_dtmax=laplace_dtmax,
        compiled_cache=None,
    )
    fm = _PlantForwardModel(
        plant=plant,
        observables=resolved_observables,
        y0=None if y0 is None else jnp.asarray(y0),
        integrator=integrator,
        diff=diff,
        time_unit=time_unit,
    )

    bundle = _build_objective(problem, fm, cfg)

    rate_theta0 = problem.rate_theta0()
    theta0 = rate_theta0  # no free-IC block in v1
    opt_bounds = _optimizer_bounds(problem, rate_theta0)

    if cfg.check_finite:
        _check_start_gradient(cfg, bundle, theta0)

    result = _run_multistart(cfg, bundle, theta0, opt_bounds)

    theta_opt = jnp.asarray(result.x)
    physical_opt = problem.physical_from_theta(theta_opt[: problem.n_rate])
    full_params = problem.p0_full.at[problem.free_indices].set(physical_opt)

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
        ) = _laplace_posterior(problem, fm, cfg, theta_opt[: problem.n_rate], jnp.zeros(0))

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
        C0_fitted=None,
        ic_named=None,
    )
