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
more output streams' channels** (via :class:`PlantObservable`), optionally
alongside **assembled-state initial conditions** (``free_ic``, naming
``(unit, species)`` slots -- the plant analogue of the reactor free-IC hook), and
over a **joint multi-batch fit** (several plant runs from different initial states
sharing the parameters). The reactor ``predictive_band`` is not yet wired for
plants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquakin.core.hints import did_you_mean
from aquakin.integrate._common import DifferentiationConfig
from aquakin.integrate.calibrate import (
    CalibrationResult,
    FreeICConfig,
    LaplaceConfig,
    OptimizerConfig,
    _build_loss,
    _build_objective,
    _build_residual,
    _CalibrationProblem,
    _check_start_gradient,
    _FitConfig,
    _free_ic_fields,
    _laplace_posterior,
    _optimizer_bounds,
    _resolve_laplace,
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
    channels: tuple | None = None


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


def _normalize_free_ic(free_ic) -> list[tuple]:
    """Coerce the free-initial-condition specification to ``(unit, species)``
    pairs. Accepts ``"unit.species"`` strings and ``(unit, species)`` tuples."""
    if not free_ic:
        return []
    out: list[tuple] = []
    for spec in free_ic:
        if isinstance(spec, str):
            unit, _, species = spec.partition(".")
            if not species:
                raise ValueError(
                    f"free_ic entry {spec!r} must be 'unit.species' or (unit, species)."
                )
            out.append((unit, species))
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            out.append((spec[0], spec[1]))
        else:
            raise TypeError(
                f"free_ic entry must be 'unit.species' or (unit, species); got {spec!r}."
            )
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

    plant: Plant
    observables: tuple  # ((endpoint, channel_index_array), ...)
    y0: jnp.ndarray | None
    integrator: object
    diff: DifferentiationConfig
    time_unit: str | None
    use_c0_as_y0: bool = False

    def solve_trajectory(self, p, C0_k, tspan, tobs):
        # ``C0_k`` is the plant's assembled ``y0`` for this dataset -- a per-batch
        # state, or a base state with fitted free-IC slots set by the generic
        # layer -- when the fit carries one; otherwise it is a placeholder and the
        # plant is warm-started from the fixed ``self.y0``.
        y0 = C0_k if self.use_c0_as_y0 else self.y0
        sol = self.plant.solve(
            tspan,
            t_eval=tobs,
            params=p,
            y0=y0,
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

    def with_dtmax(self, dtmax) -> _PlantForwardModel:
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

    plant: Plant

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


def _resolve_endpoint_species(plant: Plant, target: str):
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
    plant: Plant,
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
    free_ic,
    ic_bounds,
    ic_prior_log_std,
    y0,
) -> tuple[_CalibrationProblem, tuple, bool]:
    """Validate + coerce the plant-calibration arguments into a
    ``_CalibrationProblem`` (one or several datasets). Returns the problem, the
    resolved observables ``((endpoint, channel_index_array), ...)`` for the forward
    model, and ``use_c0_as_y0`` (whether the forward model reads the per-dataset
    ``C0`` as the plant ``y0``)."""
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

    # --- Datasets: one plant run, or several sharing the parameter vector ---
    # A multi-batch fit is the plant analogue of the reactor's joint multi-batch:
    # the batches share the plant, parameters and prior and their data terms are
    # summed, differing in the per-dataset initial state ``y0`` (and its
    # observations / window). It is detected as a *list* of observation arrays.
    def _is_multi(x) -> bool:
        return (
            isinstance(x, (list, tuple))
            and len(x) > 0
            and isinstance(x[0], (list, tuple, np.ndarray, jnp.ndarray))
        )

    multi = _is_multi(observations)
    obs_list = list(observations) if multi else [observations]
    tobs_list = list(t_obs) if multi else [t_obs]
    n_datasets = len(obs_list)
    if len(tobs_list) != n_datasets:
        raise ValueError(
            "In a multi-batch fit, observations and t_obs must be lists of equal "
            f"length; got {n_datasets} and {len(tobs_list)}."
        )

    # --- Free initial conditions (single-batch only for now) -----------
    # Each free-IC spec names a (unit, species) slot of the plant's assembled
    # state ``y0`` to fit (in log space, box-bounded). Resolve it to a flat state
    # index: a concentration unit's sub-state IS its species vector, so the slot
    # is the unit's state offset plus the species position.
    free_ic_specs = _normalize_free_ic(free_ic)
    m_ic = len(free_ic_specs)
    if multi and m_ic:
        raise ValueError("free_ic is not yet supported together with a multi-batch plant fit.")
    ic_labels: list[str] = []
    ic_flat_idx: list[int] = []
    if m_ic:
        if not (0.0 < ic_bounds[0] < ic_bounds[1]):
            raise ValueError(f"ic_bounds must satisfy 0 < lo < hi; got {ic_bounds}.")
        plant._build_state_layout()
        for unit, species in free_ic_specs:
            u = plant._unit_or_raise(unit)
            if not plant._is_concentration_unit(u):
                raise KeyError(
                    f"free_ic unit '{unit}' has no per-species state (only "
                    f"concentration units -- CSTRs, the digester -- support "
                    f"free ICs); read it as a stream instead."
                )
            if species not in u.model.species_index:
                suffix = did_you_mean(species, list(u.model.species))
                raise KeyError(
                    f"Unknown free_ic species '{species}' in unit '{unit}' "
                    f"(model '{u.model.name}').{suffix}"
                )
            start = plant._state_layout[unit][0]
            ic_flat_idx.append(start + u.model.species_index[species])
            ic_labels.append(f"{unit}.{species}")

    # Per-dataset initial states: a multi-batch fit differs in ``y0``, so it must
    # pass a list of concrete states, one per dataset; a single fit takes one
    # ``y0`` (or ``None`` -> the plant default / fixed warm start).
    if multi:
        if not isinstance(y0, (list, tuple)) or len(y0) != n_datasets:
            n_y0 = len(y0) if isinstance(y0, (list, tuple)) else "a single value"
            raise ValueError(
                "A multi-batch plant fit needs y0 as a list of initial states, one "
                f"per dataset; got {n_y0} for {n_datasets} datasets."
            )
        y0_list = [jnp.asarray(v) for v in y0]
    else:
        y0_list = [y0]

    # Per-dataset spans / sigmas (a single value broadcasts to every dataset).
    if t_span is not None and multi and isinstance(t_span[0], (list, tuple)):
        span_list = list(t_span)
    else:
        span_list = [t_span] * n_datasets
    if isinstance(sigma, (list, tuple)):
        if len(sigma) != n_datasets:
            raise ValueError(
                f"sigma list has {len(sigma)} entries but there are {n_datasets} datasets."
            )
        sigma_list = list(sigma)
    else:
        sigma_list = [sigma] * n_datasets

    # ``C0_base`` carries a real assembled ``y0`` when the fit needs a per-dataset
    # or fitted state -- a multi-batch fit (each dataset's own ``y0``) or a free-IC
    # fit (the base ``y0`` whose slots are overridden). A plain single fit leaves
    # it an unused placeholder and the forward model uses its fixed ``y0``.
    if m_ic:
        y0_base = plant.initial_state() if y0 is None else jnp.asarray(y0)
        ic_species_idx = jnp.asarray(ic_flat_idx, dtype=int)
        vals = np.clip(np.asarray(y0_base)[ic_flat_idx], ic_bounds[0], ic_bounds[1])
        ic_center_full = jnp.asarray(np.log(vals))
    else:
        ic_species_idx = jnp.asarray([], dtype=int)
        ic_center_full = jnp.zeros(0)

    datasets = []
    for ds in range(n_datasets):
        tobs_i = jnp.asarray(tobs_list[ds])
        if tobs_i.ndim != 1 or tobs_i.shape[0] < 1:
            raise ValueError(
                f"dataset {ds}: t_obs must be a non-empty 1-D array, got shape {tobs_i.shape}."
            )
        if tobs_i.shape[0] > 1 and not bool(jnp.all(jnp.diff(tobs_i) > 0)):
            raise ValueError(f"dataset {ds}: t_obs must be strictly ascending.")
        obs_i = jnp.asarray(obs_list[ds])
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
                f"{n_observed} channels were specified across the observable(s)."
            )
        span_i = span_list[ds]
        if span_i is None:
            span_i = (float(tobs_i[0]), float(tobs_i[-1]))
        tspan_i = (float(span_i[0]), float(span_i[1]))
        sig_i = jnp.asarray(sigma_list[ds]) if sigma_list[ds] is not None else None
        if multi:
            C0_i = y0_list[ds]
        elif m_ic:
            C0_i = y0_base
        else:
            C0_i = jnp.zeros(1)
        datasets.append(
            (
                C0_i,
                tobs_i,
                tspan_i,
                _build_loss(loss, obs_i, sig_i),
                _build_residual(loss, obs_i, sig_i),
            )
        )

    C0_base = tuple(d[0] for d in datasets)
    dataset_static = [(d[1], d[2], d[3], d[4]) for d in datasets]
    # The forward model reads ``C0`` as the plant ``y0`` when it carries a real
    # state (multi-batch or free-IC); otherwise it uses its fixed ``y0``.
    use_c0_as_y0 = multi or bool(m_ic)

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
        dataset_static=dataset_static,
        C0_base=C0_base,
        n_datasets=n_datasets,
        obs_species_indices=obs_species_indices,
        n_observed=n_observed,
        active_priors=active_priors,
        prior_mean=prior_mean,
        prior_std=prior_std,
        prior_mask=prior_mask,
        has_priors=bool(active_priors),
        free_ic=ic_labels,
        m_ic=m_ic,
        ic_species_idx=ic_species_idx,
        ic_center_full=ic_center_full,
        ic_prior_log_std=ic_prior_log_std,
        ic_bounds=ic_bounds,
    )
    return problem, resolved_observables, use_c0_as_y0


# --- Public entry point ------------------------------------------------


def calibrate_plant(
    plant: Plant,
    observations,
    t_obs,
    free_params: list,
    *,
    target: str = "effluent",
    observed_channels: list | None = None,
    observables: list | None = None,
    t_span: tuple | None = None,
    y0: jnp.ndarray | None = None,
    params: jnp.ndarray | None = None,
    transforms: dict | None = None,
    free_ic: FreeICConfig | None = None,
    time_unit: str | None = None,
    loss: str = "mse",
    sigma: jnp.ndarray | None = None,
    priors: dict | None = None,
    use_priors: bool = True,
    optimizer: OptimizerConfig = OptimizerConfig(),
    laplace: bool | LaplaceConfig = False,
    check_finite: bool = True,
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
    observations : array-like or list of array-like
        Observed values, shape ``(n_t,)`` for a single channel or
        ``(n_t, n_channels)``. Pass a **list** of such arrays for a joint
        multi-batch fit: each entry is one run of the plant, the batches share the
        parameter vector and prior and their data terms are summed. ``t_obs`` and
        ``y0`` must then be matching lists (one per batch).
    t_obs : array-like or list of array-like
        Observation times, shape ``(n_t,)``, in the plant's time unit (or
        ``time_unit`` if given). The solve integrates over ``t_span`` and reports
        at ``t_obs``. A list in multi-batch mode (the batches may differ in ``n_t``).
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
    t_span : tuple or list of tuple, optional
        ``(t0, t1)`` integration window. Defaults to ``(t_obs[0], t_obs[-1])``. A
        list of windows (one per batch) in multi-batch mode; a single window
        broadcasts to every batch.
    y0 : jnp.ndarray or list of jnp.ndarray, optional
        Warm-start plant state (e.g. ``bsm2_warm_start(plant)`` or a saved steady
        state). Strongly recommended for a stiff plant. In multi-batch mode this
        is **required** and must be a list of initial states, one per batch (the
        batches differ in their initial state).
    params : jnp.ndarray, optional
        Starting parameter vector. Defaults to :meth:`Plant.default_parameters`.
    transforms : dict, optional
        Per-parameter transform override (``"positive_log"`` / ``"logit"`` /
        ``"none"``). Unspecified free params fall back to the parameter's
        model-declared transform.
    free_ic : FreeICConfig, optional
        Assembled-state slots to fit alongside the parameters. ``species`` are
        ``"unit.species"`` strings (or ``(unit, species)`` pairs) naming an initial
        concentration of a concentration unit (a CSTR, the digester); they are fit
        in log space, box-bounded by ``FreeICConfig.bounds``, with an optional
        log-space prior via ``prior_log_std`` pulling each toward its starting
        value (read from ``y0`` or the plant's default initial state). The fitted
        state is returned as ``result.C0_fitted[0]`` and the pools as
        ``result.ic_named[0]``. Default ``None`` (no free ICs).
    time_unit : str, optional
        Unit ``t_obs`` / ``t_span`` are expressed in; passed to ``plant.solve``.
    loss, sigma, priors, use_priors, optimizer, laplace, check_finite :
        As in :func:`aquakin.calibrate` (the shared machinery): ``optimizer`` is an
        :class:`~aquakin.OptimizerConfig` and ``laplace`` a ``bool`` or
        :class:`~aquakin.LaplaceConfig`. ``laplace`` defaults to ``False`` here (a
        plant Hessian is expensive). ``OptimizerConfig.param_halfwidth`` is not used
        by plant fits.
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
    Fits kinetic parameters (and, optionally, assembled-state initial conditions
    via ``free_ic``) against one or more output streams, over one run of the plant
    or several joined in a multi-batch fit (pass list-valued ``observations`` /
    ``t_obs`` / ``y0``). ``free_ic`` and multi-batch are not yet combinable in one
    call.
    """
    if integrator is None:
        from aquakin.plant.plant import IntegratorConfig

        integrator = IntegratorConfig()

    if not free_params:
        raise ValueError("free_params must be non-empty.")

    # Unpack the config objects into the internal scalar knobs.
    laplace_on, lap = _resolve_laplace(laplace)
    ic_species, ic_bounds, ic_prior_log_std = _free_ic_fields(free_ic)

    observable_specs = _normalize_observables(observables, target, observed_channels)
    problem, resolved_observables, use_c0_as_y0 = _resolve_plant_problem(
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
        free_ic=ic_species,
        ic_bounds=ic_bounds,
        ic_prior_log_std=ic_prior_log_std,
        y0=y0,
    )

    # Label the (fixed, single-solve) problem's gradient path from the plant
    # DifferentiationConfig so the generic layer's finite-path reasoning is right
    # (the config owns the method -> backend decode).
    diff.validated()
    gradient = diff.gradient_backend()
    cfg = _FitConfig(
        gradient=gradient,
        ad_mode="reverse",
        check_finite=check_finite,
        stable_adjoint_max_steps=0,
        stable_adjoint_low_memory=False,
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
        compiled_cache=None,
    )
    fm = _PlantForwardModel(
        plant=plant,
        observables=resolved_observables,
        # The fixed warm-start ``y0`` is only used when ``C0`` is not the state
        # (a plain single fit); multi-batch / free-IC thread the state via ``C0``.
        y0=None if (use_c0_as_y0 or y0 is None) else jnp.asarray(y0),
        integrator=integrator,
        diff=diff,
        time_unit=time_unit,
        use_c0_as_y0=use_c0_as_y0,
    )

    bundle = _build_objective(problem, fm, cfg)

    rate_theta0 = problem.rate_theta0()
    theta0 = jnp.concatenate([rate_theta0, problem.ic_center_full]) if problem.m_ic else rate_theta0
    opt_bounds = _optimizer_bounds(problem, rate_theta0)

    if cfg.check_finite:
        _check_start_gradient(cfg, bundle, theta0)

    result = _run_multistart(cfg, bundle, theta0, opt_bounds)

    theta_opt = jnp.asarray(result.x)
    physical_opt = problem.physical_from_theta(theta_opt[: problem.n_rate])
    ic_opt = theta_opt[problem.n_rate :]
    full_params = problem.p0_full.at[problem.free_indices].set(physical_opt)

    posterior_cov = None
    posterior_std_unconstrained = None
    params_named_std = None
    hessian_unconstrained = None
    if cfg.laplace:
        # Laplace covariance is over the rate parameters; the fitted ICs are held
        # at their MAP (the same convention as the reactor calibration).
        (
            posterior_cov,
            posterior_std_unconstrained,
            params_named_std,
            hessian_unconstrained,
        ) = _laplace_posterior(problem, fm, cfg, theta_opt[: problem.n_rate], ic_opt)

    # Fitted initial state (when free ICs are active): the base ``y0`` with the
    # fitted slots set, plus the fitted values by ``"unit.species"`` label.
    C0_fitted = None
    ic_named = None
    if problem.m_ic:
        ic_vals = np.exp(np.asarray(ic_opt))
        y0_fitted = problem.C0_base[0].at[problem.ic_species_idx].set(jnp.asarray(ic_vals))
        C0_fitted = [y0_fitted]
        ic_named = [{lbl: float(v) for lbl, v in zip(problem.free_ic, ic_vals)}]

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
