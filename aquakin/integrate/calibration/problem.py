"""The calibration problem: the forward-model seam (:class:`_ReactorForwardModel`),
the resolved static problem (:class:`_CalibrationProblem`) and its resolver."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import diffrax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._common import native_time_factor
from aquakin.integrate._transforms import from_unconstrained, to_unconstrained
from aquakin.integrate.calibration.coords import (
    _resolve_ic_coords,
    _simplex_pools,
)
from aquakin.integrate.calibration.loss import (
    _build_loss,
    _build_residual,
    _coerce_sigma,
    _data_mean_square,
    _is_sigma_estimate,
    _n_obs_per_output,
    _observed_counts,
    _pooled_sse,
    _sigma_per_dataset,
)
from aquakin.integrate.calibration.observation import observation_size, observe

if TYPE_CHECKING:
    from aquakin.integrate._common import Reactor


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
    the reactor's model + single-location conditions, with the ESDIRK the
    reactor's :class:`~aquakin.IntegratorConfig` selects (its ``order``, or an
    explicit ``solver``, and its ``factormax`` step-growth cap); otherwise it calls the reactor's own diffrax ``solve``.
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
                order=self.reactor.order,
                solver=self.reactor.solver,
                factormax=self.reactor.factormax,
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
    # The observations' mean square for loss="mse" (None otherwise), which
    # bounds how far the optimiser's objective is scaled up (see
    # :func:`_objective_scale`).
    data_mean_square: float | None = None
    # What the information criteria read (see ``calibration/criteria.py``):
    # whether the data term is a Gaussian negative log-likelihood
    # (``loss="nll"``), the number of observed (non-NaN) entries over every
    # dataset, and a fingerprint of the observations and their times alone, so
    # fits of different models to the same data can be recognized as such.
    is_likelihood: bool = False
    n_observations: int = 0
    observations_key: str = ""
    # The ``sigma`` the fit was given (in model units), as given: an array, a
    # per-dataset list, ``"estimate"`` or ``None``. Read only by
    # model-discrimination design for the fit's noise level; the loss closures
    # hold their own copy, so it is not part of :meth:`struct_key`.
    fit_sigma: object = None

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
            [from_unconstrained(rate_thetas[i], self.transforms[i]) for i in range(self.n_rate)]
        )

    def rate_theta0(self) -> jnp.ndarray:
        """The unconstrained start vector for the rate block (from ``p0_full``)."""
        return jnp.stack(
            [
                to_unconstrained(self.p0_full[self.model.param_index[name]], t)
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


def _observation_record(loss: str, n_seen_per_output, obs_parts) -> dict:
    """The :class:`_CalibrationProblem` fields the information criteria read.

    Shared by the reactor and plant resolvers.

    Parameters
    ----------
    loss : str
        The loss; only ``"nll"`` is a likelihood.
    n_seen_per_output : array-like of int
        Observed (non-``NaN``) entries of each output, over every dataset.
    obs_parts : list of array-like
        Each dataset's observations and observation times, as the user's data
        (no ``sigma`` and no loss), so fits of different models to the same
        data share the fingerprint.

    Returns
    -------
    dict
        ``is_likelihood``, ``n_observations`` and ``observations_key``.
    """
    return {
        "is_likelihood": loss == "nll",
        "n_observations": int(np.sum(n_seen_per_output)),
        "observations_key": _array_digest(*obs_parts),
    }


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


def _check_start_in_domain(free_params, transforms, p0_full, param_index) -> None:
    """Refuse a starting value outside its transform's domain (``<= 0`` for
    ``positive_log``, outside ``(0, 1)`` for ``logit``)."""
    for name, t in zip(free_params, transforms):
        v = float(p0_full[param_index[name]])
        if t == "positive_log" and v <= 0.0:
            raise ValueError(
                f"Parameter '{name}' has transform 'positive_log' but initial value {v} <= 0."
            )
        if t == "logit" and not (0.0 < v < 1.0):
            raise ValueError(
                f"Parameter '{name}' has transform 'logit' but initial value {v} is not in (0, 1)."
            )


def _resolve_rate_params(model, free_params, transforms, initial_params):
    """``(transforms, p0_full, free_indices)`` for the free rate parameters.

    Each free parameter takes its explicit ``transforms`` entry, else the
    model-declared one, else ``"none"``; the start is ``initial_params`` (or the
    model defaults) and must lie in each transform's domain."""
    for name in free_params:
        if name not in model.param_index:
            raise KeyError(f"Unknown parameter '{name}'. Available: {model.parameters}")

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
    _check_start_in_domain(free_params, resolved_transforms, p0_full, model.param_index)
    return resolved_transforms, p0_full, free_indices


def _is_multi_dataset(x) -> bool:
    """Whether ``x`` is a list of per-dataset arrays (several batches sharing
    the parameter vector) rather than one dataset's array."""
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and isinstance(x[0], (list, tuple, np.ndarray, jnp.ndarray))
    )


def _observed_columns(model, observed_species, observation_fn, C0_first, p0_full, tobs_first):
    """``(obs_species_indices, n_observed, what)``: the species the observations
    are taken from, the number of observed columns, and how that number was set
    (for the column-count error)."""
    if observation_fn is not None:
        if observed_species is not None:
            raise ValueError("Pass observed_species or observation_fn, not both.")
        n_observed = observation_size(
            observation_fn, C0_first, p0_full, t0=float(np.asarray(tobs_first).reshape(-1)[0])
        )
        return jnp.arange(model.n_species), n_observed, "outputs were returned by observation_fn"
    if observed_species is None:
        return jnp.arange(model.n_species), model.n_species, "species were specified"
    obs_species_indices = jnp.asarray([model.species_index[s] for s in observed_species])
    return obs_species_indices, len(observed_species), "species were specified"


class _ResolvedDatasets(NamedTuple):
    """What :func:`_resolve_datasets` returns."""

    datasets: list
    obs_times: tuple
    digest_parts: list
    n_seen_per_output: np.ndarray
    obs_species_indices: jnp.ndarray
    n_observed: int
    data_mean_square: float | None
    obs_parts: list


def _resolve_datasets(
    model,
    C0,
    observations,
    t_obs,
    *,
    sigma,
    loss,
    time_unit,
    observed_species,
    observation_fn,
    p0_full,
) -> _ResolvedDatasets:
    """Validate the dataset(s) and build each one's
    ``(C0, t_eval, t_span, loss, residual)`` tuple, with ``t_eval`` in the model's
    native time unit."""
    multi = _is_multi_dataset(C0)
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
    obs_species_indices, n_observed, what = _observed_columns(
        model, observed_species, observation_fn, C0_list[0], p0_full, tobs_list[0]
    )

    datasets = []
    tobs_user = []
    digest_parts = []
    obs_parts = []  # the observations and their times as given, for the criteria
    obs_seen = []
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
        obs_seen.append(obs_i)
        sig_arr = _coerce_sigma(sig_i)
        digest_parts += [obs_i, sig_arr, tobs_i, tobs_user[-1]]
        obs_parts += [obs_i, tobs_user[-1]]
        datasets.append(
            (
                C0_i,
                tobs_i,
                (0.0, float(tobs_i[-1])),
                _build_loss(loss, obs_i, sig_arr),
                _build_residual(loss, obs_i, sig_arr),
            )
        )
    return _ResolvedDatasets(
        datasets,
        tuple(tobs_user),
        digest_parts,
        n_seen_per_output,
        obs_species_indices,
        n_observed,
        _data_mean_square(loss, obs_seen),
        obs_parts,
    )


def _resolve_priors(declared_priors, free_params, priors, use_priors):
    """``(active_priors, prior_mean, prior_std, prior_mask)`` over the free
    parameters: the declared priors when ``use_priors``, each overridden by an
    explicit ``priors`` entry."""
    active_priors: dict[str, tuple[float, float]] = {}
    if use_priors:
        for name in free_params:
            if name in declared_priors:
                active_priors[name] = declared_priors[name]
    if priors:
        for name, ms in priors.items():
            if name in free_params:
                active_priors[name] = (float(ms[0]), float(ms[1]))
    prior_mean = jnp.asarray([active_priors.get(n, (0.0, 1.0))[0] for n in free_params])
    prior_std = jnp.asarray([active_priors.get(n, (0.0, 1.0))[1] for n in free_params])
    prior_mask = jnp.asarray([1.0 if n in active_priors else 0.0 for n in free_params])
    return active_priors, prior_mean, prior_std, prior_mask


def _resolve_free_ic(model, free_ic, ic_bounds, ic_fixed_total, C0s):
    """``(species, ic_species_idx, ic_bounds, ic_center_full, ic_fixed_total)``
    for the fitted initial pools (none when ``free_ic`` is empty); ``C0s`` are
    the per-dataset initial states the pools start from."""
    free_ic_list = list(free_ic or [])
    for s in free_ic_list:
        if s not in model.species_index:
            raise KeyError(f"Unknown free_ic species '{s}'. Available: {model.species}")
    ic_np_idx = [model.species_index[s] for s in free_ic_list]
    ic_species_idx = jnp.asarray(ic_np_idx, dtype=int)
    ic_bounds, ic_center_full, ic_fixed_total = _resolve_ic_coords(
        free_ic_list,
        ic_bounds,
        ic_fixed_total,
        [np.asarray(C0_i, dtype=float)[ic_np_idx] for C0_i in C0s],
    )
    return free_ic_list, ic_species_idx, ic_bounds, ic_center_full, ic_fixed_total


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
    resolved_transforms, p0_full, free_indices = _resolve_rate_params(
        model, free_params, transforms, initial_params
    )
    data = _resolve_datasets(
        model,
        C0,
        observations,
        t_obs,
        sigma=sigma,
        loss=loss,
        time_unit=time_unit,
        observed_species=observed_species,
        observation_fn=observation_fn,
        p0_full=p0_full,
    )
    datasets = data.datasets
    active_priors, prior_mean, prior_std, prior_mask = _resolve_priors(
        getattr(model, "parameter_priors", {}), free_params, priors, use_priors
    )
    C0_base = tuple(C0_i for (C0_i, *_rest) in datasets)
    free_ic_list, ic_species_idx, ic_bounds, ic_center_full, ic_fixed_total = _resolve_free_ic(
        model, free_ic, ic_bounds, ic_fixed_total, C0_base
    )

    sigma_estimate = _is_sigma_estimate(sigma)
    # Each output's noise is estimated from its observed (non-NaN) entries.
    n_obs_per_output = _n_obs_per_output(sigma_estimate, data.n_seen_per_output)
    rate_bounds = _resolve_rate_bounds(
        model, free_params, resolved_transforms, bounds, use_model_bounds
    )

    dataset_static = [
        (tobs_i, tspan_i, loss_fn_i, resid_fn_i)
        for (_C0, tobs_i, tspan_i, loss_fn_i, resid_fn_i) in datasets
    ]

    return _CalibrationProblem(
        model=model,
        free_params=list(free_params),
        free_indices=free_indices,
        transforms=resolved_transforms,
        n_rate=len(free_params),
        p0_full=p0_full,
        param_halfwidth=param_halfwidth,
        datasets=datasets,
        dataset_static=dataset_static,
        C0_base=C0_base,
        n_datasets=len(datasets),
        obs_species_indices=data.obs_species_indices,
        n_observed=data.n_observed,
        active_priors=active_priors,
        prior_mean=prior_mean,
        prior_std=prior_std,
        prior_mask=prior_mask,
        has_priors=bool(active_priors),
        free_ic=free_ic_list,
        m_ic=len(free_ic_list),
        ic_species_idx=ic_species_idx,
        ic_center_full=ic_center_full,
        ic_prior_log_std=ic_prior_log_std,
        ic_bounds=ic_bounds,
        observation_fn=observation_fn,
        obs_times=data.obs_times,
        sigma_estimate=sigma_estimate,
        n_obs_per_output=n_obs_per_output,
        ic_fixed_total=ic_fixed_total,
        data_key=(loss, _array_digest(*data.digest_parts)),
        rate_bounds=rate_bounds,
        data_mean_square=data.data_mean_square,
        fit_sigma=sigma,
        **_observation_record(loss, data.n_seen_per_output, data.obs_parts),
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
