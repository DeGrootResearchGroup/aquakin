"""Lightweight least-squares parameter fitting via JAX autodiff.

A box-constrained sum-of-squares fitter (SciPy L-BFGS-B over a Diffrax adjoint
gradient) for a single batch. It is the quick point-estimate sibling of the full
:func:`aquakin.calibrate` path (which adds parameter transforms, priors, a
Laplace posterior, free initial conditions, and multi-batch support); prefer
``calibrate`` for anything beyond a point fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from aquakin.integrate._common import Reactor, native_time_factor


@dataclass
class FitResult:
    """
    Result of :func:`fit`.

    Attributes
    ----------
    params : jnp.ndarray
        Full parameter vector after optimisation (fixed params unchanged).
    params_named : dict[str, float]
        Convenience mapping ``namespaced_name -> value`` for the free params.
    loss : float
        Final loss (sum of squared residuals).
    converged : bool
        Whether scipy's optimiser reported success.
    message : str
        Optimiser status message.
    n_iter : int
        Number of iterations taken.
    """

    params: jnp.ndarray
    params_named: dict[str, float]
    loss: float
    converged: bool
    message: str
    n_iter: int


def fit(
    reactor: Reactor,
    C0: jnp.ndarray,
    observations: jnp.ndarray,
    t_obs: jnp.ndarray,
    free_params: list[str],
    *,
    method: str = "adjoint",
    initial_params: jnp.ndarray | None = None,
    observed_species: list[str] | None = None,
    time_unit: str | None = None,
) -> FitResult:
    """
    Least-squares fit of selected parameters to time-series observations.

    This is the lightweight fitter: box-constrained sum-of-squares least squares
    on a single batch, with no parameter transforms, priors, Laplace posterior,
    free initial conditions, or multi-batch support. For anything beyond a quick
    point estimate, prefer :func:`aquakin.calibrate` (use
    ``calibrate(..., laplace=False)`` for a bare point fit) --- it is a strict
    superset of this function.

    .. note::
       The reported losses are **not comparable** between the two. ``fit``
       reports :attr:`FitResult.loss` as the **sum** of squared residuals
       (SciPy ``least_squares`` cost), whereas ``calibrate`` with ``loss="mse"``
       reports the **mean** squared error. The optima coincide (a positive
       constant factor does not move the minimiser); only the absolute loss
       value differs in scale.

    Parameters
    ----------
    reactor : BatchReactor
        The reactor to integrate. (Only batch reactors are supported here; the
        PFR case can be wrapped analogously.)
    C0 : jnp.ndarray
        Initial concentration vector.
    observations : jnp.ndarray
        Observed values. Shape ``(n_t, n_observed)`` or ``(n_t,)``.
    t_obs : jnp.ndarray
        Observation times, shape ``(n_t,)``. ``C0`` is taken to be the state
        at ``t = 0``; integration runs from ``0`` to ``t_obs[-1]`` and the
        solution is sampled at ``t_obs``. In the model's native time unit
        unless ``time_unit`` is given.
    free_params : list[str]
        Namespaced parameter names to optimise. Other parameters are held at
        their default (or ``initial_params``) values.
    method : str
        Currently only ``"adjoint"`` is supported, which uses Diffrax's
        recursive-checkpoint adjoint via :func:`jax.grad` and SciPy L-BFGS-B.
    initial_params : jnp.ndarray, optional
        Starting parameter vector. Defaults to ``reactor.model.default_parameters()``.
    observed_species : list[str], optional
        Species names corresponding to columns of ``observations``. If
        ``None``, ``observations`` is assumed to be over all species in
        model order.
    time_unit : str, optional
        The time unit ``t_obs`` is expressed in (``"s"``, ``"min"``, ``"h"``,
        ``"d"``), matching :meth:`BatchReactor.solve`. ``t_obs`` is converted
        into the model's native (rate-constant) time unit before the solve, so
        a user who standardises on e.g. hours can pass the same hour-valued
        ``t_obs`` here as to ``solve``. Default ``None`` interprets ``t_obs`` in
        the native unit. The fitted rate constants are always in native units.

    Returns
    -------
    FitResult

    Notes
    -----
    Box bounds are applied per parameter from each parameter's declared
    ``bounds``. A free parameter without declared bounds is left unbounded
    (``(-inf, +inf)``) while the others keep their boxes; a warning is emitted
    in that mixed case. If no free parameter has bounds, the solve is fully
    unconstrained.
    """
    if method != "adjoint":
        raise ValueError(f"Unknown fit method {method!r}; only 'adjoint' is supported.")
    if not free_params:
        raise ValueError("free_params must be non-empty.")

    model = reactor.model
    p0_full = (
        jnp.asarray(initial_params) if initial_params is not None else model.default_parameters()
    )

    free_indices = []
    for name in free_params:
        if name not in model.param_index:
            raise KeyError(f"Unknown parameter '{name}'. Available: {model.parameters}")
        free_indices.append(model.param_index[name])
    free_indices_arr = jnp.asarray(free_indices)

    observations = jnp.asarray(observations)
    t_obs = jnp.asarray(t_obs)
    if t_obs.ndim != 1 or t_obs.shape[0] < 1:
        raise ValueError(f"t_obs must be a non-empty 1-D array, got shape {t_obs.shape}.")
    if float(t_obs[0]) < 0.0:
        raise ValueError(f"t_obs must be non-negative; got t_obs[0] = {float(t_obs[0])}.")
    if t_obs.shape[0] > 1 and not bool(jnp.all(jnp.diff(t_obs) > 0)):
        raise ValueError("t_obs must be strictly ascending.")
    # Convert t_obs into the model's native (rate-constant) time unit, the same
    # way reactor.solve(time_unit=...) does, so the data axis and the rate
    # constants share a unit. native_time_factor raises if the model has no
    # declared native unit to convert to (no silent mismatch at this boundary).
    t_obs = t_obs * native_time_factor(model.time_unit, time_unit)
    if observations.ndim == 1:
        observations = observations[:, None]
    if observations.shape[0] != t_obs.shape[0]:
        raise ValueError(
            f"observations has {observations.shape[0]} rows but t_obs has {t_obs.shape[0]} entries."
        )

    if observed_species is None:
        obs_species_indices = jnp.arange(model.n_species)
        n_observed = model.n_species
    else:
        obs_species_indices = jnp.asarray([model.species_index[s] for s in observed_species])
        n_observed = len(observed_species)
    if observations.shape[1] != n_observed:
        raise ValueError(
            f"observations has {observations.shape[1]} columns but "
            f"{n_observed} species were specified."
        )

    t_span = (0.0, float(t_obs[-1]))

    def loss_from_free(free_values):
        p = p0_full.at[free_indices_arr].set(free_values)
        sol = reactor.solve(C0, params=p, t_span=t_span, t_eval=t_obs)
        pred = sol.C[:, obs_species_indices]
        return jnp.sum((pred - observations) ** 2)

    loss_value_and_grad = jax.jit(jax.value_and_grad(loss_from_free))

    def _np_loss_and_grad(x_np):
        x = jnp.asarray(x_np)
        val, grad = loss_value_and_grad(x)
        return float(val), np.asarray(grad)

    # Per-parameter bounds: each free parameter keeps its declared box bounds;
    # a parameter without declared bounds is left unbounded as (-inf, +inf)
    # rather than dropping every other parameter's bounds.
    bounds_list = []
    unbounded = []
    for name in free_params:
        b = model.parameter_bounds.get(name)
        if b is None:
            bounds_list.append((-np.inf, np.inf))
            unbounded.append(name)
        else:
            bounds_list.append((float(b[0]), float(b[1])))
    if unbounded and len(unbounded) < len(free_params):
        warnings.warn(
            "fit(): some free parameters have no declared bounds and are left "
            f"unbounded while the others stay bounded: {unbounded}. Declare "
            "bounds on these parameters to constrain them.",
            stacklevel=2,
        )
    # All free parameters unbounded => no box constraints at all.
    use_bounds = len(unbounded) < len(free_params)

    x0_np = np.asarray(p0_full[free_indices_arr])
    result = minimize(
        _np_loss_and_grad,
        x0_np,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds_list if use_bounds else None,
    )

    final_full = p0_full.at[free_indices_arr].set(jnp.asarray(result.x))
    return FitResult(
        params=final_full,
        params_named={name: float(v) for name, v in zip(free_params, result.x)},
        loss=float(result.fun),
        converged=bool(result.success),
        message=str(result.message),
        n_iter=int(result.nit),
    )
