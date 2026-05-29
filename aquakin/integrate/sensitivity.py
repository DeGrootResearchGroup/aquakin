"""Parameter sensitivity and least-squares fitting via JAX autodiff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from aquakin.core.conditions import SpatialConditions
from aquakin.integrate._common import Reactor

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


@dataclass
class SensitivityResult:
    """
    Gradients of a scalar output with respect to parameters and conditions.

    Attributes
    ----------
    output : float
        The scalar output value at the evaluation point.
    doutput_dparams : jnp.ndarray
        Gradient w.r.t. the flat ``params`` vector, shape ``(n_params,)``.
    doutput_dconditions : dict[str, jnp.ndarray]
        Gradient w.r.t. each condition field, ``field_name -> (n_locations,)``.
    parameter_names : list[str]
        Namespaced parameter names matching ``doutput_dparams``.
    """

    output: float
    doutput_dparams: jnp.ndarray
    doutput_dconditions: dict[str, jnp.ndarray]
    parameter_names: list[str]

    def ranked_params(self) -> list[tuple[str, float]]:
        """Return ``(name, |grad|)`` pairs sorted by decreasing magnitude."""
        mags = [(n, float(jnp.abs(g))) for n, g in zip(self.parameter_names, self.doutput_dparams)]
        return sorted(mags, key=lambda kv: kv[1], reverse=True)


def sensitivity(
    reactor: Reactor,
    C0: jnp.ndarray,
    params: jnp.ndarray,
    output_fn: Callable[[Any], jnp.ndarray],
    *,
    solve_kwargs: Optional[dict] = None,
) -> SensitivityResult:
    """
    Compute gradients of a scalar output with respect to parameters and
    condition fields, via autodiff through ``reactor.solve``.

    Parameters
    ----------
    reactor : BatchReactor or PlugFlowReactor
        Any reactor exposing ``.solve(C0, params, ...)`` and a ``.conditions``
        attribute.
    C0 : jnp.ndarray
        Initial concentration vector.
    params : jnp.ndarray
        Parameter vector at which to evaluate sensitivity.
    output_fn : callable
        Maps a solution object to a scalar JAX value.
    solve_kwargs : dict, optional
        Extra keyword arguments passed through to ``reactor.solve`` (e.g.
        ``t_span``, ``t_eval`` for a batch reactor).

    Returns
    -------
    SensitivityResult
    """
    solve_kwargs = dict(solve_kwargs or {})
    base_fields = dict(reactor.conditions.fields)

    def _output_from_params(p):
        sol = reactor.solve(C0, p, **solve_kwargs)
        return jnp.asarray(output_fn(sol))

    def _output_from_field(field_name: str, field_array: jnp.ndarray):
        # Build an overlay SpatialConditions with the traced field array, and
        # pass it via the reactor's `conditions=` override. No mutation of
        # reactor state.
        overlay = SpatialConditions(
            fields={**base_fields, field_name: field_array}
        )
        sol = reactor.solve(C0, params, conditions=overlay, **solve_kwargs)
        return jnp.asarray(output_fn(sol))

    output_value = float(_output_from_params(params))
    dout_dparams = jax.grad(_output_from_params)(params)

    dout_dconditions: dict[str, jnp.ndarray] = {}
    for fname, arr in base_fields.items():
        dout_dconditions[fname] = jax.grad(
            lambda a, fn=fname: _output_from_field(fn, a)
        )(arr)

    return SensitivityResult(
        output=output_value,
        doutput_dparams=dout_dparams,
        doutput_dconditions=dout_dconditions,
        parameter_names=list(reactor.network.parameters),
    )


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
    reactor: Any,
    C0: jnp.ndarray,
    observations: jnp.ndarray,
    t_obs: jnp.ndarray,
    free_params: list[str],
    *,
    method: str = "adjoint",
    initial_params: Optional[jnp.ndarray] = None,
    observed_species: Optional[list[str]] = None,
) -> FitResult:
    """
    Least-squares fit of selected parameters to time-series observations.

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
        solution is sampled at ``t_obs``.
    free_params : list[str]
        Namespaced parameter names to optimise. Other parameters are held at
        their default (or ``initial_params``) values.
    method : str
        Currently only ``"adjoint"`` is supported, which uses Diffrax's
        recursive-checkpoint adjoint via :func:`jax.grad` and SciPy L-BFGS-B.
    initial_params : jnp.ndarray, optional
        Starting parameter vector. Defaults to ``reactor.network.default_parameters()``.
    observed_species : list[str], optional
        Species names corresponding to columns of ``observations``. If
        ``None``, ``observations`` is assumed to be over all species in
        network order.

    Returns
    -------
    FitResult
    """
    if method != "adjoint":
        raise ValueError(f"Unknown fit method {method!r}; only 'adjoint' is supported.")
    if not free_params:
        raise ValueError("free_params must be non-empty.")

    network = reactor.network
    p0_full = (
        jnp.asarray(initial_params)
        if initial_params is not None
        else network.default_parameters()
    )

    free_indices = []
    for name in free_params:
        if name not in network.param_index:
            raise KeyError(
                f"Unknown parameter '{name}'. Available: {network.parameters}"
            )
        free_indices.append(network.param_index[name])
    free_indices_arr = jnp.asarray(free_indices)

    observations = jnp.asarray(observations)
    t_obs = jnp.asarray(t_obs)
    if t_obs.ndim != 1 or t_obs.shape[0] < 1:
        raise ValueError(f"t_obs must be a non-empty 1-D array, got shape {t_obs.shape}.")
    if float(t_obs[0]) < 0.0:
        raise ValueError(f"t_obs must be non-negative; got t_obs[0] = {float(t_obs[0])}.")
    if t_obs.shape[0] > 1 and not bool(jnp.all(jnp.diff(t_obs) > 0)):
        raise ValueError("t_obs must be strictly ascending.")
    if observations.ndim == 1:
        observations = observations[:, None]
    if observations.shape[0] != t_obs.shape[0]:
        raise ValueError(
            f"observations has {observations.shape[0]} rows but t_obs has "
            f"{t_obs.shape[0]} entries."
        )

    if observed_species is None:
        obs_species_indices = jnp.arange(network.n_species)
        n_observed = network.n_species
    else:
        obs_species_indices = jnp.asarray(
            [network.species_index[s] for s in observed_species]
        )
        n_observed = len(observed_species)
    if observations.shape[1] != n_observed:
        raise ValueError(
            f"observations has {observations.shape[1]} columns but "
            f"{n_observed} species were specified."
        )

    t_span = (0.0, float(t_obs[-1]))

    def loss_from_free(free_values):
        p = p0_full.at[free_indices_arr].set(free_values)
        sol = reactor.solve(C0, p, t_span=t_span, t_eval=t_obs)
        pred = sol.C[:, obs_species_indices]
        return jnp.sum((pred - observations) ** 2)

    loss_value_and_grad = jax.jit(jax.value_and_grad(loss_from_free))

    def _np_loss_and_grad(x_np):
        x = jnp.asarray(x_np)
        val, grad = loss_value_and_grad(x)
        return float(val), np.asarray(grad)

    # Bounds (if all free params have bounds set; otherwise unbounded).
    bounds_list = []
    use_bounds = True
    for name in free_params:
        if name not in network.parameter_bounds:
            use_bounds = False
            break
        b = network.parameter_bounds[name]
        bounds_list.append((float(b[0]), float(b[1])))

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
