"""Local parameter sensitivity via JAX autodiff.

Gradients of a scalar output with respect to the model parameters and condition
fields, taken by differentiating through ``reactor.solve``. The sibling
capabilities that once shared this module now live alongside it: the
least-squares point fitter in :mod:`aquakin.integrate.fit`, and the
derivative-based global (DGSM) screen in
:mod:`aquakin.integrate.global_sensitivity`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions
from aquakin.integrate._common import (
    ConditionedReactor,
    DifferentiationConfig,
    check_finite_gradient,
    forward_adjoint,
    with_adjoint,
)


@dataclass
class SensitivityResult:
    """
    Gradients of a scalar output with respect to parameters and conditions.

    Attributes
    ----------
    output : float
        The scalar output value at the evaluation point.
    doutput_dparams : jnp.ndarray
        Gradient w.r.t. the **full** flat ``params`` vector, shape
        ``(n_params,)`` --- every model parameter, not a free subset (unlike
        :func:`fit` / :func:`~aquakin.calibrate`, which optimise a chosen
        ``free_params`` list).
    doutput_dconditions : dict[str, jnp.ndarray]
        Gradient w.r.t. each condition field, ``field_name -> (n_locations,)``.
    parameter_names : list[str]
        Namespaced parameter names matching ``doutput_dparams`` (all of them).
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
    reactor: ConditionedReactor,
    C0: jnp.ndarray,
    params: jnp.ndarray | None = None,
    output_fn: Callable[[Any], jnp.ndarray] | None = None,
    *,
    t_span: tuple[float, float] | None = None,
    t_eval: jnp.ndarray | None = None,
    solve_kwargs: dict | None = None,
    diff: DifferentiationConfig = DifferentiationConfig(),
) -> SensitivityResult:
    """
    Compute gradients of a scalar output with respect to parameters and
    condition fields, via autodiff through ``reactor.solve``.

    Parameters
    ----------
    reactor : BatchReactor or PlugFlowReactor
        Any reactor exposing ``.solve(C0, t_span, ..., params=...)`` and a ``.conditions``
        attribute.
    C0 : jnp.ndarray
        Initial concentration vector.
    params : jnp.ndarray, optional
        Parameter vector at which to evaluate sensitivity. Defaults to
        ``reactor.model.default_parameters()``.
    output_fn : callable
        Maps a solution object to a scalar JAX value, e.g.
        ``lambda sol: sol.C_named("BrO3-")[-1]``.
    t_span, t_eval : optional
        Integration window / save times, passed straight to ``reactor.solve``
        (the common batch case). Equivalent to putting them in ``solve_kwargs``;
        provide whichever reads better.
    solve_kwargs : dict, optional
        Any further keyword arguments forwarded to ``reactor.solve`` -- including
        ``time_unit=`` if ``t_span`` / ``t_eval`` are in a non-native unit
        (``solve`` converts them, so the sensitivities stay consistent).
    diff : DifferentiationConfig, optional
        Autodiff configuration. ``mode="reverse"`` (default) uses ``jax.grad``.
        ``mode="forward"`` uses ``jax.jacfwd`` and rebuilds the reactor internally
        with a forward-capable adjoint, so a *stiff* reactor whose reverse adjoint
        is non-finite can be differentiated without a ``dtmax`` cap and without the
        caller touching ``diffrax``. ``check_finite`` (default ``True``) raises a
        friendly ``RuntimeError`` if the computed sensitivities are non-finite,
        instead of returning silent ``NaN``s.

    Returns
    -------
    SensitivityResult
    """
    if output_fn is None:
        raise ValueError("output_fn is required (a solution -> scalar callable).")
    diff.validated()
    ad_mode = diff.mode
    check_finite = diff.check_finite
    if diff.forms_jacfwd():
        # Differentiate forward through the solve; needs a forward-capable
        # adjoint. Build it internally so diffrax never appears in user code.
        reactor = with_adjoint(reactor, forward_adjoint())
    _diff = jax.jacfwd if diff.forms_jacfwd() else jax.grad
    if params is None:
        params = reactor.model.default_parameters()
    solve_kwargs = dict(solve_kwargs or {})
    if t_span is not None:
        solve_kwargs.setdefault("t_span", t_span)
    if t_eval is not None:
        solve_kwargs.setdefault("t_eval", t_eval)
    base_fields = dict(reactor.conditions.fields)

    def _output_from_params(p):
        sol = reactor.solve(C0, params=p, **solve_kwargs)
        return jnp.asarray(output_fn(sol))

    def _output_from_field(field_name: str, field_array: jnp.ndarray):
        # Build an overlay SpatialConditions with the traced field array, and
        # pass it via the reactor's `conditions=` override. No mutation of
        # reactor state.
        overlay = SpatialConditions(fields={**base_fields, field_name: field_array})
        sol = reactor.solve(C0, params=params, conditions=overlay, **solve_kwargs)
        return jnp.asarray(output_fn(sol))

    output_value = float(_output_from_params(params))
    dout_dparams = _diff(_output_from_params)(params)

    dout_dconditions: dict[str, jnp.ndarray] = {}
    for fname, arr in base_fields.items():
        dout_dconditions[fname] = _diff(lambda a, fn=fname: _output_from_field(fn, a))(arr)

    if check_finite:
        remedy = (
            "Pass ad_mode='forward' (forward-mode AD is finite through a stiff "
            "solve), or build the reactor with a dtmax cap."
            if ad_mode == "reverse"
            else "Check the model and ranges; even forward-mode AD returned non-finite."
        )
        check_finite_gradient(dout_dparams, what="sensitivity", remedy=remedy)
        for arr in dout_dconditions.values():
            check_finite_gradient(arr, what="condition sensitivity", remedy=remedy)

    return SensitivityResult(
        output=output_value,
        doutput_dparams=dout_dparams,
        doutput_dconditions=dout_dconditions,
        parameter_names=list(reactor.model.parameters),
    )
