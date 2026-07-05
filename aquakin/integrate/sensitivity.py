"""Parameter sensitivity and least-squares fitting via JAX autodiff."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from aquakin.core.conditions import SpatialConditions
from aquakin.integrate._common import (
    ConditionedReactor,
    DifferentiationConfig,
    Reactor,
    check_finite_gradient,
    forward_adjoint,
    native_time_factor,
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
    params: Optional[jnp.ndarray] = None,
    output_fn: Optional[Callable[[Any], jnp.ndarray]] = None,
    *,
    t_span: Optional[tuple[float, float]] = None,
    t_eval: Optional[jnp.ndarray] = None,
    solve_kwargs: Optional[dict] = None,
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
    if diff.mode not in ("reverse", "forward"):
        raise ValueError(f"diff.mode must be 'reverse' or 'forward'; got {diff.mode!r}.")
    ad_mode = diff.mode
    check_finite = diff.check_finite
    if ad_mode == "forward":
        # Differentiate forward through the solve; needs a forward-capable
        # adjoint. Build it internally so diffrax never appears in user code.
        reactor = with_adjoint(reactor, forward_adjoint())
    _diff = jax.jacfwd if ad_mode == "forward" else jax.grad
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
    initial_params: Optional[jnp.ndarray] = None,
    observed_species: Optional[list[str]] = None,
    time_unit: Optional[str] = None,
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


# --- Derivative-based global sensitivity (DGSM) -------------------------


@dataclass
class DGSMResult:
    """Result of :func:`dgsm`.

    Attributes
    ----------
    input_names : list[str]
        Names of the uncertain inputs, matching the rows of every array.
    dgsm : jnp.ndarray
        The derivative-based global sensitivity measure
        ``nu_j = E[(d output / d z_j)^2]``, shape ``(d,)``.
    sobol_total_bound : jnp.ndarray
        Upper bound on the Sobol total-order index of each input,
        ``S_j^tot <= nu_j (b_j - a_j)^2 / (pi^2 Var(f))`` for ``z_j`` uniform on
        ``[a_j, b_j]`` (Lamboni, Sobol & Kucherenko 2013). Dimensionless and
        directly comparable across inputs -- the AD-accelerated replacement for
        a variance-based Sobol total index.
    std_error : jnp.ndarray
        Monte-Carlo standard error of ``sobol_total_bound`` (convergence
        indicator). Shrinks like ``1/sqrt(n_valid)``.
    output_variance : float
        Variance of the scalar output over the sample.
    n_samples : int
        Number of quasi-random points actually drawn (a power of two).
    n_valid : int
        Number of points with a finite output and gradient (others skipped). For
        a vector-valued ``fn`` this is counted **per output** -- a sample
        non-finite in another output is not dropped from this one -- so different
        outputs may report different ``n_valid``.
    seed : int
        Seed of the scrambled-Sobol sampler -- fixing it makes the result
        bit-for-bit reproducible.
    ranges : jnp.ndarray
        The ``(d, 2)`` input ranges used.
    """

    input_names: list[str]
    dgsm: jnp.ndarray
    sobol_total_bound: jnp.ndarray
    std_error: jnp.ndarray
    output_variance: float
    n_samples: int
    n_valid: int
    seed: int
    ranges: jnp.ndarray
    output_name: Optional[str] = None

    def ranked(self) -> list[tuple[str, float]]:
        """Return ``(name, sobol_total_bound)`` pairs sorted by decreasing bound."""
        pairs = [(n, float(b)) for n, b in zip(self.input_names, self.sobol_total_bound)]
        return sorted(pairs, key=lambda kv: kv[1], reverse=True)


# Guidance raised when a forward-mode screen hits the default reactor adjoint's
# custom_vjp (which rejects jvp). Shared by the batched and per-sample paths.
_DGSM_FORWARD_HINT = (
    "ad_mode='forward' requires forward-mode autodiff through the solve. Build "
    "the reactor inside fn with adjoint=aquakin.forward_adjoint() (dgsm cannot "
    "set the adjoint for you -- your fn constructs the reactor); the default "
    "RecursiveCheckpointAdjoint registers a custom_vjp that rejects forward mode."
)


def _validate_dgsm_ranges(ranges, input_names):
    """Coerce/validate ``ranges`` and ``input_names``.

    Returns ``(ranges_np, lo, hi, d, input_names)`` with ``input_names`` filled
    in (``z0, z1, ...``) when not supplied.
    """
    ranges_np = np.asarray(ranges, dtype=float)
    if ranges_np.ndim != 2 or ranges_np.shape[1] != 2:
        raise ValueError(f"ranges must have shape (d, 2); got {ranges_np.shape}.")
    d = ranges_np.shape[0]
    lo, hi = ranges_np[:, 0], ranges_np[:, 1]
    if not np.all(hi > lo):
        raise ValueError("each range must satisfy upper > lower.")
    if input_names is None:
        input_names = [f"z{j}" for j in range(d)]
    elif len(input_names) != d:
        raise ValueError(f"input_names has {len(input_names)} entries but ranges has d={d}.")
    return ranges_np, lo, hi, d, list(input_names)


def _sobol_sample(lo, hi, d, n_samples, seed):
    """Draw scrambled-Sobol points in the input box.

    ``n_samples`` is rounded to the nearest power of two (Sobol sequences are
    balanced there). Returns ``(Z, n_drawn)`` with ``Z`` of shape
    ``(n_drawn, d)``.
    """
    from scipy.stats import qmc

    n_pow = max(1, round(math.log2(max(n_samples, 2))))
    U = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(n_pow)
    Z = lo[None, :] + (hi - lo)[None, :] * U
    return Z, int(Z.shape[0])


def _sobol_normal_sample(mean, std, d, n_samples, seed):
    """Draw scrambled-Sobol points from independent normals ``N(mean_j, std_j^2)``.

    Maps the low-discrepancy unit points through the inverse normal CDF, so the
    design is a quasi-Monte-Carlo sample of the Gaussian rather than of a box.
    This is the input distribution for a DGSM screen under Gaussian (prior)
    inputs, whose Sobol total-index bound carries the Poincare constant
    ``std_j^2`` in place of the uniform ``(b_j-a_j)^2 / pi^2`` (Sobol & Kucherenko
    2010, Sec. 8; Lamboni et al. 2013, Thm 3.1). ``mean``/``std`` are length-``d``
    (in the space where the input is Gaussian -- e.g. log-parameter space for a
    positive rate); ``n_samples`` is rounded to a power of two. Returns
    ``(Z, n_drawn)`` with ``Z`` of shape ``(n_drawn, d)``.
    """
    import numpy as np
    from scipy.stats import norm, qmc

    n_pow = max(1, round(math.log2(max(n_samples, 2))))
    U = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(n_pow)
    U = np.clip(U, 1e-12, 1.0 - 1e-12)  # avoid +/-inf at the 0/1 endpoints
    mean = np.asarray(mean)
    std = np.asarray(std)
    Z = mean[None, :] + std[None, :] * norm.ppf(U)
    return Z, int(Z.shape[0])


def _make_dgsm_value_and_jac(fn, z0, mode):
    """Build the jitted ``(value, Jacobian)`` callable for the requested mode.

    Probes the output rank once (via :func:`jax.eval_shape`, no solve) to choose
    between scalar (``value_and_grad`` / ``jacfwd``) and vector
    (``jacrev`` / ``jacfwd``) Jacobians. Returns
    ``(value_and_jac, vector, m_out)``; the Jacobian is shape ``(d,)`` for a
    scalar output and ``(m, d)`` for a vector output.
    """
    f_arr = lambda z: jnp.asarray(fn(z))
    out_shape = jax.eval_shape(f_arr, jnp.asarray(z0)).shape
    vector = len(out_shape) == 1
    m_out = int(out_shape[0]) if vector else 1
    if mode == "reverse":
        if vector:
            value_and_jac = jax.jit(lambda z: (f_arr(z), jax.jacrev(f_arr)(z)))
        else:
            value_and_jac = jax.jit(jax.value_and_grad(f_arr))
    else:  # forward
        value_and_jac = jax.jit(lambda z: (f_arr(z), jax.jacfwd(f_arr)(z)))
    return value_and_jac, vector, m_out


def _finite_mask(v_col: np.ndarray, j_col: np.ndarray) -> np.ndarray:
    """Boolean per-sample mask: one output's value AND its Jacobian row finite.

    ``v_col`` is ``(N,)`` (a single output's value over the samples) and
    ``j_col`` is ``(N, d)`` (that output's partials). Applied **per output** so a
    sample non-finite in one output does not drop the others.
    """
    n = v_col.shape[0]
    return np.isfinite(v_col) & np.isfinite(j_col).reshape(n, -1).all(axis=1)


def _evaluate_dgsm_samples(value_and_jac, Z, mode, batched):
    """Evaluate the value/Jacobian over every sample; return the full stacked
    arrays (non-finite rows **included**).

    Finiteness is filtered downstream *per output* (see :func:`_finite_mask`), so
    this returns every drawn row -- a sample whose value/gradient is non-finite in
    one output must still contribute to the others. ``batched=True`` dispatches
    the whole sample through one :func:`jax.vmap` (a single device->host
    transfer); ``batched=False`` is the per-sample fallback (one host transfer
    each, lower peak memory). Both return identical ``(vals, jacs)`` NumPy arrays:
    ``vals`` is ``(N,)``/``(N, m)`` and ``jacs`` is ``(N, d)``/``(N, m, d)``.
    """
    if batched:
        try:
            vals, jacs = jax.vmap(value_and_jac)(jnp.asarray(Z))
        except Exception as exc:  # pragma: no cover - guidance path
            if mode == "forward":
                raise RuntimeError(_DGSM_FORWARD_HINT) from exc
            raise
        return np.asarray(vals), np.asarray(jacs)

    v_list: list[np.ndarray] = []
    j_list: list[np.ndarray] = []
    for k, z in enumerate(Z):
        try:
            v, J = value_and_jac(jnp.asarray(z))
        except Exception as exc:  # pragma: no cover - guidance path
            if mode == "forward" and k == 0:
                raise RuntimeError(_DGSM_FORWARD_HINT) from exc
            raise
        v_list.append(np.asarray(v))
        j_list.append(np.asarray(J))
    return np.asarray(v_list), np.asarray(j_list)


def dgsm(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    ranges: Any,
    *,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    n_samples: int = 64,
    seed: int = 0,
    diff: DifferentiationConfig = DifferentiationConfig(),
    batched: bool = True,
) -> Any:
    """Derivative-based global sensitivity measure via autodiff + Sobol QMC.

    Estimates, for each uncertain input ``z_j``,

        ``nu_j = E_z[ (d fn / d z_j)^2 ]``

    by averaging the squared partial derivative over scrambled-Sobol
    quasi-random points in the input ranges. ``nu_j`` bounds the Sobol
    total-order index (see :attr:`DGSMResult.sobol_total_bound`), so it is the
    AD analogue of a variance-based Sobol total index, obtained from
    derivatives rather than a variance decomposition.

    The derivatives are exact (no finite-difference truncation) and reuse the
    differentiable model, so the same machinery serves the calibration and
    identifiability analyses. The cost depends on ``ad_mode`` and on the number
    of outputs ``m`` and inputs ``d``:

    - ``ad_mode="reverse"`` (default) forms the per-sample sensitivities with
      ``m`` reverse-mode passes (one per output), each independent of ``d``.
      Best when there are few outputs relative to inputs **and** the adjoint is
      cheap. Works with any reactor adjoint.
    - ``ad_mode="forward"`` forms them with ``d`` forward-mode tangents pushed
      through a single solve, independent of ``m``. Best when there are many
      outputs, or when the reverse adjoint is expensive -- e.g. a stiff solve
      whose differentiated step must be capped (``dtmax``), which inflates the
      reverse pass. **The reactor inside ``fn`` must then be built with**
      ``adjoint=aquakin.forward_adjoint()`` (``dgsm`` cannot set the adjoint
      for you, because ``fn`` constructs the reactor): the default
      ``RecursiveCheckpointAdjoint`` registers a ``custom_vjp`` that rejects
      forward-mode autodiff.

    Both modes return identical sensitivities (to machine precision);
    ``ad_mode`` is purely a performance choice. For a single scalar output
    ``reverse`` is
    almost always cheaper; the ``forward`` advantage appears for multi-output
    screening of a stiff model.

    Parameters
    ----------
    fn : callable
        Maps an input vector (shape ``(d,)``) to either a scalar JAX value or a
        vector of ``m`` outputs (shape ``(m,)``). Must be ``jax``-differentiable
        in the requested ``mode``. For a reactor study, ``fn`` typically maps the
        uncertain inputs into a parameter vector / initial state, calls
        ``reactor.solve`` and reduces the solution to the output(s). If the
        model is stiff, build the reactor with a suitable ``dtmax`` so the
        differentiated solve stays finite. ``dgsm`` does not own the solve (your
        ``fn`` builds the reactor and chooses the ``t_eval``), so it cannot apply
        a ``time_unit`` conversion for you: any ``t_eval`` / ``t_span`` inside
        ``fn`` must be in the model's **native** time unit, or ``fn`` must pass
        ``time_unit=`` to its own ``reactor.solve`` call.
    ranges : array-like, shape (d, 2)
        ``[lower, upper]`` bound for each input; sampling is uniform within.
    input_names : list[str], optional
        Names for reporting; defaults to ``["z0", "z1", ...]``.
    output_names : list[str], optional
        Names for the ``m`` outputs when ``fn`` is vector-valued; defaults to
        ``["output0", ...]``. Ignored for a scalar ``fn``.
    n_samples : int, optional
        Target number of quasi-random points; rounded to the nearest power of
        two (Sobol sequences are balanced at powers of two). Increase until
        ``std_error`` is small relative to the ranking gaps.
    seed : int, optional
        Seed for the scrambled-Sobol sampler. Fixing it (the default ``0``)
        makes the analysis exactly reproducible.
    diff : DifferentiationConfig, optional
        Autodiff configuration. ``mode`` ({"reverse", "forward"}) selects the
        direction used to form the per-sample sensitivities (see above).
    batched : bool, optional
        When ``True`` (default) the whole sample is pushed through one
        ``jax.vmap`` dispatch and finiteness is filtered once on the stacked
        result -- one device->host transfer instead of one per point. Set
        ``False`` to evaluate point-by-point (lower peak memory for a large
        screen). Both give identical results.

    Returns
    -------
    DGSMResult or list[DGSMResult]
        A single :class:`DGSMResult` when ``fn`` is scalar-valued, or a list of
        results (one per output, in order, each carrying its ``output_name``)
        when ``fn`` is vector-valued.

    Examples
    --------
    >>> def fn(z):                       # output sensitive to z0, not z1
    ...     return 3.0 * z[0] + 0.0 * z[1]
    >>> res = aquakin.dgsm(fn, [(0.0, 1.0), (0.0, 1.0)], input_names=["a", "b"])
    >>> res.ranked()[0][0]
    'a'
    """
    if diff.mode not in ("reverse", "forward"):
        raise ValueError(f"diff.mode must be 'reverse' or 'forward'; got {diff.mode!r}.")
    mode = diff.mode

    ranges_np, lo, hi, d, input_names = _validate_dgsm_ranges(ranges, input_names)
    Z, n_drawn = _sobol_sample(lo, hi, d, n_samples, seed)
    value_and_jac, vector, m_out = _make_dgsm_value_and_jac(fn, Z[0], mode)
    vals, jacs = _evaluate_dgsm_samples(value_and_jac, Z, mode, batched)

    def _assemble(fv: np.ndarray, g2: np.ndarray, name: Optional[str]) -> DGSMResult:
        # fv: (n,) finite output values; g2: (n, d) squared partials (already
        # masked to this output's finite samples, so n is this output's n_valid).
        n = fv.shape[0]
        if n < 2:
            raise RuntimeError(
                f"DGSM needs >= 2 finite samples"
                f"{f' for output {name!r}' if name else ''}; got {n}/{n_drawn}. "
                "The output or its gradient is non-finite over the sampled ranges "
                "-- for a stiff model, cap the integrator step via the "
                "reactor's dtmax."
            )
        nu = np.mean(g2, axis=0)
        var_f = float(np.var(fv))
        if var_f > 0:
            scale = (hi - lo) ** 2 / (math.pi**2 * var_f)
        else:
            # Every sample produced an identical output (e.g. a saturated or
            # clipped response): the Sobol total-index bound is undefined
            # (0/0). Return an all-zero bound but warn, so an empty ranking is
            # not silently read as "no input matters".
            scale = np.zeros(d)
            warnings.warn(
                f"DGSM output{f' {name!r}' if name else ''} has zero variance "
                f"over the sampled ranges; the Sobol total-index bound is "
                f"undefined and reported as 0. The output may be saturated, "
                f"clipped, or insensitive to every input over these ranges.",
                stacklevel=2,
            )
        bound = nu * scale
        bound_se = (np.std(g2, axis=0) / math.sqrt(n)) * scale
        return DGSMResult(
            input_names=list(input_names),
            dgsm=jnp.asarray(nu),
            sobol_total_bound=jnp.asarray(bound),
            std_error=jnp.asarray(bound_se),
            output_variance=var_f,
            n_samples=n_drawn,
            n_valid=n,
            seed=seed,
            ranges=jnp.asarray(ranges_np),
            output_name=name,
        )

    if not vector:
        keep = _finite_mask(vals, jacs)  # vals (N,), jacs (N, d)
        return _assemble(vals[keep], jacs[keep] ** 2, None)

    if output_names is None:
        output_names = [f"output{i}" for i in range(m_out)]
    elif len(output_names) != m_out:
        raise ValueError(
            f"output_names has {len(output_names)} entries but fn returns m={m_out} outputs."
        )
    # Mask finiteness PER OUTPUT: a sample non-finite in one output (or its
    # gradient) is dropped only for that output, not jointly for all of them, so
    # each output's nu_j and n_valid are unbiased by the others' failures.
    results = []
    for i in range(m_out):
        keep_i = _finite_mask(vals[:, i], jacs[:, i, :])  # (N,)
        results.append(_assemble(vals[keep_i, i], jacs[keep_i, i, :] ** 2, output_names[i]))
    return results
