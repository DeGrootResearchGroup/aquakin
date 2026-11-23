"""Solve-time validation, ``atol`` and time-unit resolution, the friendly solve
errors, and the single-source-of-truth implicit solver / step-controller
builders with the ``diffeqsolve`` driver and the chemistry RHS built on them.

Not part of the public API; ``aquakin.integrate._common`` re-exports every name.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquakin.core.model import CompiledModel
from aquakin.core.tolerance import default_atol
from aquakin.integrate._config import FORWARD_DIFF, FORWARD_REACTOR_REMEDY


def validate_t_eval(t_eval_arr: jnp.ndarray, t0: float, t1: float) -> None:
    """Validate output times before handing them to ``SaveAt(ts=...)``.

    Diffrax silently returns NaN for save times outside ``[t0, t1]`` or a
    non-ascending sequence, so check here for a clear error instead. Value
    checks run only for concrete (non-traced) ``t_eval``; a traced array
    (e.g. differentiating with respect to the save times) skips them.

    Parameters
    ----------
    t_eval_arr : jnp.ndarray
        Candidate save times.
    t0, t1 : float
        Integration interval bounds.

    Raises
    ------
    ValueError
        If ``t_eval`` is not 1-D, lies outside ``[t0, t1]``, or is not
        strictly ascending.
    """
    if t_eval_arr.ndim != 1:
        raise ValueError(f"t_eval must be 1-D; got shape {tuple(t_eval_arr.shape)}.")
    if isinstance(t_eval_arr, jax.core.Tracer):
        return
    t_eval_np = np.asarray(t_eval_arr)
    if t_eval_np.size == 0:
        return
    lo, hi = float(t_eval_np.min()), float(t_eval_np.max())
    if lo < t0 or hi > t1:
        raise ValueError(f"t_eval must lie within t_span [{t0}, {t1}]; got values in [{lo}, {hi}].")
    if t_eval_np.size > 1 and not np.all(np.diff(t_eval_np) > 0):
        raise ValueError("t_eval must be strictly ascending.")


def require_increasing_t_span(t_span) -> tuple[float, float]:
    """Coerce a ``(t_start, t_end)`` pair to floats and require ``t_end > t_start``.

    The single home for the interval check every temporal ``solve`` /
    ``solve_sensitivity`` runs before integrating (a zero- or negative-width
    span makes the solver take no step / step backwards).

    Parameters
    ----------
    t_span : tuple of float
        ``(t_start, t_end)`` integration interval.

    Returns
    -------
    tuple of float
        ``(t0, t1)`` as Python floats.

    Raises
    ------
    ValueError
        If ``t_end`` does not strictly exceed ``t_start``.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not (t1 > t0):
        raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")
    return t0, t1


def _coerce_atol(atol, n_species: int):
    """Validate and normalise an ``atol`` argument.

    Returns either the original scalar or a ``(n_species,)`` JAX array.
    Raises ``ValueError`` if an array of the wrong shape is supplied.
    """
    arr = jnp.asarray(atol)
    if arr.ndim == 0:
        # A concrete scalar (the reactor-construction path) is returned as a
        # Python float, but a traced value is returned as the 0-d array: calling
        # ``float()`` on a tracer raises a concretization error, which would
        # otherwise prevent jitting a solve whose ``atol`` flows in under tracing.
        if isinstance(arr, jax.core.Tracer):
            return arr
        return float(arr)
    if arr.shape != (n_species,):
        raise ValueError(f"atol array must have shape ({n_species},), got {arr.shape}")
    return arr


def is_forward_mode_ad_error(exc: BaseException) -> bool:
    """True if ``exc`` is JAX's rejection of forward-mode autodiff through a
    ``custom_vjp``.

    The default ``RecursiveCheckpointAdjoint`` registers a ``custom_vjp``, so a
    ``jax.jvp`` / ``jax.jacfwd`` through the solve fails with *"can't apply
    forward-mode autodiff (jvp) to a custom_vjp function"*. This is the one
    specific failure the forward-mode sensitivity / DGSM paths convert into
    forward-mode ``diff=`` guidance; matching it narrowly keeps unrelated errors
    (bad shapes, OOM, a bug in the user's ``fn``) from being masked by that hint.
    """
    msg = str(exc).lower()
    return "forward-mode autodiff" in msg and "custom_vjp" in msg


@contextlib.contextmanager
def friendly_solve_errors(max_steps, *, what: str = "solve"):
    """Re-raise two opaque solve-time failures as domain-level errors.

    Both failures surface from JAX/Diffrax/Equinox internals with messages a
    process engineer cannot map to a fix. Wrap the *execution* of a solve in
    this context manager -- the call to the jitted solve / ``diffeqsolve``, where
    each failure actually surfaces -- to re-raise it as a plain ``RuntimeError``
    naming the remedy, with the noisy traceback suppressed (``from None``). Any
    other exception propagates unchanged.

    1. **Step-budget exhaustion.** An adaptive solve that exhausts ``max_steps``
       raises a verbose ``EquinoxRuntimeError`` ("The maximum number of solver
       steps was reached") wrapped in JAX/Equinox debugging chatter
       (``EQX_ON_ERROR``, ``kidger.site``, ...). Re-raised naming the
       warm-start / loosen-rtol / raise-max_steps remedies.
    2. **Forward-mode AD through the reverse-only adjoint.** The default
       ``RecursiveCheckpointAdjoint`` registers a ``custom_vjp``, so a
       ``jax.jacfwd`` / ``jax.jvp`` through the solve fails with JAX's
       "can't apply forward-mode autodiff (jvp) to a custom_vjp function" --
       which never mentions the cure. Re-raised naming the forward-mode
       ``diff=aquakin.DifferentiationConfig(...)`` the reactor should be built
       with.

    Parameters
    ----------
    max_steps : int
        The step budget that was hit (quoted back in the step-budget message).
    what : str
        A short label for the failing solve (e.g. ``"plant solve"``), used in
        the message.
    """
    try:
        yield
    except Exception as exc:
        msg = str(exc).lower()
        if "maximum number of solver steps" in msg:
            raise RuntimeError(
                f"The {what} hit its integrator step budget (max_steps={max_steps}) "
                "before completing. This is almost always a stiff transient, not a "
                "bug. Try, in order: (1) warm-start from a settled state -- for a "
                "plant, pass y0 from plant.run_to_steady_state (or a previous run); "
                "(2) loosen rtol (the default atol already auto-scales to the state "
                "magnitudes); or (3) raise max_steps. If none help, the model may be "
                "genuinely unstable at these parameters/inputs."
            ) from None
        if is_forward_mode_ad_error(exc):
            raise RuntimeError(
                f"Forward-mode autodiff (jax.jacfwd / jax.jvp) cannot flow through "
                f"the {what}: the default adjoint (RecursiveCheckpointAdjoint) is "
                "reverse-mode only -- it registers a custom_vjp that rejects forward "
                f"mode. To differentiate in forward mode, {FORWARD_REACTOR_REMEDY}; "
                "or take a reverse-mode gradient (jax.grad / jax.jacrev) instead. "
                f"(aquakin.sensitivity with {FORWARD_DIFF} sets this adjoint for "
                "you.)"
            ) from None
        raise


# Seconds per time unit, for converting a user-supplied ``time_unit`` to a
# model's native (rate-constant) time unit. The keys match
# ``CompiledModel.time_unit`` / ``utils.units.TIME_TOKENS``.
_TIME_UNIT_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


def native_time_factor(model_time_unit, requested_unit) -> float:
    """Factor ``f`` such that ``t_native = f * t_requested``.

    Converts a time expressed in ``requested_unit`` into the model's native
    (rate-constant) time unit, so a caller can pass ``t_span`` / ``t_eval`` in a
    convenient unit (e.g. hours) and have the solve run with the model's
    unchanged rate constants. Returns ``1.0`` when ``requested_unit is None``
    (no conversion -- the default, native-unit path).

    Parameters
    ----------
    model_time_unit : str or None
        The model's native time unit (``CompiledModel.time_unit``).
    requested_unit : str or None
        The unit the caller's times are in. ``None`` means "native".

    Raises
    ------
    ValueError
        If ``requested_unit`` is unknown, or it is given but the model's own
        time unit could not be inferred (``None``) and so there is no native
        unit to convert to.
    """
    if requested_unit is None:
        return 1.0
    if requested_unit not in _TIME_UNIT_SECONDS:
        raise ValueError(
            f"Unknown time_unit {requested_unit!r}; expected one of {sorted(_TIME_UNIT_SECONDS)}."
        )
    if model_time_unit is None:
        raise ValueError(
            f"Cannot convert times to time_unit={requested_unit!r}: this "
            "model's own time unit could not be inferred from its "
            "rate-constant units (model.time_unit is None), so there is no "
            "native unit to convert to. Pass t_span / t_eval already in the "
            "model's rate-constant time unit and omit time_unit."
        )
    # model_time_unit comes from CompiledModel.time_unit, which only ever
    # returns a known token, so it is guaranteed to be in the table.
    return _TIME_UNIT_SECONDS[requested_unit] / _TIME_UNIT_SECONDS[model_time_unit]


def to_native_time(model_time_unit, requested_unit, t_span, t_eval):
    """Scale ``(t_span, t_eval)`` from ``requested_unit`` into native time.

    Returns ``(t_span_native, t_eval_native, factor)`` where ``factor`` is the
    :func:`native_time_factor` (``t_native = factor * t_requested``); divide a
    native-unit result time by it to report back in ``requested_unit``. A
    ``factor`` of ``1.0`` (the ``requested_unit is None`` default, or a unit
    equal to the native one) leaves the inputs untouched.
    """
    factor = native_time_factor(model_time_unit, requested_unit)
    if factor == 1.0:
        return t_span, t_eval, factor
    if t_span is not None:
        t_span = (t_span[0] * factor, t_span[1] * factor)
    if t_eval is not None:
        t_eval = jnp.asarray(t_eval) * factor
    return t_span, t_eval, factor


def resolve_state_atol(model, atol):
    """Resolve the ``atol`` for a reactor whose state is one ``(n_species,)``
    concentration vector.

    ``atol=None`` -> the per-component :func:`default_atol` noise floor scaled off
    the model's reference concentrations (so a g/m³ ASM model and a mol/L
    ozone model each get sensible tolerances without hand-tuning, instead of a
    fixed scalar that is ~9 orders too tight for g/m³ states). An explicit scalar
    or ``(n_species,)`` array is validated and returned verbatim. Shared by every
    reactor with a single-concentration-vector state (Batch / PFR / Particle /
    CFD); the layered :class:`~aquakin.BiofilmReactor`, whose state spans several
    compartments, uses :func:`resolve_layered_atol` (the same per-component floor,
    tiled across compartments) instead.
    """
    return (
        default_atol(model.default_concentrations())
        if atol is None
        else _coerce_atol(atol, model.n_species)
    )


def resolve_layered_atol(model, atol, n_compartments):
    """Resolve the ``atol`` for a reactor whose state stacks ``n_compartments``
    copies of the model's ``(n_species,)`` concentration vector (the layered
    :class:`~aquakin.BiofilmReactor`: a bulk plus per-depth-layer states).

    ``atol=None`` -> the per-component :func:`default_atol` noise floor tiled
    across every compartment, so each (layer, species) slot gets the same
    per-species floor a single-vector reactor would use -- not a fixed scalar
    that is ~9 orders too tight for g/m3 ASM/ADM states. A scalar broadcasts to
    the whole state; a ``(n_species,)`` array is tiled across the compartments; a
    ``(n_compartments, n_species)`` or flat ``(n_compartments*n_species,)`` array
    is accepted verbatim. Returns the ``(n_compartments, n_species)`` array
    matching the reactor's 2-D compartment state (the sensitivity path flattens
    it for the augmented solve).
    """
    n = model.n_species
    nc = int(n_compartments)
    if atol is None:
        per_species = default_atol(model.default_concentrations())  # (n,)
        return jnp.tile(per_species, (nc, 1))
    arr = jnp.asarray(atol, dtype=float)
    if arr.ndim == 0:
        return jnp.full((nc, n), arr)
    if arr.shape == (n,):
        return jnp.tile(arr, (nc, 1))
    if arr.shape == (nc, n):
        return arr
    if arr.shape == (nc * n,):
        return arr.reshape(nc, n)
    raise ValueError(
        f"atol must be a scalar, a ({n},) per-species array, or a "
        f"({nc}, {n}) / ({nc * n},) full-state array; got shape {tuple(arr.shape)}."
    )


def init_solver_settings(reactor, model, *, rtol, integrator, diff):
    """Store the solver settings every reactor shares, on ``reactor``.

    Resolves the public :class:`IntegratorConfig` + :class:`DifferentiationConfig`
    into the primitive attributes the cache + ``_run_diffeqsolve`` path read:
    ``model``, ``rtol``, ``adjoint`` (the diffrax adjoint object, resolved from
    the differentiation config), ``dtmax``, ``max_steps``, ``order``,
    ``factormax`` and ``solver``. ``atol`` is **not** set here because its
    resolution depends on the reactor's state shape (see
    :func:`resolve_state_atol` for the single-vector case and
    :func:`resolve_layered_atol` for the layered biofilm); the caller sets
    ``reactor.atol`` itself.
    """
    reactor.model = model
    reactor.rtol = float(rtol)
    reactor.integrator = integrator
    reactor.diff = diff
    reactor.adjoint = diff.reactor_adjoint()
    reactor.dtmax = integrator.dtmax
    reactor.max_steps = int(integrator.max_steps)
    reactor.order = int(integrator.order)
    reactor.factormax = integrator.factormax
    reactor.solver = integrator.solver


def validate_C0_params(model, C0, params):
    """Raise ``ValueError`` if ``C0`` / ``params`` do not match the model.

    The shared shape check every single-vector-state reactor runs at the top of
    ``solve`` (``C0`` is ``(n_species,)``, ``params`` is ``(n_params,)``).
    """
    if C0.shape != (model.n_species,):
        raise ValueError(f"C0 has shape {C0.shape}, expected ({model.n_species},)")
    if params.shape != (model.n_params,):
        raise ValueError(f"params has shape {params.shape}, expected ({model.n_params},)")


# The per-stage Newton (root-find) tolerance is decoupled from the step tolerance
# by this factor: diffrax's stock Kvaerno root finder copies the step tolerances,
# driving each stage solve to full-step accuracy (more Newton iterations than the
# embedded error estimate needs). The step controller still enforces the solution
# accuracy at (rtol, atol); loosening only the per-stage Newton ends each stage
# solve sooner (~15-20% faster on the stiff BSM2 plant, final-state agreement
# ~1e-6).
NEWTON_TOL_FACTOR = 10.0

# The fixed set of ESDIRK orders any implicit solve may use, keyed by order. This
# is the ONLY place aquakin constructs a diffrax ESDIRK solver *object*: every
# solve path gets its solver from build_implicit_solver, which builds from this
# table, so the solver mode cannot drift between the reactor and plant paths
# (Kvaerno5 is the robust default; Kvaerno3 does less linear algebra per step).
# Enforced by tests/unit/test_solver_config_single_source.py.
_CANONICAL_SOLVERS = {3: diffrax.Kvaerno3, 5: diffrax.Kvaerno5}


def build_implicit_solver(
    rtol,
    atol,
    *,
    order=5,
    solver=None,
    colored_root_finder=None,
    force_root_finder=False,
    linear_solver=None,
):
    """The single source of truth for the implicit ODE solver, shared by EVERY
    aquakin solve so the forward and discrete-adjoint paths cannot silently drift.

    Both :func:`_run_diffeqsolve` (the forward path) and the discrete-adjoint
    forward pass (:func:`~aquakin.integrate.discrete_adjoint.esdirk_adjoint_solve`)
    build their solver here, so a per-step optimization -- the decoupled Newton
    tolerance, a colored root finder, a different ESDIRK order -- lands in one
    place and automatically reaches both modes.

    Parameters
    ----------
    rtol, atol : float
        Step-size tolerances; the per-stage Newton tolerance is these loosened by
        :data:`NEWTON_TOL_FACTOR` unless a ``colored_root_finder`` is given.
    order : int
        ESDIRK order to build when ``solver`` is ``None`` -- a key into
        :data:`_CANONICAL_SOLVERS` (``5`` -> ``Kvaerno5``, the robust default;
        ``3`` -> ``Kvaerno3``, less linear algebra per step). This is how an
        internal caller selects the solver order without constructing the diffrax
        object itself (so construction stays funneled here).
    solver : diffrax.AbstractSolver, optional
        A user-supplied ESDIRK solver -- the one escape hatch (e.g.
        ``Plant.solve(integrator=IntegratorConfig(solver=...))``). Overrides ``order``. ``None`` builds the
        canonical solver of ``order``.
    colored_root_finder : diffrax root finder, optional
        A sparsity-colored root finder (``ColoredVeryChord``) that materializes the
        per-step implicit Jacobian by one JVP per color instead of a dense
        ``jacfwd``. When given it is used verbatim (its own tolerances stand).
    force_root_finder : bool
        When ``True`` (the discrete-adjoint forward, which pairs the solver with a
        step-clipping controller that needs explicit root-finder tolerances), the
        decoupled root finder is injected even into a *supplied* ``solver``. When
        ``False`` (the forward path) a supplied ``solver`` is honoured verbatim --
        its own root finder is left untouched -- and only the *default* solver gets
        the decoupled root finder.
    linear_solver : lineax.AbstractLinearSolver, optional
        A custom linear solver for the decoupled root finder's per-stage solve.
        The forward-sensitivity augmented ``[y; S]`` solve passes the block-arrow
        :class:`~aquakin.integrate._simultaneous_corrector.SimultaneousCorrector`
        here so the decoupled Newton uses the factor-once-reuse linear algebra.
        Ignored when ``colored_root_finder`` is given (that root finder carries
        its own solver).
    """
    if colored_root_finder is not None:
        rf = colored_root_finder
    else:
        rf_kw = {} if linear_solver is None else {"linear_solver": linear_solver}
        rf = diffrax.VeryChord(
            rtol=NEWTON_TOL_FACTOR * rtol, atol=NEWTON_TOL_FACTOR * jnp.asarray(atol), **rf_kw
        )
    if solver is None:
        if order not in _CANONICAL_SOLVERS:
            raise ValueError(f"order must be one of {sorted(_CANONICAL_SOLVERS)}; got {order!r}.")
        return _CANONICAL_SOLVERS[order](root_finder=rf)
    if force_root_finder or colored_root_finder is not None:
        return eqx.tree_at(lambda s: s.root_finder, solver, rf)
    return solver  # forward path: a user-supplied solver is honoured verbatim


def build_step_controller(rtol, atol, *, factormax=None, dtmax=None):
    """The single source of truth for the PID step-size controller core, shared by
    the forward solve (which uses it directly) and the discrete adjoint (which
    wraps it in a ``ClipStepSizeController``) -- so ``factormax`` / the tolerances
    cannot drift between the two paths."""
    kw = {} if factormax is None else {"factormax": factormax}
    return diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax, **kw)


def _run_diffeqsolve(
    rhs: Callable,
    *,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    args,
    saveat: diffrax.SaveAt,
    rtol: float,
    atol,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int = 100_000,
    dtmax: float | None = None,
    event: diffrax.Event | None = None,
    progress_meter: diffrax.AbstractProgressMeter | None = None,
    solver: diffrax.AbstractSolver | None = None,
    factormax: float | None = None,
    order: int = 5,
):
    """Wrapper around the canonical Kvaerno + PIDController + adjoint setup.

    All reactors call this with their own ``rhs``. Adjusting the default
    solver, controller, or adjoint here changes behaviour for every reactor.

    ``solver`` overrides the default ``Kvaerno5`` integrator with any diffrax
    solver (``None`` keeps ``Kvaerno5``). A lower-order ESDIRK such as
    ``Kvaerno3`` does less implicit linear algebra per step (fewer stages), which
    can be faster on a large stiff system whose per-step cost is dominated by the
    Jacobian factorisation, at the cost of taking more (cheaper) steps. The
    *default* ``Kvaerno5`` is built with a **decoupled root finder** -- its
    per-stage Newton tolerance is 10x looser than the step tolerance (see the
    inline note), which makes each step cheaper at preserved accuracy; a
    user-supplied ``solver`` is used as-is.

    ``factormax`` caps the per-step growth factor of the ``PIDController`` (its
    own default is 10). A smaller cap (e.g. 3) damps the overshoot-then-reject
    oscillation that drives the step-rejection rate up on a stiff, forced system.
    ``None`` keeps the diffrax default.

    ``dtmax`` caps the integrator step size. It is ``None`` (uncapped) by
    default, which is fastest for plain forward solves. For *reverse-mode*
    differentiation of a stiff model it must be set. An L-stable solver may
    take steps far larger than the fastest reaction timescale and simply damp
    the unresolved fast modes in the primal (which stays accurate). The two AD
    modes then diverge: **forward mode** (``jax.jvp`` / ``jax.jacfwd``) stays
    finite at any step, losing only accuracy when the fast modes are
    unresolved; **reverse mode** (``jax.grad``, the discrete adjoint) returns
    **non-finite** values above a step-size threshold, an overflow in the
    backward accumulation governed by the per-step stiffness ``gamma*dt*||J||``
    (not by operator conditioning). Capping ``dtmax`` to a small multiple of
    the fastest reaction timescale bounds that product; the resulting reverse
    gradient is finite and matches both forward mode and finite differences.
    This is reverse-mode-specific and independent of the adjoint flavour. See
    the "Differentiating stiff models" discussion in CLAUDE.md.
    """
    term = diffrax.ODETerm(rhs)
    # Build the solver + controller from the shared single-source-of-truth helpers
    # (the decoupled-Newton default solver and the PID controller) so this forward
    # path and the discrete-adjoint forward pass cannot drift in their per-step
    # configuration. A user-supplied ``solver`` is honoured verbatim here.
    solver = build_implicit_solver(rtol, atol, order=order, solver=solver)
    controller = build_step_controller(rtol, atol, factormax=factormax, dtmax=dtmax)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
        event=event,
        progress_meter=(
            progress_meter if progress_meter is not None else diffrax.NoProgressMeter()
        ),
    )


def make_chemistry_rhs(
    model: CompiledModel,
    params: jnp.ndarray,
    *,
    cond_fn: Callable[[jnp.ndarray], Mapping[str, jnp.ndarray]],
    rate_scale=None,
) -> Callable:
    """Build the canonical reaction right-hand side ``rhs(t, C, args)``.

    Hoists the (parameter-dependent) stoichiometry once, then returns
    ``dC/dt = rate_scale * dCdt(C, params, cond_fn(t))``. This is the **single
    source** for the reaction RHS: both the plain solve (:func:`solve_chemistry`)
    and the event-driven segmented solve build their RHS through here, so the two
    paths cannot drift (a change to how the RHS is assembled reaches both). The
    threaded ``args`` carries the live ``params`` for the rate constants while the
    stoichiometry is the hoisted constant.

    Parameters
    ----------
    model : CompiledModel
    params : jnp.ndarray
        Parameter vector; the stoichiometry is hoisted from it once.
    cond_fn : callable
        ``cond_fn(t)`` returning the condition arrays at independent value ``t``
        (a constant dict for batch/CFD; an interpolation for PFR/particle).
    rate_scale : optional
        ``None`` (identity) or a scalar multiplying the rate (e.g. ``1/velocity``
        for the steady-state PFR).
    """
    stoich = model.compute_stoich(params)
    if rate_scale is None:

        def rhs(t, C, args):
            return model.dCdt(C, args, cond_fn(t), 0, stoich=stoich)
    else:

        def rhs(t, C, args):
            return model.dCdt(C, args, cond_fn(t), 0, stoich=stoich) * rate_scale

    return rhs


def solve_chemistry(
    model: CompiledModel,
    C0: jnp.ndarray,
    params: jnp.ndarray,
    *,
    cond_fn: Callable[[jnp.ndarray], Mapping[str, jnp.ndarray]],
    saveat: diffrax.SaveAt,
    t0,
    t1,
    rtol: float,
    atol,
    adjoint: diffrax.AbstractAdjoint | None = None,
    dtmax: float | None = None,
    max_steps: int = 100_000,
    rate_scale=None,
    order: int = 5,
    factormax: float | None = None,
    solver: diffrax.AbstractSolver | None = None,
):
    """The canonical chemistry sub-solve shared by every reactor.

    Hoists the (parameter-dependent) stoichiometry out of the per-step RHS ---
    so dynamic coefficients are evaluated once per solve, not per step --- builds
    the right-hand side ``dC/dt = rate_scale * dCdt(C, params, cond_fn(t))`` and
    runs the Kvaerno5 + ``PIDController`` solve via :func:`_run_diffeqsolve`.

    The reactors differ only in three traced-time choices, passed in here:

    - ``cond_fn(t)`` returns the condition arrays at independent-variable value
      ``t``. A batch / CFD cell passes a constant dict; a PFR or particle track
      passes an interpolation of its spatially / temporally varying fields.
    - ``rate_scale`` (``None`` = identity) multiplies the rate, e.g. ``1/velocity``
      for the steady-state PFR whose independent variable is axial position.
    - ``saveat`` / ``t0`` / ``t1`` select the output points and the span.

    Returns the diffrax ``Solution``; callers read ``sol.ts`` / ``sol.ys`` (or
    ``sol.ys[-1]`` for a single-endpoint step).
    """
    rhs = make_chemistry_rhs(model, params, cond_fn=cond_fn, rate_scale=rate_scale)

    return _run_diffeqsolve(
        rhs,
        t0=t0,
        t1=t1,
        y0=C0,
        args=params,
        saveat=saveat,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        dtmax=dtmax,
        max_steps=max_steps,
        order=order,
        factormax=factormax,
        solver=solver,
    )


def _interp_fields_to_scalar(
    t: jnp.ndarray,
    t_grid: jnp.ndarray,
    fields: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Linearly interpolate every field to a single-location array at ``t``.

    Returns a fresh dict whose values are shape-``(1,)`` arrays. Reactors
    pair this with ``loc_idx=0`` to satisfy the canonical rate-callable
    signature.
    """
    return {name: jnp.asarray([jnp.interp(t, t_grid, arr)]) for name, arr in fields.items()}
