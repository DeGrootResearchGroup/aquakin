"""Colored-AD Jacobian materialization for the implicit-stage solve.

A stiff implicit Runge--Kutta step (Kvaerno5/3) solves, once per step, a
nonlinear stage equation by a chord-Newton iteration whose linear operator is
``I - gamma.dt.J`` with ``J = df/dy`` the right-hand-side Jacobian. Diffrax's
``VeryChord`` root-finder materializes and factorizes that operator **once per
step**, reusing it across the stages and Newton iterations. For a large
flowsheet RHS (the plant) the materialization -- forming the dense ``n x n``
matrix -- dominates the per-step linear algebra, far above the factorization.

When ``J`` is sparse (the plant Jacobian is ~5-15% nonzero: dense per-unit
kinetic blocks plus sparse inter-unit flow coupling), it can be formed in far
fewer than ``n`` Jacobian-vector products by **column compression** (Curtis,
Powell & Reid 1974): group columns that share no nonzero row ("structurally
orthogonal") into colors, push one seed per color through a single forward
linearization, and scatter each color's result back to its columns using the
sparsity pattern. ``C`` colors instead of ``n`` -- for the plant ``C`` is set by
the widest dense block (the digester), ~35-45 vs ``n = 167``.

The reconstructed matrix is **identical** to the dense Jacobian (to round-off)
whenever the pattern is a true superset of the real nonzeros, so the chord
iteration -- hence the step sequence, the trajectory, and the gradient -- is
numerically unchanged; only the cost of forming ``J`` drops. A pattern that
*misses* a nonzero does not corrupt the result (the chord still converges to the
stage residual's root) but degrades convergence, so it costs steps, not
accuracy. The pattern is therefore built conservatively (see
:func:`jacobian_sparsity_pattern`) and validated against a dense Jacobian at
setup (:func:`colored_jacobian_max_error`), with the caller falling back to the
dense path on any mismatch.

This is wired into a plant solve by :meth:`aquakin.plant.plant.Plant.solve`
through ``colored_jacobian=True``; the heavy lifting (pattern, coloring,
root-finder) is built once per plant and reused.
"""

import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import lineax.internal as lxi
import numpy as np
import optimistix as optx
from diffrax._root_finder._verychord import VeryChord, _NoAux, _VeryChordState
from lineax.internal import complex_to_real_dtype

__all__ = [
    "ColoredVeryChord",
    "greedy_color",
    "jacobian_sparsity_pattern",
    "structural_sparsity_pattern",
    "build_colored_root_finder",
    "colored_jacobian_max_error",
    "materialize_colored_jacobian",
    "colored_jacobian_guard",
    "COLORED_JACOBIAN_GUARD_RTOL",
]


# Relative tolerance for the setup-time colored-vs-dense Jacobian guard: the
# colored matrix must match the dense one at the start state to within this
# fraction of the dense Jacobian's largest entry, else the caller falls back to
# the dense path. One definition shared by every guard site.
COLORED_JACOBIAN_GUARD_RTOL = 1e-8


def materialize_colored_jacobian(root_finder, f, y):
    """Materialize ``df/dy`` at ``y`` by column compression.

    Linearizes ``f`` once at ``y`` (a single primal pass through the full RHS),
    pushes the color seeds through that one tangent map, and scatters each
    color's Jacobian-vector product back to its columns on ``root_finder``'s
    sparsity pattern. Equal to ``jax.jacfwd(f)(y)`` on the pattern's support (to
    round-off) when the pattern is a true superset of the real nonzeros.

    This is the single definition of the colored materialization shared by the
    per-step solve (:meth:`ColoredVeryChord.init`), the setup guard
    (:func:`colored_jacobian_max_error`), and the discrete-adjoint / steady-state
    Jacobian builders.

    Parameters
    ----------
    root_finder : ColoredVeryChord
        Carries the ``seed_matrix`` ``(n, C)``, the ``color_of`` ``(n,)`` column
        map, and the ``(n, n)`` ``pattern``.
    f : Callable
        ``y -> f(y)``, the function whose Jacobian is formed.
    y : jnp.ndarray
        Point at which to linearize.

    Returns
    -------
    jnp.ndarray
        The ``(n, n)`` Jacobian on the pattern's support.
    """
    _, lin = jax.linearize(f, y)
    JS = jax.vmap(lin, in_axes=1, out_axes=1)(root_finder.seed_matrix)  # (n, C)
    return JS[:, root_finder.color_of] * root_finder.pattern  # (n, n)


def greedy_color(pattern: np.ndarray) -> np.ndarray:
    """Greedy CPR (forward) column coloring of a Jacobian sparsity pattern.

    Two columns may share a color iff they are *structurally orthogonal* -- no
    row has a nonzero in both -- so the column-compressed forward AD can scatter
    a color's combined Jacobian-vector product back to the individual columns
    unambiguously. Columns are colored most-constrained-first (descending
    conflict degree), which keeps the color count near the chromatic number.

    Parameters
    ----------
    pattern : np.ndarray
        Boolean ``(n, n)`` sparsity pattern; ``pattern[i, j]`` is True if column
        ``j`` may have a nonzero in row ``i``.

    Returns
    -------
    np.ndarray
        Integer ``(n,)`` array of colors in ``0 .. C-1``.
    """
    P = np.asarray(pattern).astype(np.int32)
    n = P.shape[1]
    # columns j, k conflict iff they share a nonzero row: (P^T P)[j, k] > 0.
    conflict = (P.T @ P) > 0
    np.fill_diagonal(conflict, False)
    order = np.argsort(-conflict.sum(axis=1))  # most-constrained first
    color = -np.ones(n, dtype=int)
    for j in order:
        used = {color[k] for k in np.where(conflict[j])[0] if color[k] >= 0}
        c = 0
        while c in used:
            c += 1
        color[j] = c
    return color


def jacobian_sparsity_pattern(
    rhs: Callable,
    y0: jnp.ndarray,
    *,
    n_probe: int = 24,
    seed: int = 0,
    rel_tol: float = 1e-11,
) -> np.ndarray:
    """Conservative (superset) sparsity pattern of ``J = d rhs / dy`` near ``y0``.

    The pattern is the union of the numerical nonzeros of ``J`` over a set of
    **strictly-positive** probe states drawn at two scales, plus ``y0`` itself.
    Two failure modes must both be covered, and a single probe scale covers only
    one:

    - *Depleted-at-y0 components.* A species sitting at zero zeroes the
      columns/rows that couple through it, so a probe there misses structurally
      present entries. Lifting every component to ``|y0| + 1`` and jittering
      reveals those couplings (the historical scheme).
    - *Small-natural-scale components.* A species whose physical scale is far
      below one (the ADM1 dissolved hydrogen ``S_h2`` sits at ~``1e-7`` at its
      inhibition knee, where its Jacobian column is enormous) is pushed by the
      ``|y0| + 1`` lift into a *saturated* regime where its column goes flat and
      its huge near-zero gradient vanishes. Probing each component around its
      **own** magnitude keeps such a state in its physical regime so its column
      is captured.

    The pattern therefore unions, at ``rel_tol`` relative threshold: the
    Jacobian at ``y0`` (the actual operating regime, which makes the start-state
    guard pass by construction), ``n_probe`` own-scale multiplicative-jitter
    probes, and ``n_probe`` lifted multiplicative-jitter probes. The full
    diagonal is always included (the implicit-stage residual ``I - c.J`` has a
    nonzero diagonal regardless of ``J``).

    The result is a superset of the true pattern with overwhelming reliability;
    a missed entry costs solver steps, not accuracy, and the setup-time guard
    (:func:`colored_jacobian_max_error`) catches a gross miss at the start state.

    Parameters
    ----------
    rhs : Callable
        ``y -> dy/dt``, the (already condition/parameter-bound) right-hand side.
    y0 : jnp.ndarray
        Reference state, shape ``(n,)``; sets the per-component probe scale.
    n_probe : int, optional
        Number of probe states *per scale* (default 24); the total is
        ``2*n_probe + 1`` (own-scale, lifted, and ``y0``).
    seed : int, optional
        RNG seed for reproducibility (default 0).
    rel_tol : float, optional
        Relative magnitude threshold for counting an entry nonzero.

    Returns
    -------
    np.ndarray
        Boolean ``(n, n)`` superset pattern (diagonal included).
    """
    y0 = jnp.asarray(y0)
    n = y0.shape[0]
    fj = jax.jit(jax.jacfwd(rhs))
    rng = np.random.default_rng(seed)
    ay0 = np.abs(np.asarray(y0))
    # Own-scale = each component's own magnitude; an exactly-zero component has no
    # scale to jitter around, so it is floored to a tiny fraction of the typical
    # nonzero magnitude (small enough to stay in the depleted regime, positive so
    # the rate kinetics are well defined).
    typ = float(np.median(ay0[ay0 > 0])) if np.any(ay0 > 0) else 1.0
    own = np.where(ay0 > 0.0, ay0, typ * 1e-6)
    lifted = ay0 + 1.0  # lift depleted components above 1
    P = np.eye(n, dtype=bool)

    def _accumulate(ys):
        J = np.asarray(fj(jnp.asarray(ys)))
        nonlocal P
        P |= np.abs(J) > rel_tol * (np.abs(J).max() + 1e-300)

    # The actual operating regime at y0 (zeros lifted to the tiny own floor so
    # the rates are evaluated at a strictly-positive state near y0). This makes
    # the start-state guard pass and captures small-natural-scale columns.
    _accumulate(np.where(ay0 > 0.0, ay0, own))
    for _ in range(n_probe):  # own-scale: physical regime
        _accumulate(own * np.exp(rng.normal(0.0, 1.0, size=n)))
    for _ in range(n_probe):  # lifted: depleted-coupling regime
        _accumulate(lifted * np.exp(rng.normal(0.0, 1.0, size=n)))
    return P


def structural_sparsity_pattern(model, params=None) -> np.ndarray:
    """Exact **structural** Jacobian sparsity pattern from a model's equations.

    Unlike :func:`jacobian_sparsity_pattern`, which thresholds ``|J| > tol`` at
    sampled states, this is **state-free**: it asks which species each rate
    *equation references*, not which couplings are numerically active at some
    probe point. That distinction matters because a saturated Monod term
    ``S/(K+S)`` with ``S >> K`` has a tiny but nonzero sensitivity ``~K/S^2`` --
    structurally present, numerically below threshold -- so a probe taken at one
    operating regime (e.g. a warm-start steady state) **drops couplings that
    activate in another** (a dynamic load excursion drives substrates into the
    Monod-limiting range, inhibitors across their thresholds, pH through a
    ``pH_switch`` pKa). The dropped couplings are the stiff/fast ones, so a stale
    pattern wrecks the chord-Newton convergence (see the issue history). The
    structural pattern cannot go stale: it is a superset of every coupling the
    equations *can* express, for any influent.

    Construction. For each reaction ``r`` the species it can affect are the
    nonzeros of its stoichiometry row (static ``stoich_matrix`` **and** the
    symbolic ``stoich_dynamic`` entries), and the species its rate can depend on
    are the syntactic species of its rate AST (:meth:`ASTNode.species`) plus --
    when the rate reads a derived condition such as a state-derived ``pH`` -- the
    species feeding that derived field. The derived-field dependencies are found
    by one forward-AD pass of the (always-on, non-saturating) charge-balance
    ``derived_condition_fn``: ``d(pH)/d(total_i)`` is nonzero for every total the
    charge balance carries, regardless of regime, so a single evaluation gives
    the exact structural set. The Jacobian diagonal is always included.

    Parameters
    ----------
    model : CompiledModel
        The compiled model (its ``rate_asts``, ``stoich_matrix`` /
        ``stoich_dynamic``, ``species_index`` and optional
        ``derived_condition_fn`` / ``derived_fields``).
    params : jnp.ndarray, optional
        Parameter vector for evaluating the derived-condition AD (default
        ``model.default_parameters()``); only its structure matters, not its
        values.

    Returns
    -------
    np.ndarray
        Boolean ``(n_species, n_species)`` structural superset (diagonal
        included).
    """
    n = model.n_species
    si = model.species_index
    P = np.eye(n, dtype=bool)
    if params is None:
        params = model.default_parameters()

    # affects[r] = species reaction r can change (static + symbolic stoichiometry)
    stoich = np.asarray(model.stoich_matrix)
    affects = [set(np.nonzero(stoich[r])[0].tolist()) for r in range(stoich.shape[0])]
    for r, j, _fn in model.stoich_dynamic:
        affects[r].add(int(j))

    # derived condition (pH, ...) -> the species that feed it. The charge balance
    # depends on every total it carries at any state, so one forward-AD pass of
    # the always-on derived fn gives the exact structural dependency set.
    derived_deps: dict[str, set[int]] = {}
    if model.derived_condition_fn is not None and model.derived_fields:
        conds = {k: jnp.asarray(v) for k, v in model.default_conditions().fields.items()}
        c_generic = jnp.maximum(jnp.abs(model.default_concentrations()), 1.0)
        fields = list(model.derived_fields)

        def _derived(c):
            out = model.derived_condition_fn(c, params, conds, 0)
            return jnp.stack([jnp.reshape(out[f], ()) for f in fields])

        jac_derived = np.asarray(jax.jacfwd(_derived)(c_generic))  # (n_fields, n)
        for k, f in enumerate(fields):
            derived_deps[f] = set(np.nonzero(jac_derived[k] != 0.0)[0].tolist())

    for r, ast in enumerate(model.rate_asts):
        deps = {si[s] for s in ast.species()}
        for cond in ast.condition_names():
            if cond in derived_deps:
                deps |= derived_deps[cond]
        if deps:
            dep_idx = list(deps)
            for i in affects[r]:
                P[i, dep_idx] = True
    return P


class ColoredVeryChord(VeryChord):
    """:class:`diffrax.VeryChord` that materializes the per-step Jacobian by
    column-compressed forward AD instead of the dense ``lineax`` linearization.

    Only :meth:`init` (which forms and factorizes the operator once per step)
    differs from the parent; ``step`` / ``terminate`` / ``postprocess`` are
    inherited unchanged, so the chord iteration is identical -- it just receives
    a Jacobian formed in ``C`` Jacobian-vector products (one per color) rather
    than ``n``. The materialized matrix equals the dense Jacobian to round-off
    when ``pattern`` is a true superset of the real nonzeros.

    The linear solver defaults to a plain dense ``lineax.LU`` (the colored matrix
    is explicit and the implicit operator ``I - gamma.dt.J`` is well-conditioned),
    avoiding the well-posedness probing of the parent's ``AutoLinearSolver``.

    Assumes a flat ``(n,)`` state ``y`` (the plant's concatenated state vector).

    Fields
    ------
    seed_matrix : jax.Array
        ``(n, C)`` color seed matrix; ``seed_matrix[j, color_of[j]] = 1``.
    color_of : jax.Array
        ``(n,)`` int column -> color map.
    pattern : jax.Array
        ``(n, n)`` 0/1 sparsity superset used to scatter each color's
        Jacobian-vector product back to its columns (diagonal included).
    """

    seed_matrix: jax.Array = None
    color_of: jax.Array = None
    pattern: jax.Array = None
    linear_solver: lx.AbstractLinearSolver = lx.LU()

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        try:
            return options["init_state"]
        except KeyError:
            pass
        g = _NoAux(fn)
        # Linearize ONCE (a single primal pass through the full RHS, incl. the
        # recycle and pH solves) and push the C color seeds through that one
        # tangent map, scattering each color's JVP back to its columns. Using
        # jax.jvp per color would redo the expensive nonlinear primal C times.
        J = materialize_colored_jacobian(self, lambda z: g(z, args), y)
        jac = lx.MatrixLinearOperator(J, tags=tags)
        init_later_state = self.linear_solver.init(jac, options={})
        dynamic, static = eqx.partition(init_later_state, eqx.is_array)
        dynamic = lax.stop_gradient(dynamic)
        init_later_state = eqx.combine(dynamic, static)
        linear_state = (jac, init_later_state)
        y_leaves = jtu.tree_leaves(y)
        if len(y_leaves) == 0:
            y_dtype = lxi.default_floating_dtype()
        else:
            y_dtype = jnp.result_type(*y_leaves)
        diff_dtype = complex_to_real_dtype(y_dtype)
        return _VeryChordState(
            linear_state=linear_state,
            diff=jtu.tree_map(lambda x: jnp.full(x.shape, jnp.inf, x.dtype), y),
            diffsize=jnp.array(jnp.inf, dtype=diff_dtype),
            diffsize_prev=jnp.array(1.0, dtype=diff_dtype),
            result=optx.RESULTS.successful,
            step=jnp.array(0),
        )


def build_colored_root_finder(
    rhs: Callable,
    y0: jnp.ndarray,
    *,
    rtol: float,
    atol: float,
    n_probe: int = 24,
    seed: int = 0,
    extra_pattern: "np.ndarray | None" = None,
    probe_pattern: "np.ndarray | None" = None,
) -> tuple[ColoredVeryChord, int]:
    """Build a :class:`ColoredVeryChord` for ``rhs`` linearized near ``y0``.

    Derives the sparsity pattern (:func:`jacobian_sparsity_pattern`), colors it
    (:func:`greedy_color`), and packs the seed matrix / color map / pattern into
    the root finder. ``rtol`` / ``atol`` are the chord-Newton tolerances (the
    plant passes the decoupled, 10x-loosened step tolerances, matching the
    default solver).

    ``extra_pattern`` (an ``(n, n)`` boolean array) is unioned with the probed
    pattern before coloring. The plant passes the **structural** (equation-derived)
    Jacobian blocks here (:func:`structural_sparsity_pattern`): the numerical
    probe at ``y0`` captures the linear always-on couplings (flow/recycle) but
    drops kinetic couplings that are saturated at ``y0`` and only activate off it,
    so the structural blocks restore them and the pattern cannot go stale.

    Returns ``(root_finder, n_colors)``.
    """
    if probe_pattern is not None:
        P = np.asarray(probe_pattern, dtype=bool)  # caller already probed
    else:
        P = jacobian_sparsity_pattern(rhs, y0, n_probe=n_probe, seed=seed)
    if extra_pattern is not None:
        P = P | np.asarray(extra_pattern, dtype=bool)
    color = greedy_color(P)
    n_colors = int(color.max() + 1)
    n = P.shape[0]
    S = np.zeros((n, n_colors))
    S[np.arange(n), color] = 1.0
    rf = ColoredVeryChord(
        rtol=rtol,
        atol=atol,
        seed_matrix=jnp.asarray(S),
        color_of=jnp.asarray(color),
        pattern=jnp.asarray(P, dtype=jnp.result_type(float)),
    )
    return rf, n_colors


def colored_jacobian_max_error(
    rhs: Callable,
    y: jnp.ndarray,
    root_finder: ColoredVeryChord,
) -> float:
    """Max absolute difference between the colored and dense ``d rhs / dy`` at ``y``.

    The setup-time correctness guard: a small value confirms the coloring +
    pattern reconstruct the true Jacobian at ``y``; a large value means the
    pattern missed a nonzero, and the caller should fall back to the dense
    solver. (Validates ``rhs``'s Jacobian directly; the implicit-stage residual
    ``I - c.J`` shares this pattern plus the always-included diagonal.)
    """
    Jd = jax.jacfwd(rhs)(y)
    Jc = materialize_colored_jacobian(root_finder, rhs, y)
    return float(jnp.max(jnp.abs(Jd - Jc)))


def colored_jacobian_guard(
    rhs: Callable,
    y0: jnp.ndarray,
    root_finder: ColoredVeryChord,
    *,
    context: str,
    stacklevel: int = 3,
) -> bool:
    """Setup-time guard: does the colored Jacobian match the dense one at ``y0``?

    Compares the colored and dense ``d rhs / dy`` at ``y0``
    (:func:`colored_jacobian_max_error`) against
    :data:`COLORED_JACOBIAN_GUARD_RTOL` times the dense Jacobian's largest entry.
    Returns ``True`` when they agree; on a mismatch (the sparsity pattern missed
    a nonzero) it ``warnings.warn``s -- with ``context`` naming the caller -- and
    returns ``False`` so the caller can fall back to the dense path. The single
    definition of the colored-Jacobian correctness check shared by the forward
    solver, the stable-adjoint backward and the steady-state PTC builders.

    Parameters
    ----------
    rhs : Callable
        ``y -> f(y)`` whose Jacobian is colored.
    y0 : jnp.ndarray
        Start state to validate at.
    root_finder : ColoredVeryChord
        The colored root finder / Jacobian materializer to validate.
    context : str
        Short label naming the caller for the warning (e.g.
        ``"colored_jacobian=True"``).
    stacklevel : int, optional
        ``warnings.warn`` stack level (default 3, to point at the caller's
        caller).
    """
    err = colored_jacobian_max_error(rhs, y0, root_finder)
    jscale = float(jnp.max(jnp.abs(jax.jacfwd(rhs)(y0)))) + 1e-300
    ok = err <= COLORED_JACOBIAN_GUARD_RTOL * jscale
    if not ok:
        warnings.warn(
            f"{context}: the derived Jacobian sparsity pattern disagrees with "
            f"the dense Jacobian at the start state (max abs error {err:.2e}, "
            f"scale {jscale:.2e}); falling back to dense. This indicates the "
            f"structural pattern missed a nonzero -- please report it.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return ok
