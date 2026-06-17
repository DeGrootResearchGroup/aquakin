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
from lineax.internal import complex_to_real_dtype

from diffrax._root_finder._verychord import VeryChord, _NoAux, _VeryChordState


__all__ = [
    "ColoredVeryChord",
    "greedy_color",
    "jacobian_sparsity_pattern",
    "build_colored_root_finder",
    "colored_jacobian_max_error",
]


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
    order = np.argsort(-conflict.sum(axis=1))   # most-constrained first
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

    The pattern is the union of the numerical nonzeros of ``J`` over ``n_probe``
    **strictly-positive** probe states. Probing at positive states (every
    component floored above zero, then log-normally jittered) is essential: at a
    state with a *depleted* component (a species at zero), the columns/rows that
    couple through that component vanish, so a probe there misses structurally
    present entries. Floored-positive probes reveal every term that can be
    nonzero. The full diagonal is always included (the implicit-stage residual
    ``I - c.J`` has a nonzero diagonal regardless of ``J``).

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
        Number of probe states (default 24).
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
    base = np.abs(np.asarray(y0)) + 1.0          # strictly-positive per-component floor
    P = np.eye(n, dtype=bool)
    for _ in range(n_probe):
        ys = jnp.asarray(base * np.exp(rng.normal(0.0, 1.0, size=n)))   # always > 0
        J = np.asarray(fj(ys))
        scale = np.abs(J).max() + 1e-300
        P |= np.abs(J) > rel_tol * scale
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
        # recycle and pH solves), then push the C color seeds through the SAME
        # linear tangent map. Using jax.jvp per color would redo the expensive
        # nonlinear primal C times.
        _, lin = jax.linearize(lambda z: g(z, args), y)
        JS = jax.vmap(lin, in_axes=1, out_axes=1)(self.seed_matrix)   # (n, C)
        # Scatter: column j takes its color's combined JVP, masked to the rows
        # where (i, j) is structurally present. Exact because columns sharing a
        # color share no pattern row.
        J = JS[:, self.color_of] * self.pattern                      # (n, n)
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
) -> tuple[ColoredVeryChord, int]:
    """Build a :class:`ColoredVeryChord` for ``rhs`` linearized near ``y0``.

    Derives the sparsity pattern (:func:`jacobian_sparsity_pattern`), colors it
    (:func:`greedy_color`), and packs the seed matrix / color map / pattern into
    the root finder. ``rtol`` / ``atol`` are the chord-Newton tolerances (the
    plant passes the decoupled, 10x-loosened step tolerances, matching the
    default solver).

    Returns ``(root_finder, n_colors)``.
    """
    P = jacobian_sparsity_pattern(rhs, y0, n_probe=n_probe, seed=seed)
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
    _, lin = jax.linearize(rhs, y)
    JS = jax.vmap(lin, in_axes=1, out_axes=1)(root_finder.seed_matrix)
    Jc = JS[:, root_finder.color_of] * root_finder.pattern
    return float(jnp.max(jnp.abs(Jd - Jc)))
