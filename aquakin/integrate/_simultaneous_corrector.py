"""CVODES-style simultaneous-corrector linear solver for forward sensitivity.

The forward-sensitivity solve integrates the augmented state ``z = [y; vec(S)]``
(the state plus its sensitivity ``S = dy/dtheta`` with respect to ``k`` free
parameters). A stiff ESDIRK step (Kvaerno5) solves, per Newton iteration, a
linear system with operator

    M = I - gamma.dt.(dF/dz)

where ``F`` is the augmented right-hand side. Because ``S`` does not feed back
into ``y`` and the sensitivity columns do not couple to each other, ``dF/dz`` is
**block-lower-triangular in "arrow" form**: every diagonal block equals the
state Jacobian ``J = df/dy``, each sensitivity column ``S_j`` couples only to
``y`` (an off-diagonal block ``L_j``), and ``S_i``/``S_j`` blocks are zero. So
``M`` has identical diagonal blocks ``D = I - gamma.dt.J`` and solving ``M x = b``
needs only

    x_y    = D^{-1} b_y
    x_{Sj} = D^{-1} (b_{Sj} - L_j x_y)

i.e. **one ``n x n`` factorization of ``D`` reused across all ``k+1`` blocks**
(forward substitution), instead of factorizing the full
``n(1+k) x n(1+k)`` system whose cost grows like ``(1+k)^3``. This is CVODES'
*simultaneous corrector* (Hindmarsh et al. 2005; Maly & Petzold 1996).

Diffrax's ``Kvaerno5`` runs its implicit stages through a ``VeryChord``
root-finder that calls ``linear_solver.init`` once per step (the natural home
for the factorization) and ``linear_solver.compute`` once per Newton iteration
(the back-solve), so pointing that root-finder at :class:`SimultaneousCorrector`
gives the factorization reuse for free, with no changes to diffrax.

The solver sees only the matrix-free augmented operator ``M`` (a
``FunctionLinearOperator`` that diffrax builds). Probing ``M.mv`` on the ``n``
state-unit tangents yields each full augmented column at once: the state rows
give the diagonal block ``D`` and the sensitivity rows give the off-diagonal
coupling blocks ``L_j`` -- so a single ``vmap`` of ``n`` matrix-vector products
in ``init`` materialises ``D`` (factorised once) and every ``L_j`` (small,
``n x n``) with no extra cost, and the per-iteration ``compute`` is then pure
BLAS (one triangular back-solve per block plus an ``L_j x_y`` matmul). The
resulting Newton step is **exact** -- identical to a dense LU of ``M`` -- so the
root-finder converges identically; only the per-step linear-algebra cost
changes. The state layout it assumes is the column-major augmented vector
``[y (n); S_0 (n); ... ; S_{k-1} (n)]`` produced by
:func:`aquakin.integrate.forward_sensitivity.augmented_forward_sensitivity`.

Only plain arrays (the LU factors and the ``L_j`` stack) are stored in the
solver state -- never the operator itself, whose closure-converted function
carries a ``Jaxpr`` that diffrax cannot route through the ``lax.cond`` /
``stop_gradient`` it uses to cache the Jacobian across stages.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from lineax import RESULTS


class SimultaneousCorrector(lx.AbstractLinearSolver):
    """Block-arrow linear solver: factorize the diagonal block once, reuse it.

    Parameters
    ----------
    ndof : int
        Size ``n`` of the state block (degrees of freedom of ``y``).
    n_sens : int
        Number ``k`` of sensitivity columns. The full augmented system has size
        ``ndof * (1 + n_sens)``.
    """

    ndof: int = eqx.field(static=True)
    n_sens: int = eqx.field(static=True)

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]):
        del options
        n, k = self.ndof, self.n_sens
        in_struct = operator.in_structure()
        N = n * (1 + k)
        if in_struct.shape != (N,):
            raise ValueError(
                f"SimultaneousCorrector expects a flat augmented operator of size "
                f"{N} = ndof*(1+n_sens); got in_structure {in_struct.shape}."
            )
        # Probe M.mv on the ndof state-unit tangents. Column i of the result is
        # M e_i: its state rows are column i of the diagonal block D, and its
        # S_j rows are column i of the off-diagonal coupling block L_j. So one
        # vmap materialises D and every L_j together.
        basis = jnp.eye(N, n, dtype=in_struct.dtype)  # columns e_0..e_{n-1}
        cols = jax.vmap(operator.mv, in_axes=1, out_axes=1)(basis)  # (N, n)
        D = cols[:n]  # (n, n)
        L = cols[n:].reshape(k, n, n)  # L[j] = (n, n) block
        lu = jsl.lu_factor(D)
        return (lu, L, eqxi.Static(False))

    def compute(self, state, vector, options):
        del options
        lu, L, transposed = state
        n, k = self.ndof, self.n_sens
        b_y = vector[:n]
        b_S = vector[n:].reshape(k, n)  # row j = b for S_j

        if not transposed.value:
            # Forward substitution on the lower-arrow system:
            #   x_y    = D^{-1} b_y;  x_{Sj} = D^{-1} (b_{Sj} - L_j x_y).
            x_y = jsl.lu_solve(lu, b_y)
            coupling = jnp.einsum("jab,b->ja", L, x_y)  # row j = L_j x_y
            x_S = jax.vmap(lambda r: jsl.lu_solve(lu, r))(b_S - coupling)
        else:
            # Transposed (upper-arrow) system M^T x = b: solve the S blocks
            # first, then y, with the coupling sum_j L_j^T x_{Sj}.
            x_S = jax.vmap(lambda r: jsl.lu_solve(lu, r, trans=1))(b_S)
            coupling_y = jnp.einsum("jab,ja->b", L, x_S)  # sum_j L_j^T x_{Sj}
            x_y = jsl.lu_solve(lu, b_y - coupling_y, trans=1)

        x = jnp.concatenate([x_y, x_S.reshape(-1)])
        return x, RESULTS.successful, {}

    def transpose(self, state, options):
        lu, L, transposed = state
        new_state = (lu, L, eqxi.Static(not transposed.value))
        return new_state, options

    def conj(self, state, options):
        # Real-valued use only; conjugation is a no-op on real data. Conjugate
        # every array leaf so the contract conj(init(op)) == init(conj(op)) holds
        # generically without special-casing.
        state = jax.tree_util.tree_map(lambda x: jnp.conj(x) if eqx.is_array(x) else x, state)
        return state, options

    def assume_full_rank(self) -> bool:
        # D = I - gamma.dt.J is regularized away from singularity by the implicit
        # step, so the augmented operator is full rank.
        return True
