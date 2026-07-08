"""Constrained design optimization.

:func:`optimize_design` minimises (or maximises) an objective over a bounded
design space subject to inequality :class:`Constraint`\\ s, using autodiff
gradients through a gradient-based constrained NLP solver (SciPy). The objective
and each constraint share the ``fn(x) -> scalar`` contract of
:func:`aquakin.monte_carlo` -- they build the params / initial state and run the
solve themselves -- and must be JAX-differentiable, since their gradients are
taken by autodiff and handed to the optimizer. Quasi-random multistart draws its
starts from the shared :mod:`aquakin.integrate._qmc` unit sampler.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._qmc import _unit_sample

# --- Constrained design optimization -----------------------------------------


@dataclass
class Constraint:
    """An inequality constraint on a design optimization.

    ``fn(x)`` (a scalar function of the design vector, e.g. an effluent
    concentration or a total-nitrogen metric) must satisfy ``lower <= fn(x) <=
    upper``. Give an ``upper`` (a permit ceiling), a ``lower`` (a floor), or
    both. ``fn`` should be JAX-differentiable -- its gradient is taken by autodiff
    and handed to the optimizer.
    """

    fn: Callable
    upper: float | None = None
    lower: float | None = None
    name: str | None = None

    def __post_init__(self):
        if self.upper is None and self.lower is None:
            raise ValueError("a Constraint needs an 'upper' and/or 'lower' bound.")


@dataclass
class OptimizeResult:
    """Result of :func:`optimize_design`.

    Attributes
    ----------
    input_names : list[str]
        Names of the design variables (order of ``x``).
    x : np.ndarray
        The optimal design vector.
    objective : float
        Objective value at ``x`` (in the original sense -- already un-negated for
        a ``maximize`` run).
    constraint_values : dict
        ``name -> fn(x)`` for every constraint at the optimum.
    feasible : bool
        Whether every constraint is satisfied at ``x`` (within ``constraint_tol``).
    success : bool
        The optimizer reported convergence for the chosen start.
    message : str
        Optimizer status message.
    n_iter, n_starts : int
        Iterations of the winning run; number of multistart runs.
    """

    input_names: list[str]
    x: np.ndarray
    objective: float
    constraint_values: dict
    feasible: bool
    success: bool
    message: str
    n_iter: int
    n_starts: int

    @property
    def x_named(self) -> dict:
        """The optimal design as a ``name -> value`` dict."""
        return {n: float(v) for n, v in zip(self.input_names, self.x)}

    def report(self) -> str:
        lines = [
            f"optimize_design: objective = {self.objective:.6g} "
            f"({'feasible' if self.feasible else 'INFEASIBLE'}, "
            f"{'converged' if self.success else 'not converged'})"
        ]
        for n, v in self.x_named.items():
            lines.append(f"  {n} = {v:.6g}")
        for n, v in self.constraint_values.items():
            lines.append(f"  [{n}] = {v:.6g}")
        return "\n".join(lines)


def _jax_value_and_grad(f):
    """Return numpy ``value(x)`` and ``grad(x)`` callables for a scalar JAX fn."""
    vg = jax.jit(jax.value_and_grad(lambda x: jnp.asarray(f(x), dtype=float).reshape(())))

    def value(x):
        return float(vg(jnp.asarray(x, dtype=float))[0])

    def grad(x):
        return np.asarray(vg(jnp.asarray(x, dtype=float))[1], dtype=float)

    return value, grad


def optimize_design(
    objective: Callable,
    bounds: Sequence,
    *,
    input_names: Sequence[str] | None = None,
    constraints: Sequence[Constraint] = (),
    x0: Sequence[float] | None = None,
    maximize: bool = False,
    method: str = "SLSQP",
    n_starts: int = 1,
    seed: int = 0,
    tol: float = 1e-6,
    constraint_tol: float = 1e-4,
) -> OptimizeResult:
    """Minimise (or maximise) an objective over a bounded design space subject to
    inequality constraints, using autodiff gradients.

    The canonical use is "size a design to a permit at minimum cost": ``objective``
    is an operational-cost / energy metric and each :class:`Constraint` is an
    effluent-quality ceiling. ``objective`` and the constraint functions share the
    ``fn(x) -> scalar`` contract of :func:`monte_carlo` -- they build the params /
    initial state and run the solve themselves -- and must be JAX-differentiable,
    since their gradients are taken by autodiff and passed to the optimizer (a
    gradient-based, constrained NLP solver via SciPy).

    Parameters
    ----------
    objective : callable
        ``objective(x) -> scalar`` to minimise (or maximise; see ``maximize``).
    bounds : sequence of (low, high)
        Box bounds for each design variable, length ``d``.
    input_names : sequence of str, optional
        Design-variable names (defaults to ``x0..``).
    constraints : sequence of Constraint
        Inequality constraints ``lower <= c.fn(x) <= upper``.
    x0 : sequence of float, optional
        Starting point. Defaults to the box centre; with ``n_starts > 1`` it is
        ignored in favour of quasi-random starts.
    maximize : bool
        Maximise instead of minimise.
    method : str
        SciPy constrained method (default ``"SLSQP"``; ``"trust-constr"`` also
        works with bounds + constraints).
    n_starts : int
        Multistart count -- quasi-random (Sobol) starts in the box; the best
        feasible optimum is returned. Escapes local minima on multimodal designs.
    seed : int
        Seed for the multistart sampler (reproducible).
    tol, constraint_tol : float
        Optimizer tolerance and the slack within which a constraint counts as
        satisfied when judging feasibility / picking the multistart winner.

    Returns
    -------
    OptimizeResult
    """
    from scipy.optimize import minimize

    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    d = len(bounds)
    if input_names is None:
        input_names = [f"x{j}" for j in range(d)]
    elif len(input_names) != d:
        raise ValueError(f"input_names has {len(input_names)} entries but bounds has d={d}.")
    input_names = list(input_names)
    sign = -1.0 if maximize else 1.0

    obj_val, obj_grad = _jax_value_and_grad(lambda x: sign * objective(x))
    con_val_grad = [(c, *_jax_value_and_grad(c.fn)) for c in constraints]

    # SciPy inequality constraints: g(x) >= 0. upper -> upper - fn >= 0;
    # lower -> fn - lower >= 0. Jacobians come from the autodiff gradient.
    scipy_cons = []
    for c, cval, cgrad in con_val_grad:
        if c.upper is not None:
            scipy_cons.append(
                {
                    "type": "ineq",
                    "fun": (lambda x, cv=cval, u=c.upper: u - cv(x)),
                    "jac": (lambda x, cg=cgrad: -cg(x)),
                }
            )
        if c.lower is not None:
            scipy_cons.append(
                {
                    "type": "ineq",
                    "fun": (lambda x, cv=cval, lo=c.lower: cv(x) - lo),
                    "jac": (lambda x, cg=cgrad: cg(x)),
                }
            )

    # Starting points: x0 (or box centre) for a single start; quasi-random
    # otherwise.
    if n_starts > 1:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        U, _ = _unit_sample(d, n_starts, "sobol", seed)
        starts = lo[None, :] + (hi - lo)[None, :] * U[:n_starts]
    else:
        if x0 is None:
            starts = np.array([[0.5 * (b[0] + b[1]) for b in bounds]])
        else:
            starts = np.asarray(x0, dtype=float).reshape(1, d)

    def _feasible(x):
        for c, cval, _ in con_val_grad:
            v = cval(x)
            if c.upper is not None and v > c.upper + constraint_tol:
                return False
            if c.lower is not None and v < c.lower - constraint_tol:
                return False
        return True

    best = None
    for s in starts:
        res = minimize(
            obj_val,
            np.asarray(s, dtype=float),
            jac=obj_grad,
            bounds=bounds,
            constraints=scipy_cons,
            method=method,
            tol=tol,
        )
        feas = _feasible(res.x)
        # Prefer a feasible point; among equally feasible, a converged optimizer
        # result over a non-converged one (don't let a failed solve with a lower
        # objective win); then the lower objective.
        key = (not feas, not bool(res.success), float(res.fun))
        if best is None or key < best[0]:
            best = (key, res, feas)

    _, res, feas = best
    cvals = {(c.name or f"c{i}"): float(cval(res.x)) for i, (c, cval, _) in enumerate(con_val_grad)}
    return OptimizeResult(
        input_names=input_names,
        x=np.asarray(res.x, dtype=float),
        objective=sign * float(res.fun),
        constraint_values=cvals,
        feasible=bool(feas),
        success=bool(res.success),
        message=str(res.message),
        n_iter=int(getattr(res, "nit", 0)),
        n_starts=int(starts.shape[0]),
    )
