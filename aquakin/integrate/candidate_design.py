"""Optimal design over a discrete pool of candidate measurements.

Each candidate ``c`` carries its own Fisher information ``F_c`` (one sampling
site, one composite sample, one experiment), and information is additive, so a
design that takes candidate ``c`` ``n_c`` times has

    F(n) = base + sum_c n_c F_c.

:func:`design_from_candidates` chooses the counts ``n_c`` to optimize a design
criterion of ``F(n)`` (see :mod:`aquakin.integrate.experiment_design`) subject
to a budget ``sum_c n_c cost_c <= budget``, a cap on repeats, and candidates
that must be included. It combines three methods:

- **relaxed**: let the counts be real weights ``w_c`` with
  ``sum_c w_c cost_c = budget``. For D (and D on a subset) and A the criterion
  is concave in ``w``, so projected-gradient ascent reaches the global optimum.
  The Frank-Wolfe duality gap at the solution bounds how far it is from that
  optimum, and the relaxed optimum bounds every integer design from above. With
  no base, no required candidates and no binding cap, the Kiefer-Wolfowitz
  equivalence theorem gives the classic check: at the optimum the largest
  standardized variance ``budget * tr(F^{-1} F_c) / cost_c`` equals the number
  of scored parameters.
- **greedy**: add, one at a time, the affordable candidate with the largest
  gain (per unit cost when costs differ). For D with a positive-definite base
  and unit costs, log det is monotone submodular, so the greedy design is
  within a factor ``1 - 1/e`` of the best subset's gain over the base.
- **exchange**: from a starting design, repeatedly make the single swap (or
  addition) that improves the criterion most, until none does. This reaches a
  local optimum, which need not be the best design.

When the pool is small enough to list every feasible design, the default
method scores them all and so returns the exact optimum.

While a design has fewer informative candidates than parameters its matrix is
singular and every D, A or E value is the worst possible, so greedy cannot
tell candidates apart. When the base is missing or singular, the integer
methods therefore select on ``F(n) + ridge * I`` with a tiny ridge, and report
the criterion of ``F(n)`` itself. A pool that cannot determine the scored
quantities even with every candidate included raises an error instead.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate.experiment_design import (
    _LARGER_IS_BETTER,
    _check_criterion,
    _criterion,
    _subset_indices,
)
from aquakin.integrate.fisher import FisherInformation

METHODS = ("auto", "relaxed", "greedy", "exchange")
# Criteria that are concave in the design weights, so the relaxed problem has
# a global optimum that projected-gradient ascent finds.
_RELAXABLE = ("D", "A")
# Criteria that never get worse when information is added, so a design keeps
# spending budget even when the best next candidate gains nothing.
_MONOTONE = ("D", "A", "E")
# Ridge for the integer methods' selection, relative to the mean eigenvalue of
# the most informative candidate.
_RIDGE = 1e-6


@dataclass
class CandidateDesign:
    """Result of :func:`design_from_candidates`.

    Attributes
    ----------
    counts : np.ndarray
        ``(n_candidates,)`` integer count of each candidate in the design.
    selected : list
        Labels of the candidates with a non-zero count, in pool order.
    labels : list
        Label of every candidate: the keys of a dict pool, else its indices.
    value : float
        Criterion of the design (natural units: larger is better for D and E,
        smaller for A and modified-E), averaged over any parameter draws.
    cost : float
        Total cost of the design.
    matrix : np.ndarray
        ``(p, p)`` Fisher information of the design (the mean over parameter
        draws, if any).
    method : str
        Method that produced the design: ``"relaxed"`` (rounded),
        ``"greedy"``, ``"exchange"`` or ``"exhaustive"``.
    gains : np.ndarray
        ``(n_candidates,)`` improvement of the criterion from adding one more
        of each candidate to the design, signed so that positive is better.
        Ignores the budget and the cap, so it shows what to add next.
    relaxed_weights : np.ndarray or None
        ``(n_candidates,)`` relaxed optimum, if the relaxed problem was solved.
    relaxed_value : float or None
        Criterion at the relaxed optimum.
    bound : float or None
        Bound on the criterion of any feasible design: the relaxed value plus
        its duality gap (an upper bound for D, a lower bound for A).
    efficiency : float or None
        Efficiency of the design relative to ``bound``, in ``(0, 1]``: for D
        ``exp((value - bound) / q)`` with ``q`` the number of scored
        parameters, for A ``bound / value``.
    gap : float or None
        Frank-Wolfe duality gap of the relaxed solution, in criterion units.
    equivalence : tuple of float or None
        ``(max variance, q)`` from the Kiefer-Wolfowitz equivalence theorem
        when it applies (D, no base, no required candidates, no binding cap).
        The relaxed design is optimal when the two are equal.
    regularized : bool
        Whether the integer methods selected on a ridge-regularized matrix
        because the base was missing or singular.
    """

    counts: np.ndarray
    selected: list
    labels: list
    value: float
    cost: float
    matrix: np.ndarray
    method: str
    gains: np.ndarray
    relaxed_weights: np.ndarray | None = None
    relaxed_value: float | None = None
    bound: float | None = None
    efficiency: float | None = None
    gap: float | None = None
    equivalence: tuple[float, float] | None = None
    regularized: bool = False


def design_from_candidates(
    candidates,
    *,
    budget: float,
    names: list[str] | None = None,
    criterion: str = "D",
    subset: list[str] | None = None,
    costs=None,
    required: Sequence[Hashable] | None = None,
    max_repeats: int | Sequence[int] | None = 1,
    base=None,
    method: str = "auto",
    param_draws_axis: int | None = None,
    exhaustive_limit: int = 20000,
    tol: float = 1e-9,
    max_iter: int = 20000,
) -> CandidateDesign:
    """Choose the best set of measurements from a pool of candidates.

    Parameters
    ----------
    candidates : array-like, list or dict of FisherInformation
        The information of each candidate: an ``(n_c, p, p)`` stack, a list of
        :class:`FisherInformation` (with the same names), or a dict of them by
        label, such as the groups returned by
        :func:`aquakin.fisher_information_from_fn`. A stack may carry a further
        axis of parameter draws (see ``param_draws_axis``).
    budget : float
        Total budget: the number of candidates to choose, or with ``costs`` the
        total cost.
    names : list of str, optional
        Parameter names of a bare stack, for ``subset``. Taken from the
        ``FisherInformation`` otherwise.
    criterion : {"D", "A", "E", "modified_E"}
        See :func:`aquakin.design_criterion`. The relaxed method needs D or A.
    subset : list of str, optional
        Score only these parameters, treating the others as nuisance (D on a
        subset is Ds-optimality).
    costs : array-like, optional
        ``(n_c,)`` positive cost of each candidate. Default 1 each.
    required : sequence, optional
        Candidates (labels of a dict pool, else indices) that must be in the
        design at least once.
    max_repeats : int, sequence of int, or None
        Most times a candidate may be chosen: one value, one per candidate, or
        ``None`` for no cap. Default 1 (each candidate at most once).
    base : array-like or FisherInformation, optional
        Information already in hand (a prior, experiments already run):
        ``(p, p)``, or ``(n_draw, p, p)`` with parameter draws.
    method : {"auto", "relaxed", "greedy", "exchange"}
        ``"relaxed"`` solves the relaxed problem and rounds it; ``"greedy"``
        selects forward; ``"exchange"`` improves the greedy design by swaps.
        ``"auto"`` (default) runs the rounded relaxed design and the greedy
        design each through exchange and keeps the better, or, when there are
        at most ``exhaustive_limit`` feasible designs, scores every one and
        returns the exact optimum (``method`` ``"exhaustive"``). With D or A
        it also solves the relaxed problem for the bound.
    param_draws_axis : int, optional
        Axis of a 4-D candidate stack that indexes parameter draws (0 or 1).
        The criterion is then averaged over the draws (robust design).
    exhaustive_limit : int
        Most feasible designs ``"auto"`` will enumerate. For D, A and E only
        designs that cannot afford another candidate are counted.
    tol : float
        Convergence tolerance of the relaxed solver on its duality gap,
        relative to the criterion's magnitude (at least 1). The solver also
        stops when round-off prevents further progress; ``gap`` reports where
        it ended either way.
    max_iter : int
        Iteration cap of the relaxed solver.

    Returns
    -------
    CandidateDesign
        The chosen counts and their criterion, with the relaxed optimum, its
        bound and the design's efficiency when the relaxed problem was solved.

    Raises
    ------
    ValueError
        If the budget cannot pay for the required candidates or buys nothing,
        the pool cannot determine the scored parameters even in full, a shape
        or option is invalid, or the relaxed method is asked for E or
        modified-E.

    Examples
    --------
    >>> total, sites = aquakin.fisher_information_from_fn(
    ...     fn, theta, names=names, sigma=0.1, groups=site_labels)
    >>> design = aquakin.design_from_candidates(
    ...     sites, budget=4, criterion="D", subset=["zone_1", "zone_2"],
    ...     required=["WWTP"])
    >>> design.selected, design.efficiency
    """
    _check_criterion(criterion)
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}.")
    if method == "relaxed" and criterion not in _RELAXABLE:
        raise ValueError(
            f"The relaxed method needs a concave criterion ({_RELAXABLE}); "
            f"use method='greedy' or 'exchange' for {criterion!r}."
        )
    F, labels, names = _candidate_stack(candidates, names, param_draws_axis)
    n_d, n_c, p, _ = F.shape
    B0 = _base_stack(base, n_d, p)
    if subset is not None and names is None:
        raise ValueError("Pass names= to score a subset of a bare candidate stack.")
    sub, nui = _subset_indices(names, subset)
    q = p if subset is None else len(subset)

    cost = np.ones(n_c) if costs is None else np.asarray(costs, dtype=float).reshape(-1)
    if cost.shape != (n_c,) or not np.all(cost > 0.0):
        raise ValueError(f"costs must be {n_c} positive values, one per candidate.")
    budget = float(budget)
    if not budget > 0.0:
        raise ValueError(f"budget must be positive; got {budget}.")
    if max_repeats is None:
        upper = np.full(n_c, np.inf)
    else:
        upper = np.broadcast_to(np.asarray(max_repeats, dtype=float), (n_c,)).copy()
        if not np.all((upper >= 1) & (upper == np.floor(upper))):
            raise ValueError("max_repeats must be a positive integer (or None for no cap).")
    lower = np.zeros(n_c)
    for r in required or ():
        if r not in labels:
            raise KeyError(f"Unknown required candidate {r!r}. Available: {labels}")
        lower[labels.index(r)] = 1.0
    if lower @ cost > budget * (1 + 1e-12):
        raise ValueError(
            f"The required candidates cost {lower @ cost:g}, more than the budget {budget:g}."
        )
    if not lower.any() and cost.min() > budget:
        raise ValueError(
            f"The budget {budget:g} buys no candidate; the cheapest costs {cost.min():g}."
        )

    sign = 1.0 if _LARGER_IS_BETTER[criterion] else -1.0
    strict = _utility_fn(criterion, sub, nui, rtol=None, sign=sign)

    # The pool in full (each candidate at its cap, or once if uncapped) must
    # determine the scored quantities, or no design can.
    full = B0 + np.einsum("c,dcij->dij", np.where(np.isinf(upper), 1.0, upper), F)
    if not np.isfinite(strict(full[None])[0]):
        what = "the parameters" if subset is None else f"the subset {list(subset)}"
        raise ValueError(
            f"The candidate pool cannot determine {what} even with every candidate "
            "included: the information is singular. Add information already in hand "
            "(a prior) with base=, or score fewer parameters with subset=."
        )

    regularized = not np.isfinite(strict(B0[None])[0])
    if regularized:
        scale = max(float(np.max(np.trace(F, axis1=-2, axis2=-1))) / p, np.finfo(float).tiny)
        ridge = _RIDGE * scale * np.eye(p)
        select = _utility_fn(criterion, sub, nui, rtol=0.0, sign=sign)
        B_sel = B0 + ridge
    else:
        select, B_sel = strict, B0

    def design_matrix(n, B=B_sel):
        return B + np.einsum("c,dcij->dij", n, F)

    def utility(n):
        return float(select(design_matrix(n)[None])[0])

    # Candidates' information summed into every possible one-unit addition:
    # (n_c, n_d, p, p) on top of a design matrix.
    F_add = np.moveaxis(F, 1, 0)

    def add_utilities(n):
        return np.asarray(select(design_matrix(n)[None] + F_add))

    def determined(n):
        return bool(np.isfinite(strict(design_matrix(n, B0)[None])[0]))

    ctx = _Context(n_c, cost, budget, lower, upper, criterion)
    designs = []
    relaxed = None
    if method in ("auto", "relaxed") and criterion in _RELAXABLE:
        relaxed = _solve_relaxed(
            F, B0, cost, budget, lower, upper, criterion, sub, nui, tol, max_iter
        )
        rounded = _greedy(
            np.maximum(np.floor(relaxed[0] + 1e-9), lower), ctx, utility, add_utilities
        )
        designs.append(("relaxed", rounded))
        if method == "auto":
            designs.append(("exchange", _exchange(rounded, ctx, utility, add_utilities)))
    every = _enumerate(ctx, exhaustive_limit) if method == "auto" else None
    if every is not None:
        u_all = np.concatenate(
            [
                select(np.einsum("kc,dcij->kdij", chunk, F) + B_sel)
                for chunk in np.array_split(every, max(1, len(every) // 2048))
            ]
        )
        designs = [("exhaustive", every[int(np.argmax(u_all))])]
    elif method in ("auto", "greedy", "exchange"):
        greedy = _greedy(lower.copy(), ctx, utility, add_utilities, determined)
        if method == "greedy":
            designs.append(("greedy", greedy))
        else:
            designs.append(("exchange", _exchange(greedy, ctx, utility, add_utilities)))

    # Keep the best; on a tie, the earlier entry (the rounded relaxed design
    # before its exchange refinement, which is then identical).
    best_u, best = -np.inf, None
    for name, n in designs:
        u = utility(n)
        if best is None or u > best_u + 1e-12 * max(1.0, abs(best_u)):
            best_u, best = u, (name, n)
    name, counts = best

    M = B0 + np.einsum("c,dcij->dij", counts, F)
    u_final = float(strict(M[None])[0])
    value = sign * u_final
    with np.errstate(invalid="ignore"):
        gains = np.asarray(strict(M[None] + F_add)) - u_final

    result = CandidateDesign(
        counts=counts.astype(int),
        selected=[labels[i] for i in np.flatnonzero(counts)],
        labels=labels,
        value=value,
        cost=float(counts @ cost),
        matrix=M.mean(axis=0),
        method=name,
        gains=gains,
        regularized=regularized,
    )
    if relaxed is not None:
        w, f_rel, gap, grad = relaxed
        result.relaxed_weights = w
        result.relaxed_value = sign * f_rel
        result.gap = gap
        result.bound = sign * (f_rel + gap)
        if criterion == "D":
            result.efficiency = float(np.exp(min(0.0, (value - result.bound) / q)))
        else:
            result.efficiency = float(min(1.0, result.bound / value))
        caps_bind = np.any(w > upper - 1e-9)
        if criterion == "D" and base is None and not lower.any() and not caps_bind:
            result.equivalence = (float(np.max(budget * grad / cost)), float(q))
    return result


def _candidate_stack(candidates, names, param_draws_axis):
    """The candidates as a ``(n_draw, n_c, p, p)`` array, their labels and the
    parameter names."""
    if isinstance(candidates, dict):
        labels = list(candidates)
        items = list(candidates.values())
    elif (
        isinstance(candidates, (list, tuple))
        and candidates
        and all(isinstance(c, FisherInformation) for c in candidates)
    ):
        labels, items = list(range(len(candidates))), list(candidates)
    else:
        items = None
    if items is not None:
        if not items or not all(isinstance(c, FisherInformation) for c in items):
            raise ValueError("A dict or list pool must hold FisherInformation objects.")
        first = items[0].names
        if any(c.names != first for c in items):
            raise ValueError("Every candidate's FisherInformation must have the same names.")
        if names is not None and list(names) != list(first):
            raise ValueError("names= disagrees with the candidates' own names.")
        if param_draws_axis is not None:
            raise ValueError("param_draws_axis applies only to an array of candidates.")
        F = np.stack([np.asarray(c.matrix, dtype=float) for c in items])[None]
        return 0.5 * (F + np.swapaxes(F, -1, -2)), labels, list(first)

    F = np.asarray(candidates, dtype=float)
    if param_draws_axis is None:
        if F.ndim != 3:
            raise ValueError(
                f"candidates must be an (n_c, p, p) stack; got shape {F.shape}. Pass "
                "param_draws_axis= for a stack with parameter draws."
            )
        F = F[None]
    else:
        if F.ndim != 4 or param_draws_axis not in (0, 1):
            raise ValueError("With param_draws_axis, candidates must be 4-D and the axis 0 or 1.")
        F = np.moveaxis(F, param_draws_axis, 0)
    if F.shape[1] == 0 or F.shape[2] != F.shape[3] or F.shape[2] == 0:
        raise ValueError(f"candidates must be square, non-empty matrices; got shape {F.shape}.")
    p = F.shape[2]
    if names is not None:
        names = list(names)
        if len(names) != p:
            raise ValueError(f"names has {len(names)} entries; the matrices are {p} x {p}.")
    return 0.5 * (F + np.swapaxes(F, -1, -2)), list(range(F.shape[1])), names


def _base_stack(base, n_d, p):
    """The base information as ``(n_d, p, p)`` (zeros without a base)."""
    if base is None:
        return np.zeros((n_d, p, p))
    B = np.asarray(base.matrix if isinstance(base, FisherInformation) else base, dtype=float)
    if B.shape == (p, p):
        B = np.broadcast_to(B, (n_d, p, p))
    elif B.shape != (n_d, p, p):
        raise ValueError(f"base must be ({p}, {p}) or ({n_d}, {p}, {p}); got shape {B.shape}.")
    return 0.5 * (B + np.swapaxes(B, -1, -2))


def _utility_fn(criterion, sub, nui, *, rtol, sign):
    """Jitted ``(k, n_d, p, p) -> (k,)``: the criterion of each design, signed
    so that larger is better and averaged over the draws. ``rtol=None`` uses
    the default singular rule."""
    kw = {} if rtol is None else {"rtol": rtol}

    def one(M):
        return sign * _criterion(M, criterion, sub, nui, **kw)

    @jax.jit
    def batch(Ms):
        k, n_d, p, _ = Ms.shape
        vals = jax.vmap(one)(Ms.reshape(k * n_d, p, p)).reshape(k, n_d)
        return jnp.mean(vals, axis=1)

    return lambda Ms: np.asarray(batch(jnp.asarray(Ms)))


@dataclass
class _Context:
    """The constraints the integer methods share."""

    n_c: int
    cost: np.ndarray
    budget: float
    lower: np.ndarray
    upper: np.ndarray
    criterion: str

    def can_add(self, n):
        """Candidates that one more unit of keeps within the cap and budget."""
        slack = self.budget - n @ self.cost
        return (n < self.upper) & (self.cost <= slack * (1 + 1e-12) + 1e-12)


def _greedy(n, ctx, utility, add_utilities, determined=lambda n: True):
    """Forward selection from ``n``: add the best affordable candidate (by gain
    per unit cost) until none is affordable, or, for a criterion that can get
    worse with more information, none improves once the design ``determined``
    the scored parameters (a singular design always takes the best candidate).
    Ties go to the lower index."""
    n = n.astype(float).copy()
    while True:
        ok = ctx.can_add(n)
        if not ok.any():
            return n
        u = utility(n)
        u_add = add_utilities(n)
        gain = np.where(ok & np.isfinite(u_add), (u_add - u) / ctx.cost, -np.inf)
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]):
            return n
        if ctx.criterion not in _MONOTONE and not gain[j] > 0.0 and determined(n):
            return n
        n[j] += 1


def _exchange(n, ctx, utility, add_utilities, max_passes=500):
    """Fedorov exchange from ``n``: make the single swap (drop one unit of
    ``i``, add one of ``j``) or addition that improves the criterion most,
    until none does. Ties go to the lower indices."""
    n = n.astype(float).copy()
    for _ in range(max_passes):
        u = utility(n)
        thresh = u + 1e-12 * max(1.0, abs(u)) if np.isfinite(u) else u
        best, move = thresh, None
        ok = ctx.can_add(n)
        if ok.any():
            u_add = np.where(ok, add_utilities(n), -np.inf)
            j = int(np.argmax(u_add))
            if u_add[j] > best:
                best, move = u_add[j], (None, j)
        for i in np.flatnonzero(n > ctx.lower):
            m = n.copy()
            m[i] -= 1
            ok = ctx.can_add(m)
            ok[i] = False
            if not ok.any():
                continue
            u_swap = np.where(ok, add_utilities(m), -np.inf)
            j = int(np.argmax(u_swap))
            if u_swap[j] > best:
                best, move = u_swap[j], (i, j)
        if move is None:
            return n
        i, j = move
        if i is not None:
            n[i] -= 1
        n[j] += 1
    return n


def _enumerate(ctx, limit):
    """Every feasible design as a ``(k, n_c)`` array, in lexicographic order,
    or ``None`` if there are more than ``limit``. For a monotone criterion only
    the designs that cannot afford one more candidate are listed, since adding
    information never makes them worse."""
    out, n = [], ctx.lower.copy()
    monotone = ctx.criterion in _MONOTONE

    def rec(c, left):
        if len(out) > limit:
            return
        if c == ctx.n_c:
            if not monotone or not ctx.can_add(n).any():
                out.append(n.copy())
            return
        top = ctx.upper[c] - ctx.lower[c]
        k = 0
        while k <= top and k * ctx.cost[c] <= left * (1 + 1e-12) + 1e-12:
            n[c] = ctx.lower[c] + k
            rec(c + 1, left - k * ctx.cost[c])
            k += 1
        n[c] = ctx.lower[c]

    rec(0, ctx.budget - ctx.lower @ ctx.cost)
    if len(out) > limit or not out:
        return None
    return np.asarray(out)


def _value_and_grad(w, F, B0, criterion, sub, nui):
    """The relaxed objective (signed so larger is better, averaged over draws)
    and its gradient in the weights; ``-inf`` where the matrix is singular."""
    M = B0 + np.einsum("c,dcij->dij", w, F)
    try:
        if criterion == "D":
            sgn, ld = np.linalg.slogdet(M)
            G = np.linalg.inv(M)
            if sub is not None and len(nui):
                nn = np.ix_(nui, nui)
                sgn_n, ld_n = np.linalg.slogdet(M[:, nui][:, :, nui])
                if np.any(sgn_n <= 0):
                    return -np.inf, None
                ld = ld - ld_n
                G[(slice(None),) + nn] -= np.linalg.inv(M[:, nui][:, :, nui])
            if np.any(sgn <= 0):
                return -np.inf, None
            f = float(np.mean(ld))
            W = G
        else:
            G = np.linalg.inv(M)
            if np.any(np.linalg.eigvalsh(M)[:, 0] <= 0):
                return -np.inf, None
            s = np.arange(M.shape[-1]) if sub is None else np.asarray(sub)
            Gs = G[:, :, s]
            f = -float(np.mean(np.trace(G[:, s][:, :, s], axis1=-2, axis2=-1)))
            W = Gs @ np.swapaxes(Gs, -1, -2)
    except np.linalg.LinAlgError:
        return -np.inf, None
    if not np.isfinite(f):
        return -np.inf, None
    grad = np.einsum("dij,dcij->c", W, F) / F.shape[0]
    return f, grad


def _project(v, cost, budget, lower, upper):
    """Euclidean projection of ``v`` onto ``{lower <= w <= upper, cost . w =
    budget}``: ``clip(v - lam * cost)`` with ``lam`` found by bisection."""

    def spend(lam):
        return np.clip(v - lam * cost, lower, upper) @ cost

    lo = float(np.min((v - np.where(np.isinf(upper), v + budget, upper)) / cost)) - 1.0
    while spend(lo) < budget:
        lo = 2.0 * lo - 1.0
    hi = float(np.max((v - lower) / cost)) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if spend(mid) > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, abs(lo)):
            break
    return np.clip(v - 0.5 * (lo + hi) * cost, lower, upper)


def _fw_vertex(grad, cost, budget, lower, upper):
    """The vertex of the feasible polytope that maximizes ``grad . w``: the
    lower bounds, then the remaining budget spent on the best gradient per
    unit cost (a fractional knapsack)."""
    order = np.argsort(-grad / cost, kind="stable")
    room = ((upper - lower) * cost)[order]
    before = np.concatenate([[0.0], np.cumsum(room)[:-1]])
    spent = np.clip(budget - lower @ cost - before, 0.0, room)
    s = lower.copy()
    s[order] += spent / cost[order]
    return s


def _solve_relaxed(F, B0, cost, budget, lower, upper, criterion, sub, nui, tol, max_iter):
    """Maximize the concave relaxed criterion over the weights by projected
    gradient ascent with backtracking. Returns the weights, the signed value,
    the Frank-Wolfe duality gap and the gradient at the solution."""
    capacity = upper @ cost
    budget = min(budget, capacity) if np.isfinite(capacity) else budget
    w = _project(np.full(cost.shape, budget / cost.sum()), cost, budget, lower, upper)
    f, g = _value_and_grad(w, F, B0, criterion, sub, nui)
    if g is None:
        raise ValueError(
            "The relaxed design is singular: spreading the budget over the pool "
            "does not determine the scored parameters. Add information with base=."
        )
    step = budget / max(float(np.max(np.abs(g))) * cost.sum(), 1e-300)
    gap = np.inf
    for _ in range(max_iter):
        gap = float(g @ (_fw_vertex(g, cost, budget, lower, upper) - w))
        if gap <= tol * max(1.0, abs(f)):
            break
        while True:
            w_new = _project(w + step * g, cost, budget, lower, upper)
            f_new, g_new = _value_and_grad(w_new, F, B0, criterion, sub, nui)
            if g_new is not None and f_new >= f + 1e-4 * g @ (w_new - w):
                break
            step *= 0.5
            if step < 1e-300:
                break
        if g_new is None or np.allclose(w_new, w, rtol=0.0, atol=1e-15):
            break
        w, f, g = w_new, f_new, g_new
        step *= 2.0
    gap = max(0.0, float(g @ (_fw_vertex(g, cost, budget, lower, upper) - w)))
    return w, f, gap, g
