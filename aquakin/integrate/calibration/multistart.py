"""Starting points, optimizer bounds and the multistart loop."""

from __future__ import annotations

import warnings
from collections import namedtuple
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from aquakin.integrate._transforms import to_unconstrained
from aquakin.integrate.calibration.loss import _objective_scale

if TYPE_CHECKING:
    from aquakin.integrate.calibration.objective import _FitConfig, _ObjectiveBundle
    from aquakin.integrate.calibration.problem import _CalibrationProblem


# Uniform optimiser output so the multistart loop and downstream code are
# agnostic to which backend (L-BFGS-B or Gauss-Newton least-squares) ran.
_OptOut = namedtuple("_OptOut", "x fun success message nit")


def _scale_objective(bundle: _ObjectiveBundle, x_start) -> None:
    """Set the factor the optimiser's objective is divided by from its value
    at ``x_start`` (see :func:`_objective_scale`; only ``loss="mse"`` is
    scaled, which a bundle marks with a ``data_mean_square``).

    Each start of a multistart is scaled by its own starting loss, so a start
    runs exactly as it would as the only start of a fit, and an explicit start
    list returns what the best of separate fits from its starts would. A start
    whose solve fails is left unscaled: the multistart runner reports that
    failure."""
    if bundle.data_mean_square is None:
        return
    try:
        value = bundle.value_and_grad(jnp.asarray(x_start))[0]
    except Exception as exc:
        if not _is_solve_failure(exc):
            raise
        bundle.scale_box[0] = 1.0
        return
    start_loss = float(value) * bundle.loss_scale
    bundle.scale_box[0] = _objective_scale(bundle.data_mean_square, start_loss)


# Outer iterations of the Gauss-Newton noise re-estimation (``sigma="estimate"``);
# each is a warm-started least-squares solve, and the noise level typically
# settles in a handful.
_IRLS_MAX_ITER = 50


def _optimizer_bounds(problem: _CalibrationProblem, rate_theta0):
    """Box bounds in unconstrained space.

    Rate dims carry the absolute bounds ``problem.rate_bounds`` (model-declared
    and/or explicit, already mapped through each transform), intersected with
    ``theta0 +/- param_halfwidth`` when that is given; the halfwidth box is
    centred on the start moved inside the absolute bounds, so the intersection
    is never empty. Free-IC dims are log-bounded by ``ic_bounds`` (one pair, or
    one per pool), and unbounded for a ``fixed_total`` split.
    Returns ``(bounds, lb, ub, ls_bounds, has_bounds)`` -- ``bounds`` the
    L-BFGS-B list (``None`` if unbounded), ``ls_bounds`` the least-squares
    ``(lb, ub)`` pair."""
    rate_th0 = np.asarray(rate_theta0, dtype=float)
    if problem.rate_bounds is not None:
        rate_lb = np.asarray(problem.rate_bounds[0], dtype=float)
        rate_ub = np.asarray(problem.rate_bounds[1], dtype=float)
    else:
        rate_lb = np.full(problem.n_rate, -np.inf)
        rate_ub = np.full(problem.n_rate, np.inf)
    if problem.param_halfwidth is not None:
        centre = np.clip(rate_th0, rate_lb, rate_ub)
        rate_lb = np.maximum(rate_lb, centre - problem.param_halfwidth)
        rate_ub = np.minimum(rate_ub, centre + problem.param_halfwidth)
    if problem.ic_fixed_total:
        # The log-ratio coordinates of a split are unbounded (softmax keeps every
        # pool positive and the total fixed).
        n_ic = problem.m_ic_theta * problem.n_datasets
        ic_lb = np.full(n_ic, -np.inf)
        ic_ub = np.full(n_ic, np.inf)
    else:
        # ``ic_bounds`` is one (lo, hi) pair or per-pool (lo, hi) arrays.
        lo = np.broadcast_to(np.asarray(problem.ic_bounds[0], dtype=float), (problem.m_ic,))
        hi = np.broadcast_to(np.asarray(problem.ic_bounds[1], dtype=float), (problem.m_ic,))
        ic_lb = np.tile(np.log(lo), problem.n_datasets)
        ic_ub = np.tile(np.log(hi), problem.n_datasets)
    lb = np.concatenate([rate_lb, ic_lb])
    ub = np.concatenate([rate_ub, ic_ub])
    has_bounds = bool(np.isfinite(lb).any() or np.isfinite(ub).any())
    if has_bounds:
        bounds = [
            (None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
            for lo, hi in zip(lb, ub)
        ]
    else:
        bounds = None
    return bounds, lb, ub, (lb, ub), has_bounds


def _clip_start(problem: _CalibrationProblem, theta0, opt_bounds):
    """Move the start inside the optimiser bounds, warning (by name) for any
    free rate parameter whose starting value lay outside its bounds."""
    _bounds, lb, ub, _ls, has_bounds = opt_bounds
    if not has_bounds:
        return theta0
    th = np.asarray(theta0, dtype=float)
    clipped = np.clip(th, lb, ub)
    outside = [problem.free_params[i] for i in range(problem.n_rate) if clipped[i] != th[i]]
    if outside:
        warnings.warn(
            f"calibrate: the starting value of {outside} lies outside its bounds; "
            "the optimiser starts from the nearest bound instead.",
            RuntimeWarning,
            stacklevel=4,  # past _initial_point and the calibrate entry point
        )
    return jnp.asarray(clipped)


_SOLVE_FAILURE_HINT = (
    "This usually means the optimiser stepped into a stiff or infeasible region "
    "of parameter space. Bound the rate constants with "
    "OptimizerConfig(bounds={name: (lo, hi)}) or "
    "OptimizerConfig(param_halfwidth=...) (e.g. np.log(100) for two decades "
    "either side of the start), start from a better initial guess, or tighten "
    "the free_ic bounds."
)


def _is_solve_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is a runtime failure of the forward solve (as opposed to
    a programming error).

    A solve that goes bad inside compiled code (a non-finite linear solve, a
    step-count overrun) is reported through ``equinox.error_if``, which raises
    from a host callback. It reaches Python either as equinox's own runtime
    error or, under a plain ``jax.jit``, wrapped in a JAX runtime error (or a
    ``ValueError`` on older JAX) whose message carries the callback failure.
    """
    if type(exc).__name__ in ("EquinoxRuntimeError", "_EquinoxRuntimeError"):
        return True
    return isinstance(exc, (RuntimeError, ValueError)) and "Callback error" in str(exc)


def _resolve_starts(problem: _CalibrationProblem, starts) -> list:
    """The explicit ``OptimizerConfig.starts`` as unconstrained rate-theta
    vectors. Each entry is a full physical parameter vector (only the free
    entries are read) or a ``{name: value}`` dict over free parameters overlaid
    on ``p0_full``."""
    if isinstance(starts, (np.ndarray, jnp.ndarray)) and np.ndim(starts) == 1:
        raise ValueError(
            "optimizer.starts must be a list of starts (vectors or dicts), not a "
            "single vector; pass initial_params for one start."
        )
    starts = list(starts)
    if not starts:
        raise ValueError("optimizer.starts must contain at least one start.")
    model = problem.model
    n_params = int(problem.p0_full.shape[0])
    out = []
    for s_i, start in enumerate(starts):
        if isinstance(start, dict):
            p = problem.p0_full
            for name, value in start.items():
                if name not in problem.free_params:
                    raise ValueError(
                        f"optimizer.starts[{s_i}]: '{name}' is not a free parameter; "
                        f"a start sets free parameters only ({problem.free_params})."
                    )
                p = p.at[model.param_index[name]].set(float(value))
        else:
            p = jnp.asarray(start, dtype=float)
            if p.shape != (n_params,):
                raise ValueError(
                    f"optimizer.starts[{s_i}] has shape {p.shape}; a vector start must "
                    f"be a full parameter vector of shape ({n_params},), or pass a "
                    "{name: value} dict."
                )
        for name, t in zip(problem.free_params, problem.transforms):
            v = float(p[model.param_index[name]])
            if (t == "positive_log" and v <= 0.0) or (t == "logit" and not 0.0 < v < 1.0):
                raise ValueError(
                    f"optimizer.starts[{s_i}]: '{name}' = {v} is outside the domain "
                    f"of its transform {t!r}."
                )
        out.append(
            jnp.stack(
                [
                    to_unconstrained(p[model.param_index[name]], t)
                    for name, t in zip(problem.free_params, problem.transforms)
                ]
            )
        )
    return out


def _run_multistart(
    cfg: _FitConfig, bundle: _ObjectiveBundle, theta0, opt_bounds, starts=None, progress=None
) -> _OptOut:
    """Run the optimiser from the start point, then from ``n_starts - 1``
    deterministic jittered restarts, keeping the lowest finite loss. With an
    explicit ``starts`` list (full unconstrained theta vectors), run from each
    of those instead (``theta0`` is then unused).

    ``progress`` (see :mod:`aquakin.progress`) is told of every objective
    evaluation, so a caller can follow the fit and cancel it between two.

    A start whose forward solve fails part-way (see :func:`_is_solve_failure`)
    is skipped with a ``RuntimeWarning``; if every start fails, a
    ``RuntimeError`` with remedies is raised, chained to the first failure.
    """
    from aquakin.progress import report

    bounds, lb, ub, ls_bounds, _has = opt_bounds
    n_planned = len(starts) if starts is not None else cfg.n_starts
    evaluations = [0]
    current = [0]

    def _tick():
        evaluations[0] += 1
        report(
            progress,
            "fit",
            evaluations[0],
            None,
            f"start {current[0] + 1} of {n_planned}, evaluation {evaluations[0]}",
        )

    def _np_loss_and_grad(x_np):
        _tick()
        x = jnp.asarray(x_np)
        val, grad = bundle.value_and_grad(x)
        return float(val), np.asarray(grad)

    def _run_from(x_start):
        if cfg.optimizer == "lbfgsb":
            r = minimize(
                _np_loss_and_grad,
                np.asarray(x_start),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": cfg.max_iter, "gtol": cfg.tol},
            )
            return _OptOut(
                np.asarray(r.x), float(r.fun), bool(r.success), str(r.message), int(r.nit)
            )

        def _lsq(x0):
            def _residual(x):
                _tick()
                return np.asarray(bundle.res_j(jnp.asarray(x)), dtype=float)

            r = least_squares(
                _residual,
                np.asarray(x0),
                jac=lambda x: np.asarray(bundle.jac_j(jnp.asarray(x)), dtype=float),
                method="trf",
                bounds=ls_bounds,
                max_nfev=cfg.max_iter,
                xtol=cfg.tol,
                ftol=cfg.tol,
            )
            # ``n_iter`` reports optimiser ITERATIONS, consistent with L-BFGS-B's
            # ``nit``. scipy.least_squares exposes no iteration count, so use the
            # Jacobian-evaluation count ``njev`` (~one per trust-region iteration).
            return r, int(r.njev) if r.njev is not None else int(r.nfev)

        if bundle.sigma_hat_of is None:
            r, n_iter_ls = _lsq(x_start)
            return _OptOut(
                np.asarray(r.x), float(r.cost), bool(r.success), str(r.message), n_iter_ls
            )
        # Concentrated likelihood: alternate a least-squares solve at a fixed
        # noise level with re-estimating it from the residuals. At the fixed
        # point the weighted-least-squares gradient equals the gradient of the
        # concentrated objective, so this is a stationary point of it.
        x = np.asarray(x_start, dtype=float)
        sig = np.asarray(bundle.sigma_hat_of(x))
        n_iter_total, converged = 0, False
        for _ in range(_IRLS_MAX_ITER):
            bundle.set_sigma_hat(sig)
            r, n_it = _lsq(x)
            n_iter_total += n_it
            x = np.asarray(r.x)
            sig_new = np.asarray(bundle.sigma_hat_of(x))
            converged = bool(np.max(np.abs(sig_new / sig - 1.0)) <= max(cfg.tol, 1e-12))
            sig = sig_new
            if converged:
                break
        bundle.set_sigma_hat(sig)
        message = str(r.message) if converged else "noise re-estimation did not converge"
        fun = float(bundle.value_and_grad(jnp.asarray(x))[0])
        return _OptOut(x, fun, bool(r.success) and converged, message, n_iter_total)

    failures = []  # (start index, exception) of starts whose solve failed

    def _try_from(s, x_start):
        current[0] = s
        try:
            _scale_objective(bundle, x_start)
            out = _run_from(x_start)
        except Exception as exc:
            if not _is_solve_failure(exc):
                raise
            failures.append((s, exc))
            return None
        # Starts may run at different scales: compare them on the loss itself.
        return out._replace(fun=out.fun * bundle.loss_scale)

    def _better(cand, best):
        return best is None or (np.isfinite(cand.fun) and cand.fun < best.fun)

    if starts is not None:
        result = None
        for s, x_start in enumerate(starts):
            x_np = np.asarray(x_start, dtype=float)
            if bounds is not None:
                x_np = np.clip(x_np, lb, ub)
            cand = _try_from(s, x_np)
            if cand is not None and _better(cand, result):
                result = cand
        n_run = len(starts)
    else:
        result = _try_from(0, theta0)
        n_run = cfg.n_starts
    if starts is None and cfg.n_starts > 1:
        theta0_np = np.asarray(theta0)
        # Default: a single Gaussian stream of scale ``jitter`` seeded once.
        # ``jitter_schedule`` (a tuple of scales): start s uses scale
        # schedule[(s-1) % len] with its own generator (seed + s).
        seq_rng = None if cfg.jitter_schedule else np.random.default_rng(cfg.seed)
        for s in range(1, cfg.n_starts):
            if cfg.jitter_schedule:
                jit = cfg.jitter_schedule[(s - 1) % len(cfg.jitter_schedule)]
                noise = np.random.default_rng(cfg.seed + s).normal(0.0, jit, size=theta0_np.shape)
            else:
                noise = seq_rng.normal(0.0, cfg.jitter, size=theta0_np.shape)
            perturbed = theta0_np + noise
            if bounds is not None:
                perturbed = np.clip(perturbed, lb, ub)
            cand = _try_from(s, perturbed)
            if cand is None:
                continue
            if _better(cand, result):
                result = cand
    if result is None:
        what = "the optimiser run" if n_run == 1 else f"all {n_run} optimiser starts"
        raise RuntimeError(
            f"Calibration failed: the forward solve failed in {what}, at a parameter "
            f"point the optimiser reached. {_SOLVE_FAILURE_HINT} The first solve "
            "error is the direct cause above."
        ) from failures[0][1]
    if failures:
        idx = ", ".join(str(s) for s, _exc in failures)
        which = "indexing optimizer.starts" if starts is not None else "0 is the unperturbed start"
        warnings.warn(
            f"calibrate: the forward solve failed in {len(failures)} of {n_run} "
            f"optimiser starts (start index {idx}; {which}); "
            f"they were skipped and the best remaining start was kept. {_SOLVE_FAILURE_HINT}",
            RuntimeWarning,
            stacklevel=3,
        )
    return result
