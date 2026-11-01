"""Probe the stage Newton iteration of the lean loop on the BSM2 window.

Each implicit Kvaerno3 stage of ``forward_solve`` / ``staggered_sensitivity_solve``
solves ``R(x) = x - f(t_i, base + h*gamma*x) = 0`` by iterating with the
factorization ``M = I - h*gamma*J_n`` of the Jacobian at the step start. Near the
root the plain iteration's error obeys ``e_{m+1} = T e_m`` with

    T = I - M^{-1} A(x),   A(x) = dR/dx = I - h*gamma*J(Y),

so ``T`` measures how far the Jacobian has drifted between the step start and
the stage value, in the directions that matter to the iteration. This probe
wraps the stage solver and, at every state stage (sensitivity stages share the
same ``T``), forms ``T`` at the final iterate in the solver's own WRMS weighting.
Per outcome (converged in 2 iterations / converged slowly / hit the iteration
cap / diverged) it reports:

- the spectral radius and the counts of singular values / eigenvalues of ``T``
  above 0.05, 0.1, 0.3 and 1 -- how many directions the factored Jacobian is
  wrong in;
- the iterations needed for a 1e-3 error reduction by the plain iteration
  (powers of ``T``) and by GMRES preconditioned with ``M`` (Krylov on
  ``I - T``), on three random right-hand sides (the largest);
- the plant units carrying the dominant singular directions.

For a stage that failed, the final iterate is not the root, so ``T`` there is
the linearization where the iteration stopped. The Krylov count is for one
linearized Newton step.

Usage (the steady state is cached next to ``--out``)::

    python scripts/probe_newton_drift.py --days 0.25 --settings plain  # no Anderson
    python scripts/probe_newton_drift.py --days 0.25  # the shipped settings
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np
import scipy.sparse.linalg as spla
from _bsm2_window import ATOL, RTOL, bsm2_window
from _lean_settings import LeanSettings

import aquakin.integrate.forward_solve as forward_solve

REDUCTION = 1e-3
THRESHOLDS = (0.05, 0.1, 0.3, 1.0)
CLASSES = ("fast", "slow", "maxiter", "diverged", "nonfinite")


def _plain_iterations(T, b):
    v, n0 = b.copy(), np.linalg.norm(b)
    for m in range(1, 101):
        v = T @ v
        if np.linalg.norm(v) < REDUCTION * n0:
            return m
    return 101  # not within 100 (reported as such)


def _gmres_iterations(B, b):
    count = [0]

    def cb(_):
        count[0] += 1

    spla.gmres(
        B,
        b,
        rtol=REDUCTION,
        atol=0.0,
        restart=B.shape[0],
        maxiter=1,
        callback=cb,
        callback_type="pr_norm",
    )
    return count[0]


class StageRecorder:
    """Collect one record per state stage solve (host side)."""

    def __init__(self):
        self.records = []

    def __call__(self, iterations, conv, T, scale):
        iterations, conv = int(iterations), bool(conv)
        T, scale = np.asarray(T), np.asarray(scale)
        if not (np.all(np.isfinite(T)) and np.all(np.isfinite(scale))):
            self.records.append({"cls": "nonfinite", "it": iterations})
            return
        if conv:
            cls = "fast" if iterations <= 2 else "slow"
        else:
            cls = "maxiter" if iterations >= forward_solve._MAXNEWT else "diverged"
        Tw = T * scale[None, :] / scale[:, None]  # D^{-1} T D, D = diag(scale)
        U, s, Vt = np.linalg.svd(Tw)
        eig = np.abs(np.linalg.eigvals(Tw))
        rng = np.random.default_rng(len(self.records))
        bs = [rng.standard_normal(T.shape[0]) for _ in range(3)]
        B = np.eye(T.shape[0]) - Tw
        self.records.append(
            {
                "cls": cls,
                "it": iterations,
                "rho": eig.max(),
                "s1": s[0],
                "n_s": {th: int((s > th).sum()) for th in THRESHOLDS},
                "n_e": {th: int((eig > th).sum()) for th in THRESHOLDS},
                "plain": max(_plain_iterations(Tw, b) for b in bs),
                "gmres": max(_gmres_iterations(B, b) for b in bs),
                "top": [int(i) for i in np.argsort(np.abs(Vt[0]))[-3:]]
                + [int(i) for i in np.argsort(np.abs(U[:, 0]))[-3:]],
            }
        )


def install(recorder):
    """Wrap the lean loop's stage solver so every state stage is recorded."""
    solve_stage = forward_solve._stage_solve

    def probed(residual, guess, lu, scale):
        x, rate, conv, iterations = solve_stage(residual, guess, lu, scale)
        if guess.ndim == 1:
            eye = jnp.eye(guess.shape[0], dtype=guess.dtype)
            T = eye - jsla.lu_solve(lu, jax.jacfwd(residual)(x))
            jax.debug.callback(recorder, iterations, conv, T, jnp.broadcast_to(scale, x.shape))
        return x, rate, conv, iterations

    forward_solve._stage_solve = probed


def _summary(values):
    v = np.asarray(values, dtype=float)
    return f"median {np.median(v):.3g}  p90 {np.percentile(v, 90):.3g}  max {v.max():.3g}"


def report(records, state_units):
    by = collections.defaultdict(list)
    for r in records:
        by[r["cls"]].append(r)
    for cls in CLASSES:
        rs = by.get(cls, [])
        print(f"\n== {cls}: {len(rs)} stage solves", flush=True)
        if not rs or cls == "nonfinite":
            continue
        print("  iterations taken:", _summary([r["it"] for r in rs]))
        print("  spectral radius of T:", _summary([r["rho"] for r in rs]))
        print("  largest singular value:", _summary([r["s1"] for r in rs]))
        for th in THRESHOLDS:
            print(
                f"  #singular values > {th:<4}:",
                _summary([r["n_s"][th] for r in rs]),
                f"| #|eigenvalues| > {th:<4}:",
                _summary([r["n_e"][th] for r in rs]),
            )
        print(
            f"  plain iterations for a {REDUCTION:g} reduction:", _summary([r["plain"] for r in rs])
        )
        print(
            f"  GMRES iterations for a {REDUCTION:g} reduction:", _summary([r["gmres"] for r in rs])
        )
        units = collections.Counter(state_units[i] for r in rs for i in r["top"])
        print("  units in the dominant directions:", units.most_common(6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=0.25)
    ap.add_argument("--settings", default="default", help="a _lean_settings.py spec")
    ap.add_argument("--out", type=Path, default=Path("probe_newton_drift.txt"))
    args = ap.parse_args()
    settings = LeanSettings.parse(args.settings)
    if settings.solve_kwargs:
        ap.error("the drift probe runs the state solve (k=0); noerrcon does not apply")
    settings.apply()

    plant, params, y_ss, _ = bsm2_window(args.out.with_name("bsm2_open_loop_ss.npy"))
    recorder = StageRecorder()
    install(recorder)
    start = time.perf_counter()
    *_, stats = plant.solve_sensitivity(
        params,
        [],
        t_span=(0.0, args.days),
        t_eval=jnp.array([0.0, args.days]),
        y0=y_ss,
        rtol=RTOL,
        atol=ATOL,
        max_steps=200_000,
        return_stats=True,
    )
    print(
        f"days={args.days} settings={args.settings} "
        f"solve {time.perf_counter() - start:.0f}s (probe included) "
        f"attempted/accepted/nonconverged/error-rejected {[int(v) for v in stats]}",
        flush=True,
    )
    state_units = []
    for unit, (_, size) in sorted(plant._state_layout.items(), key=lambda kv: kv[1][0]):
        state_units += [unit] * size
    report(recorder.records, state_units)


if __name__ == "__main__":
    main()
