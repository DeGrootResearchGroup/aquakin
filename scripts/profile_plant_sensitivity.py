"""Profile ``Plant.solve_sensitivity`` on a BSM2 dynamic window (issue #654).

Runs the open-loop BSM2 plant (``build_bsm2`` defaults, BSM2-configured ASM1 +
ADM1) from its constant-influent steady state over a window of the synthesised
dry-weather influent (``load_bsm2_influent("dry")``), with ``rtol=1e-4`` /
``atol=1e-3``, and reports per sensitivity-parameter count ``k`` (the first
``k`` ASM1 parameters):

- the wall time of a first call (compile + run) and a second call (run only,
  the compiled solve reused from the plant's cache);
- the step counts: attempted, non-converged (Newton divergence / no
  convergence) and error-test rejections.

For the staggered corrector the counts come from the solve itself
(``return_stats=True``). For the simultaneous corrector they are counted by
wrapping ``diffrax.Kvaerno3.step`` / ``PIDController.adapt_step_size`` with a
``jax.debug.callback`` (result codes 7/8 are the root finder's max-steps /
divergence failures; every other rejection is an error-test failure).

The steady state is computed once and cached next to ``--out``.

Usage::

    python scripts/profile_plant_sensitivity.py --corrector staggered --k 0 1 16
    python scripts/profile_plant_sensitivity.py --corrector simultaneous --k 0 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import aquakin
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm.bsm2 import (
    bsm2_asm1_model,
    bsm2_constant_influent,
    bsm2_parameters,
    build_bsm2,
)
from aquakin.plant.influent import load_bsm2_influent

_NONLINEAR_FAILURE_CODES = (7, 8)  # diffrax RESULTS: nonlinear max steps / divergence


def _steady_state(asm1, adm1, params, cache: Path):
    if cache.exists():
        return jnp.asarray(np.load(cache))
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    sol = plant.solve(
        t_span=(0.0, 150.0),
        t_eval=jnp.array([0.0, 150.0]),
        params=params,
        y0=bsm2_warm_start(plant),
        rtol=1e-5,
        atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=500_000),
    )
    y_ss = sol.state[-1]
    np.save(cache, np.asarray(y_ss))
    return y_ss


class _DiffraxStepCounter:
    """Count Kvaerno3 step results and controller rejections of diffrax solves."""

    def __init__(self):
        import diffrax

        self.attempted = self.nonconverged = self.rejected = 0
        step, adapt = diffrax.Kvaerno3.step, diffrax.PIDController.adapt_step_size

        def counted_step(solver, *a, **kw):
            out = step(solver, *a, **kw)
            jax.debug.callback(self._on_step, out[-1]._value)
            return out

        def counted_adapt(controller, *a, **kw):
            out = adapt(controller, *a, **kw)
            jax.debug.callback(self._on_adapt, out[0])
            return out

        diffrax.Kvaerno3.step = counted_step
        diffrax.PIDController.adapt_step_size = counted_adapt

    def _on_step(self, code):
        self.attempted += 1
        self.nonconverged += int(int(code) in _NONLINEAR_FAILURE_CODES)

    def _on_adapt(self, keep):
        self.rejected += int(not bool(keep))

    def reset(self):
        self.attempted = self.nonconverged = self.rejected = 0

    def counts(self):
        return self.attempted, self.nonconverged, self.rejected - self.nonconverged


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corrector", choices=("staggered", "simultaneous"), default="staggered")
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 16])
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--factormax", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("profile_plant_sensitivity.txt"))
    args = ap.parse_args()

    asm1, adm1 = bsm2_asm1_model(), aquakin.load_model("adm1")
    params = bsm2_parameters(asm1, adm1)
    y_ss = _steady_state(asm1, adm1, params, args.out.with_name("bsm2_open_loop_ss.npy"))
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", load_bsm2_influent("dry", asm1))
    asm_params = [n for n in plant.parameter_names() if n.startswith("asm1.")]
    counter = _DiffraxStepCounter() if args.corrector == "simultaneous" else None
    t_eval = jnp.linspace(0.0, args.days, int(args.days) + 1)

    with args.out.open("a") as log:
        for k in args.k:
            for call in (1, 2):
                if counter is not None:
                    counter.reset()
                start = time.perf_counter()
                out = plant.solve_sensitivity(
                    params,
                    asm_params[:k],
                    t_span=(0.0, args.days),
                    t_eval=t_eval,
                    y0=y_ss,
                    rtol=1e-4,
                    atol=1e-3,
                    factormax=args.factormax,
                    max_steps=200_000,
                    corrector=args.corrector,
                    return_stats=True,
                )
                jax.block_until_ready(out[2])
                wall = time.perf_counter() - start
                stats = out[3]
                if counter is not None:
                    attempted, nonconverged, error_rejected = counter.counts()
                else:
                    attempted = int(stats.attempted)
                    nonconverged = int(stats.nonconverged)
                    error_rejected = int(stats.error_rejected)
                line = (
                    f"{args.corrector} days={args.days} factormax={args.factormax} k={k} "
                    f"call={call} wall={wall:.1f}s attempted={attempted} "
                    f"nonconverged={nonconverged} error_rejected={error_rejected} "
                    f"finite={bool(np.all(np.isfinite(np.asarray(out[2]))))}"
                )
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                np.save(args.out.with_name(f"{args.corrector}_k{k}_S.npy"), np.asarray(out[2]))
                np.save(args.out.with_name(f"{args.corrector}_k{k}_y.npy"), np.asarray(out[1]))


if __name__ == "__main__":
    main()
