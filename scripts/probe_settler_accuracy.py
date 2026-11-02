"""Why the lean loop and diffrax differ on BSM2's Takacs settler layers.

The case of ``test_forward_fast.py::test_forward_fast_matches_diffrax_bsm2``:
BSM2 with temperature correction and the influent bypass, the synthesised dry
influent, 8 days from ``bsm2_warm_start``, saved every 0.01 d. Three commands:

- ``run --solver lean|diffrax [--rtol R] [--exact-stages]`` solves it and records
  every step attempt: its start, size, whether it was accepted, and (diffrax) whether
  its error estimate was finite -- diffrax marks a failed root find with an infinite
  one. ``--exact-stages`` solves the lean loop's stages to ``_KAPPA = 1e-7``.
  ``atol`` scales with ``rtol`` (``1e-3`` at ``rtol = 1e-4``).
- ``reference [--rtol 1e-8]`` is ``run --solver lean`` at a tight tolerance.
- ``report`` compares every run with the reference: the worst component's error
  and the median component's (each relative to the component's peak, floored at
  ``1e-3``), the settler layers 2-4, and the step counts; and, along the
  reference, how often the settling-flux minimum below the feed switches branch
  between saves (the Takacs rule takes the smaller of two adjacent layers'
  fluxes, so the right-hand side has a kink wherever they cross).

Usage (each command one process, one at a time -- each holds a whole-plant program)::

    python scripts/probe_settler_accuracy.py run --solver lean --out DIR
    python scripts/probe_settler_accuracy.py run --solver diffrax --out DIR
    python scripts/probe_settler_accuracy.py reference --out DIR
    python scripts/probe_settler_accuracy.py report --out DIR
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import aquakin
import aquakin.integrate.forward_solve as forward_solve
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm.bsm2 import (
    InfluentBypass,
    bsm2_asm1_model,
    bsm2_parameters,
    build_bsm2,
)
from aquakin.plant.influent import load_bsm2_influent
from aquakin.plant.plant import PlantSolution

T_END, N_SAVES = 8.0, 801


def build():
    asm1 = bsm2_asm1_model()
    adm1 = aquakin.load_model("adm1")
    plant = build_bsm2(
        asm1_model=asm1,
        adm1_model=adm1,
        do_temperature_correction=True,
        bypass=InfluentBypass(threshold=60000.0),
    )
    plant.add_influent("feed", load_bsm2_influent("dry", asm1))
    plant._build_state_layout()
    return plant, bsm2_parameters(asm1, adm1)


def record_lean(steps):
    """Record each lean-loop attempt as ``(t, h, accepted, 1)``."""
    integrate = forward_solve._integrate

    def recorded(attempt, z0, t0, t1, t_eval, control):
        def recorded_attempt(t, z, h):
            a = attempt(t, z, h)
            jax.debug.callback(
                lambda tt, hh, ok: steps.append((float(tt), float(hh), bool(ok), True)),
                t,
                h,
                a.conv & (a.en <= 1.0),
                ordered=True,
            )
            return a

        return integrate(recorded_attempt, z0, t0, t1, t_eval, control)

    forward_solve._integrate = recorded


def record_diffrax(steps):
    """Record each diffrax attempt as ``(t, h, accepted, error finite)``."""
    adapt = diffrax.PIDController.adapt_step_size

    def recorded(controller, t0, t1, y0, y1, args, y_error, *a, **kw):
        res = adapt(controller, t0, t1, y0, y1, args, y_error, *a, **kw)
        finite = jnp.all(jnp.isfinite(jnp.asarray(y_error)))
        jax.debug.callback(
            lambda s, e, keep, fin: steps.append((float(s), float(e - s), bool(keep), bool(fin))),
            t0,
            t1,
            res[0],
            finite,
            ordered=True,
        )
        return res

    diffrax.PIDController.adapt_step_size = recorded


def run(solver, rtol, exact_stages, out: Path, name: str):
    steps = []
    if solver == "diffrax":
        record_diffrax(steps)
    else:
        record_lean(steps)
        if exact_stages:
            forward_solve._KAPPA, forward_solve._MAXNEWT = 1e-7, 40
    plant, params = build()
    kw = dict(
        t_span=(0.0, T_END),
        t_eval=jnp.linspace(0.0, T_END, N_SAVES),
        params=params,
        y0=bsm2_warm_start(plant),
        rtol=rtol,
        atol=1e-3 * rtol / 1e-4,
    )
    start = time.perf_counter()
    if solver == "diffrax":
        config = aquakin.IntegratorConfig(max_steps=50_000_000, colored_jacobian=True)
        sol = plant.solve(**kw, integrator=config)
    else:
        config = aquakin.IntegratorConfig(max_steps=50_000_000)
        sol = plant.solve(**kw, forward_fast=True, integrator=config)
    ys = np.asarray(sol.state)
    jax.effects_barrier()
    np.savez(out / f"{name}.npz", y=ys, steps=np.array(steps, dtype=float).reshape(-1, 4))
    print(f"{name}: {time.perf_counter() - start:.0f}s, {len(steps)} attempts", flush=True)


def kink_switches(plant, y):
    """Branch switches of the settling-flux minimum below the feed, per interface."""
    settler = plant.units["settler"]
    s0, _ = plant._state_layout["settler"]
    te = jnp.linspace(0.0, T_END, y.shape[0])
    feed = plant.stream(
        PlantSolution(t=te, state=jnp.asarray(y), plant=plant), "tank5_split.to_settler"
    )
    C = np.asarray(feed.C)
    x_min = float(settler._fns) * (
        C[:, np.asarray(settler._part_idx_arr)] @ np.asarray(settler._factors_arr)
    )
    X = y[:, s0 : s0 + settler.n_layers]  # lumped TSS per layer, bottom first
    excess = X - x_min[:, None]
    v = settler._v0 * (np.exp(-settler._rh * excess) - np.exp(-settler._rp * excess))
    F = np.clip(v, 0.0, settler._vmax) * X
    print(f"settler TSS at t={T_END:g} d, bottom to top: {np.round(X[-1], 1)}")
    for i in range(settler.feed_layer):
        upper = F[:, i + 1] < F[:, i]
        gap = np.median(np.abs(F[:, i + 1] - F[:, i]) / np.maximum(F[:, i], 1.0))
        print(
            f"  interface {i}|{i + 1}: {int(np.sum(upper[1:] != upper[:-1]))} branch switches "
            f"in {y.shape[0] - 1} save intervals; median flux gap {gap:.1e}"
        )


def report(out: Path):
    plant, _ = build()
    s0, _ = plant._state_layout["settler"]
    ref = np.load(out / "reference.npz")["y"]
    peak = np.maximum(np.max(np.abs(ref), axis=0), 1e-3)
    for f in sorted(out.glob("*.npz")):
        if f.stem == "reference":
            continue
        data = np.load(f)
        e = np.max(np.abs(data["y"] - ref), axis=0) / peak
        s = data["steps"]
        acc, fin = s[:, 2] > 0, s[:, 3] > 0
        print(
            f"{f.stem:28s} attempts {len(s):6d}  accepted {acc.sum():6d}  rejected: root find "
            f"{np.sum(~acc & ~fin):5d} error test {np.sum(~acc & fin):5d}  median accepted h "
            f"{np.median(s[acc, 1]):.1e} | error worst {e.max():.1e} median {np.median(e):.1e} "
            f"settler layers 2-4 {e[s0 + 2]:.1e} {e[s0 + 3]:.1e} {e[s0 + 4]:.1e}"
        )
    kink_switches(plant, ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "reference", "report"))
    ap.add_argument("--solver", choices=("lean", "diffrax"), default="lean")
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--exact-stages", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        report(args.out)
    elif args.command == "reference":
        run("lean", args.rtol or 1e-8, False, args.out, "reference")
    else:
        rtol = args.rtol or 1e-4
        name = f"{args.solver}_rtol{rtol:g}" + ("_exact" if args.exact_stages else "")
        run(args.solver, rtol, args.exact_stages, args.out, name)


if __name__ == "__main__":
    main()
