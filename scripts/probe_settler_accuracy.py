"""Why the lean loop and diffrax differ on BSM2's Takacs settler layers.

The case of ``test_forward_fast.py::test_forward_fast_matches_diffrax_bsm2``:
BSM2 with temperature correction and the influent bypass, the synthesised dry
influent, 8 days from ``bsm2_warm_start``, saved every 0.01 d. Every command takes
``--interface-flux takacs|godunov`` (the settler's ``interface_flux``; default
``takacs``), and runs and references are kept per flux. Three commands:

- ``run --solver lean|diffrax [--rtol R] [--exact-stages]`` solves it and records
  every step attempt: its start, size, whether it was accepted, and (diffrax) whether
  its error estimate was finite -- diffrax marks a failed root find with an infinite
  one. ``--exact-stages`` solves the lean loop's stages to ``_KAPPA = 1e-7``.
  ``atol`` scales with ``rtol`` (``1e-3`` at ``rtol = 1e-4``).
- ``reference [--rtol 1e-8]`` is ``run --solver lean`` at a tight tolerance.
- ``report`` (no ``--interface-flux``: it reports both) compares every run with the
  reference of its own flux: the worst component's error and the median
  component's (each relative to the component's peak, floored at ``1e-3``), the
  settler layers 2-4, and the step counts. Along each reference it counts how
  often the upper of two adjacent layers below the feed changes from the more
  dilute to the denser one between saves -- where the Takacs minimum switches
  between the two layers' fluxes (a kink in the right-hand side) -- and how much
  the blanket layers 2-5 move from one save to the next (the pulses). With both
  references present it also compares the two models on the effluent, wastage
  and RAS streams.

Usage (each command one process, one at a time -- each holds a whole-plant program)::

    python scripts/probe_settler_accuracy.py run --solver lean --out DIR
    python scripts/probe_settler_accuracy.py run --solver diffrax --out DIR
    python scripts/probe_settler_accuracy.py reference --out DIR
    python scripts/probe_settler_accuracy.py run --solver lean --interface-flux godunov --out DIR
    python scripts/probe_settler_accuracy.py reference --interface-flux godunov --out DIR
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
FLUXES = ("takacs", "godunov")
STREAMS = ("effluent", "wastage", "ras")
SPECIES = ("SNH", "SNO", "SS", "XS", "XB_H", "XB_A", "SND", "XND")


def build(interface_flux):
    asm1 = bsm2_asm1_model()
    adm1 = aquakin.load_model("adm1")
    plant = build_bsm2(
        asm1_model=asm1,
        adm1_model=adm1,
        do_temperature_correction=True,
        bypass=InfluentBypass(threshold=60000.0),
        settler_interface_flux=interface_flux,
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


def run(solver, rtol, exact_stages, interface_flux, out: Path, name: str):
    steps = []
    if solver == "diffrax":
        record_diffrax(steps)
    else:
        record_lean(steps)
        if exact_stages:
            forward_solve._KAPPA, forward_solve._MAXNEWT = 1e-7, 40
    plant, params = build(interface_flux)
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


def blanket_behaviour(plant, y):
    """Switches of the upper/denser ordering below the feed, and layer movement."""
    settler = plant.units["settler"]
    s0, _ = plant._state_layout["settler"]
    X = y[:, s0 : s0 + settler.n_layers]  # lumped TSS per layer, bottom first
    print(f"  settler TSS at t={T_END:g} d, bottom to top: {np.round(X[-1], 1)}")
    for i in range(settler.feed_layer):
        upper_denser = X[:, i + 1] > X[:, i]
        flips = int(np.sum(upper_denser[1:] != upper_denser[:-1]))
        print(
            f"  interface {i}|{i + 1}: upper layer denser at {np.mean(upper_denser):.0%} of "
            f"saves, {flips} changes in {y.shape[0] - 1} save intervals"
        )
    move = np.abs(np.diff(X[:, 2:6], axis=0)) / X[:-1, 2:6]
    print(
        "  layers 2-5, change between consecutive saves (relative): median "
        f"{np.median(move):.1e}, p99 {np.percentile(move, 99):.1e}, max {move.max():.1e}"
    )


def streams(plant, y):
    te = jnp.linspace(0.0, T_END, y.shape[0])
    sol = PlantSolution(t=te, state=jnp.asarray(y), plant=plant)
    return {name: plant.stream(sol, name) for name in STREAMS}


def compare_models(out: Path):
    """The Godunov model against the Takacs one, both at the reference tolerance."""
    plant, _ = build("takacs")
    a = streams(plant, np.load(out / "reference_takacs.npz")["y"])
    b = streams(plant, np.load(out / "reference_godunov.npz")["y"])
    cols = [plant.units["settler"].model.species_index[s] for s in SPECIES]
    print("godunov vs takacs (reference tolerance), relative to the takacs peak:")
    for name in STREAMS:
        ca, cb = np.asarray(a[name].C)[:, cols], np.asarray(b[name].C)[:, cols]
        peak = np.maximum(np.max(np.abs(ca), axis=0), 1e-3)
        worst = np.max(np.abs(cb - ca), axis=0) / peak
        mean = np.abs(cb[1:].mean(axis=0) - ca[1:].mean(axis=0)) / np.abs(ca[1:].mean(axis=0))
        k = int(np.argmax(worst))
        print(
            f"  {name}: worst species {SPECIES[k]} {worst[k]:.1e} (8-day mean "
            f"{mean[k]:.1e}); all species' mean differences up to {mean.max():.1e}"
        )


def report(out: Path):
    for flux in FLUXES:
        ref_path = out / f"reference_{flux}.npz"
        if not ref_path.exists():
            continue
        plant, _ = build(flux)
        s0, _ = plant._state_layout["settler"]
        ref = np.load(ref_path)["y"]
        peak = np.maximum(np.max(np.abs(ref), axis=0), 1e-3)
        print(f"== {flux} (reference: {ref_path.name})")
        for f in sorted(out.glob(f"{flux}_*.npz")):
            data = np.load(f)
            e = np.max(np.abs(data["y"] - ref), axis=0) / peak
            s = data["steps"]
            acc, fin = s[:, 2] > 0, s[:, 3] > 0
            print(
                f"  {f.stem:30s} attempts {len(s):6d}  accepted {acc.sum():6d}  rejected: "
                f"root find {np.sum(~acc & ~fin):5d} error test {np.sum(~acc & fin):5d}  "
                f"median accepted h {np.median(s[acc, 1]):.1e} | error worst {e.max():.1e} "
                f"median {np.median(e):.1e} settler layers 2-4 {e[s0 + 2]:.1e} "
                f"{e[s0 + 3]:.1e} {e[s0 + 4]:.1e}"
            )
        blanket_behaviour(plant, ref)
    if all((out / f"reference_{flux}.npz").exists() for flux in FLUXES):
        compare_models(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "reference", "report"))
    ap.add_argument("--solver", choices=("lean", "diffrax"), default="lean")
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--exact-stages", action="store_true")
    ap.add_argument("--interface-flux", choices=FLUXES, default="takacs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    flux = args.interface_flux
    if args.command == "report":
        report(args.out)
    elif args.command == "reference":
        run("lean", args.rtol or 1e-8, False, flux, args.out, f"reference_{flux}")
    else:
        rtol = args.rtol or 1e-4
        name = f"{flux}_{args.solver}_rtol{rtol:g}" + ("_exact" if args.exact_stages else "")
        run(args.solver, rtol, args.exact_stages, flux, args.out, name)


if __name__ == "__main__":
    main()
