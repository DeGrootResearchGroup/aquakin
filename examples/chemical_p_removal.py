"""Combined biological + chemical phosphorus removal (ASM2d A²O plant).

On a phosphorus-rich, VFA-limited influent the biological P removal alone cannot
meet a tight effluent-phosphorus permit. Dosing a ferric salt into the aerobic
zone precipitates the residual phosphate as metal phosphate (the ASM2d
``XMeOH``/``XMeP`` precipitation), polishing the effluent -- the standard
combined bio-P + chemical-P strategy. This sweeps the ferric dose and shows the
effluent phosphate falling toward a permit limit, with the phosphorus mass
balance closing across the plant.

Run:
    python examples/chemical_p_removal.py
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant import build_a2o, a2o_influent, a2o_warm_start, FerricDose

PERMIT_TP = 1.0  # g P / m³ effluent target


def steady_effluent(ferric):
    net = aquakin.load_network("asm2d")
    # A phosphorus-rich, VFA-limited influent: biological P removal alone leaves
    # residual phosphate above the permit.
    influent = a2o_influent(net, overrides={"SPO4": 15.0, "SA": 20.0})
    plant = build_a2o(net, ferric=ferric)
    plant.add_influent("feed", influent)
    y0 = a2o_warm_start(plant)
    sol = plant.solve(t_span=(0.0, 200.0), t_eval=jnp.array([200.0]), y0=y0,
                      forward_fast=True, rtol=1e-5, atol=1e-3, max_steps=8_000_000)
    eff = plant.stream(sol, "effluent")
    return float(eff.C_named("SPO4")[-1]), float(eff.C_named("XMeP")[-1])


def main():
    print("Combined bio-P + chemical-P removal (P-rich, VFA-limited influent)")
    print(f"  effluent permit: {PERMIT_TP:.1f} g P/m3\n")
    print(f"  {'ferric dose (m3/d)':>20s} | {'effluent SPO4':>14s} | {'XMeP formed':>12s} | permit")
    print("  " + "-" * 62)
    for ferric in [None, FerricDose(flow=2.0), FerricDose(flow=5.0),
                   FerricDose(flow=10.0)]:
        spo4, xmep = steady_effluent(ferric)
        dose = 0.0 if ferric is None else ferric.flow
        ok = "PASS" if spo4 <= PERMIT_TP else "fail"
        tag = "  (bio-P only)" if ferric is None else ""
        print(f"  {dose:20.1f} | {spo4:11.2f}    | {xmep:9.2f}    | {ok}{tag}")
    print("\nBiological P removal alone leaves residual phosphate above the permit;"
          "\nferric dosing precipitates it as metal phosphate to meet the limit.")


if __name__ == "__main__":
    main()
