"""A²O biological-nutrient-removal plant (ASM2d): simultaneous C/N/P removal.

Builds the default Anaerobic–Anoxic–Oxic flowsheet on the ASM2d network, warm-
started from a healthy enhanced-biological-phosphorus-removal (EBPR) sludge, and
reports the effluent and the bio-P signature (anaerobic phosphate release +
aerobic luxury uptake / poly-P storage).

This is the first phosphorus-capable plant in aquakin — the BSM benchmarks run
the P-free ASM1 — and the substrate for the chemical-P (metal-salt dosing)
demonstration.

Run:
    python examples/a2o_bio_p.py
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant import build_a2o, a2o_influent, a2o_warm_start


def main():
    net = aquakin.load_network("asm2d")
    plant = build_a2o(net)
    influent = a2o_influent(net)
    plant.add_influent("feed", influent)

    # Start from an established EBPR sludge and settle to steady state.
    y0 = a2o_warm_start(plant)
    sol = plant.solve(t_span=(0.0, 200.0), t_eval=jnp.linspace(0.0, 200.0, 41),
                      y0=y0, rtol=1e-5, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=4_000_000))

    eff = plant.stream(sol, "effluent")
    inflC = influent.at(0.0).C
    def i(sp):
        return float(inflC[net.species_index[sp]])
    def e(sp):
        return float(eff.C_named(sp)[-1])

    print("A2O plant — steady-state effluent (influent -> effluent):")
    print(f"  Ammonia   SNH4 : {i('SNH4'):6.1f} -> {e('SNH4'):6.2f} g N/m3")
    print(f"  Nitrate   SNO3 : {i('SNO3'):6.1f} -> {e('SNO3'):6.2f} g N/m3")
    print(f"  Phosphate SPO4 : {i('SPO4'):6.1f} -> {e('SPO4'):6.2f} g P/m3"
          f"   ({100 * (i('SPO4') - e('SPO4')) / i('SPO4'):.0f}% P removal)")
    print(f"  Sol. COD  SF+SA: {i('SF') + i('SA'):6.1f} -> {e('SF') + e('SA'):6.2f} g COD/m3")

    # Bio-P signature: phosphate released in the anaerobic selector, taken back up
    # (and stored as poly-P) in the aerobic zone.
    states = plant.states_by_unit(sol.final_state)
    def zone(unit, sp):
        return float(states[unit][net.species_index[sp]])
    print("\nBiological-P signature (phosphate by zone):")
    print(f"  anaerobic SPO4 : {zone('anaer2', 'SPO4'):6.1f} g P/m3   (release)")
    print(f"  aerobic   SPO4 : {zone('aer3', 'SPO4'):6.2f} g P/m3   (uptake)")
    print(f"  aerobic   XPAO : {zone('aer3', 'XPAO'):6.0f} g COD/m3  (PAO biomass)")
    print(f"  aerobic   XPP  : {zone('aer3', 'XPP'):6.0f} g P/m3    (stored poly-P)")


if __name__ == "__main__":
    main()
