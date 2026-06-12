"""BSM1 dry-weather demo.

Builds the canonical BSM1 reference plant (5 ASM1 reactors + secondary
clarifier, internal + RAS recycles), drives it with the synthesised
dry-weather influent, and prints the headline plant state plus simple
metrics. End-to-end demonstration of the ``aquakin.plant`` sub-package.

Note: the shipped influent CSV is *synthesised* to match the
BSM1 statistical profile, not the canonical IWA file. For published
EQI / OCI comparisons, replace ``BSM1_dry.csv`` under
``aquakin/plant/bsm/data/`` with the official BSM1 file.
"""

import jax.numpy as jnp

import aquakin
from aquakin import effluent_averages, evaluate_bsm1
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent


def main() -> None:
    network = aquakin.load_network("asm1")

    plant = build_bsm1(network=network)
    inf = load_bsm1_influent("dry", network)
    plant.add_influent("feed", inf, to="inlet_mix.fresh")

    print("BSM1 plant:")
    print(f"  units: {list(plant.units.keys())}")
    print(f"  state size: {sum(u.state_size for u in plant.units.values())}")
    print(f"  influent: dry weather, {inf.t.shape[0]} samples over "
          f"{float(inf.t[-1]):.1f} days, avg Q = {float(inf.Q.mean()):.0f} m³/d")
    print()

    print("Solving for 15 days (open-loop kLa, ideal clarifier) ...")
    sol = plant.solve(
        t_span=(0.0, 15.0),
        t_eval=jnp.linspace(0.0, 15.0, 16),
        rtol=1e-4, atol=1e-3,
    )

    print()
    print("Tank-5 effluent state at simulation end:")
    # Units come from the network (carried through compile), not a hand-kept
    # name->unit table -- no risk of mislabelling N as COD.
    for sp in ("SS", "SNH", "SNO", "SO", "XB_H", "XB_A"):
        val = float(sol.C_named("tank5", sp)[-1])
        print(f"  {sp:5s} = {val:7.2f}  {network.units_of(sp)}")

    print()
    print("Per-tank XB_H (heterotrophic biomass) at simulation end:")
    for name in ("tank1", "tank2", "tank3", "tank4", "tank5"):
        val = float(sol.C_named(name, "XB_H")[-1])
        print(f"  {name}: {val:7.1f}  {network.units_of('XB_H')}")

    # Effluent metrics: reconstruct the clarifier overflow stream over the saved
    # states (the plant integrates unit states, not the inter-unit streams).
    # The metric kernels take the reconstructed stream directly.
    eff = plant.stream(sol, "clarifier.overflow")
    avgs = effluent_averages(eff)
    print()
    print("Time-averaged effluent quality:")
    # Real species (SNH, SNO) get their units from the network; the lumped
    # aggregate metrics (COD/BOD/TSS/TKN) are not species, so they carry an
    # explicit label.
    aggregate_units = {"COD": "g_COD/m3", "BOD": "g_COD/m3",
                       "TSS": "g_SS/m3", "TKN": "g_N/m3"}
    for key, val in avgs.items():
        unit = (network.units_of(key) if key in network.species_index
                else aggregate_units[key])
        print(f"  {key:5s} = {val:7.2f}  {unit}")

    # Headline BSM1 performance indices (EQI / OCI and component terms).
    ev = evaluate_bsm1(plant, sol)
    print()
    print("BSM1 performance indices:")
    print(f"  EQI = {ev.eqi:8.1f}  kg pollutant/d")
    print(f"  OCI = {ev.oci:8.1f}  (AE + PE + 5*sludge)")
    print(f"    aeration energy   = {ev.aeration_energy:8.1f}  kWh/d")
    print(f"    pumping energy    = {ev.pumping_energy:8.1f}  kWh/d")
    print(f"    sludge production = {ev.sludge_production:8.1f}  kg TSS/d")


if __name__ == "__main__":
    main()
