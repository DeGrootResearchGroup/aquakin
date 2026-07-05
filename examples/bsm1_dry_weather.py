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
    model = aquakin.load_model("asm1")

    plant = build_bsm1(model=model)
    inf = load_bsm1_influent("dry", model)
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
    # final_named reads the last-point values for several species in one call
    # (no per-species [-1] slice). Units come from the model (carried through
    # compile), not a hand-kept name->unit table -- no risk of mislabelling N as COD.
    finals = sol.final_named("tank5", ("SS", "SNH", "SNO", "SO", "XB_H", "XB_A"))
    for sp, val in finals.items():
        print(f"  {sp:5s} = {val:7.2f}  {model.units_of(sp)}")

    print()
    print("Per-tank XB_H (heterotrophic biomass) at simulation end:")
    for name in ("tank1", "tank2", "tank3", "tank4", "tank5"):
        val = float(sol.C_named(name, "XB_H")[-1])
        print(f"  {name}: {val:7.1f}  {model.units_of('XB_H')}")

    # Effluent metrics: reconstruct the effluent stream over the saved states
    # (the plant integrates unit states, not the inter-unit streams). The
    # semantic shortcut reads the right port (see plant.list_streams()); the
    # metric kernels take the reconstructed stream directly.
    eff = plant.effluent_stream(sol)             # == plant.stream(sol, "effluent")
    avgs = effluent_averages(eff)
    print()
    print("Time-averaged effluent quality:")
    # Real species (SNH, SNO) get their units from the model; the lumped
    # aggregate metrics (COD/BOD/TSS/TKN) are not species, so they carry an
    # explicit label.
    aggregate_units = {"COD": "g_COD/m3", "BOD": "g_COD/m3",
                       "TSS": "g_SS/m3", "TKN": "g_N/m3"}
    for key, val in avgs.items():
        unit = (model.units_of(key) if key in model.species_index
                else aggregate_units[key])
        print(f"  {key:5s} = {val:7.2f}  {unit}")

    # Headline BSM1 performance indices (EQI / OCI). The evaluation prints a
    # labeled, units-annotated breakdown -- each OCI term with its value, units
    # and signed contribution to the index, plus the OCI-definition caveat -- so
    # the headline numbers aren't bare floats to misread against published values.
    ev = evaluate_bsm1(plant, sol)
    print()
    print(ev.report())          # == print(ev); fields (ev.eqi, ev.oci, ...) too


if __name__ == "__main__":
    main()
