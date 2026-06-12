"""BSM2 open-loop steady state.

Builds the full BSM2 plant -- the BSM1 activated-sludge core wrapped with the
sludge train (primary clarifier, thickener, ADM1 anaerobic digester with the
ASM1<->ADM1 interfaces, and dewatering, with the two reject streams recycled to
the front) -- and drives it with the published constant influent to its
open-loop steady state. This is a genuinely two-network plant (ASM1 water line +
ADM1 digester) integrated under one monolithic solve.

It prints the headline activated-sludge reactor state and the digester biogas,
which match the published BSM2 reference steady state to within a few percent
(see ``tests/validation/test_bsm2_steadystate.py``).

The plant is warm-started from a healthy activated-sludge biomass so the slow
digester (HRT ~19 d) settles quickly, using ``plant.initial_state(overrides=...)``
to seed the reactor units by name.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
)

# A healthy activated-sludge biomass to seed the five AS reactors (g/m³),
# roughly the reference BSM2 reactor composition. The digester and the rest of
# the plant start from network defaults and settle in.
WARM_AS = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
           "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
           "SND": 0.7, "XND": 3.0, "SALK": 5.0}


def main() -> None:
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")

    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    params = bsm2_parameters(asm1, adm1)

    # Seed the AS reactors with a healthy biomass (warm start) so the slow
    # digester settles quickly. The rest of the plant starts from its defaults.
    warm = asm1.concentrations(WARM_AS)
    tanks = ("tank1", "tank2", "tank3", "tank4", "tank5")
    y0 = plant.initial_state(overrides={t: warm for t in tanks})

    print("Settling BSM2 to open-loop steady state (constant influent) ...")
    sol = plant.solve(
        t_span=(0.0, 200.0), t_eval=jnp.array([0.0, 200.0]),
        params=params, y0=jnp.asarray(y0),
        rtol=1e-5, atol=1e-3, max_steps=500_000,
    )

    print()
    print("Activated-sludge reactor 5 (effluent) at steady state:")
    # Units come from the ASM1 network (carried through compile), not a
    # hand-kept name->unit table.
    for sp in ("SNH", "SNO", "SO", "XB_H", "XB_A", "XI"):
        val = float(sol.C_named("tank5", sp)[-1])
        print(f"  {sp:5s} = {val:8.2f}  {asm1.units_of(sp)}")

    print()
    print("Anaerobic digester headspace (biogas) at steady state:")
    for sp, label in (("S_gas_ch4", "methane"), ("S_gas_co2", "CO₂"),
                      ("S_gas_h2", "hydrogen")):
        if sp in adm1.species_index:
            val = float(sol.C_named("digester", sp)[-1])
            print(f"  {label:8s} ({sp}) = {val:10.4g}")


if __name__ == "__main__":
    main()
