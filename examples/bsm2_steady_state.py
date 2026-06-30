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
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
)


def main() -> None:
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")

    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    params = bsm2_parameters(asm1, adm1)

    # Seed the AS reactors with a healthy biomass (warm start) so the slow
    # digester settles quickly. The rest of the plant starts from its defaults.
    y0 = bsm2_warm_start(plant)

    print("Settling BSM2 to open-loop steady state (constant influent) ...")
    sol = plant.solve(
        t_span=(0.0, 200.0), t_eval=jnp.array([0.0, 200.0]),
        params=params, y0=jnp.asarray(y0),
        rtol=1e-5, atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=500_000),
    )

    print()
    print("Activated-sludge reactor 5 (effluent) at steady state:")
    # final_named reads the steady-state (last-point) values for several species
    # in one call. Units come from the ASM1 network (carried through compile),
    # not a hand-kept name->unit table.
    finals = sol.final_named("tank5", ("SNH", "SNO", "SO", "XB_H", "XB_A", "XI"))
    for sp, val in finals.items():
        print(f"  {sp:5s} = {val:8.2f}  {asm1.units_of(sp)}")

    print()
    print("Anaerobic digester biogas at steady state:")
    # The biogas is a *derived* output (computed from the ADM1 headspace state,
    # not a material port); plant.digester_gas reconstructs flow + composition.
    gas = plant.digester_gas(sol)
    print(f"  total flow  Q_gas = {float(gas.Q[-1]):10.4g} m³/d")
    print(f"  methane           = {gas.methane_production():10.4g} kg CH₄/d")
    print(f"  partial p (bar): CH₄ {float(gas.p_ch4[-1]):.3f}  "
          f"CO₂ {float(gas.p_co2[-1]):.3f}  H₂ {float(gas.p_h2[-1]):.2e}")


if __name__ == "__main__":
    main()
