"""BSM2 results-level mass-balance closure.

The first check an engineer runs on a plant result: does what went in equal what
came out, plus what left as gas, plus what accumulated? ``plant.mass_balance``
accounts COD / N / P over the simulation window -- inflow, outflow (effluent +
disposal cake), gas (aeration oxygen, digester biogas, denitrification N2) and
the change in plant inventory -- and reports the closure imbalance.

It works across the two-network BSM2 (ASM1 water line + ADM1 digester): the
inventories and fluxes are summed on one canonical gram basis via the shipped
``aquakin.composition_table``, so the g/m3 water line and the kg/m3, kmol/m3
digester add up. At steady state the balance closes to well under 1%.

(This is also the tool that uncovered a nitrogen-stoichiometry transcription
error in the ADM1 disintegration/decay reactions, since corrected against the
official BSM2 source.)
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
    bsm2_warm_start,
)


def main() -> None:
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")

    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    params = bsm2_parameters(asm1, adm1)

    print("Settling BSM2 to steady state ...")
    ss = plant.run_to_steady_state(
        params=params, y0=jnp.asarray(bsm2_warm_start(plant)), max_time=400.0)

    # Score the balance over a short window at the operating point.
    sol = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 9),
                      params=params, y0=ss.state)
    mb = plant.mass_balance(sol, components=("COD", "N"), params=params)

    print()
    print(mb.summary())
    print()
    print("Gas / reaction breakdown (canonical g over the window):")
    for name, value in mb.gas_detail.items():
        print(f"  {name:18s} = {value:12.4g}")


if __name__ == "__main__":
    main()
