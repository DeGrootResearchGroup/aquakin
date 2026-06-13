"""BSM2 scheduled (timed) wastage.

The waste-sludge pump can follow a time schedule instead of a constant flow:
``build_bsm2(wastage_schedule=bsm2_wastage_schedule())`` steps the wastage
between a low and a high rate over the year (the BSM2 strategy for managing the
sludge inventory). Wasting more sludge lowers the reactor biomass (a shorter
solids retention time); wasting less lets it build back up.

This script drives the plant across the first schedule step (day 182, where the
wastage rises from 300 to 450 m³/d) and reports the waste flow and the reactor-5
heterotroph biomass on either side, showing the inventory response.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm import (
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    bsm2_wastage_schedule,
)


def main() -> None:
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)

    schedule = bsm2_wastage_schedule()
    plant = build_bsm2(asm1, adm1, wastage_schedule=schedule)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")

    y0 = bsm2_warm_start(plant)

    t_eval = jnp.array([0.0, 90.0, 180.0, 220.0, 300.0])
    print("Driving BSM2 with the scheduled wastage (step up at day 182) ...")
    sol = plant.solve(t_span=(0.0, 300.0), t_eval=t_eval, params=params,
                      y0=jnp.asarray(y0), rtol=1e-5, atol=1e-3, max_steps=900_000)

    waste = plant.stream(sol, "underflow_split.waste", params)
    print()
    print(f"{'day':>6}{'Qw (m³/d)':>12}{'tank5 XB_H (g/m³)':>20}")
    print("-" * 38)
    for i, day in enumerate(t_eval):
        qw = float(waste.Q[i])
        xbh = float(sol.C_named("tank5", "XB_H")[i])
        print(f"{float(day):>6.0f}{qw:>12.1f}{xbh:>20.1f}")

    print()
    print("The wastage steps 300 -> 450 m³/d at day 182; wasting more sludge "
          "draws the reactor biomass (XB_H) down -- a shorter solids retention "
          "time, the lever the wastage timer manages.")


if __name__ == "__main__":
    main()
