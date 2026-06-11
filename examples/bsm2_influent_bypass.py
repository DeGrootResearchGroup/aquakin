"""BSM2 hydraulic influent bypass under wet-weather flow.

Builds the BSM2 plant with the influent bypass enabled
(``build_bsm2(influent_bypass=True)``) and drives it with a storm flow well
above the bypass threshold. Raw influent flow above the threshold is diverted
around the whole treatment train and rejoined with the clarified effluent, so
the plant's hydraulics are protected at the cost of releasing untreated
wastewater -- the classic wet-weather trade-off.

The script reports the flow split and contrasts the clarified (treated) stream
with the final (treated + bypassed) effluent, showing the pollutant load the
bypass adds.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import (
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    evaluate_bsm2,
)
from aquakin.plant.bsm.bsm2 import BSM2_BYPASS_Q
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.metrics import derived_COD, derived_TSS

# A healthy activated-sludge biomass to seed the five AS reactors (g/m³).
WARM_AS = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
           "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
           "SND": 0.7, "XND": 3.0, "SALK": 5.0}


def main() -> None:
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)

    # A sustained storm flow at 1.5x the bypass threshold.
    Q_storm = 1.5 * BSM2_BYPASS_Q
    C = bsm2_constant_influent(asm1).C[0]
    influent = InfluentSeries(t=jnp.array([0.0, 1e4]), Q=jnp.full((2,), Q_storm),
                              C=jnp.tile(C, (2, 1)), network=asm1)

    plant = build_bsm2(asm1, adm1, influent_bypass=True)
    plant.add_influent("feed", influent, to="bypass_split.in")

    warm = asm1.concentrations(WARM_AS)
    tanks = ("tank1", "tank2", "tank3", "tank4", "tank5")
    y0 = plant.initial_state(overrides={t: warm for t in tanks})

    print(f"Driving BSM2 (influent bypass on) at a storm flow "
          f"{Q_storm:.0f} m³/d (threshold {BSM2_BYPASS_Q:.0f}) ...")
    sol = plant.solve(t_span=(0.0, 30.0), t_eval=jnp.array([0.0, 30.0]),
                      params=params, y0=jnp.asarray(y0),
                      rtol=1e-5, atol=1e-3, max_steps=500_000)

    bp = plant.stream(sol, "bypass_split.bypass", params)
    pl = plant.stream(sol, "bypass_split.to_plant", params)
    treated = plant.stream(sol, "settler.overflow", params)
    eff = plant.stream(sol, "effluent_mix.out", params)

    print()
    print(f"Flow split (m³/d): bypassed = {float(bp.Q[-1]):8.0f}   "
          f"to plant = {float(pl.Q[-1]):8.0f}")
    print()
    print(f"{'stream':<22}{'Q (m³/d)':>12}{'COD (g/m³)':>14}{'TSS (g/m³)':>14}")
    print("-" * 62)
    for label, s in (("clarified (treated)", treated), ("final effluent", eff)):
        cod = float(derived_COD(s.C[-1], asm1))
        tss = float(derived_TSS(s.C[-1], asm1))
        print(f"{label:<22}{float(s.Q[-1]):>12.0f}{cod:>14.1f}{tss:>14.1f}")

    ev = evaluate_bsm2(plant, sol, params)
    print()
    print(f"EQI (final effluent) = {ev.eqi:.0f} kg/d  "
          f"-- the untreated bypass dominates the pollutant load.")


if __name__ == "__main__":
    main()
