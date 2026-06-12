"""BSM2 with seasonal influent temperature.

Temperature is carried algebraically through the flowsheet: a temperature-
carrying influent flows through the plant (mixers heat-balance it, every other
unit passes it through), each activated-sludge reactor reads its inlet
temperature, and the ASM1 temperature corrections slow the kinetics in the cold.
Nitrification -- the most temperature-sensitive process -- slows most, so colder
water leaves more residual ammonia in the effluent.

Use ``bsm2_asm1_network()`` (its temperature corrections are referenced to the
BSM2 15 °C base) for *both* the plant and the influent, so a constant-15 °C run
reproduces the validated steady state and a temperature-carrying influent drives
it away from there.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.influent import InfluentSeries

WARM_AS = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
           "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
           "SND": 0.7, "XND": 3.0, "SALK": 5.0}


def _steady_snh(asm1, adm1, params, T_kelvin):
    """Steady-state tank-5 effluent ammonia at a constant influent temperature."""
    # The published constant influent, re-tagged with the season's temperature.
    base = bsm2_constant_influent(asm1)
    influent = InfluentSeries(t=base.t, Q=base.Q, C=base.C, network=asm1,
                              T=jnp.full_like(base.Q, float(T_kelvin)))

    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", influent, to="front_mix.fresh")

    warm = asm1.concentrations(WARM_AS)
    tanks = ("tank1", "tank2", "tank3", "tank4", "tank5")
    y0 = plant.initial_state(overrides={t: warm for t in tanks})

    sol = plant.solve(t_span=(0.0, 150.0), t_eval=jnp.array([0.0, 150.0]),
                      params=params, y0=jnp.asarray(y0),
                      rtol=1e-5, atol=1e-3, max_steps=500_000)
    return float(sol.C_named("tank5", "SNH")[-1])


def main() -> None:
    asm1 = bsm2_asm1_network()           # corrections referenced to 15 °C
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)

    print("BSM2 effluent ammonia vs influent temperature:")
    snh_units = asm1.units_of("SNH")     # units from the network, not hardcoded
    print(f"  {'season':<10} {'T [°C]':>7} {f'SNH [{snh_units}]':>14}")
    for season, T_c in (("winter", 10.0), ("spring", 15.0), ("summer", 20.0)):
        snh = _steady_snh(asm1, adm1, params, 273.15 + T_c)
        print(f"  {season:<10} {T_c:>7.1f} {snh:>14.3f}")

    print()
    print("Colder influent -> slower nitrification -> higher residual ammonia.")


if __name__ == "__main__":
    main()
