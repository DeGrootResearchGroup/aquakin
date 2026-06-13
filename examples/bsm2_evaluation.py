"""BSM2 performance evaluation: open- vs closed-loop DO control.

Builds the full BSM2 plant twice -- once open-loop (fixed aeration kLa) and once
with the closed dissolved-oxygen loop (``build_bsm2(do_control=True)``, a PI
controller holding reactor-4 oxygen at the 2.0 g/m³ setpoint by manipulating
kLa) -- runs both from the same warm-started biomass under the published
constant influent, and reports the headline BSM2 performance indices with
``evaluate_bsm2``:

- **EQI** (Effluent Quality Index, kg pollutant/day): effluent pollutant load,
  lower is better.
- **OCI** (Operational Cost Index): aeration + pumping + 5x sludge production.

The control objective is to hold reactor-4 DO at the setpoint regardless of load:
the open-loop reactor-4 oxygen floats with the fixed kLa, while the closed loop
pins it at 2.0. The energy consequence depends on the operating point -- if the
fixed kLa over-aerates, the controller throttles down and saves aeration energy;
if it under-aerates (as at this warm, high-biomass load), the controller spends
more aeration to reach the setpoint and nitrifies further. The table makes that
trade-off concrete; ``evaluate_bsm2`` reads the *actual* kLa over the run, so
the aeration energy reflects the controller's manipulation.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import (
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    bsm2_warm_start,
    evaluate_bsm2,
)


def _run_and_evaluate(do_control, asm1, adm1, params):
    plant = build_bsm2(asm1, adm1, do_control=do_control)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    # Each plant builds its own warm start: the closed-loop plant carries one
    # extra state (the controller integral, seeded at its default 0).
    y0 = bsm2_warm_start(plant)
    sol = plant.solve(
        t_span=(0.0, 30.0), t_eval=jnp.linspace(0.0, 30.0, 31),
        params=params, y0=jnp.asarray(y0),
        rtol=1e-5, atol=1e-3, max_steps=500_000,
    )
    so4 = float(sol.C_named("tank4", "SO")[-1])
    return evaluate_bsm2(plant, sol, params), so4


def main() -> None:
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)

    print("Evaluating BSM2 open-loop vs closed-loop DO control "
          "(constant influent) ...")
    open_ev, open_so4 = _run_and_evaluate(False, asm1, adm1, params)
    closed_ev, closed_so4 = _run_and_evaluate(True, asm1, adm1, params)

    print()
    header = f"{'index':<22}{'open-loop':>14}{'closed-loop':>14}{'change':>12}"
    print(header)
    print("-" * len(header))
    # Species units come from the network; the plant-level metric units
    # (EQI/OCI/energy/sludge) are not species, so they keep explicit labels.
    rows = [
        (f"reactor-4 SO ({asm1.units_of('SO')})", open_so4, closed_so4),
        ("EQI (kg/d)", open_ev.eqi, closed_ev.eqi),
        ("OCI (full BSM2)", open_ev.oci, closed_ev.oci),
        ("  aeration (kWh/d)", open_ev.aeration_energy, closed_ev.aeration_energy),
        ("  pumping (kWh/d)", open_ev.pumping_energy, closed_ev.pumping_energy),
        ("  mixing (kWh/d)", open_ev.mixing_energy, closed_ev.mixing_energy),
        ("  sludge (kg TSS/d)", open_ev.sludge_production, closed_ev.sludge_production),
        ("  carbon (kg COD/d)", open_ev.carbon_mass, closed_ev.carbon_mass),
        ("  methane (kg CH₄/d)", open_ev.methane_production, closed_ev.methane_production),
        ("  heating (kWh/d)", open_ev.heating_energy, closed_ev.heating_energy),
        (f"effluent SNH ({asm1.units_of('SNH')})", open_ev.effluent["SNH"], closed_ev.effluent["SNH"]),
        (f"effluent SNO ({asm1.units_of('SNO')})", open_ev.effluent["SNO"], closed_ev.effluent["SNO"]),
    ]
    for label, o, c in rows:
        pct = 100.0 * (c - o) / (abs(o) + 1e-12)
        print(f"{label:<22}{o:>14.2f}{c:>14.2f}{pct:>11.1f}%")

    print()
    print(f"Aerated reactors: {open_ev.aerated_tanks}")
    print(f"Note: {open_ev.oci_note}")


if __name__ == "__main__":
    main()
