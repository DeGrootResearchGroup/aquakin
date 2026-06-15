"""BSM2 greenhouse-gas and cost reporting + scenario KPI comparison.

Builds on ``evaluate_bsm2`` (EQI / OCI) to produce the two deliverables a
carbon-footprint / cost trade-off study actually reports:

- a **carbon footprint** (``carbon_footprint``): the plant's CO2-equivalent
  emission rate -- indirect energy CO2e (aeration + pumping + mixing at a grid
  carbon intensity), the digester methane fugitive emission, the avoided-emission
  biogas credit, and the direct N2O (``direct_n2o_emission`` -- 0 here because
  the standard BSM2 plant runs ASM1, which does not resolve nitrous oxide; an
  N2O-capable activated-sludge network gives a non-zero term);
- a monetised **operating cost** (``operating_cost``): energy / carbon / sludge
  OPEX, the biogas value credit, an optional CAPEX and carbon charge.

Two operating scenarios (open-loop fixed aeration vs a closed dissolved-oxygen
control loop) are then put side by side with ``kpi_comparison``, the
standardized KPI table that tabulates heterogeneous report objects (the
evaluation, the footprint, the cost) per scenario -- the client-deliverable form
of the comparison.

The grid factor, GWPs and unit prices here are representative defaults; set them
to a site's actual values for a real estimate.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import (
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    bsm2_warm_start,
    direct_n2o_emission,
    evaluate_bsm2,
)

GRID_FACTOR = 0.4   # kg CO2e / kWh
COST = aquakin.CostFactors(energy_price=0.12, carbon_price=0.50,
                           sludge_disposal_price=0.35, biogas_value=0.20,
                           ghg_price=0.05)


def _scenario(do_control, asm1, adm1, params):
    """Build, warm-start and solve a BSM2 plant; return its report objects."""
    plant = build_bsm2(asm1, adm1, do_control=do_control)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    y0 = bsm2_warm_start(plant)
    sol = plant.solve(
        t_span=(0.0, 30.0), t_eval=jnp.linspace(0.0, 30.0, 31),
        params=params, y0=jnp.asarray(y0),
        rtol=1e-5, atol=1e-3, max_steps=500_000,
    )
    ev = evaluate_bsm2(plant, sol, params)
    n2o = direct_n2o_emission(plant, sol, params)   # 0 for ASM1 (no SN2O)
    fp = aquakin.carbon_footprint(
        ev.total_energy(), grid_factor=GRID_FACTOR, n2o_emission=n2o,
        methane_production=ev.methane_production, ch4_fugitive_fraction=0.015,
    )
    oc = aquakin.operating_cost(
        energy_kwh_per_d=ev.total_energy(), carbon_kg_cod_per_d=ev.carbon_mass,
        sludge_kg_tss_per_d=ev.sludge_production,
        methane_kg_per_d=ev.methane_production, factors=COST,
        co2e_per_d=fp.total_co2e,
    )
    return ev, fp, oc


def main() -> None:
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)

    print("Running BSM2 open-loop vs closed-loop DO control "
          "(constant influent) ...\n")
    open_ev, open_fp, open_oc = _scenario(False, asm1, adm1, params)
    ctrl_ev, ctrl_fp, ctrl_oc = _scenario(True, asm1, adm1, params)

    print(open_fp, "\n")
    print(open_oc, "\n")

    # Side-by-side KPI tables: performance, GHG, cost.
    print("Performance:")
    print(aquakin.kpi_comparison({"open-loop": open_ev,
                                  "DO-control": ctrl_ev}).table(), "\n")
    print("Carbon footprint:")
    print(aquakin.kpi_comparison({"open-loop": open_fp,
                                  "DO-control": ctrl_fp}).table(), "\n")
    print("Operating cost:")
    cost_cmp = aquakin.kpi_comparison({"open-loop": open_oc,
                                       "DO-control": ctrl_oc})
    print(cost_cmp.table())
    print(f"\nLowest annual cost: {cost_cmp.best('Annual cost (USD/yr)')}")


if __name__ == "__main__":
    main()
