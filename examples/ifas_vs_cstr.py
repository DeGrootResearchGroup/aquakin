"""IFAS / MBBR intensification demo: a biofilm tank vs a plain CSTR.

Builds two single-tank ASM1 plants fed the same influent at the same volume and
aeration -- one a plain suspended-growth ``CSTRUnit``, one an ``IFASUnit`` whose
carrier media host an attached biofilm -- and compares their steady-state
effluent. The IFAS tank removes more soluble COD and nitrifies markedly more:
the attached biomass adds treatment capacity in the same footprint, which is the
point of an MBBR/IFAS intensification retrofit.

The ``IFASUnit`` wires the depth-resolved ``BiofilmReactor`` (1-D
diffusion-reaction over biofilm depth) into the flowsheet; oxygen and substrate
diffuse from the bulk into the biofilm, react against the attached biomass, and
the well-mixed bulk leaves as the effluent.
"""

import aquakin
from aquakin.plant import Aeration, CSTRUnit, IFASUnit, Plant


def _single_tank_plant(unit, model):
    plant = Plant("demo")
    plant.add_unit(unit)
    plant.add_influent(
        "feed",
        model.influent({"SS": 60.0, "SNH": 25.0, "XB_H": 50.0, "SO": 2.0},
                         Q=500.0),
        to="r.in",
    )
    return plant


def main() -> None:
    net = aquakin.load_model("asm1")
    aeration = Aeration(kla=600.0)

    cstr = CSTRUnit("r", net, volume=1000.0, input_port_names=["in"],
                    conditions={"T": 293.15}, aeration=aeration)
    ifas = IFASUnit(
        "r", net, volume=1000.0, input_port_names=["in"],
        specific_surface_area=500.0,   # carrier SSA, m^2 / m^3 of media
        fill_fraction=0.4,             # 40% of the tank filled with carriers
        biofilm_thickness=5e-4,        # 0.5 mm biofilm
        n_layers=4,
        conditions={"T": 293.15},
        aeration=aeration,
        # mature attached-biomass inventory seeded onto the carriers
        biofilm_initial=net.concentrations({"XB_H": 3000.0, "XB_A": 150.0}),
    )

    rows = []
    for label, unit in (("CSTR", cstr), ("IFAS", ifas)):
        plant = _single_tank_plant(unit, net)
        result = plant.run_to_steady_state()
        eff = plant.stream(result.solution, "r.out")
        rows.append((label, float(eff.C_named("SS")[-1]),
                     float(eff.C_named("SNH")[-1]), result.converged))

    print(f"{'unit':6}  {'effluent SS':>12}  {'effluent SNH':>13}  converged")
    for label, ss, snh, ok in rows:
        print(f"{label:6}  {ss:12.2f}  {snh:13.2f}  {ok}")
    print("\nThe IFAS tank's attached biofilm removes more soluble COD and "
          "nitrifies\nmore than the suspended-growth CSTR in the same footprint.")


if __name__ == "__main__":
    main()
