"""Override a plant parameter by name.

A single model has ``model.parameter_values({"O3_Br_direct.k1": 175.0})``,
but a ``Plant`` concatenates the parameter vectors of all its kinetic models
into one flat vector. ``Plant.parameter_values`` gives that flat vector the same
friendly by-name API, keyed by ``"<model>.<param>"`` -- so to bump one ASM1
rate in BSM2 (ASM1 water line + ADM1 digester) you don't hunt the block offset
and index by hand.

This script builds the BSM2 plant, lists a few parameter names, overrides two
rates from different models in one call, and -- for code that differentiates
with respect to a parameter -- shows the companion ``parameter_index``.
"""

import aquakin
from aquakin.plant.bsm import build_bsm2, bsm2_asm1_model


def main() -> None:
    asm1 = bsm2_asm1_model()
    adm1 = aquakin.load_model("adm1")
    plant = build_bsm2(asm1, adm1)

    names = plant.parameter_names()
    print(f"The BSM2 plant has {len(names)} calibratable parameters, addressed "
          "by '<model>.<param>':")
    print("  ASM1 (water line): ", [n for n in names if n.startswith("asm1.")][:5])
    print("  ADM1 (digester):   ", [n for n in names if n.startswith("adm1.")][:5])
    print()

    # Bump the heterotroph max growth rate (ASM1) and a hydrolysis rate (ADM1)
    # in a single call -- no block offsets, no manual indexing.
    defaults = plant.default_parameters()
    params = plant.parameter_values({"asm1.muH": 8.0, "adm1.k_hyd_ch": 12.0})

    for name in ("asm1.muH", "adm1.k_hyd_ch"):
        i = plant.parameter_index(name)
        print(f"  {name:16} {float(defaults[i]):8.3f}  ->  {float(params[i]):8.3f}")
    print()
    print("Pass the returned vector straight to solve():")
    print("    plant.solve(t_span=(0.0, 200.0), params=params, ...)")
    print()
    # parameter_index is the companion for AD: jax.grad needs the position, not a
    # rebuilt vector (parameter_values materialises concrete values).
    print(f"For jax.grad w.r.t. one rate, its flat index is "
          f"plant.parameter_index('adm1.k_m_ac') = "
          f"{plant.parameter_index('adm1.k_m_ac')}.")

    # An unknown name fails loudly with a close-match hint.
    try:
        plant.parameter_values({"asm1.mu_H": 6.0})
    except KeyError as exc:
        print()
        print(f"A typo is caught: {exc}")


if __name__ == "__main__":
    main()
