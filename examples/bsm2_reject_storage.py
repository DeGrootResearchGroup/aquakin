"""BSM2 reject equalisation (storage) tank.

The reject-water recycle (thickener overflow + dewatering reject) is a
high-ammonia load returned to the plant front. ``build_bsm2(reject=RejectStorage())``
routes it through a variable-volume :class:`StorageTank` -- a completely-mixed
CSTR with no reactions whose liquid volume is a state -- that holds the reject
and releases it at a controlled rate ``storage_output_flow``, with a level-gated
overflow bypass protecting the tank's volume limits:

- normal level: release the requested flow, store the rest;
- full and filling (inflow > release): divert the whole inflow (don't overfill);
- full and draining (inflow <= release): release normally;
- empty: stop releasing and just fill.

This script runs the plant to steady state with the **default zero release** and
shows the consequence: the tank fills to its upper limit and bypasses the entire
reject stream, so it is a faithful pass-through -- the activated-sludge steady
state is identical to the no-storage plant. (With a fixed nonzero release the
tank simply fills or drains, since the benchmark's wastage pump ``Qw`` is fixed
and the reject flow is nearly constant; genuine equalisation -- holding a
mid-level and *timing* the reject return -- needs a level-based release
controller, the deferred closed-loop reject-control piece.)
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm import (
    build_bsm2,
    RejectStorage,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.bsm.bsm2 import BSM2_STORAGE_VOLUME


def _steady(plant, asm1, params):
    y0 = bsm2_warm_start(plant)
    return plant.solve(t_span=(0.0, 80.0), t_eval=jnp.array([0.0, 80.0]),
                       params=params, y0=jnp.asarray(y0),
                       rtol=1e-5, atol=1e-3,
                       integrator=aquakin.IntegratorConfig(max_steps=600_000))


def main() -> None:
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)

    # No-storage reference: the reject recycles directly to the front.
    base = build_bsm2(asm1, adm1)
    base.add_influent("feed", bsm2_constant_influent(asm1))
    sol_base = _steady(base, asm1, params)

    # Storage tank on the reject line (default zero release).
    plant = build_bsm2(asm1, adm1, reject=RejectStorage())
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    sol = _steady(plant, asm1, params)

    level = float(sol.unit_state("reject_storage")[-1, -1])
    released = float(plant.stream(sol, "reject_storage.out", params).Q[-1])
    bypassed = float(plant.stream(sol, "reject_storage.bypass", params).Q[-1])

    print("BSM2 reject storage tank at steady state (zero release):")
    print(f"  tank level      = {level:7.1f} m³   "
          f"(of {BSM2_STORAGE_VOLUME:.0f} m³ max; the 0.9 upper limit)")
    print(f"  reject released = {released:7.1f} m³/d")
    print(f"  reject bypassed = {bypassed:7.1f} m³/d   (the whole reject stream)")
    print()
    print(f"{'tank-5 state':<14}{'no storage':>12}{'with storage':>14}")
    print("-" * 40)
    for sp in ("XB_H", "XB_A", "SNH", "SNO"):
        a = float(sol_base.C_named("tank5", sp)[-1])
        b = float(sol.C_named("tank5", sp)[-1])
        print(f"{sp:<14}{a:>12.2f}{b:>14.2f}")
    print()
    print("The tank sits full and bypasses all reject, so the activated-sludge "
          "steady state is unchanged -- a faithful pass-through.")


if __name__ == "__main__":
    main()
