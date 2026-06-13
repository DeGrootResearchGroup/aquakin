"""BSM2 closed-loop reject control (storage-tank level controller).

The reject-water storage tank holds the recycled reject and returns it to the
plant front. Run **open-loop** (``reject=RejectStorage()``, zero release) it simply
fills to its limit and bypasses the whole reject stream -- the tank does no
buffering. Run **closed-loop** (``reject=RejectStorage(control=True)``) a proportional level
controller manipulates the release: the tank settles at a mid-level and releases
the reject smoothly through the controlled pump, with no overflow bypass. It is
then a functioning equalisation tank -- the reject returns as a controlled,
buffered flow instead of an uncontrolled spill.

This script runs both to steady state and contrasts the tank operating point.
The net reject returned to the plant is the same in both (so the
activated-sludge steady state is unchanged), but the *path* differs: bypass
overflow vs controlled release.
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


def _run(asm1, adm1, params, **kw):
    plant = build_bsm2(asm1, adm1, **kw)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    y0 = bsm2_warm_start(plant)
    sol = plant.solve(t_span=(0.0, 80.0), t_eval=jnp.array([0.0, 80.0]),
                      params=params, y0=jnp.asarray(y0),
                      rtol=1e-5, atol=1e-3, max_steps=600_000)
    return plant, sol


def main() -> None:
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)

    print("BSM2 reject storage tank: open-loop bypass vs closed-loop control")
    print()
    header = (f"{'mode':<24}{'tank level (m³)':>16}{'released':>11}"
              f"{'bypassed':>11}{'tank5 XB_H':>12}")
    print(header)
    print("-" * len(header))
    for label, kw in (("open-loop (release=0)", dict(reject=RejectStorage())),
                      ("closed-loop control", dict(reject=RejectStorage(control=True)))):
        plant, sol = _run(asm1, adm1, params, **kw)
        V = float(sol.unit_state("reject_storage")[-1, -1])
        out = float(plant.stream(sol, "reject_storage.out", params).Q[-1])
        byp = float(plant.stream(sol, "reject_storage.bypass", params).Q[-1])
        xbh = float(sol.C_named("tank5", "XB_H")[-1])
        print(f"{label:<24}{V:>16.1f}{out:>11.1f}{byp:>11.1f}{xbh:>12.1f}")

    print()
    print(f"Open-loop: the tank sits full ({0.9 * BSM2_STORAGE_VOLUME:.0f} m³) "
          f"and bypasses all reject.")
    print("Closed-loop: the controller holds a mid-level and releases the reject "
          "through the pump (zero bypass) -- a functioning equalisation tank, "
          "same net reject and the same activated-sludge steady state.")


if __name__ == "__main__":
    main()
