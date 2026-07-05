"""BSM2 influent hydraulic delay.

A length of sewer or channel ahead of the works delays and smooths the influent
flow and load. ``build_bsm2(hydraulic_delay=HydraulicDelay())`` puts a
:class:`HydraulicDelayUnit` -- a first-order lag on flow and load with time
constant ``tau`` -- on the raw influent, so a flow pulse reaches the plant
delayed and rounded.

This script drives the plant with a short influent flow pulse and prints the raw
influent flow against the delayed flow the plant actually sees, showing the lag.
The delay is a pass-through at steady state, so it leaves the steady operating
point unchanged.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm import (
    build_bsm2,
    HydraulicDelay,
    bsm2_asm1_model,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.bsm.bsm2 import BSM2_Q_REF
from aquakin.plant.influent import InfluentSeries


def _pulse_influent(asm1):
    """A flow pulse (2x for ~0.1 d) on the constant influent composition."""
    C = bsm2_constant_influent(asm1).C[0]
    t = jnp.array([0.0, 0.50, 0.5001, 0.60, 0.6001, 2.0])
    Q = jnp.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0]) * BSM2_Q_REF
    return InfluentSeries(t=t, Q=Q, C=jnp.tile(C, (t.shape[0], 1)), model=asm1)


def main() -> None:
    asm1 = bsm2_asm1_model()
    adm1 = aquakin.load_model("adm1")
    params = bsm2_parameters(asm1, adm1)

    # A long-ish lag so the delay is visible against the 0.1-day pulse.
    plant = build_bsm2(asm1, adm1, hydraulic_delay=HydraulicDelay(tau=0.05))
    plant.add_influent("feed", _pulse_influent(asm1))

    y0 = bsm2_warm_start(plant)

    t_eval = jnp.linspace(0.4, 1.0, 13)
    print("Driving BSM2 with a 2x influent flow pulse (days 0.5-0.6), "
          "delay tau=0.05 d ...")
    sol = plant.solve(t_span=(0.0, 2.0), t_eval=t_eval, params=params,
                      y0=jnp.asarray(y0), rtol=1e-6, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=600_000))

    raw = plant.influents["feed"]
    delayed = plant.stream(sol, "influent_delay.out", params)
    print()
    print(f"{'day':>6}{'raw Q (m³/d)':>16}{'delayed Q (m³/d)':>20}")
    print("-" * 42)
    for i, day in enumerate(t_eval):
        print(f"{float(day):>6.2f}{float(raw.at(day).Q):>16.0f}"
              f"{float(delayed.Q[i]):>20.0f}")

    print()
    print("The raw influent steps sharply; the delayed flow the plant sees rises "
          "and falls smoothly behind it -- the hydraulic lag of the upstream "
          "channel.")


if __name__ == "__main__":
    main()
