"""Hit a target sludge age (SRT) by solving for the wastage flow Qw.

SRT is not a knob you set -- it is an emergent property of how fast you waste
sludge. So to operate an activated-sludge plant at a target SRT, you choose the
wastage flow ``Qw`` that produces it. This example closes that loop with the
``aquakin.plant.design`` layer:

1. **Forward design** with :func:`size_activated_sludge`: from an SRT/HRT target
   and the design flow, get the aeration volume and a first-guess ``Qw``.
2. **Achieved metrics** with ``plant.sludge_age`` (= :func:`sludge_metrics`):
   build the BSM1 plant at a given ``Qw``, solve to steady state, and read back
   the SRT/HRT/F:M the model actually produced.
3. **Solve for Qw**: a secant iteration on ``achieved_SRT(Qw) − target`` lands
   the wastage flow that hits the target sludge age -- the by-hand iteration the
   design layer is meant to replace.
"""

import jax.numpy as jnp

import aquakin
from aquakin import size_activated_sludge
from aquakin.plant.bsm import build_bsm1
from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
from aquakin.plant.influent import InfluentSeries

TARGET_SRT = 10.0  # days


def _influent(network):
    """The documented BSM1 average influent (Table 5.1 composition)."""
    C0 = network.concentrations({
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
        "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0,
    })
    return InfluentSeries(
        t=jnp.array([0.0, 300.0]), Q=jnp.full((2,), BSM1_Q_AVG),
        C=jnp.tile(C0, (2, 1)), network=network,
    )


def achieved_srt(network, influent, Qw):
    """Build BSM1 at wastage flow ``Qw``, solve to steady state, return SRT."""
    plant = build_bsm1(network=network, wastage_flow=float(Qw))
    plant.add_influent("feed", influent, to="inlet_mix.fresh")
    sol = plant.solve(
        t_span=(0.0, 80.0), t_eval=jnp.linspace(70.0, 80.0, 6),
        rtol=1e-4, atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=300_000),
    )
    return plant.sludge_age(sol).SRT


def main() -> None:
    network = aquakin.load_network("asm1")
    influent = _influent(network)

    # ----- 1. Forward design gives the aeration volume + a first-guess Qw. -----
    sizing = size_activated_sludge(
        SRT=TARGET_SRT, HRT_h=7.8, Q=BSM1_Q_AVG, n_tanks=5,
        internal_recycle_ratio=3.0, ras_ratio=1.0)
    print(sizing.summary())
    print()

    # The forward Qw assumes mixed-liquor wasting; BSM1 wastes from the
    # thickened underflow, so the realised SRT differs -- which is exactly why
    # we close the loop on the model.
    print(f"Target SRT = {TARGET_SRT:.1f} d. Solving for the wastage flow Qw "
          f"that the BSM1 model needs to hit it ...\n")

    # ----- 2-3. Secant iteration on achieved_SRT(Qw) - target. -----
    # SRT decreases monotonically with Qw (waste more -> younger sludge), so the
    # secant converges quickly. Bracket-ish starting pair around the design Qw.
    Qw0, Qw1 = 250.0, 500.0
    f0 = achieved_srt(network, influent, Qw0) - TARGET_SRT
    f1 = achieved_srt(network, influent, Qw1) - TARGET_SRT
    print(f"  Qw = {Qw0:6.1f} m3/d -> SRT = {f0 + TARGET_SRT:6.2f} d")
    print(f"  Qw = {Qw1:6.1f} m3/d -> SRT = {f1 + TARGET_SRT:6.2f} d")

    for _ in range(6):
        if abs(f1 - f0) < 1e-9:
            break
        Qw2 = Qw1 - f1 * (Qw1 - Qw0) / (f1 - f0)   # secant step
        Qw2 = float(jnp.clip(Qw2, 10.0, 2000.0))
        f2 = achieved_srt(network, influent, Qw2) - TARGET_SRT
        print(f"  Qw = {Qw2:6.1f} m3/d -> SRT = {f2 + TARGET_SRT:6.2f} d")
        Qw0, f0, Qw1, f1 = Qw1, f1, Qw2, f2
        if abs(f2) < 0.05:  # within 0.05 d of the target
            break

    print()
    print(f"Converged: Qw = {Qw1:.1f} m3/d gives an SRT of "
          f"{f1 + TARGET_SRT:.2f} d (target {TARGET_SRT:.1f} d).")
    print(f"(The forward-design first guess was Qw = {sizing.wastage_flow:.1f} "
          f"m3/d for mixed-liquor wasting.)")


if __name__ == "__main__":
    main()
