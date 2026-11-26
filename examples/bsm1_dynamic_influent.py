"""BSM1 with a dynamic (time-varying) influent.

Builds the canonical BSM1 plant, settles it to a steady state under the
average dry-weather load, then plays a 14-day *dynamic* influent through it and
tracks how the tank-5 (effluent) state responds. Two profiles are compared --
dry weather and a wet-weather rain event -- warm-started from the same steady
state so the difference is purely the influent forcing.

The recycle pumps hold the plant throughput bounded under the influent swing
(the fixed-flow-pump model), so the monolithic solve stays efficient rather
than chasing a near-singular recycle gain. See ``examples/bsm1_dry_weather.py``
for the open-loop steady-state version.

Note: the shipped influent CSVs are *synthesised* to match the BSM1 statistical
profile, not the canonical IWA files. For published EQI / OCI comparisons,
replace the files under ``aquakin/plant/bsm/data/``.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent
from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG

# The BSM1 average dry-weather load (Copp 2002), held constant for the settle.
BSM1_AVERAGE_FEED = {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
                     "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}


def _settle(model):
    """Run the plant to a steady state under the constant average load (the
    14-day dynamic record ends long before a 100-day settle would) and return
    the final plant state."""
    plant = build_bsm1(model=model)
    plant.add_influent("feed", model.influent(BSM1_AVERAGE_FEED, Q=BSM1_Q_AVG),
                       to="inlet_mix.fresh")
    sol = plant.solve(t_span=(0.0, 100.0), t_eval=jnp.array([0.0, 100.0]),
                      rtol=1e-5, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=300_000))
    return sol.state[-1]


def _run_dynamic(model, profile, y0, t_end=14.0):
    """Play a dynamic influent through a fresh plant, warm-started from y0."""
    plant = build_bsm1(model=model)
    plant.add_influent("feed", load_bsm1_influent(profile, model),
                       to="inlet_mix.fresh")
    sol = plant.solve(
        t_span=(0.0, t_end),
        t_eval=jnp.linspace(0.0, t_end, 8 * int(t_end) + 1),  # ~3-hourly
        y0=y0, rtol=1e-5, atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=300_000),
    )
    return sol


def main() -> None:
    model = aquakin.load_model("asm1")

    print("Settling BSM1 to a dry-weather steady state ...")
    y_ss = _settle(model)

    print("Playing 14-day dynamic influents (dry vs rain), warm-started ...")
    sols = {p: _run_dynamic(model, p, y_ss) for p in ("dry", "rain")}

    # Effluent ammonia is the headline regulated quantity; track it over time.
    print()
    print("Tank-5 effluent response (SNH = ammonia, SNO = nitrate):")
    print(f"  {'profile':<8} {'SNH peak':>10} {'SNH mean':>10} "
          f"{'SNO peak':>10}  [{model.units_of('SNH')}]")
    for profile, sol in sols.items():
        snh = sol.C_named("tank5", "SNH")
        sno = sol.C_named("tank5", "SNO")
        print(f"  {profile:<8} {float(jnp.max(snh)):>10.3f} "
              f"{float(jnp.mean(snh)):>10.3f} {float(jnp.max(sno)):>10.3f}")

    print()
    print("Dry-weather diurnal SNH trace (every ~1.75 d):")
    sol = sols["dry"]
    for i in range(0, sol.t.shape[0], 14):
        print(f"  t = {float(sol.t[i]):5.2f} d   "
              f"SNH = {float(sol.C_named('tank5', 'SNH')[i]):6.3f} "
              f"{model.units_of('SNH')}   "
              f"SO = {float(sol.C_named('tank5', 'SO')[i]):6.3f} "
              f"{model.units_of('SO')}")


if __name__ == "__main__":
    main()
