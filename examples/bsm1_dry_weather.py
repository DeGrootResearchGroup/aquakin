"""BSM1 dry-weather demo.

Builds the canonical BSM1 reference plant (5 ASM1 reactors + secondary
clarifier, internal + RAS recycles), drives it with the synthesised
dry-weather influent, and prints the headline plant state plus simple
metrics. End-to-end demonstration of the ``aquakin.plant`` sub-package.

Note: the shipped influent CSV is *synthesised* to match the
BSM1 statistical profile, not the canonical IWA file. For published
EQI / OCI comparisons, replace ``BSM1_dry.csv`` under
``aquakin/plant/bsm/data/`` with the official BSM1 file.
"""

import jax.numpy as jnp

import aquakin
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent
from aquakin.plant.streams import Stream
from aquakin.plant.metrics import effluent_averages


def main() -> None:
    network = aquakin.load_network("asm1")

    plant = build_bsm1(network=network)
    inf = load_bsm1_influent("dry", network)
    plant.add_influent("feed", inf, to="inlet_mix.fresh")

    print("BSM1 plant:")
    print(f"  units: {list(plant.units.keys())}")
    print(f"  state size: {sum(u.state_size for u in plant.units.values())}")
    print(f"  influent: dry weather, {inf.t.shape[0]} samples over "
          f"{float(inf.t[-1]):.1f} days, avg Q = {float(inf.Q.mean()):.0f} m³/d")
    print()

    print("Solving for 15 days (open-loop kLa, ideal clarifier) ...")
    sol = plant.solve(
        t_span=(0.0, 15.0),
        t_eval=jnp.linspace(0.0, 15.0, 16),
        rtol=1e-4, atol=1e-3,
    )

    print()
    print("Tank-5 effluent state at simulation end:")
    print(f"  SS    = {float(sol.C_named('tank5', 'SS')[-1]):7.2f}  g_COD/m³")
    print(f"  SNH   = {float(sol.C_named('tank5', 'SNH')[-1]):7.2f}  g_N/m³")
    print(f"  SNO   = {float(sol.C_named('tank5', 'SNO')[-1]):7.2f}  g_N/m³")
    print(f"  SO    = {float(sol.C_named('tank5', 'SO')[-1]):7.3f}  g_O2/m³")
    print(f"  XB_H  = {float(sol.C_named('tank5', 'XB_H')[-1]):7.1f}  g_COD/m³")
    print(f"  XB_A  = {float(sol.C_named('tank5', 'XB_A')[-1]):7.2f}  g_COD/m³")

    print()
    print("Per-tank XB_H (heterotrophic biomass) at simulation end:")
    for name in ("tank1", "tank2", "tank3", "tank4", "tank5"):
        print(f"  {name}: {float(sol.C_named(name, 'XB_H')[-1]):7.1f}  g_COD/m³")

    # Effluent metrics: reconstruct overflow stream at every save time.
    clar = plant.units["clarifier"]
    n_t = sol.state.shape[0]
    C_eff = []
    Q_eff = []
    for i in range(n_t):
        tank5_start, tank5_size = plant._state_layout["tank5"]
        tank5_C = sol.state[i, tank5_start:tank5_start + tank5_size]
        Q_in_t = float(inf.at(jnp.asarray(sol.t[i])).Q)
        Q_clar = 2.0 * Q_in_t  # = 2/5 * 5*Q_in
        out = clar.compute_outputs(
            jnp.asarray(sol.t[i]), jnp.zeros((0,)),
            {"inlet": Stream(Q=jnp.asarray(Q_clar), C=tank5_C, network=network)},
            plant.default_parameters(),
        )
        C_eff.append(out["overflow"].C)
        Q_eff.append(out["overflow"].Q)
    C_eff = jnp.stack(C_eff)
    Q_eff = jnp.stack(Q_eff)

    avgs = effluent_averages(sol.t, C_eff, Q_eff, network)
    print()
    print("Time-averaged effluent quality:")
    for key, val in avgs.items():
        unit = "g_N/m³" if key in ("SNH", "SNO", "TKN") else "g/m³"
        print(f"  {key:5s} = {val:7.2f}  {unit}")


if __name__ == "__main__":
    main()
