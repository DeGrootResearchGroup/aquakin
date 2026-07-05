"""Located events: scheduled re-dosing + a regulatory cut-off in batch ozonation.

Ozone decays as it oxidises bromide, so a contactor is often **re-dosed** to
hold a residual -- a discontinuous on/off operation. Bromate (BrO3-) is a
regulated disinfection by-product, so the run should **stop** once it reaches
the limit. Both are exactly the located discontinuities ``events=`` handles:

- **Time events** (``at_times=``) inject a fresh ozone dose at scheduled times
  by resetting the ``[O3]`` state -- the segment boundaries are known, so this
  path is AD-safe.
- A **state event** (``cond_fn=``) fires when ``[BrO3-]`` crosses the regulatory
  limit and **terminates** the solve, located exactly by a root find rather than
  smoothed or grid-snapped.

The same facility drives SBR fill/react/settle/decant phase switches and on/off
/ relay control in the plant solve (``plant.solve(events=...)``).
"""

import jax.numpy as jnp

import aquakin

# Bromate cut-off (M_BrO3- = 128 g/mol). Set above the 10 ug/L drinking-water
# limit so the scheduled re-doses fire first and the demo shows both event kinds.
BROMATE_LIMIT = 30e-6 / 128.0       # ~2.3e-7 mol/L (30 ug/L)
O3_DOSE = 5e-5                       # mol/L added per re-dose


def main() -> None:
    net = aquakin.load_model("ozone_bromate")
    cond = aquakin.SpatialConditions.uniform(pH=7.5, T=293.15, OH_scavenging=1.0e5)
    reactor = aquakin.BatchReactor(net, cond)

    C0 = net.concentrations({"O3": 1.0e-4, "Br-": 1.0e-6})
    i_o3 = net.species_index["O3"]
    i_bro3 = net.species_index["BrO3-"]
    t_eval = jnp.linspace(0.0, 600.0, 61)   # seconds

    # Re-dose ozone every 200 s; stop if bromate reaches the limit.
    redose = aquakin.Event(
        at_times=[200.0, 400.0],
        apply=lambda t, C, p: C.at[i_o3].add(O3_DOSE),
        name="re-dose O3",
    )
    cutoff = aquakin.Event(
        cond_fn=lambda t, C, p: C[i_bro3] - BROMATE_LIMIT,
        direction=1, terminal=True, name="bromate limit",
    )

    sol = reactor.solve(C0, (0.0, 600.0), t_eval, events=[redose, cutoff])

    print("Batch ozonation with scheduled re-dosing + bromate cut-off")
    print("=" * 58)
    print("Fired events (time s, name):")
    for t, name in sol.events_log:
        print(f"  t = {t:7.2f} s   {name}")

    o3 = sol.C_named("O3")
    bro3 = sol.C_named("BrO3-")
    print("\n  t (s)     [O3] (uM)   [BrO3-] (ug/L)")
    for k in range(0, len(sol.t), 5):
        print(f"  {float(sol.t[k]):6.0f}   {float(o3[k])*1e6:9.2f}   "
              f"{float(bro3[k])*128.0*1e6:9.2f}")

    fired = [n for _, n in sol.events_log]
    if "bromate limit" in fired:
        t_stop = [t for t, n in sol.events_log if n == "bromate limit"][0]
        print(f"\nSolve terminated at t = {t_stop:.1f} s: bromate reached the "
              f"{BROMATE_LIMIT*128.0*1e6:.0f} ug/L limit.")
    else:
        print(f"\nBromate stayed below the limit "
              f"({float(bro3[-1])*128.0*1e6:.2f} ug/L at 600 s).")


if __name__ == "__main__":
    main()
