"""Secondary-clarifier design checks for the A²O plant (``build_a2o``).

Solves ``build_a2o`` to a steady state for one design (clarifier area, RAS
ratio, wastage flow), then reports the standard secondary-clarifier checks next
to the solved plant: the surface overflow rate, the solids loading rate, the
limiting flux of a state-point analysis on the settler's own Takács settling
curve, the sludge age, the MLSS, the settler's layer TSS profile (the sludge
blanket) and the effluent quality. It changes no shipped default: the area is
passed to the builder by patching the module constant for this process only.

Usage::

    python scripts/probe_a2o_clarifier.py                     # the shipped design
    python scripts/probe_a2o_clarifier.py --area 2500 --wastage 200 --ras 1.0

Each run prints one JSON line (append them to a file to compare designs).
"""

from __future__ import annotations

import argparse
import json
import time

import jax.numpy as jnp
import numpy as np

import aquakin
from aquakin.plant import a2o
from aquakin.plant.takacs import TakacsClarifier

T_END = 200.0  # days: about ten sludge ages of every design probed


def state_point(clar: TakacsClarifier, X_f: float, q_over: float, u: float) -> dict:
    """State-point analysis on the settler's Takács curve.

    ``X_f`` is the feed TSS (g/m³), ``q_over`` the overflow rate and ``u`` the
    underflow velocity (m/d). Returns the batch flux at the feed, the limiting
    flux (the local minimum of the total flux ``v_s(X)·X + u·X`` on the
    hindered branch; none when ``u`` is so large the total flux has no minimum)
    and the concentration it sits at.
    """
    X_min = clar._fns * X_f
    X = np.linspace(X_f, 40_000.0, 400_001)
    excess = X - X_min
    v = np.clip(clar._v0 * (np.exp(-clar._rh * excess) - np.exp(-clar._rp * excess)), 0, clar._vmax)
    G_total = v * X + u * X
    interior = (G_total[1:-1] < G_total[:-2]) & (G_total[1:-1] <= G_total[2:])
    idx = np.flatnonzero(interior)
    G_L = float(G_total[idx[0] + 1]) if idx.size else float("nan")
    X_L = float(X[idx[0] + 1]) if idx.size else float("nan")
    v_f = float(v[0])
    return {
        "batch_flux_at_feed": v_f * X_f,
        "v_s_at_feed": v_f,
        "overflow_flux_at_feed": q_over * X_f,
        "limiting_flux": G_L,
        "X_limiting": X_L,
    }


def run(area: float, ras: float, wastage: float) -> dict:
    """Solve one design to steady state and return its checks."""
    a2o.A2O_CLARIFIER_AREA = area  # this process only
    net = aquakin.load_model("asm2d")
    plant = a2o.build_a2o(net, ras_ratio=ras, wastage_flow=wastage)
    plant.add_influent("feed", a2o.a2o_influent(net))
    t0 = time.perf_counter()
    sol = plant.solve(
        t_span=(0.0, T_END),
        t_eval=jnp.array([T_END - 10.0, T_END]),
        y0=a2o.a2o_warm_start(plant),
        rtol=1e-5,
        atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=8_000_000),
    )
    wall = time.perf_counter() - t0

    clar = plant.units["clarifier"]
    layers = clar._layered(plant.states_by_unit(sol.final_state)["clarifier"])
    tss_layers = [float(clar._tss(layer)) for layer in layers]  # bottom -> top
    eff = plant.stream(sol, "effluent")
    feed = plant.stream(sol, "aer_split.to_clarifier")
    under = plant.stream(sol, "clarifier.underflow")
    named = {s: float(eff.C_named(s)[-1]) for s in ("SNH4", "SNO3", "SPO4", "XTSS")}
    drift = {
        s: float(abs(eff.C_named(s)[-1] - eff.C_named(s)[0])) for s in ("SNH4", "SPO4", "XTSS")
    }
    X_f = float(feed.C_named("XTSS")[-1])
    Q_feed = float(feed.Q[-1])
    Q_u = float(under.Q[-1])
    Q_over = float(eff.Q[-1])
    # Sludge age on TSS: the reactors' solids over the solids leaving in the
    # wastage and the effluent (the clarifier's blanket counted separately).
    states = plant.states_by_unit(sol.final_state)
    xtss = net.species_index["XTSS"]
    reactor_mass = sum(
        plant.units[u].volume * float(states[u][xtss]) for u in plant.activated_sludge_reactors()
    )
    waste = plant.stream(sol, "wastage")
    out_waste = float(waste.Q[-1] * waste.C_named("XTSS")[-1])
    out_eff = Q_over * named["XTSS"]
    blanket_mass = float(clar.solids_mass(states["clarifier"]))
    return {
        "area": area,
        "ras_ratio": ras,
        "wastage": wastage,
        "wall_s": round(wall, 1),
        "MLSS": X_f,
        "SRT_d": reactor_mass / (out_waste + out_eff),
        "SRT_wastage_only_d": reactor_mass / out_waste,
        "effluent_solids_share": out_eff / (out_waste + out_eff),
        "blanket_mass_share": blanket_mass / (reactor_mass + blanket_mass),
        "SOR_m_per_d": Q_over / area,
        "SLR_g_per_m2_d": X_f * Q_feed / area,
        "u_underflow_m_per_d": Q_u / area,
        **state_point(clar, X_f, Q_over / area, Q_u / area),
        "underflow_TSS": float(under.C_named("XTSS")[-1]),
        "layer_TSS_bottom_to_top": [round(x, 1) for x in tss_layers],
        "effluent": named,
        "drift_last_10_d": drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--area", type=float, default=a2o.A2O_CLARIFIER_AREA)
    parser.add_argument("--ras", type=float, default=a2o.A2O_RAS_RATIO)
    parser.add_argument("--wastage", type=float, default=a2o.A2O_WASTAGE_FLOW)
    args = parser.parse_args()
    print(json.dumps(run(args.area, args.ras, args.wastage)))


if __name__ == "__main__":
    main()
