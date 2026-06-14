"""Scenario comparison and Monte-Carlo uncertainty propagation.

Two engineering deliverables built on the same ``fn(x) -> outputs`` contract as
``aquakin.dgsm``: compare a model under several named operating points, and
propagate uncertain inputs through it to get output percentiles. Here the model
is a 1-day aerobic nitrification batch (asm3_2step); the inputs are the AOB
maximum growth rate and its ammonia affinity, and the outputs are the effluent
ammonia and nitrate.
"""
import jax.numpy as jnp

import aquakin

net = aquakin.load_network("asm3_2step")
reactor = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(T=293.15))
C0 = net.concentrations({"SO2": 300.0, "SNH4": 30.0, "XAOB": 80.0,
                         "XNOB": 80.0, "SALK": 0.05})
i_mu, i_k = net.param_index["muAOB"], net.param_index["KAOBNH4"]


def fn(x):
    """Map (muAOB, KAOBNH4) -> (effluent NH4, effluent NO3)."""
    p = net.default_parameters().at[i_mu].set(x[0]).at[i_k].set(x[1])
    sol = reactor.solve(C0, params=p, t_span=(0.0, 1.0),
                        t_eval=jnp.linspace(0.0, 1.0, 6))
    return jnp.array([sol.C_named("SNH4")[-1], sol.C_named("SNO3")[-1]])


# --- Scenario comparison: three operating assumptions, side by side ----------
scenarios = aquakin.compare_scenarios(
    fn,
    {
        "nominal": {},
        "fast_AOB": {"muAOB": 1.2},          # a faster-growing community
        "low_affinity": {"KAOBNH4": 0.8},    # a poorer ammonia affinity
    },
    input_names=["muAOB", "KAOBNH4"],
    baseline=[0.9, 0.14],
    output_names=["eff_NH4", "eff_NO3"],
)
print("Scenario comparison (g N/m3):")
print(scenarios.table())
print(f"  lowest effluent ammonia: {scenarios.best('eff_NH4')}\n")

# --- Monte-Carlo: uncertain kinetics -> effluent distribution ----------------
mc = aquakin.monte_carlo(
    fn,
    {
        "muAOB": {"dist": "normal", "mean": 0.9, "std": 0.15},
        "KAOBNH4": {"dist": "lognormal", "mean": 0.14, "std": 0.05},
    },
    output_names=["eff_NH4", "eff_NO3"],
    n_samples=256, sampler="sobol", seed=0,
)
print(mc.summary())
lo, med, hi = mc.percentiles((2.5, 50.0, 97.5))
print(f"\n95% effluent-ammonia interval: "
      f"{lo[mc.output_names.index('eff_NH4')]:.2f} - "
      f"{hi[mc.output_names.index('eff_NH4')]:.2f} g N/m3 "
      f"(median {med[mc.output_names.index('eff_NH4')]:.2f})")
