"""Microbenchmark: un-jitted vs jit-wrapped reactor.solve.

Diffrax's diffeqsolve is internally jit-able but is not auto-jitted, so Python
overhead is paid per RK stage unless the caller wraps the integration in
`jax.jit`. This script measures the difference on the ozone/bromate model.

Run:
    .venv/bin/python scripts/benchmark_solve.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import aquakin


def setup():
    model = aquakin.load_model("ozone_bromate")
    conditions = aquakin.SpatialConditions.uniform(
        n_locations=1, pH=7.5, T=293.15, OH_scavenging=5.0e4
    )
    atol = jnp.full((model.n_species,), 1e-12)
    atol = atol.at[model.species_index["OH"]].set(1e-20)
    reactor = aquakin.BatchReactor(model, conditions, atol=atol)
    C0 = model.default_concentrations()
    C0 = C0.at[model.species_index["O3"]].set(1.0e-4)
    C0 = C0.at[model.species_index["Br-"]].set(1.0e-5)
    params = model.default_parameters()
    t_eval = jnp.linspace(0.0, 600.0, 121)
    return reactor, C0, params, t_eval


def time_calls(fn, n_warmup: int = 1, n_timed: int = 5) -> float:
    """Return median wall time of fn() over n_timed runs after warm-up."""
    for _ in range(n_warmup):
        result = fn()
        jax.block_until_ready(result)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    reactor, C0, params, t_eval = setup()
    t_span = (0.0, 600.0)

    def un_jitted_call():
        sol = reactor.solve(C0, params, t_span=t_span, t_eval=t_eval)
        return sol.C

    @jax.jit
    def jitted_solve(C0_, params_):
        sol = reactor.solve(C0_, params_, t_span=t_span, t_eval=t_eval)
        return sol.C

    def jitted_call():
        return jitted_solve(C0, params)

    t_un = time_calls(un_jitted_call, n_warmup=1, n_timed=5)
    t_jt = time_calls(jitted_call, n_warmup=2, n_timed=5)

    print(f"Un-jitted solve:  {t_un*1000:8.2f} ms / call (median of 5)")
    print(f"Jit-wrapped solve:{t_jt*1000:8.2f} ms / call (median of 5)")
    print(f"Speed-up:         {t_un / t_jt:6.2f}x")


if __name__ == "__main__":
    main()
