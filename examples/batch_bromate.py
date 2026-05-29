"""Batch reactor demo: ozone/bromate formation with explicit OH chemistry."""

import jax.numpy as jnp

import aquakin


def main() -> None:
    network = aquakin.load_network("ozone_bromate")
    print(network.summary())

    conditions = aquakin.SpatialConditions.uniform(
        n_locations=1, pH=7.5, T=293.15, OH_scavenging=5.0e4
    )

    # Per-species absolute tolerance: OH lives at ~1e-12 M, others at ~1e-4 M.
    atol = jnp.full((network.n_species,), 1e-12)
    atol = atol.at[network.species_index["OH"]].set(1e-20)
    reactor = aquakin.BatchReactor(network, conditions, atol=atol)

    C0 = network.default_concentrations()
    C0 = C0.at[network.species_index["O3"]].set(1.0e-4)
    C0 = C0.at[network.species_index["Br-"]].set(1.0e-5)

    t_eval = jnp.linspace(0.0, 600.0, 121)
    sol = reactor.solve(C0, network.default_parameters(), t_span=(0.0, 600.0), t_eval=t_eval)

    print()
    print(
        f"{'t [s]':>8}  {'O3 [M]':>12}  {'Br- [M]':>12}  "
        f"{'HOBr [M]':>12}  {'OH [M]':>12}  {'BrO3- [M]':>12}"
    )
    for i in [0, 10, 30, 60, 120]:
        print(
            f"{float(sol.t[i]):8.1f}  "
            f"{float(sol.C_named('O3')[i]):12.4e}  "
            f"{float(sol.C_named('Br-')[i]):12.4e}  "
            f"{float(sol.C_named('HOBr')[i]):12.4e}  "
            f"{float(sol.C_named('OH')[i]):12.4e}  "
            f"{float(sol.C_named('BrO3-')[i]):12.4e}"
        )


if __name__ == "__main__":
    main()
