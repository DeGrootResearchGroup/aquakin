"""Batch reactor demo: ozone/bromate formation with explicit OH chemistry."""

import jax.numpy as jnp

import aquakin


def main() -> None:
    model = aquakin.load_model("ozone_bromate")
    print(model.summary())

    conditions = aquakin.OperatingConditions(pH=7.5, T=293.15, OH_scavenging=5.0e4)

    # Per-species absolute tolerance: OH lives at ~1e-12 M, others at ~1e-4 M.
    atol = model.atol({"OH": 1e-20}, default=1e-12)
    reactor = aquakin.BatchReactor(model, conditions, atol=atol)

    C0 = model.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})

    t_eval = jnp.linspace(0.0, 600.0, 121)
    sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_eval)

    species = ["O3", "Br-", "HOBr", "OH", "BrO3-"]
    sample_rows = [0, 10, 30, 60, 120]
    print()

    try:
        # Preferred path: hand the whole trajectory to a DataFrame, then select
        # the species and sample times of interest -- no per-cell float() casts.
        df = sol.to_dataframe()
    except ImportError:
        # pandas is the optional `dataframe` extra; fall back to a manual table.
        print("(install aquakin[dataframe] for tabular export; manual table:)")
        header = "  ".join([f"{'t [s]':>8}"] + [f"{s + ' [M]':>12}" for s in species])
        print(header)
        for i in sample_rows:
            cells = [f"{float(sol.t[i]):8.1f}"] + [
                f"{float(sol.C_named(s)[i]):12.4e}" for s in species
            ]
            print("  ".join(cells))
    else:
        units = df.attrs["units"]["O3"]   # all bromate species share mol/L
        print(f"Concentrations [{units}] at selected times (via to_dataframe):")
        table = df[species].iloc[sample_rows]
        print(table.to_string(float_format=lambda v: f"{v:.4e}"))
        # to_csv embeds the units in the header so the file is self-describing.
        print("\nto_csv() header:  " + sol.to_csv().splitlines()[0])


if __name__ == "__main__":
    main()
