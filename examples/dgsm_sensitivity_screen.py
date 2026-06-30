"""Derivative-based global sensitivity (DGSM) screen.

Ranks how strongly each of several rate constants controls the final bromate
yield of the ozone/bromate network, using ``aquakin.dgsm`` -- an AD analogue of
the Sobol total-order index. It draws scrambled-Sobol quasi-random samples
across each parameter's uncertainty range, accumulates the mean-squared
gradient at each, and reports an upper bound on the Sobol total index per input.

The same screen is run in both AD modes -- reverse (one adjoint pass per output)
and forward (one tangent per input through a single solve) -- to show they agree
to machine precision; the choice is purely performance. ``dgsm`` builds the
reactor inside ``fn``, so it cannot set the differentiation mode for you:
forward mode needs the reactor built with
``aquakin.DifferentiationConfig(mode="forward", method="through_solve")``.
"""

import jax.numpy as jnp

import aquakin


def main() -> None:
    network = aquakin.load_network("ozone_bromate")
    conditions = aquakin.OperatingConditions(pH=7.5, T=293.15,
                                             OH_scavenging=5.0e4)

    # Per-species atol: OH lives in the 1e-12 M band.
    atol = network.atol({"OH": 1e-20}, default=1e-12)
    # forward mode so the *forward*-mode screen works (finite at any step).
    reactor = aquakin.BatchReactor(
        network, conditions, atol=atol,
        diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve"))

    C0 = network.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})
    base = network.default_parameters()

    # Screen these rate constants; each varies +/- a factor of 3 about its default.
    screened = ["O3_Br_direct.k1", "O3_OBr_oxidation.k2",
                "O3_BrO2_oxidation.k3", "O3_decay_OH.k_dec"]
    idx = jnp.array([network.param_index[n] for n in screened])
    defaults = jnp.array([float(base[network.param_index[n]]) for n in screened])
    ranges = [(float(d) / 3.0, float(d) * 3.0) for d in defaults]

    t_eval = jnp.linspace(0.0, 600.0, 31)

    def bromate_yield(z):
        """Final bromate, given the screened rate constants z (the rest fixed)."""
        params = base.at[idx].set(z)
        sol = reactor.solve(C0, params=params, t_span=(0.0, 600.0), t_eval=t_eval)
        return sol.C_named("BrO3-")[-1]

    print("DGSM screen of final bromate yield vs four rate constants")
    print(f"(scrambled-Sobol QMC, ranges = default x[1/3, 3])\n")

    for ad_mode in ("reverse", "forward"):
        res = aquakin.dgsm(bromate_yield, ranges, input_names=screened,
                           n_samples=64, seed=0,
                           diff=aquakin.DifferentiationConfig(mode=ad_mode))
        print(f"  ad_mode = {ad_mode}:")
        for name, bound in res.ranked():
            print(f"    {name:<24s}  Sobol-total bound = {bound:.3e}")
        print()


if __name__ == "__main__":
    main()
