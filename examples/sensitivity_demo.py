"""End-to-end calibration + uncertainty demo.

1. Load the shipped ozone/bromate network.
2. Synthesise noisy observations of bromate at known parameter values.
3. Calibrate selected rate constants via MAP with positive-log transforms.
4. Compute the Laplace covariance and report per-parameter standard
   deviations in physical space (delta-method projection).
5. Print parameter-ranked sensitivities from the same fit.
"""

import jax.numpy as jnp
import numpy as np

import aquakin


def main() -> None:
    rng = np.random.default_rng(0)

    network = aquakin.load_network("ozone_bromate")
    # Per-species atol: OH lives in the 1e-12 M band.
    atol = network.atol({"OH": 1e-20}, default=1e-12)
    conditions = aquakin.OperatingConditions(pH=7.5, T=293.15, OH_scavenging=5.0e4)
    reactor = aquakin.BatchReactor(network, conditions, atol=atol)

    C0 = network.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})

    t_obs = jnp.linspace(30.0, 600.0, 20)
    sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_obs)
    clean = np.asarray(sol.C_named("BrO3-"))
    sigma_value = 0.05 * float(np.max(clean))  # 5% relative noise
    noisy = clean + sigma_value * rng.standard_normal(clean.shape)

    free_params = [
        "O3_Br_direct.k1",
        "O3_BrO2_oxidation.k3",
        "O3_decay_OH.k_dec",
    ]

    print("Calibrating with MAP + Laplace posterior ...")
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=jnp.asarray(noisy),
        t_obs=t_obs,
        free_params=free_params,
        observed_species=["BrO3-"],
        loss="nll",
        sigma=jnp.asarray(sigma_value),
        laplace=True,
    )

    truth = {name: float(network.default_parameters()[network.param_index[name]])
             for name in free_params}

    print()
    print(f"  {'parameter':<28s} {'truth':>10s} {'fit':>10s} {'std':>10s}  rel%")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}  ----")
    for name in free_params:
        true_v = truth[name]
        fit_v = result.params_named[name]
        std_v = result.params_named_std[name]
        rel = 100.0 * std_v / abs(fit_v)
        print(f"  {name:<28s} {true_v:>10.3e} {fit_v:>10.3e} {std_v:>10.3e}  {rel:4.1f}")

    print()
    print(f"Final loss (Gaussian NLL): {result.loss:.4f}")
    print(f"Optimiser iterations:      {result.n_iter}")

    print()
    print("Sensitivity ranking at the MAP (|dBrO3-(t_end)/d_param|):")
    sens = aquakin.sensitivity(
        reactor,
        C0,
        result.params,
        output_fn=lambda s: s.C_named("BrO3-")[-1],
        t_span=(0.0, 600.0),
        t_eval=t_obs,
    )
    for name, mag in sens.ranked_params()[:5]:
        print(f"  {name:<32s}  {mag:.3e}")


if __name__ == "__main__":
    main()
