"""Calibrate sewer nitrate-dosing kinetics from a synthetic batch.

Reproduces, on shipped library pieces, the shape of a sewer nitrate-dosing
calibration: an anoxic batch is dosed with nitrate, which drives the biological
re-oxidation of dissolved sulfide (and the intermediate elemental sulfur) back
to sulfate. Two rate constants govern that pathway --
``k_12_no`` (nitrate-driven sulfide oxidation) and ``k_s0_anox_f`` (elemental-S
oxidation) -- and they are what we recover here.

Workflow:
1. Pick "true" values for the two rates and simulate the batch.
2. Add measurement noise to the sulfide / sulfate / nitrate series.
3. Recover the rates with ``aquakin.calibrate`` (MAP, positive-log transforms),
   using the Gauss-Newton optimiser with a forward-mode Jacobian -- robust on
   the stiff WATS kinetics, whose reverse-mode adjoint needs a step cap -- and a
   Gauss-Newton Laplace posterior for parameter uncertainty.

The model carries a state-derived (charge-balance) pH, so no pH condition is
supplied. ``DifferentiationConfig(mode="forward")`` asks ``calibrate`` to form
the Jacobian in forward mode and to build the forward-capable solver adjoint
internally -- so the script needs no ``diffrax`` import. (A ``dtmax`` cap is still set on the reactor:
this WATS variant is stiff enough that the *forward solve itself* needs a bounded
step to stay finite, independent of how the gradient is formed.)
"""

import jax.numpy as jnp
import numpy as np

import aquakin


# Anoxic nitrate-dosing batch initial state (mg/L as the model's COD/N/S units).
BATCH_IC = {
    "X_BH": 30.0,    # heterotrophic biomass
    "S_B": 40.0,     # fermentable substrate
    "S_VFA": 20.0,   # volatile fatty acids
    "S_NO": 30.0,    # dosed nitrate
    "sumS": 10.0,    # dissolved sulfide present at dosing
    "S_SO4": 4.0,    # sulfate
    "S_NH": 25.0,    # ammonia background
    "S_CO2": 60.0,   # inorganic carbon (sets the pH via the charge balance)
}
OBSERVED = ["sumS", "S_SO4", "S_NO"]          # the measured series
FREE = ["k_12_no", "k_s0_anox_f"]             # the rates to recover
TRUE = {"k_12_no": 12.0, "k_s0_anox_f": 1.5}  # ground truth (defaults: 7.95, 2.64)
T_END = 5.0 / 24.0                            # 5-hour batch (days)


def main() -> None:
    rng = np.random.default_rng(0)
    model = aquakin.load_model("wats_sewer_extended")
    conditions = model.default_conditions()
    reactor = aquakin.BatchReactor(
        model, conditions,
        integrator=aquakin.IntegratorConfig(dtmax=1e-3))

    C0 = model.concentrations(BATCH_IC)        # YAML defaults + the dosed batch
    t_obs = jnp.linspace(0.0, T_END, 13)

    # 1-2. Truth simulation + noisy observations.
    true_params = model.parameter_values(TRUE)  # defaults with the two rates set
    clean = reactor.solve(C0, params=true_params, t_span=(0.0, T_END), t_eval=t_obs)
    obs_idx = [model.species_index[s] for s in OBSERVED]
    clean_obs = np.asarray(clean.C[:, obs_idx])
    sigma = 0.05 * np.maximum(clean_obs.max(axis=0), 1.0)   # 5% per-series noise
    noisy = clean_obs + sigma * rng.standard_normal(clean_obs.shape)

    print("Recovering nitrate-driven sulfur-oxidation rates from a noisy batch ...")
    calib = aquakin.calibrate(
        reactor, C0,
        observations=noisy, t_obs=t_obs,      # calibrate coerces NumPy inputs
        free_params=FREE, observed_species=OBSERVED,
        transforms={name: "positive_log" for name in FREE},
        loss="nll", sigma=sigma,
        optimizer=aquakin.OptimizerConfig(method="gauss_newton"),
        diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve"),
        laplace=aquakin.LaplaceConfig(method="gauss_newton"),
    )

    print()
    print(f"  {'rate':<16} {'truth':>10} {'fit':>10} {'std':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
    for name in FREE:
        print(f"  {name:<16} {TRUE[name]:>10.3f} "
              f"{calib.params_named[name]:>10.3f} "
              f"{calib.params_named_std[name]:>10.3f}")
    print()
    print(f"Final loss (Gaussian NLL): {calib.loss:.3f}   converged: {calib.converged}")


if __name__ == "__main__":
    main()
