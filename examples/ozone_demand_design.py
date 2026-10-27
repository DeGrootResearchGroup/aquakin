"""Worked example: identifiability and experiment design for ozone demand.

A water sample removes ozone in two ways: a fast, capacity-limited demand D
(used up within seconds) and a slow reaction with organic matter M (in large
excess over ozone). The unknowns are the two rate constants and the two initial
pools [D]0 and [M]0. The steps follow the "Worked example: designing an
experiment" page of the documentation, which includes the code between the
``# [name]`` / ``# [end name]`` markers below and the output this script
prints:

1. Diagnose a simple design (five ozone samples) with the Fisher information
   and an identifiability report.
2. Compare ways of improving the run.
3. Optimize the sampling times and the ozone dose.
4. Calibrate on synthetic data from the optimized design and check the result
   against the true values, including a profile likelihood.
5. Redesign for a second run, averaging over the calibration's posterior.

Run it with ``python examples/ozone_demand_design.py``. Pass a directory to
also write the figures and the printed output there, as the documentation
build does: ``python examples/ozone_demand_design.py docs/images``.
"""

import contextlib
import io
import sys

# [imports]
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import aquakin

# [end imports]


# [observation]
def make_ozone_and_uv(model, eps):
    """Build the observation function for ozone and UV254.

    aquakin calls an observation function as ``f(C, p)``: the concentration
    vector at one time and the parameter vector. Everything else it needs
    (where ozone and M sit in ``C``, and the specific absorbance ``eps``) is
    bound here, so the returned function has no hidden inputs.
    """
    i_o3, i_m = model.species_index["O3"], model.species_index["M"]

    def ozone_and_uv(C, p):
        """Dissolved ozone (umol/L) and UV254 absorbance (1/cm) at one time."""
        return jnp.stack([C[i_o3], eps * C[i_m]])

    return ozone_and_uv


# [end observation]


def run_example():
    """Run every step, printing the results. Returns the objects the figures
    and the test need."""
    # [setup]
    # The model file sits next to this script.
    model = aquakin.load_model_from_file(Path(__file__).parent / "ozone_demand.yaml")
    reactor = aquakin.BatchReactor(model, model.default_conditions(), rtol=1e-8, atol=1e-8)

    # The true composition of the sample, in umol/L, with a 50 umol/L ozone
    # dose. The synthetic data in step 4 are generated from these values.
    C0 = model.concentrations({"O3": 50.0, "D": 15.0, "M": 1000.0})
    true_params = model.default_parameters()  # k1 = 0.01, k2 = 2e-6 L/umol/s

    # The four unknowns: two rate constants and the two initial pools.
    free_params = ["fast.k1", "slow.k2"]
    free_ic = aquakin.FreeICConfig(["D", "M"], bounds=(1e-2, 1e5))

    # Measurements. Ozone has a standard deviation of 1 umol/L. UV254 is the
    # absorbance at 254 nm per cm of path; it is proportional to the organic
    # matter, A254 = eps * [M], with eps assumed known.
    sigma_o3 = 1.0  # umol/L
    sigma_uv = 0.001  # 1/cm
    eps = 2.0e-4  # 1/cm per umol/L
    ozone_and_uv = make_ozone_and_uv(model, eps)
    sigma_both = jnp.asarray([sigma_o3, sigma_uv])  # one per output of ozone_and_uv

    def percent(x):
        """A relative standard deviation as text; inf means not identifiable."""
        if not np.isfinite(x):
            return "inf"
        return f"{100 * x:.0f}%" if x >= 1.0 else f"{100 * x:.2g}%"

    # [end setup]

    print("== Step 1 ==")
    # [step1]
    # A run is the initial state, the sampling times (s), the measurement
    # standard deviation, and what is measured.
    base = aquakin.Experiment(
        C0, jnp.linspace(60.0, 600.0, 5), sigma=sigma_o3, observed_species=["O3"]
    )
    fim = aquakin.fisher_information(
        reactor, base, free_params, params=true_params, free_ic=free_ic
    )
    rep = aquakin.identifiability(fim)

    print("Eigenvalues relative to the largest:")
    for d in rep.directions:
        print(f"  {d.relative_eigenvalue:9.1e}  identifiable: {d.identifiable}")
    print("Directions flagged as not identifiable (two largest loadings):")
    for d in rep.unidentifiable:
        top = ", ".join(f"{n} {v:+.2f}" for n, v in list(d.loadings.items())[:2])
        print(f"  relative eigenvalue {d.relative_eigenvalue:8.1e}: {top}")
    print("Relative standard deviations:")
    for name, value in rep.relative_std.items():
        print(f"  {name:8s} {percent(value):>8s}")
    # [end step1]

    print("== Step 2 ==")
    # [step2]
    t_base = np.linspace(60.0, 600.0, 5)  # s
    early = np.linspace(2.0, 20.0, 5)  # s
    late = np.linspace(700.0, 1800.0, 5)  # s
    designs = {
        "base": aquakin.Experiment(C0, t_base, sigma=sigma_o3, observed_species=["O3"]),
        "+ late samples": aquakin.Experiment(
            C0, np.r_[t_base, late], sigma=sigma_o3, observed_species=["O3"]
        ),
        "+ early samples": aquakin.Experiment(
            C0, np.r_[early, t_base], sigma=sigma_o3, observed_species=["O3"]
        ),
        "+ UV254": aquakin.Experiment(C0, t_base, sigma=sigma_both, observation_fn=ozone_and_uv),
        "+ early samples + UV254": aquakin.Experiment(
            C0, np.r_[early, t_base], sigma=sigma_both, observation_fn=ozone_and_uv
        ),
    }

    print(f"{'design':24s}" + "".join(f"{n:>10s}" for n in rep.names))
    comparison = {}
    for label, run in designs.items():
        fim = aquakin.fisher_information(
            reactor, run, free_params, params=true_params, free_ic=free_ic
        )
        rel = aquakin.identifiability(fim).relative_std
        comparison[label] = rel
        print(f"{label:24s}" + "".join(f"{percent(rel[n]):>10s}" for n in rep.names))

    # The same designs ranked by the D-criterion (log det F, larger is better;
    # -inf means the information matrix is singular).
    ranked = aquakin.rank_experiments(
        reactor, list(designs.values()), free_params, params=true_params, free_ic=free_ic
    )
    labels = list(designs)
    print("Ranked by the D-criterion:")
    for c in ranked:
        print(f"  {labels[c.index]:24s} D = {c.value:7.2f}")
    # [end step2]

    print("== Step 3 ==")
    # [step3]
    # Start from 8 evenly spaced samples of ozone and UV254.
    start = aquakin.Experiment(
        C0, jnp.linspace(5.0, 900.0, 8), sigma=sigma_both, observation_fn=ozone_and_uv
    )
    opt = aquakin.optimize_experiment(
        reactor,
        start,
        free_params,
        params=true_params,
        free_ic=free_ic,
        criterion="D",
        time_bounds=(1.0, 1800.0),  # s
        min_spacing=5.0,  # s between consecutive samples
        initial_bounds={"O3": (20.0, 150.0)},  # dose range, umol/L
    )
    designed = opt.experiments[0]
    print(f"converged: {opt.success}")
    print(f"D-criterion: {opt.initial_value:.2f} -> {opt.value:.2f}")
    print(f"sampling times (s): {np.round(np.asarray(designed.t_obs), 1).tolist()}")
    print(f"ozone dose: {float(designed.C0[model.species_index['O3']]):.0f} umol/L")

    rel_start = aquakin.identifiability(
        aquakin.fisher_information(reactor, start, free_params, params=true_params, free_ic=free_ic)
    ).relative_std
    rel_designed = aquakin.identifiability(
        aquakin.fisher_information(
            reactor, designed, free_params, params=true_params, free_ic=free_ic
        )
    ).relative_std
    print(f"{'':10s}{'start':>10s}{'designed':>10s}")
    for n in rep.names:
        print(f"{n:10s}{percent(rel_start[n]):>10s}{percent(rel_designed[n]):>10s}")
    # [end step3]

    print("== Step 4 ==")
    # [step4-data]
    # Synthetic measurements: solve with the true values, measure, add noise.
    t_obs = jnp.asarray(designed.t_obs)
    C0_designed = jnp.asarray(designed.C0)
    sol = reactor.solve(
        C0_designed, t_span=(0.0, float(t_obs[-1])), t_eval=t_obs, params=true_params
    )
    clean = np.asarray(jnp.stack([ozone_and_uv(c, true_params) for c in sol.C]))
    rng = np.random.default_rng(0)
    data = clean + rng.standard_normal(clean.shape) * np.asarray(sigma_both)
    # [end step4-data]

    # [step4-fit]
    # Start the fit away from the truth: the rate constants a factor of 2-3
    # off and the pools at round guesses.
    guess = true_params.at[model.param_index["fast.k1"]].set(0.003)
    guess = guess.at[model.param_index["slow.k2"]].set(5.0e-6)
    C0_guess = C0_designed.at[model.species_index["D"]].set(5.0)
    C0_guess = C0_guess.at[model.species_index["M"]].set(500.0)

    result = aquakin.calibrate(
        reactor,
        C0_guess,
        data,
        t_obs,
        free_params,
        initial_params=guess,
        observation_fn=ozone_and_uv,
        loss="nll",
        sigma=sigma_both,
        free_ic=free_ic,
        laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True, eig_keep=1e-10),
        # Keep each rate constant within a factor of 100 of its starting
        # guess, so the optimizer cannot step into regions where the solve fails.
        optimizer=aquakin.OptimizerConfig(
            method="gauss_newton", n_starts=4, param_halfwidth=np.log(100.0)
        ),
    )
    truth = {"fast.k1": 0.01, "slow.k2": 2.0e-6, "D": 15.0, "M": 1000.0}
    print(f"converged: {result.converged}")
    print(f"{'':10s}{'estimate':>12s}{'std dev':>12s}{'relative':>10s}{'true':>12s}")
    for n in free_params:
        est, sd = result.params_named[n], result.params_named_std[n]
        print(f"{n:10s}{est:12.4g}{sd:12.2g}{percent(sd / est):>10s}{truth[n]:12.4g}")
    for s in ["D", "M"]:
        est, sd = result.ic_named[0][s], result.ic_named_std[0][s]
        name = f"[{s}]0"
        print(f"{name:10s}{est:12.4g}{sd:12.2g}{percent(sd / est):>10s}{truth[s]:12.4g}")
    # [end step4-fit]

    # [step4-profile]
    k2_hat = result.params_named["slow.k2"]
    profile = aquakin.profile_likelihood(
        reactor,
        result.C0_fitted[0],
        data,
        t_obs,
        free_params,
        grid=jnp.asarray(k2_hat * np.exp(np.linspace(-0.05, 0.05, 11))),
        profile_param="slow.k2",
        initial_params=result.params,
        observation_fn=ozone_and_uv,
        loss="nll",
        sigma=sigma_both,
        free_ic=free_ic,
        optimizer=aquakin.OptimizerConfig(method="gauss_newton", param_halfwidth=np.log(100.0)),
    )
    lo, hi = profile.ci
    sd = result.params_named_std["slow.k2"]
    print("Profile likelihood:")
    print(f"profile-likelihood 95% interval for k2: {lo:.4g} to {hi:.4g}")
    print(
        f"Laplace 95% interval for k2:            {k2_hat - 1.96 * sd:.4g} to "
        f"{k2_hat + 1.96 * sd:.4g}"
    )
    # [end step4-profile]

    print("== Step 5 ==")
    # [step5]
    draws = result.posterior_draws(model, n_draw=10, seed=1)  # (10, n_params)
    robust = aquakin.optimize_experiment(
        reactor,
        start,  # the same 8 evenly spaced samples as in step 3
        free_params,
        param_draws=draws,
        free_ic=free_ic,
        time_bounds=(1.0, 1800.0),
        min_spacing=5.0,
        initial_bounds={"O3": (20.0, 150.0)},
    )
    second = robust.experiments[0]
    print(f"sampling times (s): {np.round(np.asarray(second.t_obs), 1).tolist()}")
    print(f"ozone dose: {float(second.C0[model.species_index['O3']]):.0f} umol/L")
    # [end step5]
    print("== End ==")

    return {
        "model": model,
        "reactor": reactor,
        "true_params": true_params,
        "eps": eps,
        "base": base,
        "step1": rep,
        "step2": comparison,
        "step3": opt,
        "rel_designed": rel_designed,
        "step4": result,
        "profile": profile,
        "step5": robust,
    }


def make_figures(out, res):
    """The two figures on the documentation page."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    model, reactor, true_p, eps = res["model"], res["reactor"], res["true_params"], res["eps"]
    designed = res["step3"].experiments[0]
    i_o3 = model.species_index["O3"]

    # Figure 1: the true curves with the base and designed sampling times.
    t = jnp.linspace(0.0, 1800.0, 721)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for dose, style in [(50.0, "-"), (float(designed.C0[i_o3]), "--")]:
        C0 = model.concentrations({"O3": dose, "D": 15.0, "M": 1000.0})
        sol = reactor.solve(C0, t_span=(0.0, 1800.0), t_eval=t, params=true_p)
        label = f"{dose:.0f} µmol/L dose" + (" (designed)" if style == "--" else "")
        ax1.plot(t, sol.C_named("O3"), style, color="0.3", lw=1.2, label=label)
        ax2.plot(t, eps * sol.C_named("M"), style, color="0.3", lw=1.2, label=label)
    for ax in (ax1, ax2):
        for x in np.asarray(res["base"].t_obs):
            ax.axvline(x, color="tab:blue", lw=0.8, alpha=0.6, ymax=0.08)
        for x in np.asarray(designed.t_obs):
            ax.axvline(x, color="tab:red", lw=0.8, alpha=0.8, ymin=0.92)
        ax.set_xlabel("time (s)")
    ax1.set_ylabel("dissolved ozone (µmol/L)")
    ax2.set_ylabel("UV254 absorbance (1/cm)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("ozone", fontsize=10)
    ax2.set_title("UV254", fontsize=10)
    fig.savefig(out / "ozone_demand_design.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: profile likelihood of k2 with the Laplace interval.
    prof, result = res["profile"], res["step4"]
    k2, sd = result.params_named["slow.k2"], result.params_named_std["slow.k2"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    ax.plot(
        np.asarray(prof.grid) * 1e6,
        np.asarray(prof.delta_loss),
        "o-",
        color="0.3",
        label="profile",
    )
    ax.axhline(prof.delta, color="tab:red", lw=0.8, ls="--", label="95 % threshold")
    ax.axvspan(
        (k2 - 1.96 * sd) * 1e6,
        (k2 + 1.96 * sd) * 1e6,
        color="tab:blue",
        alpha=0.15,
        label="Laplace 95 % interval",
    )
    ax.axvline(
        float(true_p[model.param_index["slow.k2"]]) * 1e6, color="k", lw=1.0, label="true value"
    )
    ax.set_xlabel(r"$k_2$ ($10^{-6}$ L µmol$^{-1}$ s$^{-1}$)")
    ax.set_ylabel("loss - minimum loss")
    ax.set_ylim(top=1.7 * float(np.nanmax(prof.delta_loss)))
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.savefig(out / "ozone_demand_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


class _Tee(io.TextIOBase):
    """Write to the terminal and keep a copy."""

    def __init__(self, stream):
        self.stream, self.copy = stream, io.StringIO()

    def write(self, s):
        self.stream.write(s)
        return self.copy.write(s)


def main(output_dir=None):
    """Run the example. With ``output_dir``, also write the figures and the
    printed output (``ozone_demand_output.txt``) there."""
    tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        res = run_example()
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ozone_demand_output.txt").write_text(tee.copy.getvalue())
        make_figures(out, res)
        print(f"Figures and output written to {out}")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
