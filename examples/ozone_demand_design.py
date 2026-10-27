"""Worked example: identifiability and experiment design for ozone demand.

A water sample removes ozone in two ways: a fast, capacity-limited demand D
(used up within seconds) and a slow reaction with organic matter M (in large
excess over ozone). The unknowns are the two rate constants and the two initial
pools [D]0 and [M]0. This script follows the steps of the "Worked example:
designing an experiment" page of the documentation:

1. Diagnose a simple design (five ozone samples) with the Fisher information
   and an identifiability report.
2. Compare ways of improving the run.
3. Optimise the sampling times and the ozone dose.
4. Calibrate on synthetic data from the optimised design and check the result
   against the true values, including a profile likelihood.
5. Redesign for a second run, averaging over the calibration's posterior.

Run it with ``python examples/ozone_demand_design.py``; pass a directory to
also write the figures, e.g. ``python examples/ozone_demand_design.py docs/images``.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import aquakin
from aquakin import Experiment, FreeICConfig, LaplaceConfig, OptimizerConfig

HERE = Path(__file__).parent

FREE_PARAMS = ["fast.k1", "slow.k2"]
FREE_IC = FreeICConfig(["D", "M"], bounds=(1e-2, 1e5))
SIGMA_O3 = 1.0  # umol/L
SIGMA_UV = 0.001  # absorbance units (1/cm)
# UV254 is the absorbance at 254 nm per cm of path; by Beer-Lambert it is
# proportional to concentration, A254 = EPS_UV * [M].
EPS_UV = 2.0e-4  # specific UV254 absorbance of M, 1/cm per umol/L (assumed known)


def setup():
    """Model, reactor and the true sample composition."""
    model = aquakin.load_model_from_file(HERE / "ozone_demand.yaml")
    reactor = aquakin.BatchReactor(model, model.default_conditions(), rtol=1e-8, atol=1e-8)
    return model, reactor


def ozone_and_uv(model):
    """Observation function: dissolved ozone and UV254 absorbance."""
    i_o3, i_m = model.species_index["O3"], model.species_index["M"]
    return lambda C, p: jnp.stack([C[i_o3], EPS_UV * C[i_m]])


def run(model, t_obs, *, dose=50.0, uv=False):
    """One batch run: an ozone dose and the sampling times, measuring ozone
    alone or ozone and UV254."""
    C0 = model.default_concentrations().at[model.species_index["O3"]].set(dose)
    t_obs = jnp.asarray(t_obs, dtype=float)
    if uv:
        return Experiment(
            C0,
            t_obs,
            sigma=jnp.asarray([SIGMA_O3, SIGMA_UV]),
            observation_fn=ozone_and_uv(model),
        )
    return Experiment(C0, t_obs, sigma=SIGMA_O3, observed_species=["O3"])


def candidate_designs(model):
    """The base run and the ways of improving it compared in step 2."""
    base = np.linspace(60.0, 600.0, 5)
    early = np.linspace(2.0, 20.0, 5)
    late = np.linspace(700.0, 1800.0, 5)
    return {
        "base: 5 ozone samples, 60-600 s": run(model, base),
        "+ 5 late samples, 700-1800 s": run(model, np.r_[base, late]),
        "+ 5 early samples, 2-20 s": run(model, np.r_[early, base]),
        "+ UV254 at the same times": run(model, base, uv=True),
        "+ early samples and UV254": run(model, np.r_[early, base], uv=True),
    }


def report(reactor, experiment, params=None):
    """Fisher information and identifiability report of one design."""
    fim = aquakin.fisher_information(
        reactor, experiment, FREE_PARAMS, params=params, free_ic=FREE_IC
    )
    return fim, aquakin.identifiability(fim)


def step1_diagnose(model, reactor):
    designs = candidate_designs(model)
    fim, rep = report(reactor, designs["base: 5 ozone samples, 60-600 s"])
    return {
        "names": rep.names,
        "relative_std": rep.relative_std,
        "unidentifiable": rep.unidentifiable,
    }


def step2_compare(model, reactor):
    rows = {}
    for label, exp in candidate_designs(model).items():
        fim, rep = report(reactor, exp)
        rows[label] = {
            "relative_std": rep.relative_std,
            "D": aquakin.design_criterion(fim, "D"),
        }
    return rows


def step3_optimise(model, reactor):
    start = run(model, np.linspace(5.0, 900.0, 8), uv=True)
    res = aquakin.optimize_experiment(
        reactor,
        start,
        FREE_PARAMS,
        free_ic=FREE_IC,
        time_bounds=(1.0, 1800.0),
        min_spacing=5.0,
        initial_bounds={"O3": (20.0, 150.0)},
    )
    _, rep_start = report(reactor, start)
    _, rep_opt = report(reactor, res.experiments[0])
    return {
        "start": start,
        "result": res,
        "relative_std_start": rep_start.relative_std,
        "relative_std_opt": rep_opt.relative_std,
    }


def synthetic_data(reactor, experiment, *, seed=0):
    """Noisy ozone and UV254 measurements of the true sample."""
    sol = reactor.solve(
        jnp.asarray(experiment.C0),
        t_span=(0.0, float(experiment.t_obs[-1])),
        t_eval=experiment.t_obs,
    )
    p = reactor.model.default_parameters()
    clean = np.asarray(jnp.stack([experiment.observation_fn(c, p) for c in sol.C]))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(clean.shape) * np.asarray(experiment.sigma)[None, :]
    return clean + noise


def step4_calibrate(model, reactor, experiment, *, seed=0):
    data = synthetic_data(reactor, experiment, seed=seed)
    # Start the fit away from the truth: rate constants a factor of 2-3 off,
    # pools guessed at round numbers.
    guess = model.default_parameters()
    guess = guess.at[model.param_index["fast.k1"]].set(0.003)
    guess = guess.at[model.param_index["slow.k2"]].set(5.0e-6)
    C0_guess = jnp.asarray(experiment.C0).at[model.species_index["D"]].set(5.0)
    C0_guess = C0_guess.at[model.species_index["M"]].set(500.0)
    common = dict(
        observation_fn=experiment.observation_fn,
        loss="nll",
        sigma=experiment.sigma,
        free_ic=FREE_IC,
        transforms=dict.fromkeys(FREE_PARAMS, "positive_log"),
    )
    result = aquakin.calibrate(
        reactor,
        C0_guess,
        data,
        experiment.t_obs,
        FREE_PARAMS,
        initial_params=guess,
        # The information spans about four orders of magnitude here ([M]0 is
        # pinned by UV254, k1 only by the few early samples), so keep every
        # direction: the default eig_keep=1e-2 would drop the k1 direction and
        # report a misleadingly small standard deviation for it.
        laplace=LaplaceConfig(method="gauss_newton", include_ic=True, eig_keep=1e-10),
        optimizer=OptimizerConfig(method="gauss_newton", n_starts=4),
        **common,
    )
    k2 = result.params_named["slow.k2"]
    grid = jnp.asarray(k2 * np.exp(np.linspace(-0.05, 0.05, 11)))
    profile = aquakin.profile_likelihood(
        reactor,
        result.C0_fitted[0],
        data,
        experiment.t_obs,
        FREE_PARAMS,
        grid=grid,
        profile_param="slow.k2",
        initial_params=result.params,
        optimizer=OptimizerConfig(method="gauss_newton"),
        **common,
    )
    return {"data": data, "result": result, "profile": profile}


def step5_redesign(model, reactor, calibration, *, n_draw=10):
    draws = calibration["result"].posterior_draws(model, n_draw=n_draw, seed=1)
    start = run(model, np.linspace(5.0, 900.0, 8), uv=True)
    res = aquakin.optimize_experiment(
        reactor,
        start,
        FREE_PARAMS,
        free_ic=FREE_IC,
        time_bounds=(1.0, 1800.0),
        min_spacing=5.0,
        initial_bounds={"O3": (20.0, 150.0)},
        param_draws=draws,
    )
    return {"draws": draws, "result": res}


def _fmt(x):
    if not np.isfinite(x):
        return "not identifiable"
    if x >= 1.0:
        return f"{x * 100:.0f} %"
    return f"{x * 100:.2g} %"


def make_figures(model, reactor, out, opt, calibration):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    true_p = model.default_parameters()

    # Figure 1: the true ozone and UV254 curves with the two sets of sampling times.
    exp_opt = opt["result"].experiments[0]
    t = jnp.linspace(0.0, 1800.0, 721)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for dose, style, label in [(50.0, "-", "50 µmol/L dose"), (float(exp_opt.C0[0]), "--", "")]:
        C0 = model.default_concentrations().at[0].set(dose)
        sol = reactor.solve(C0, t_span=(0.0, 1800.0), t_eval=t, params=true_p)
        lab = label or f"{dose:.0f} µmol/L dose (optimised)"
        ax1.plot(t, sol.C_named("O3"), style, color="0.3", lw=1.2, label=lab)
        ax2.plot(t, EPS_UV * sol.C_named("M"), style, color="0.3", lw=1.2, label=lab)
    base_t = np.linspace(60.0, 600.0, 5)
    for ax in (ax1, ax2):
        for x in base_t:
            ax.axvline(x, color="tab:blue", lw=0.8, alpha=0.6, ymax=0.08)
        for x in np.asarray(exp_opt.t_obs):
            ax.axvline(x, color="tab:red", lw=0.8, alpha=0.8, ymin=0.92)
        ax.set_xlabel("time (s)")
    ax1.set_ylabel("dissolved ozone (µmol/L)")
    ax2.set_ylabel("UV254 absorbance (1/cm)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("ozone", fontsize=10)
    ax2.set_title("UV254", fontsize=10)
    fig.text(
        0.5,
        -0.04,
        "ticks: base sampling times (blue, bottom) and optimised times (red, top)",
        ha="center",
        fontsize=8,
    )
    fig.savefig(out / "ozone_demand_design.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: profile likelihood of k2 with the Laplace interval.
    prof = calibration["profile"]
    res = calibration["result"]
    k2 = res.params_named["slow.k2"]
    sd = res.params_named_std["slow.k2"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    grid = np.asarray(prof.grid)
    ax.plot(grid * 1e6, np.asarray(prof.delta_loss), "o-", color="0.3", label="profile")
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


def main(figure_dir=None):
    model, reactor = setup()
    true_p = model.default_parameters()

    print("Step 1: the base design (5 ozone samples, 60-600 s)")
    s1 = step1_diagnose(model, reactor)
    for d in s1["unidentifiable"]:
        top = list(d.loadings.items())[:2]
        print(f"  flagged direction, relative eigenvalue {d.relative_eigenvalue:.1e}: {top}")

    print("\nStep 2: relative standard deviation for each way of improving the run")
    s2 = step2_compare(model, reactor)
    for label, row in s2.items():
        cells = ", ".join(f"{n}: {_fmt(v)}" for n, v in row["relative_std"].items())
        print(f"  {label:34s} D = {row['D']:7.2f} | {cells}")

    print("\nStep 3: optimised sampling times and dose (ozone + UV254, 8 samples)")
    s3 = step3_optimise(model, reactor)
    e = s3["result"].experiments[0]
    print(
        f"  converged: {s3['result'].success}, D: {s3['result'].initial_value:.2f} -> "
        f"{s3['result'].value:.2f}"
    )
    print(f"  times (s): {np.round(np.asarray(e.t_obs), 1).tolist()}")
    print(f"  dose: {float(e.C0[0]):.1f} umol/L")
    for n in s3["relative_std_opt"]:
        print(
            f"  {n:10s} {_fmt(s3['relative_std_start'][n]):>10s} -> "
            f"{_fmt(s3['relative_std_opt'][n]):>10s}"
        )

    print("\nStep 4: calibration on synthetic data from the optimised design")
    s4 = step4_calibrate(model, reactor, e)
    res = s4["result"]
    truth = {n: float(true_p[model.param_index[n]]) for n in FREE_PARAMS}
    truth.update({"D": 15.0, "M": 1000.0})
    for n in FREE_PARAMS:
        print(
            f"  {n:8s} {res.params_named[n]:.4g} +/- {res.params_named_std[n]:.2g}"
            f"  (true {truth[n]:.4g})"
        )
    for s in ["D", "M"]:
        print(
            f"  [{s}]0     {res.ic_named[0][s]:.4g} +/- {res.ic_named_std[0][s]:.2g}"
            f"  (true {truth[s]:.4g})"
        )
    print(f"  profile-likelihood 95 % interval for k2: {s4['profile'].ci}")

    print("\nStep 5: robust redesign over 10 posterior draws")
    s5 = step5_redesign(model, reactor, s4)
    e5 = s5["result"].experiments[0]
    print(f"  times (s): {np.round(np.asarray(e5.t_obs), 1).tolist()}")
    print(f"  dose: {float(e5.C0[0]):.1f} umol/L")

    if figure_dir is not None:
        make_figures(model, reactor, figure_dir, s3, s4)
        print(f"\nFigures written to {figure_dir}")
    return {"step1": s1, "step2": s2, "step3": s3, "step4": s4, "step5": s5}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
