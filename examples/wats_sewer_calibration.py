"""AD-based calibration of the ``wats_sewer`` model to a batch nitrate test.

The published study fit the influential anoxic parameters to a batch
nitrate-dosing experiment with a MATLAB least-squares routine. This script does
the same with gradient-based optimization through ``aquakin.calibrate()`` —
the gradient flows through the stiff Diffrax solve via the ``dtmax`` cap (see
CLAUDE.md, "Differentiating stiff networks").

It fits the three influential RATE constants (sulfide and elemental-S anoxic
oxidation, biofilm denitrification) to the measured sulfide, sulfate and
nitrate of the "calibration" batch, then reports the fit on both batches.

Two findings the run reproduces (both consistent with the paper):
  * Identifiability — fitting the rate constants together with their nitrate /
    sulfide saturation constants is ill-posed (a rate and its saturation
    constant trade off), so the saturation constants are held fixed at the
    published calibrated values and only the three rates are freed. Freeing all
    six lets correlated pairs run off to absurd magnitudes.
  * Calibration/validation tension — the two batches favour slightly different
    sulfur kinetics, so a fit to one overshoots the other's sulfate. VFA is the
    weak point in both (as reported in the source study).

    python examples/wats_sewer_calibration.py            # fit + summary
    python examples/wats_sewer_calibration.py --plot     # also save overlays
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import jax.numpy as jnp
import numpy as np

import aquakin
from aquakin import BatchReactor, SpatialConditions, calibrate

_HERE = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_CSV = os.path.join(_HERE, "data", "wats_sewer_batch_reference.csv")

_BASE_IC = dict(
    S_B=120.0, X_BH=11.8, X_BA=0.0, X_S1=28.32, X_S2=100.84, S_O=0.0,
    S_NH=40.0, S_PO4=46.0, S_CH4=0.0, S_H2=0.0, S_CO2=20.0, S_I=50.0,
    X_I=33.04, X_S0=15.0,
)
IC = {
    "calibration": {**_BASE_IC, "S_VFA": 28.2, "S_NO": 30.0, "sumS": 9.0, "S_SO4": 4.0},
    "validation":  {**_BASE_IC, "S_VFA": 16.0, "S_NO": 33.0, "sumS": 16.0, "S_SO4": 2.0},
}
CONDITIONS = dict(T=20.0, A_V=56.7, X_BF=10.0)
T_END_DAYS = 5.0 / 24.0
DTMAX = 5.0e-4  # cap needed for finite gradients through the stiff solve

# Fit these influential rate constants; saturation constants stay fixed
# (identifiability). Per-species weights balance the disparate scales.
FREE_PARAMS = ["k_sII_anox_f", "k_s0_anox_f", "k_12_no"]
FIT_SPECIES = [("sumS", "sulfide"), ("S_SO4", "sulfate"), ("S_NO", "nitrate")]
FIT_SIGMA = jnp.asarray([1.0, 2.0, 1.5])  # mgS, mgS, mgN

# (panel title, aquakin species, y-label, reference species)
PLOT_SPECIES = [
    ("Sulfide", "sumS", "Sulfide (mgS/L)", "sulfide"),
    ("Sulfate", "S_SO4", "Sulfate (mgS/L)", "sulfate"),
    ("VFA", "S_VFA", "VFA (mgCOD/L)", "VFA"),
    ("Nitrate", "S_NO", "Nitrate (mgN/L)", "nitrate"),
]


def _load_measured():
    """Return ``{case: {species: (times_h, values)}}`` from the bundled CSV."""
    out = defaultdict(lambda: defaultdict(lambda: ([], [])))
    with open(_REFERENCE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["series"] != "measured":
                continue
            t, v = out[row["case"]][row["species"]]
            t.append(float(row["time_h"]))
            v.append(float(row["value"]))
    return out


def _C0(network, case):
    c = network.default_concentrations()
    for name, value in IC[case].items():
        c = c.at[network.species_index[name]].set(value)
    return c


def _rmse(reactor, network, case, params, measured):
    t_h = np.array(measured[case]["sulfide"][0])
    sol = reactor.solve(_C0(network, case), params, t_span=(0.0, T_END_DAYS),
                        t_eval=jnp.asarray(t_h / 24.0))
    return {
        ref: float(np.sqrt(np.mean((np.array(sol.C_named(sp)) - np.array(measured[case][ref][1])) ** 2)))
        for sp, ref in FIT_SPECIES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="save overlay figures (needs matplotlib)")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    network = aquakin.load_network("wats_sewer")
    reactor = BatchReactor(network, SpatialConditions.uniform(1, **CONDITIONS),
                           rtol=1e-6, atol=1e-9, dtmax=DTMAX)
    measured = _load_measured()

    # observations for the calibration batch (shared 11-point time grid)
    t_h = np.array(measured["calibration"]["sulfide"][0])
    t_obs = jnp.asarray(t_h / 24.0)
    obs = jnp.asarray(np.stack([measured["calibration"][ref][1] for _, ref in FIT_SPECIES], axis=1))

    p0 = network.default_parameters()
    print("initial (published-calibrated) rate constants:")
    for f in FREE_PARAMS:
        print(f"  {f} = {float(p0[network.param_index[f]]):.3f}")

    result = calibrate(
        reactor, _C0(network, "calibration"), obs, t_obs, FREE_PARAMS,
        transforms={f: "positive_log" for f in FREE_PARAMS},
        observed_species=[sp for sp, _ in FIT_SPECIES],
        loss="wmse", sigma=FIT_SIGMA, laplace=False, max_iter=120,
    )
    print(f"\nconverged={result.converged}  n_iter={result.n_iter}  loss={result.loss:.4g}")
    print("fitted rate constants:")
    for k, v in result.params_named.items():
        print(f"  {k} = {v:.3f}")

    print("\nRMSE per species (initial -> fitted):")
    for case in ("calibration", "validation"):
        ri = _rmse(reactor, network, case, p0, measured)
        rf = _rmse(reactor, network, case, result.params, measured)
        print(f"  {case}:")
        for _, ref in FIT_SPECIES:
            print(f"    {ref:8s} {ri[ref]:5.2f} -> {rf[ref]:5.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for case in ("calibration", "validation"):
            t_eval = jnp.linspace(0.0, T_END_DAYS, 400)
            sol = reactor.solve(_C0(network, case), result.params, t_span=(0.0, T_END_DAYS), t_eval=t_eval)
            th = [float(t) * 24.0 for t in sol.t]
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
            for ax, (title, sp, ylab, ref) in zip(axes.ravel(), PLOT_SPECIES):
                ax.plot(th, sol.C_named(sp), "-", color="C2", lw=2.2, label="aquakin (AD-calibrated)")
                ax.plot(measured[case][ref][0], measured[case][ref][1], "o", color="C3", ms=5, label="measured")
                ax.set_title(title); ax.set_xlabel("time (h)"); ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.set_xlim(0, 5)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"wats_sewer AD-calibrated vs measured — {case}", fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            out = os.path.join(args.out_dir, f"wats_sewer_calibrated_{case}.png")
            fig.savefig(out, dpi=130)
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
