"""Reproduce sewer nitrate-dosing batch tests with the ``wats_sewer`` network.

Runs the two anoxic batch experiments (a "calibration" and a "validation"
dataset) used to fit and check the extended-WATS sewer model: reactor liquor is
dosed with nitrate at t=0 and sulfide, sulfate, VFA and nitrate are followed for
~5 h. Nitrate first drives biological sulfide oxidation
(sulfide -> elemental S -> sulfate), suppressing sulfide; once the nitrate is
exhausted, sulfate reduction resumes and sulfide rebuilds.

The cases are closed batches (no flow), so they map directly onto
``aquakin.BatchReactor``. Initial states and the calibrated rate constants come
from the published model setup; nothing here is re-fitted. pH is a spectator in
this anoxic regime (dissolved O2 is zero, so the pH-dependent oxidation terms
vanish), so the shipped network is used as-is.

Measured data and the original model curves are bundled in
``examples/data/wats_sewer_batch_reference.csv`` so the overlay needs no
external files.

    python examples/wats_sewer_batch_validation.py          # integrate + summary
    python examples/wats_sewer_batch_validation.py --plot    # also save overlay figures
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import jax.numpy as jnp

import aquakin

_HERE = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_CSV = os.path.join(_HERE, "data", "wats_sewer_batch_reference.csv")

# Batch initial states (mg/L) for the 18 reactive species, from the published
# experiment setup. Species not listed default to the network's reference value.
_BASE_IC = dict(
    S_B=120.0, X_BH=11.8, X_BA=0.0, X_S1=28.32, X_S2=100.84, S_O=0.0,
    S_NH=40.0, S_PO4=46.0, S_CH4=0.0, S_H2=0.0, S_CO2=20.0, S_I=50.0,
    X_I=33.04, X_S0=15.0,
)
CASES = {
    "calibration": {
        "ic": {**_BASE_IC, "S_VFA": 28.2, "S_NO": 30.0, "sumS": 9.0, "S_SO4": 4.0},
        # calibration uses the network's default (calibrated) rate constants
        "params": {},
    },
    "validation": {
        "ic": {**_BASE_IC, "S_VFA": 16.0, "S_NO": 33.0, "sumS": 16.0, "S_SO4": 2.0},
        # the published validation run re-tuned four constants
        "params": {
            "k_12_no": 8.0, "k_h2s_acid": 2.5, "k_sII_ox_f": 18.0, "k_sII_anox_f": 25.3,
        },
    },
}

# Reactor operating conditions and run length (model time is in days; 5 h test).
CONDITIONS = dict(T=20.0, A_V=56.7, X_BF=10.0)
T_END_DAYS = 5.0 / 24.0

# (panel title, aquakin species, y-label, reference-CSV species name)
PLOT_SPECIES = [
    ("Sulfide", "sumS", "Sulfide (mgS/L)", "sulfide"),
    ("Sulfate", "S_SO4", "Sulfate (mgS/L)", "sulfate"),
    ("VFA", "S_VFA", "VFA (mgCOD/L)", "VFA"),
    ("Nitrate", "S_NO", "Nitrate (mgN/L)", "nitrate"),
]


def run_case(network, case_name, n_out=400):
    """Integrate one batch case; return (time_hours, {aquakin_species: trajectory})."""
    spec = CASES[case_name]
    C0 = network.default_concentrations()
    for name, value in spec["ic"].items():
        C0 = C0.at[network.species_index[name]].set(value)
    params = network.default_parameters()
    for name, value in spec["params"].items():
        params = params.at[network.param_index[name]].set(value)

    reactor = aquakin.BatchReactor(network, aquakin.SpatialConditions.uniform(1, **CONDITIONS))
    t_eval = jnp.linspace(0.0, T_END_DAYS, n_out)
    sol = reactor.solve(C0, params, t_span=(0.0, T_END_DAYS), t_eval=t_eval)
    t_hours = [float(t) * 24.0 for t in sol.t]
    return t_hours, {sp: sol.C_named(sp) for _, sp, _, _ in PLOT_SPECIES}


def load_reference(case_name, path=_REFERENCE_CSV):
    """Load bundled measured points and original model curves for ``case_name``.

    Returns ``{csv_species: {"measured": (t, v), "model": (t, v)}}`` or ``None``
    if the bundled CSV is missing.
    """
    if not os.path.isfile(path):
        return None
    series = defaultdict(lambda: defaultdict(lambda: ([], [])))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["case"] != case_name:
                continue
            t, v = series[row["species"]][row["series"]]
            t.append(float(row["time_h"]))
            v.append(float(row["value"]))
    return {sp: dict(kinds) for sp, kinds in series.items()}


def plot_case(case_name, t_hours, traj, reference, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (title, sp, ylabel, ref_sp) in zip(axes.ravel(), PLOT_SPECIES):
        ax.plot(t_hours, traj[sp], "-", color="C2", lw=2.2, label="aquakin")
        if reference is not None and ref_sp in reference:
            tn, vn = reference[ref_sp].get("model", ([], []))
            te, ve = reference[ref_sp].get("measured", ([], []))
            ax.plot(tn, vn, "--", color="C0", lw=1.4, label="original model")
            ax.plot(te, ve, "o", color="C3", ms=5, label="measured")
        ax.set_title(title)
        ax.set_xlabel("time (h)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max(t_hours))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"WATS sewer nitrate-dosing batch — {case_name}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="save overlay figures (needs matplotlib)")
    parser.add_argument("--out-dir", default=".", help="where to write figures")
    args = parser.parse_args()

    network = aquakin.load_network("wats_sewer")
    for case_name in CASES:
        print(f"\n=== {case_name} ===")
        t_hours, traj = run_case(network, case_name)
        for title, sp, _, _ in PLOT_SPECIES:
            arr = traj[sp]
            print(f"  {title:8s} t0={float(arr[0]):7.2f}  min={float(arr.min()):7.2f}  "
                  f"end={float(arr[-1]):7.2f}  (mg/L)")
        if args.plot:
            reference = load_reference(case_name)
            if reference is None:
                print("  (bundled reference CSV not found; plotting aquakin only)")
            plot_case(case_name, t_hours, traj, reference,
                      os.path.join(args.out_dir, f"wats_sewer_{case_name}.png"))


if __name__ == "__main__":
    main()
