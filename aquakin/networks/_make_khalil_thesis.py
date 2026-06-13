"""Generate wats_sewer_khalil_thesis.yaml: the complete WATS sewer-process model
(Hvitved-Jacobsen, Vollertsen & Nielsen 2013, Tables 9.1-9.4 -- aerobic, anoxic
and anaerobic heterotrophic carbon turnover plus the full sulfur cycle) extended
with the nitrate-dosing additions of Khalil's thesis Table 4-1 (methanogenesis,
elemental-sulfur reduction, and the nitrate-driven two-step sulfide -> S0 ->
sulfate oxidation).

This is the *complete* WATS model, not a reduced subset: aerobic and anoxic
growth (bulk + biofilm) on both fermentable substrate and VFA, maintenance,
fast/slow hydrolysis in all three redox regimes, fermentation, sulfate reduction,
and chemical/biological sulfide oxidation in bulk and biofilm. The reaction
structure is the same as the shipped `wats_sewer_extended` network -- here we
  (1) drop the two pieces that are NOT part of the WATS process matrix or the
      thesis (the charge-balance pH solver and nitrification/autotrophs), and
  (2) revert the kinetic parameters to the values reported in the thesis/paper
      (single heterotroph yield 0.55; faster hydrolysis k_h1=12, k_h2=5;
      q_ferm=2; the Table 4-1 nitrate sulfur kinetics),
to test whether the published-fidelity model reproduces the batch nitrate-dosing
results.

The pH-dependent bulk sulfide-oxidation rates (textbook Eqs 5.14-5.15) remain,
but pH is supplied as a *fixed* condition rather than solved from the state --
the thesis uses a fixed operating pH and does not include the charge-balance
speciation machinery.

Run from this directory:  python _make_khalil_thesis.py
"""
from __future__ import annotations

import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "wats_sewer_extended.yaml")
OUT = os.path.join(HERE, "wats_sewer_khalil_thesis.yaml")

# Reactions in wats_sewer_extended that are NOT part of the WATS process matrix
# (Tables 9.1-9.4) or the thesis heterotrophic base model -- nitrification and
# autotroph decay are autotrophic nitrogen processes the thesis does not model.
# FeS precipitation is an improvement beyond the published thesis model (iron was
# commented out of Khalil's code); it lives only in wats_sewer_extended and the
# _balanced variant, not in the faithful thesis reproduction.
DROP_REACTIONS = {"nitrification", "nitrifier_decay", "FeS_precipitation"}
DROP_SPECIES = {"S_Fe2", "X_FeS"}

# Fixed operating pH for the batch (thesis uses a fixed pH, not a charge-balance
# solver). Near-neutral wastewater; below the biological-oxidation optimum (8.0).
PH_FIXED = 7.5

# Thesis / paper parameter values (revert the wats_sewer_extended deviations).
THESIS = {
    # single heterotroph yield Y_H = 0.55 for aerobic AND anoxic growth
    "y_h_anox": 0.55, "y_hw": 0.55, "y_hf": 0.55,
    # thesis hydrolysis is much faster than the paper's printed 5 / 0.5
    "k_h1": 12.0, "k_h2": 5.0, "k_fe": 20.0,
    "q_ferm": 2.0,
    "mu_h": 7.0,           # thesis Table 3-1 max aerobic growth rate (paper uses 6.7)
    "eta_an": 0.18,        # thesis anaerobic hydrolysis correction (paper uses 0.21)
    "k_12_o2": 4.0,        # thesis Table 3-1 aerobic biofilm half-order growth const (paper uses 18)
    "k_no": 2.0,            # bulk nitrate half-saturation (paper)
    "k_h2s_acid": 2.5,      # sulfate reduction (paper)
    "k_s0_acid": 15.5,      # elemental-sulfur reduction (paper, as printed)
    "k_sII_anox_f": 12.1,   # thesis-calibrated anoxic sulfide oxidation
    "k_s0_anox_f": 2.2,     # anoxic elemental-sulfur oxidation
}


def main():
    net = yaml.safe_load(open(BASE))
    net["network"]["name"] = "wats_sewer_khalil_thesis"
    net["network"]["description"] = (
        "Complete WATS sewer-process model (Hvitved-Jacobsen, Vollertsen & "
        "Nielsen 2013, Tables 9.1-9.4: aerobic/anoxic/anaerobic heterotrophic "
        "carbon turnover -- growth bulk+biofilm on fermentable substrate and "
        "VFA, maintenance, fast/slow hydrolysis in all redox regimes, "
        "fermentation -- plus the sulfur cycle: sulfate reduction and "
        "chemical/biological sulfide oxidation in bulk and biofilm) extended "
        "with Khalil's thesis Table 4-1 nitrate-dosing additions "
        "(methanogenesis, elemental-sulfur reduction, and nitrate-driven "
        "two-step sulfide->S0->sulfate oxidation). Same reaction structure as "
        "wats_sewer_extended, with the non-WATS/non-thesis pieces removed (charge-balance "
        "pH solver replaced by a fixed operating pH; nitrification/autotrophs "
        "dropped) and parameters reverted to thesis/paper values (single yield "
        "0.55; k_h1=12, k_h2=5; q_ferm=2; Table 4-1 sulfur kinetics). Generated "
        "by _make_khalil_thesis.py to test reproduction of the published batch "
        "nitrate-dosing results.")

    # (1) Drop the charge-balance pH solver; supply pH as a fixed condition so
    #     the pH-dependent bulk sulfide-oxidation rates still compile.
    net.pop("speciation", None)
    cond_names = {c["name"] for c in net["conditions"]}
    if "pH" not in cond_names:
        net["conditions"].append({
            "name": "pH",
            "description": "Fixed operating pH of the batch (thesis uses a "
                           "fixed pH rather than a charge-balance solver).",
            "units": "-",
            "default": PH_FIXED,
        })

    # (2) Drop nitrification / autotroph decay (not in the WATS process matrix
    #     or the thesis heterotrophic model).
    net["reactions"] = [rx for rx in net["reactions"]
                        if rx["name"] not in DROP_REACTIONS]
    net["species"] = [sp for sp in net["species"]
                      if sp["name"] not in DROP_SPECIES]
    net["parameters"].pop("k_fes_p", None)

    # (3) Revert kinetic parameters to thesis/paper values.
    for k, v in THESIS.items():
        if k in net["parameters"]:
            net["parameters"][k]["value"] = v

    with open(OUT, "w") as f:
        f.write("# Auto-generated from wats_sewer_extended.yaml by _make_khalil_thesis.py "
                "-- do not edit by hand.\n")
        yaml.safe_dump(net, f, sort_keys=False, default_flow_style=False, width=100)
    print(f"wrote {os.path.basename(OUT)} "
          f"({net['network']['name']}, {len(net['reactions'])} reactions, "
          f"{len(net['conditions'])} conditions)")


if __name__ == "__main__":
    main()
