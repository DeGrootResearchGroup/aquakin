"""Generate wats_sewer_khalil_paper_balanced_biofilm_multispecies.yaml: the full
multispecies depth-resolved biofilm model, in which the sulfur and methane
processes are driven by their OWN per-layer growing/decaying functional-group
biomass instead of being interim-coupled to the heterotroph X_BH.

Starting point
--------------
``wats_sewer_khalil_paper_balanced_biofilm_biomass`` (increment 1) already made
the heterotroph X_BH an explicit per-layer state driving the carbon backbone, but
tied the sulfur/methane processes to [X_BH] as a stand-in. That biases any
calibration: the optimizer would abuse those [X_BH]-scaled terms to fit
sulfide/sulfate, even though SRB, methanogens and S-oxidizers stratify
differently from heterotrophs (Sun et al. 2014: SRB outer, methanogens inner;
Jiang et al. 2009 Fig 6: SRB peak deep). This generator gives those processes
their own biomass so the calibration is physical.

Functional groups added (each a per-layer state; composition = active biomass:
COD 1, N i_n_bio, P i_p_bio, S 0.01)
-----------------------------------------------------------------------------
- ``X_SRB`` -- sulfate- and elemental-sulfur-reducing bacteria. Grow on the
  fermentable donor S_B, reducing S_SO4->sumS and X_S0->sumS. (Jiang 2009 Table
  2: mu_SRB, Y_SRB=0.596, b_SRB=0.192, K_SO4=6.4.)
- ``X_MA`` -- methanogens (acetoclastic on S_VFA + hydrogenotrophic on S_H2),
  nitrate-inhibited. (ADM1 / Sun 2014 range: low yield, slow growth.)
- ``X_SOB`` -- sulfide-oxidizing bacteria (nitrate-driven and aerobic), oxidising
  sumS->X_S0->S_SO4. Chemolithotrophic: the reduced sulfur is the electron
  donor, nitrate/O2 the acceptor. (Mohanakrishnan 2009; Nielsen 2005.)

Each process keeps its existing Monod rate form but is driven by [X_group] and
produces biomass at yield Y; every reaction's stoichiometry is re-derived to
conserve COD/S/N (and -> the original electron balance as Y->0). Each group gets
a first-order decay X_group -> X_I releasing N/P. The old areal rate constants
(k_h2s_acid, k_sII_anox_f, ...) are superseded by growth rates ``mu_*`` and
auto-pruned. The heterotroph carbon backbone (growth, maintenance, hydrolysis,
fermentation) is unchanged on X_BH.

Run in :class:`aquakin.BiofilmReactor` with ``biofilm_reactions=None`` (single
phase), a stratified initial state (high X_BH / X_SRB / X_MA / X_SOB in the
layers, low in the bulk), and ``fixed_mask`` holding only the inert solids X_I.
The functional-group gradients then evolve and stratify by their own kinetics.

Parameter note: like the increment-1 model, the absolute growth rate and the
biofilm biomass density are confounded (only their product is grounded), so the
``mu_*`` defaults here are literature-range placeholders -- the variant is meant
to be re-calibrated. What is fixed from the literature are the YIELDS and the
electron stoichiometry (which set the conserving mass balance), not the rates.

Run from this directory:  python _make_khalil_balanced_biofilm_multispecies.py
"""
from __future__ import annotations

import os
import re
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "wats_sewer_khalil_paper_balanced_biofilm_biomass.yaml")
OUT = os.path.join(HERE, "wats_sewer_khalil_paper_balanced_biofilm_multispecies.yaml")

# Active-biomass composition shared by every functional group (matches X_BH):
# COD=1 with the biomass N / P / S micro-contents, so the conserved-quantity table
# (check_conservation) closes the growth/decay balances for these groups too.
_BIOMASS_SPECIES = lambda name, desc: {
    "name": name, "description": desc, "units": "gCOD/m3",
    "default_concentration": 1.0,
    "composition": {"COD": 1.0, "N": 0.07, "P": 0.02, "S": 0.01},
}

# New network-level parameters (yields fixed from the literature; growth rates
# and decay are literature-range placeholders pending re-calibration).
NEW_PARAMS = {
    # --- sulfate-reducing bacteria (Jiang et al. 2009 Table 2) ---
    "Y_srb": {"value": 0.596, "units": "gCOD/gCOD", "bounds": [0.3, 0.7]},
    "mu_srb": {"value": 0.8, "units": "1/d", "bounds": [0.05, 50.0]},
    "mu_srb_s0": {"value": 2.0, "units": "1/d", "bounds": [0.05, 50.0]},
    "b_srb": {"value": 0.192, "units": "1/d", "bounds": [0.02, 1.0]},
    # --- methanogens (ADM1 / Sun et al. 2014 range) ---
    "Y_ma": {"value": 0.05, "units": "gCOD/gCOD", "bounds": [0.02, 0.1]},
    "Y_ma_h2": {"value": 0.06, "units": "gCOD/gCOD", "bounds": [0.02, 0.1]},
    "mu_ma": {"value": 0.3, "units": "1/d", "bounds": [0.02, 20.0]},
    "mu_ma_h2": {"value": 2.0, "units": "1/d", "bounds": [0.05, 50.0]},
    "b_ma": {"value": 0.03, "units": "1/d", "bounds": [0.005, 0.5]},
    # --- sulfide-oxidising bacteria (Mohanakrishnan 2009; Nielsen 2005) ---
    "Y_sob": {"value": 0.10, "units": "gCOD/gCOD", "bounds": [0.02, 0.4]},
    "mu_sob": {"value": 1.0, "units": "1/d", "bounds": [0.05, 50.0]},
    "mu_sob_s0": {"value": 0.5, "units": "1/d", "bounds": [0.05, 50.0]},
    "mu_sob_aer": {"value": 2.0, "units": "1/d", "bounds": [0.05, 50.0]},
    "mu_sob_s0_aer": {"value": 0.0, "units": "1/d", "bounds": [0.0, 50.0]},
    "b_sob": {"value": 0.05, "units": "1/d", "bounds": [0.005, 0.5]},
    # sulfate-reduction acceptor affinity (Jiang Table 2; was 2.0)
    "K_SO4": {"value": 6.4, "units": "gS/m3", "bounds": [0.5, 20.0]},
}

# Replacement reactions. Stoichiometry strings use the network parameters; each
# conserves COD/S/N and reduces to the original electron balance as Y -> 0.
# i_n_bio / i_p_bio carry the biomass N/P demand.
NN, NP = "0.0 - i_n_bio", "0.0 - i_p_bio"
REPLACE = {
    # ---- X_SRB ---------------------------------------------------------------
    "sulfate_reduction_SB": dict(
        name="sulfate_reduction",
        rate="mu_srb * monod([S_B], k_srb) * monod([S_SO4], K_SO4) * no_gate * [X_SRB]",
        stoichiometry={
            "X_SRB": 1, "S_B": "0.0 - 1.0 / Y_srb",
            "sumS": "(1.0 - Y_srb) / (2.0 * Y_srb)",
            "S_SO4": "0.0 - (1.0 - Y_srb) / (2.0 * Y_srb)",
            "S_NH": NN, "S_PO4": NP,
        }),
    "elemental_S_reduction_SB": dict(
        name="elemental_S_reduction",
        rate="mu_srb_s0 * monod([S_B], k_srb) * monod([X_S0], K_S0) * no_gate * [X_SRB]",
        stoichiometry={
            "X_SRB": 1, "S_B": "0.0 - 1.0 / Y_srb",
            "sumS": "2.0 * (1.0 - Y_srb) / Y_srb",
            "X_S0": "0.0 - 2.0 * (1.0 - Y_srb) / Y_srb",
            "S_NH": NN, "S_PO4": NP,
        }),
    # ---- X_MA ----------------------------------------------------------------
    "methanogenesis_VFA": dict(
        name="methanogenesis_VFA",
        rate="mu_ma * monod([S_VFA], k_vfa_mb) * monod_inh([S_NO], K_NO_f) * [X_MA]",
        stoichiometry={
            "X_MA": 1, "S_VFA": "0.0 - 1.0 / Y_ma",
            "S_CH4": "(1.0 - Y_ma) / Y_ma", "S_NH": NN, "S_PO4": NP,
        }),
    "methanogenesis_H2": dict(
        name="methanogenesis_H2",
        rate="mu_ma_h2 * monod([S_H2], k_h2_mb) * monod_inh([S_NO], K_NO_f) * [X_MA]",
        stoichiometry={
            "X_MA": 1, "S_H2": "0.0 - 1.0 / Y_ma_h2",
            "S_CH4": "(1.0 - Y_ma_h2) / Y_ma_h2", "S_NH": NN, "S_PO4": NP,
        }),
    # ---- X_SOB ---------------------------------------------------------------
    # sumS -> X_S0 releases 0.5 gCOD/gS; X_S0 -> SO4 releases 1.5 gCOD/gS. Per
    # gCOD biomass the donor must supply 1/Y_sob gCOD of electrons.
    "sulfide_oxidation_anoxic": dict(
        name="sulfide_oxidation_anoxic",
        rate="mu_sob * monod([sumS], K_S2) * monod([S_NO], K_NO_S_f) * [X_SOB]",
        stoichiometry={
            "X_SOB": 1, "sumS": "0.0 - 2.0 / Y_sob", "X_S0": "2.0 / Y_sob",
            "S_NO": "0.0 - (1.0 / Y_sob - 1.0) / 2.86", "S_NH": NN, "S_PO4": NP,
        }),
    "elemental_S_oxidation_anoxic": dict(
        name="elemental_S_oxidation_anoxic",
        rate="mu_sob_s0 * monod([X_S0], K_S0) * monod([S_NO], K_NO_S_f) * [X_SOB]",
        stoichiometry={
            "X_SOB": 1, "X_S0": "0.0 - 2.0 / (3.0 * Y_sob)",
            "S_SO4": "2.0 / (3.0 * Y_sob)",
            "S_NO": "0.0 - (1.0 / Y_sob - 1.0) / 2.86", "S_NH": NN, "S_PO4": NP,
        }),
    "sulfide_biooxidation_aerobic": dict(
        name="sulfide_biooxidation_aerobic",
        rate="mu_sob_aer * monod([sumS], K_S2) * monod([S_O], k_o) * [X_SOB]",
        stoichiometry={
            "X_SOB": 1, "sumS": "0.0 - 2.0 / Y_sob", "X_S0": "2.0 / Y_sob",
            "S_O": "0.0 - (1.0 / Y_sob - 1.0)", "S_NH": NN, "S_PO4": NP,
        }),
    "elemental_S_oxidation_aerobic": dict(
        name="elemental_S_oxidation_aerobic",
        rate="mu_sob_s0_aer * monod([X_S0], K_S0) * monod([S_O], k_o) * [X_SOB]",
        stoichiometry={
            "X_SOB": 1, "X_S0": "0.0 - 2.0 / (3.0 * Y_sob)",
            "S_SO4": "2.0 / (3.0 * Y_sob)",
            "S_O": "0.0 - (1.0 / Y_sob - 1.0)", "S_NH": NN, "S_PO4": NP,
        }),
}

# First-order decay (inactivation) of each functional group -> inert X_I, N/P
# released. Rate constant per group.
DECAYS = [
    ("decay_srb", "b_srb", "X_SRB"),
    ("decay_ma", "b_ma", "X_MA"),
    ("decay_sob", "b_sob", "X_SOB"),
]


def main():
    net = yaml.safe_load(open(BASE))
    net["network"]["name"] = "wats_sewer_khalil_paper_balanced_biofilm_multispecies"
    net["network"]["description"] = (
        "Full multispecies depth-resolved biofilm model: the sulfur and methane "
        "processes are driven by their own per-layer growing/decaying biomass "
        "(X_SRB sulfate/elemental-S reducers, X_MA methanogens, X_SOB "
        "sulfide-oxidisers) rather than the heterotroph X_BH. Each process keeps "
        "its Monod form but grows its group at a literature yield with "
        "COD/S/N-conserving stoichiometry; each group decays first-order to inert. "
        "Run with aquakin.BiofilmReactor (biofilm_reactions=None), a stratified "
        "initial state, and fixed_mask holding only X_I. Generated by "
        "networks/_make_khalil_balanced_biofilm_multispecies.py from the "
        "increment-1 _biofilm_biomass model."
    )

    # 1. New functional-group biomass species (insert after X_BA).
    species = net["species"]
    names = [s["name"] for s in species]
    insert_at = names.index("X_BA") + 1
    new_species = [
        _BIOMASS_SPECIES("X_SRB", "Sulfate/elemental-sulfur-reducing biomass"),
        _BIOMASS_SPECIES("X_MA", "Methanogenic biomass"),
        _BIOMASS_SPECIES("X_SOB", "Sulfide-oxidising biomass"),
    ]
    net["species"] = species[:insert_at] + new_species + species[insert_at:]

    # 2. New parameters.
    net.setdefault("parameters", {})
    for pname, spec in NEW_PARAMS.items():
        net["parameters"][pname] = dict(spec)

    # 3. Replace the eight sulfur/methane reactions with growth versions.
    out = []
    n_repl = 0
    for rx in net["reactions"]:
        if rx["name"] in REPLACE:
            spec = REPLACE[rx["name"]]
            new = {"name": spec["name"],
                   "description": rx.get("description", rx["name"]),
                   "rate": spec["rate"], "stoichiometry": spec["stoichiometry"]}
            if "reference" in rx:
                new["reference"] = rx["reference"]
            out.append(new)
            n_repl += 1
        else:
            out.append(rx)
    # 4. Add decay reactions.
    for rname, bcoef, grp in DECAYS:
        out.append({
            "name": rname,
            "description": f"First-order decay (inactivation) of {grp} to inert.",
            "rate": f"{bcoef} * [{grp}]",
            "stoichiometry": {grp: -1, "X_I": 1, "S_NH": "i_n_bio", "S_PO4": "i_p_bio"},
        })
    net["reactions"] = out

    # 5. Prune parameters orphaned by the replacement (the old areal rate
    #    constants k_h2s_acid, k_sII_anox_f, ... are no longer referenced).
    referenced = set()
    for rx in net["reactions"]:
        referenced |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rx["rate"]))
        for v in rx.get("stoichiometry", {}).values():
            if isinstance(v, str):
                referenced |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", v))
    for v in (net.get("expressions") or {}).values():
        referenced |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", v))
    n_pruned = 0
    for pname in [p for p in net["parameters"] if p not in referenced]:
        del net["parameters"][pname]
        n_pruned += 1

    with open(OUT, "w") as f:
        f.write("# Auto-generated from wats_sewer_khalil_paper_balanced_biofilm_biomass.yaml "
                "by _make_khalil_balanced_biofilm_multispecies.py -- do not edit by hand.\n")
        yaml.safe_dump(net, f, sort_keys=False, default_flow_style=False, width=100)

    print(f"wrote {os.path.basename(OUT)} ({len(net['reactions'])} reactions, "
          f"{len(net['species'])} species; replaced {n_repl} processes, added "
          f"{len(DECAYS)} decays, pruned {n_pruned} dead params)")
    for grp in ("X_SRB", "X_MA", "X_SOB"):
        rxns = [r["name"] for r in net["reactions"] if f"[{grp}]" in r["rate"]]
        print(f"  {grp}: {rxns}")


if __name__ == "__main__":
    main()
