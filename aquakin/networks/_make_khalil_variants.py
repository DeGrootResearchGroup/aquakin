"""Generate the structural variants of the Khalil sewer model, for BOTH the
paper-faithful base (`wats_sewer_khalil_paper`) and the mass/electron-balanced
base (`wats_sewer_khalil_paper_balanced`). Run from this directory:

    python _make_khalil_variants.py

The structural changes come from reviewer feedback on the original model:
  - halforder     : half-order (square-root) biofilm sulfur-oxidation kinetics.
  - directsulfate : nitrate-driven sulfide oxidation goes straight to sulfate,
                    bypassing the elemental-sulfur intermediate.
  - srbsubstrate  : sulfate / elemental-S reduction consume readily-biodegradable
                    substrate (S_B) rather than VFA (faithful base only -- the
                    balanced base already makes this change by design).
  - combined      : all of the base's variants applied together.

Produces, for each base, wats_sewer_khalil_paper[_balanced]_{halforder,
directsulfate,srbsubstrate,combined}.yaml. The directsulfate nitrate demand is
computed from the base's own two-step coefficients, so it stays electron-balanced
on whichever base it is applied to.
"""
from __future__ import annotations

import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# AD-safe half-order (sqrt) terms: x * (x + eps)^(-1/2) ~= sqrt(x), finite at 0.
# eps=1e-2 (rather than 1e-3) softens the steep slope at low concentration so the
# stiff-solver Jacobian / adjoint stays well-conditioned during gradient-based fits.
HALFORDER_EXPR = {
    "ho_sumS_half": "[sumS] * ([sumS] + 1.0e-2) ** (0.0 - 0.5)",
    "ho_XS0_half":  "[X_S0] * ([X_S0] + 1.0e-2) ** (0.0 - 0.5)",
    "ho_SNO_half":  "[S_NO] * ([S_NO] + 1.0e-2) ** (0.0 - 0.5)",
}


def rxn(net, name):
    for r in net["reactions"]:
        if r["name"] == name:
            return r
    raise KeyError(name)


def apply_halforder(net):
    net["expressions"].update(HALFORDER_EXPR)
    rxn(net, "sulfide_oxidation_anoxic_biofilm")["rate"] = \
        "k_sII_anox_f * ho_sumS_half * ho_SNO_half * {A_V}"
    rxn(net, "elemental_S_oxidation_anoxic_biofilm")["rate"] = \
        "k_s0_anox_f * ho_XS0_half * ho_SNO_half * {A_V}"


def apply_directsulfate(net):
    # Sulfide is oxidized straight to sulfate in one step, bypassing the
    # elemental-sulfur intermediate. The nitrate demand is the sum of the two
    # steps' coefficients, taken from the base itself so the one-step reaction
    # carries the same total electron acceptance (and stays electron-balanced).
    s1 = rxn(net, "sulfide_oxidation_anoxic_biofilm")["stoichiometry"]
    s2 = rxn(net, "elemental_S_oxidation_anoxic_biofilm")["stoichiometry"]
    total_no = s1["S_NO"] + s2["S_NO"]
    rxn(net, "sulfide_oxidation_anoxic_biofilm")["stoichiometry"] = \
        {"sumS": -1, "S_SO4": 1, "S_NO": total_no}


def apply_srbsubstrate(net):
    # Sulfate / elemental-S reduction consume readily-biodegradable substrate
    # (S_B) rather than VFA, so VFA accumulates as a fermentation product.
    net["reactions"] = [r for r in net["reactions"]
                        if r["name"] not in ("sulfate_reduction_VFA_biofilm",
                                             "elemental_S_reduction_VFA_biofilm")]
    net["reactions"] += [
        {"name": "sulfate_reduction_SB_biofilm",
         "description": "S_B-driven sulfate reduction to sulfide, biofilm.",
         "rate": "k_h2s_acid * monod([S_B], k_srb) * monod([S_SO4], k_so4) * no_gate * {A_V}",
         "stoichiometry": {"S_B": -2, "sumS": 1, "S_SO4": -1}},
        {"name": "elemental_S_reduction_SB_biofilm",
         "description": "S_B-driven reduction of elemental sulfur to sulfide, biofilm.",
         "rate": "k_s0_acid * monod([S_B], k_srb) * monod([X_S0], K_S0) * no_gate * {A_V}",
         # X_S0 (COD 1.5) -> sulfide (COD 2) needs 0.5 gCOD of donor per gS.
         "stoichiometry": {"S_B": -0.5, "sumS": 1, "X_S0": -1}},
    ]


def apply_stopatS0(net):
    # The nitrate-driven oxidation stops at elemental sulfur: the second step
    # (S0 -> sulfate) is removed, so the dosing pathway produces no sulfate. A
    # falsification test of whether the measured sulfate rise requires that step.
    net["reactions"] = [r for r in net["reactions"]
                        if r["name"] != "elemental_S_oxidation_anoxic_biofilm"]


VARIANT_FNS = {
    "halforder": apply_halforder,
    "directsulfate": apply_directsulfate,
    "srbsubstrate": apply_srbsubstrate,
    "stopatS0": apply_stopatS0,
}
VARIANT_DESC = {
    "halforder": ("Half-order (square-root) kinetics for the nitrate-driven biofilm "
                  "sulfur-oxidation reactions, in place of the Monod terms."),
    "directsulfate": ("Nitrate-driven sulfide oxidation proceeds directly to sulfate "
                      "in one step, bypassing the elemental-sulfur intermediate."),
    "srbsubstrate": ("Sulfate and elemental-sulfur reduction consume readily-"
                     "biodegradable substrate (S_B) rather than VFA."),
    "stopatS0": ("The nitrate-driven oxidation stops at elemental sulfur: the "
                 "second step, elemental sulfur to sulfate, is removed, so the "
                 "dosing pathway produces no sulfate."),
    "combined": "All of the base's structural changes applied together.",
}

# Per base: which single-change variants to emit. The combined variant applies
# all of them. The balanced base already makes the srbsubstrate change, so it is
# omitted there.
BASES = {
    "wats_sewer_khalil_paper": ["halforder", "directsulfate", "srbsubstrate"],
    "wats_sewer_khalil_paper_balanced": ["halforder", "directsulfate"],
}
# Standalone variants emitted per base but NOT folded into 'combined' (they are
# mutually exclusive with the combined changes -- stopatS0 contradicts
# directsulfate, which routes sulfide straight to sulfate).
STANDALONE = {
    "wats_sewer_khalil_paper": ["stopatS0"],
    "wats_sewer_khalil_paper_balanced": ["stopatS0"],
}


def build(base_name, variant_keys, standalone_keys=()):
    base = yaml.safe_load(open(os.path.join(HERE, base_name + ".yaml")))
    # single-change variants, a combined variant applying all of them, and any
    # standalone variants (emitted on their own, not folded into 'combined')
    for key in list(variant_keys) + ["combined"] + list(standalone_keys):
        net = copy.deepcopy(base)
        net["network"]["name"] = f"{base_name}_{key}"
        net["network"]["description"] = (
            f"Structural variant of {base_name}. {VARIANT_DESC[key]} All other "
            f"reactions, parameters, species and conditions are identical to the "
            f"base model. Auto-generated by _make_khalil_variants.py.")
        fns = ([VARIANT_FNS[k] for k in variant_keys] if key == "combined"
               else [VARIANT_FNS[key]])
        for fn in fns:
            fn(net)
        out = os.path.join(HERE, f"{base_name}_{key}.yaml")
        with open(out, "w") as f:
            f.write(f"# Auto-generated from {base_name}.yaml by "
                    "_make_khalil_variants.py -- do not edit by hand.\n")
            yaml.safe_dump(net, f, sort_keys=False, default_flow_style=False, width=100)
        print(f"wrote {os.path.basename(out)}  ({len(net['reactions'])} reactions)")


def main():
    for base_name, variant_keys in BASES.items():
        build(base_name, variant_keys, STANDALONE.get(base_name, ()))


if __name__ == "__main__":
    main()
