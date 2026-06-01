"""Generate wats_sewer_khalil_paper.yaml: the paper-faithful re-implementation of the
Khalil et al. (2025) sewer nitrate-dosing model, expressed as the *full* WATS
model (Hvitved-Jacobsen, Vollertsen & Nielsen 2013, Tables 9.1-9.4) plus the
paper's stated additions and modifications -- not a hand-trimmed subset.

The paper states (Sec. 2.2) the model is "based on the WATS model with
extensions". This script keeps the paper's active anoxic/anaerobic process set
(whose kinetics match the paper's Table 2 exactly and reproduce the batch
figures) and splices in, from the full-WATS parent `wats_sewer_extended.yaml`, the base
WATS sulfur-cycle pieces the paper carries but that are dormant in the air-sealed
anoxic batch: the aerobic O2-driven chemical and biological *bulk* sulfide
oxidation (pH-dependent, Table 9.4) and the aerobic biological *biofilm* sulfide
/ elemental-sulfur oxidation. So the model is provably "full WATS + paper
modifications", not a reduced re-derivation.

Paper modifications applied: half-order WATS biofilm terms -> Monod (Sec. 2.2),
so the spliced aerobic biofilm oxidation is Monod-ized; no temperature
correction (batch at 20 C, theta=1); pH supplied as a fixed operating condition
(no charge-balance solver). Every spliced aerobic-oxidation rate carries an
[S_O] factor and S_O = 0 in the batch, so the batch trajectories are unchanged;
the additions make the network the structurally complete WATS model.

Comment-preserving (ruamel round-trip) so the per-parameter provenance comments
that gen_appendix.py parses are retained. The paper-active "core" of this file is
edited in place; re-running re-splices the WATS pieces idempotently.

Run from this directory (needs ruamel.yaml):  python _make_khalil_paper.py
"""
from __future__ import annotations

import os
from ruamel.yaml import YAML

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, "wats_sewer_extended.yaml")
CORE = os.path.join(HERE, "wats_sewer_khalil_paper.yaml")

ADD_SPECIES = ["X_BA", "S_NH", "S_PO4", "S_CH4", "S_CO2", "S_I", "X_I"]
ADD_REACTIONS = [
    "sulfide_chemoxidation_bulk",
    "sulfide_biooxidation_bulk",
    "sulfide_biooxidation_aerobic_biofilm",
    "elemental_S_oxidation_aerobic_biofilm",
]
MONOD_RATE = {
    "sulfide_biooxidation_aerobic_biofilm":
        "k_sII_ox_f * monod([sumS], K_S2) * monod([S_O], k_o) * {A_V}",
    "elemental_S_oxidation_aerobic_biofilm":
        "k_s0_ox_f * monod([X_S0], K_S0) * monod([S_O], k_o) * {A_V}",
}
ADD_PARAMS = ["k_h2sc", "k_hsc", "n1", "n2", "k_sII_b_pHopt",
              "omega_sII_b", "pH_optimum", "k_sII_ox_f", "k_s0_ox_f"]
ADD_EXPRS = ["ho_sumS_n1", "ho_SO_n2"]
PH_FIXED = 7.5


def by_name(seq, name):
    for it in seq:
        if it["name"] == name:
            return it
    raise KeyError(name)


def copy_eol(dst_map, src_map, key):
    """Carry an end-of-line comment from one mapping key to another."""
    items = getattr(src_map.ca, "items", {})
    if key in items:
        dst_map.ca.items[key] = items[key]


def _value_consistent(value, constraint, kind):
    """Is ``value`` compatible with a parent ``prior``/``bounds`` declaration?

    A prior/bounds is only propagated onto a core parameter when its (paper)
    value is consistent, so deliberately different paper values are not
    penalised. Bounds: value must lie within. Range prior: value within the
    range expanded by half its width on each side (~the +-2 sigma support).
    Mean/std prior: value within 4 sigma of the mean.
    """
    value = float(value)
    if kind == "bounds":
        lo, hi = float(constraint[0]), float(constraint[1])
        return lo <= value <= hi
    if "range" in constraint:
        lo, hi = float(constraint["range"][0]), float(constraint["range"][1])
        hw = 0.5 * (hi - lo)
        return (lo - hw) <= value <= (hi + hw)
    if "mean" in constraint:
        mean = float(constraint["mean"])
        std = float(constraint.get("std", abs(mean) or 1.0))
        return abs(value - mean) <= 4.0 * std
    return True


def main():
    yaml = YAML()  # round-trip
    yaml.preserve_quotes = True
    yaml.width = 4096
    with open(CORE) as f:
        core = yaml.load(f)
    with open(PARENT) as f:
        parent = yaml.load(f)

    # Idempotency: strip what we add, leaving the paper-active core.
    core["species"] = [s for s in core["species"] if s["name"] not in ADD_SPECIES]
    core["reactions"] = [r for r in core["reactions"] if r["name"] not in ADD_REACTIONS]
    core["conditions"] = [c for c in core["conditions"] if c["name"] != "pH"]
    for p in ADD_PARAMS:
        core["parameters"].pop(p, None)
    for e in ADD_EXPRS:
        core["expressions"].pop(e, None)

    # (1) Full-WATS state species (carried; largely inert under the batch).
    for sp in ADD_SPECIES:
        core["species"].append(by_name(parent["species"], sp))

    # (2) Fixed operating pH condition for the pH-dependent bulk oxidation.
    ph = {"name": "pH",
          "description": "Fixed operating pH (no charge-balance solver; used by "
                         "the pH-dependent bulk sulfide-oxidation rates).",
          "default": PH_FIXED}
    core["conditions"].append(ph)

    # (3) Expressions + parameters the added reactions reference (with comments).
    for e in ADD_EXPRS:
        core["expressions"][e] = parent["expressions"][e]
        copy_eol(core["expressions"], parent["expressions"], e)
    for p in ADD_PARAMS:
        core["parameters"][p] = parent["parameters"][p]
        copy_eol(core["parameters"], parent["parameters"], p)

    # (4) Dormant aerobic sulfur-cycle oxidation (biofilm half-order -> Monod).
    for name in ADD_REACTIONS:
        rxn = by_name(parent["reactions"], name)
        if name in MONOD_RATE:
            rxn["rate"] = MONOD_RATE[name]
        core["reactions"].append(rxn)

    # (5) Methane is now a tracked species: let methanogenesis produce it.
    for r in core["reactions"]:
        if r["name"] in ("methanogenesis_VFA", "methanogenesis_H2"):
            r["stoichiometry"]["S_CH4"] = 1

    # (6) Propagate literature priors/bounds from the full-WATS parent onto the
    # shared core parameters that lack them, so the calibration stays physical
    # (e.g. the hydrolysis constants cannot drift far past their book ranges).
    # Guarded by value-consistency: where the paper deliberately uses a value
    # outside the parent's literature range (e.g. K_NO=2.0, k_S2-,S0=15.5 as
    # printed), the parent constraint is NOT imposed -- the paper choice stands.
    # The parent is authoritative for shared-parameter priors/bounds: strip any
    # (possibly stale) copy from the core, then re-copy from the parent when the
    # paper value is consistent. This keeps re-runs idempotent so edits to the
    # parent's priors propagate even after a prior generation baked them in.
    for name, cpar in core["parameters"].items():
        ppar = parent["parameters"].get(name)
        if ppar is None:
            continue  # core-only parameter keeps its own prior/bounds
        val = cpar.get("value")
        for key in ("prior", "bounds"):
            cpar.pop(key, None)
            if key in ppar and val is not None \
                    and _value_consistent(val, ppar[key], key):
                cpar[key] = ppar[key]

    with open(CORE, "w") as f:
        yaml.dump(core, f)
    print(f"wrote wats_sewer_khalil_paper.yaml "
          f"({len(core['reactions'])} reactions, {len(core['species'])} species, "
          f"{len(core['parameters'])} params, "
          f"conditions={[c['name'] for c in core['conditions']]})")


if __name__ == "__main__":
    main()
