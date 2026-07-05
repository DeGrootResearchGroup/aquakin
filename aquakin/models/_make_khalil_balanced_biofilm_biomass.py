"""Generate wats_sewer_khalil_paper_balanced_biofilm_biomass.yaml: a depth-
resolved biofilm variant in which biomass is an explicit, per-layer, growing/
decaying STATE rather than a uniform lumped activity condition.

Motivation
----------
The first depth-resolved variant (wats_sewer_khalil_paper_balanced_biofilm)
keeps the lumped model's areal device: biofilm reactions carry a single
spatially-uniform activity factor ``eps*{X_BF}*{A_V}`` (or ``{A_V}``), identical
in every layer. That cannot represent a biomass *gradient* -- and the central
mechanism for a sewer biofilm (fermenters resident deep where there is no
electron acceptor, heterotroph/denitrifier activity high near the surface where
nitrate penetrates) is precisely a biomass gradient (Jiang et al. 2009 Figs 6,
8, 9; Sun et al. 2014). With a uniform activity factor every layer ferments and
denitrifies in lockstep, so depth resolution adds only solute penetration, not
the production/consumption *separation* that protects an intermediate such as
acetate. This variant makes biomass a real per-layer state so that gradient can
exist.

Transform (this increment: the heterotroph X_BH)
------------------------------------------------
Starting from wats_sewer_khalil_paper_balanced (the well-mixed model), where
heterotroph reactions exist in two forms -- a bulk-suspended form on the local
biomass ``[X_BH]`` and a biofilm form on the areal lump ``{A_V}`` -- we:

  1. drop the four ``*_biofilm`` heterotroph GROWTH duplicates (the ``[X_BH]``
     bulk twins are retained and run in every compartment);
  2. rewrite hydrolysis_fast/slow and fermentation from the composite
     ``bio_hf = [X_BH] + eps*{X_BF}*{A_V}`` to the local ``[X_BH]``;
  3. rewrite the remaining biofilm processes (sulfate / elemental-S reduction,
     methanogenesis, nitrate-driven and aerobic sulfide / elemental-S oxidation)
     from ``{A_V}`` to the local ``[X_BH]``. These are an INTERIM coupling --
     their rate is scaled by the local biofilm biomass pending their own
     functional-group biomass states (X_SRB, methanogens, S-oxidizers), to be
     added in a later increment;
  4. remove the now-unused ``A_V`` / ``X_BF`` conditions and the ``bio_hf``
     expression.

Every reaction then runs on a LOCAL, volumetric, per-layer biomass; there is no
bulk/biofilm phase split (the biomass concentration itself -- low in the bulk,
high in the biofilm layers -- carries the distinction). Run it in
:class:`aquakin.BiofilmReactor` with ``biofilm_reactions=None`` (single phase,
all reactions everywhere), a stratified initial state (high ``X_BH`` in the
layers, low in the bulk), and ``fixed_mask`` holding only the truly inert solids
(``X_I``); ``X_BH`` then grows/decays and the gradient evolves.

Scale note (IMPORTANT): this conversion does NOT preserve the original rate
magnitudes, and the rate constants MUST be re-calibrated before any quantitative
use. The lumped biofilm activity was ``eps*{X_BF}*{A_V}`` with the base defaults
``eps=0.15, X_BF=10, A_V=56.7`` (so ~85 for hydrolysis/fermentation; ~56.7 for
the bare-``{A_V}`` sulfur/methane terms), whereas a biofilm-density biomass seed
is O(1e3) gCOD/m^3 -- so the converted ``[X_BH]``-scaled rates are inflated by
~1-2 orders of magnitude. Only the PRODUCT (areal rate constant x biofilm biomass
density) is grounded by the lumped calibration, not the split; the biomass
density and the rate constant are confounded. Treat absolute concentrations from
this variant as uncalibrated and rely only on qualitative penetration /
stratification behaviour until it is re-calibrated against the data.

Run from this directory:  python _make_khalil_balanced_biofilm_biomass.py
"""

from __future__ import annotations

import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "wats_sewer_khalil_paper_balanced.yaml")
OUT = os.path.join(HERE, "wats_sewer_khalil_paper_balanced_biofilm_biomass.yaml")

# Heterotroph growth biofilm-duplicates to drop (their [X_BH] bulk twins remain).
DROP = {
    "anox_growth_SB_biofilm",
    "anox_growth_VFA_biofilm",
    "aer_growth_SB_biofilm",
    "aer_growth_VFA_biofilm",
}
LOCAL_BIOMASS = "[X_BH]"


def _strip_phase_suffix(name: str) -> str:
    for suf in ("_bulk", "_biofilm"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def main():
    net = yaml.safe_load(open(BASE))
    net["model"]["name"] = "wats_sewer_khalil_paper_balanced_biofilm_biomass"
    net["model"]["description"] = (
        "Depth-resolved biofilm variant of wats_sewer_khalil_paper_balanced in "
        "which biomass is an explicit per-layer growing/decaying state rather "
        "than a uniform lumped activity factor. The heterotroph X_BH drives "
        "every biofilm process via the LOCAL volumetric biomass [X_BH] (no {A_V} "
        "/ {X_BF}); the sulfur and methane processes are interim-coupled to "
        "[X_BH] pending their own functional-group biomass (X_SRB, methanogens, "
        "S-oxidizers). Run with aquakin.BiofilmReactor (biofilm_reactions=None, "
        "single phase), a stratified initial state, and fixed_mask holding only "
        "the inert solids X_I; the biomass gradient then evolves. Generated by "
        "models/_make_khalil_balanced_biofilm_biomass.py."
    )

    out = []
    n_drop = n_av = n_hf = 0
    for rx in net["reactions"]:
        if rx["name"] in DROP:
            n_drop += 1
            continue
        rate = rx["rate"]
        if "bio_hf" in rate:
            rate = rate.replace("bio_hf", LOCAL_BIOMASS)
            n_hf += 1
        if "{A_V}" in rate:
            rate = rate.replace("{A_V}", LOCAL_BIOMASS)
            n_av += 1
        rx = dict(rx)
        rx["rate"] = rate
        rx["name"] = _strip_phase_suffix(rx["name"])
        out.append(rx)
    net["reactions"] = out

    # Drop the now-unused lumped-activity conditions; keep pH.
    net["conditions"] = [c for c in net.get("conditions", []) if c["name"] not in ("A_V", "X_BF")]
    # bio_hf is fully inlined away.
    if "expressions" in net and "bio_hf" in net["expressions"]:
        del net["expressions"]["bio_hf"]

    # Sanity: no reaction should still reference the removed factors.
    for rx in net["reactions"]:
        for tok in ("{A_V}", "{X_BF}", "bio_hf"):
            assert tok not in rx["rate"], f"{rx['name']} still references {tok}"

    names = [r["name"] for r in net["reactions"]]
    assert len(names) == len(set(names)), "duplicate reaction names after suffix strip"

    # Prune model-level parameters orphaned by the transform. Dropping the four
    # biofilm growth duplicates orphans their areal rate constants (k_12_no,
    # k_12_o2, k_sf), and inlining bio_hf orphans eps. A dead but calibratable
    # parameter is a trap (e.g. k_12_no is in the lumped model's free-rate list),
    # so remove any model-level parameter no longer referenced by a rate, a
    # string stoichiometry coefficient, or an expression.
    referenced = set()
    for rx in net["reactions"]:
        referenced |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rx["rate"]))
        for v in rx.get("stoichiometry", {}).values():
            if isinstance(v, str):
                referenced |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", v))
    for v in (net.get("expressions") or {}).values():
        referenced |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", v))
    n_pruned = 0
    if isinstance(net.get("parameters"), dict):
        for pname in [p for p in net["parameters"] if p not in referenced]:
            del net["parameters"][pname]
            n_pruned += 1

    with open(OUT, "w") as f:
        f.write(
            "# Auto-generated from wats_sewer_khalil_paper_balanced.yaml by "
            "_make_khalil_balanced_biofilm_biomass.py -- do not edit by hand.\n"
        )
        yaml.safe_dump(net, f, sort_keys=False, default_flow_style=False, width=100)

    print(
        f"wrote {os.path.basename(OUT)} "
        f"({len(net['reactions'])} reactions, {len(net['species'])} species; "
        f"dropped {n_drop} growth duplicates, converted {n_av} {{A_V}} + "
        f"{n_hf} bio_hf rates to {LOCAL_BIOMASS}, pruned {n_pruned} dead params)"
    )
    biomass_rxns = [r["name"] for r in net["reactions"] if "[X_BH]" in r["rate"]]
    print(f"  reactions on [X_BH] ({len(biomass_rxns)}): {biomass_rxns}")


if __name__ == "__main__":
    main()
