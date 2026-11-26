The sewer nitrate-dosing models are no longer shipped: `wats_sewer_extended`,
its structural variants and `wats_sewer_extended_v0`, the
`wats_sewer_khalil_paper*` family (including the depth-resolved biofilm
variants) and `wats_sewer_khalil_thesis`. They were built for one study and
now live with it. `aquakin.load_model` no longer finds them; to keep using one,
load a copy of its YAML with `aquakin.load_model_from_file`. The reference WATS
model, `wats_sewer`, is still shipped.
