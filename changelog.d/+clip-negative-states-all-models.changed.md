Every shipped model now evaluates its reaction rates with negative
concentrations treated as zero (`clip_negative_states: true`). A stiff solve
can step a nearly depleted species slightly below zero; in `adm1`, `asm3`,
`ozone_bromate`, `uv_h2o2` and the `wats_sewer*` family a rate could then turn
slightly negative, running the reaction backwards. Those rates now take their
value at zero, as the ASM1, ASM2d and precipitation models already did.
Solutions where every concentration stays non-negative are unchanged; the BSM2
digester and whole-plant steady states are bit-identical.
