`build_bsm1` on a model that tracks its suspended solids as a state (the ASM2d
and ASM3 families, such as `asm3_2step_n2o`) weighs the settler's solids by the
model's own TSS factors (`iTSS_*` / `iSS_*`), as `build_a2o` does, instead of
ASM1's 0.75 g TSS per g COD with stored polyphosphate and precipitates at 0.
Such a plant's steady state moves slightly: on the default BSM1 configuration
the effluent TSS falls by 4–7 % and the sludge age rises by 2–3 %. Pass
`tss_species=` to keep the ASM1 weighting. ASM1 plants are unchanged.
