`build_bsm1` runs activated-sludge models other than ASM1, such as an N₂O model
that splits the autotrophs into AOB and NOB. The clarifier's settling and TSS
species are derived from the model (new `settling_species=` / `tss_species=`
arguments override them), and the new `aeration=` argument replaces the
aeration of named tanks, for example with DO control and gas stripping.
