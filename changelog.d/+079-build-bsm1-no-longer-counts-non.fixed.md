`build_bsm1` no longer counts non-COD particulates as suspended solids for a
model without a composition table. `solids_species` now reads the declared
units: only particulates in a COD mass per volume count towards TSS, so the
metal precipitates (mol/m3), stored polyphosphate (g P/m3) and the lumped
`XTSS` state of `asm2d_chemp` no longer inflate the settler's solids (#672).
