`build_a2o(model=...)` runs on a model that extends `asm2d` and keeps its
species names, such as `asm2d_chemp`: the clarifier now settles the model's own
particulates instead of a fixed `asm2d` list, which raised on `XMeOH`. A model
with other species names (`asm2d_tud`, `asm3_biop`) is refused with an error
naming the species it lacks, as is `ferric=` on a model without `XMeOH`.
