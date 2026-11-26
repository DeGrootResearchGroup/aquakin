`asm2d`, `asm2d_tud`, `asm3` and `asm3_biop` (and `asm2d_chemp`, which extends
`asm2d`) have realistic default concentrations instead of 1.0 for every species,
so `model.default_concentrations()` is a usable starting point: `asm3` takes
the aerated mixed liquor `asm1` declares, and the bio-P models an established
bio-P mixed liquor (about 4300 g TSS/m³). A run that starts from these
defaults gives different results; pass your own initial concentrations to keep
the previous ones.
