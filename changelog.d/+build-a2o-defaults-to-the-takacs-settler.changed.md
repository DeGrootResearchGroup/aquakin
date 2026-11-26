`build_a2o` now builds the ten-layer Takács secondary clarifier by default
(`use_takacs=True`), as `build_bsm1` and `build_bsm2` do, instead of the
stateless `IdealClarifier`. A²O results change (the effluent and the reactor
states move, since the settler holds solids of its own) and the plant state is
longer (the settler's layer states), so a `y0` saved from the old default does
not fit it; a flowsheet file keeps the clarifier it was saved with. Solves take
longer. Pass `build_a2o(use_takacs=False)` for the previous ideal clarifier.
