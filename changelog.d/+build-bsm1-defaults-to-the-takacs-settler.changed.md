`build_bsm1` now builds the benchmark's ten-layer Takács secondary clarifier by
default (`use_takacs=True`), the same settler `build_bsm2` uses, instead of the
stateless `IdealClarifier`. BSM1 results change: the steady state, the dynamic
effluent, EQI / OCI and the sludge age all move, and the plant state is longer
(the settler's layer states), so a `y0` saved from the old default does not fit
it; a flowsheet file keeps the clarifier it was saved with. Solves take longer.
Pass `build_bsm1(use_takacs=False)` for the previous ideal clarifier and its
results.
