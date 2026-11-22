The layered secondary clarifier (`TakacsClarifier`, and the settler of
`build_bsm1(use_takacs=True)` / `build_bsm2`) now settles solids between its
layers with the Godunov flux of the settling conservation law (Bürger, Diehl &
Nopens 2011) instead of the Takács minimum-flux rule of the BSM benchmark
settlers. This is a deliberate departure from the benchmark. The two rules
differ only where an upper layer is denser than the one below it; there the
Takács rule switches between the two layers' fluxes whenever their order
flips, which made the sludge blanket pulse and forced small steps. The BSM2
steady state is unchanged, and on the dynamic BSM2 influent the effluent's
mean concentrations differ by under 0.1%, while a forward solve took about a
third of the steps and wall time and the default diffrax solve about a fifth
of the steps. `interface_flux="takacs"` (`settler_interface_flux="takacs"` on
the builders) reproduces the benchmark settler exactly.
