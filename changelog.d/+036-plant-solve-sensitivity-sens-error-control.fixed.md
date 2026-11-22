`Plant.solve_sensitivity(sens_error_control=False)` chooses the steps on the
state's error alone, leaving the sensitivity out of the error test (the CVODES
`errconS = false` option; staggered corrector only). The sensitivity is still
converged in every step but its error is no longer bounded; where the
sensitivity forces smaller steps than the state it is much faster (about half
the time on a dynamic BSM2 window with 16 parameters), for screening runs or
otherwise very long ones. `dynamic_sensitivity`
and `dynamic_dgsm` pass it through in forward mode. The default (`True`) is
unchanged.
