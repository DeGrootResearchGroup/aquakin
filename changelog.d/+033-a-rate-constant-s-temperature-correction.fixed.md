A rate constant's temperature correction can name a model parameter as its
`theta` (`temperature: {theta: theta_A, ref_T: 293.15}`) instead of a fixed
number. The temperature sensitivity is then part of the parameter vector, so
`calibrate` can estimate it and `sensitivity`, `fisher_information`,
`identifiability`, `dgsm` and Monte Carlo include it, with no recompile
needed to sweep it. Several rate constants may share one `theta` parameter.
Models with numeric `theta` values behave exactly as before.
