`Plant.solve_sensitivity` now uses the staggered corrector by default (see
above). Its results agree with the previous solve to the solver tolerance. It
is forward-only: it needs concrete `params` and `y0`, and differentiating
through it raises an error that points to `corrector="simultaneous"`.
`factormax=None` now means the staggered controller's own growth cap of 2.
