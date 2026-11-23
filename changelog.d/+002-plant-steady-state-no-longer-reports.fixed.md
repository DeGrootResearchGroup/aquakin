`Plant.steady_state` no longer reports a loose forward fallback as converged.
When pseudo-transient continuation did not converge, the forward run it fell
back on stopped on `run_to_steady_state`'s default norm test and ignored the
`tol` and `scale_floor` you passed, so a result with `method="ptc->forward"`
could say `converged=True` with a scaled residual orders of magnitude above
`tol` (about 2e-3 against `tol=1e-8` on BSM1). The fallback now stops on the
same scaled-residual test with your `tol` and `scale_floor` (still
overridable through `fallback_kwargs`), and reports `converged` only when the
returned residual meets that tolerance.
