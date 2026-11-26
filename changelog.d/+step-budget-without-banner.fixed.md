A reactor or plant solve that runs out of integrator steps raises aquakin's
`RuntimeError` naming the remedies without first printing a JAX callback
traceback and Equinox's banner to the log. This covers every solve: the batch,
plug-flow, particle-track, CFD and biofilm reactors and their
`solve_sensitivity`, located events, and the plant's forward, `forward_fast`,
`solve_sensitivity` and steady-state solves. Under `jax.grad` or `jax.jit` the
solve fails as before.
