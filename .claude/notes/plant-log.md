# Plant: measurements moved out of the rule

Moved from `.claude/rules/plant.md` on 2026-11-23 during the #824 condensing.

## Vectorized stream reconstruction

`Plant.stream` / `plant.outputs_at` reconstruct the inter-unit streams from the
saved states (the plant integrates unit states only). The whole output sweep is
reconstructed in one `jax.vmap` pass over the saved times, each time's
`_resolve_streams` sweep batched into one XLA program, instead of a Python loop
of per-step sweeps. Measured effect: evaluating a 609-day hourly BSM2 dynamic run
dropped from about 20 minutes to a few seconds. `evaluate_bsm2`'s
digester-feed-temperature and closed-loop kLa histories are vmapped the same way.

## Primary clarifier inflow filter (`t_m`)

`PrimaryClarifier` with `t_m > 0` computes its HRT from a first-order filtered
inflow (one trailing state seeded at `initial_flow`) while the flow split stays
on the instantaneous inflow; `build_bsm2` sets `t_m = 0.125 d`,
`initial_flow = Q_ref`. On the 609-day BSM2 dynamic influent, running without
the filter put the average effluent S_NH 2.8% above the MATLAB benchmark
implementation; with the filter the gap was 0.1% (#646).
