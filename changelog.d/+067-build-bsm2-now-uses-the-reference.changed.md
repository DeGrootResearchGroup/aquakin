`build_bsm2` now uses the reference filter (`t_m=0.125` d, seeded at
`Q_ref`). Steady states are unchanged. Dynamic BSM2 results change: on the
609-day influent the average effluent ammonia was 2.8% above the reference
without the filter and is 0.1% above it with the filter. The primary's state
is now one entry longer, so a hand-built `initial_state(overrides=...)` entry
for `"primary"` needs the filtered flow appended.
