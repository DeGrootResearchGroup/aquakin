`Plant(name, time_unit="d")` sets the plant's clock: `t_span`, flows (m3 per
that unit) and every unit parameter with a time in it are in that unit.
Everything that has its own time unit is converted to it before the solve:
each kinetic model's rates (so a model in seconds can run beside one in
days, with its parameters left in their own units), each influent's times
and flows, and the constants built into the UV reactor, the primary
clarifier and the Takács settler's default settling velocities, which
previously required a plant in days. The BSM1, BSM2 and A²O builders use
days, and `InfluentSeries.in_time_unit(...)` moves a series onto another
clock.
