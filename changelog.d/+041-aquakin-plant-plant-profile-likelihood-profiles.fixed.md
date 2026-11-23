`aquakin.plant.plant_profile_likelihood` profiles one plant parameter. It
re-fits the others with `Plant.calibrate` at each grid value and returns a
`ProfileResult` with the likelihood-ratio interval. The sweep is the same as
for `profile_likelihood`, and every grid point reuses one compiled
objective.
