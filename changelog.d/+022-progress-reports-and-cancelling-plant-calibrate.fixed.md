Progress reports and cancelling: `Plant.calibrate`,
`plant_profile_likelihood`, `plant_rank_sensor_sets` and
`plant_fisher_information` take `progress=`, a callable that receives an
`aquakin.Progress` at each objective evaluation, profile point or run, and
can raise `aquakin.Cancelled` to stop the analysis.
