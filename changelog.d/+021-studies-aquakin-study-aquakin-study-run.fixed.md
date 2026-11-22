Studies (`aquakin.study`, `aquakin study run|check|freeze|unfreeze|copy`): an
analysis of a flowsheet project saved in the project with its results: a
simulation (steady state or dynamic), a Fisher information, a calibration
against a data table, a profile likelihood, or a ranking of sensor sets. Results are
plain YAML / CSV files keyed by name. A study is reported stale when the
flowsheet, a model or an influent table it ran on changes (saying which), and
a newer aquakin is noted without making it stale. A frozen study runs on a
snapshot of its plant with every model flattened (shipped models too), so
neither project edits nor upgrades change it; unfreezing deletes its results.
A run replaces a study's results only once it succeeds: a run that fails or
is cancelled (or is killed) keeps the earlier results and notes the attempt.
Calibration and profile studies always fit the Gaussian likelihood of the
data, so they need each channel's measurement error (`sigma`, one value for
all or one per channel) or `estimate_sigma: true`; least squares, which
ignores the error and leaves uncertainties, profile intervals and priors
without a scale, remains available from Python (`Plant.calibrate`).
A study can also be set up from plain settings: `study_form()` describes
every setting, `plant_choices(spec)` lists the streams, parameters and
measurable channels a study of that flowsheet may name, and `create_study`,
`read_study` and `update_study` write and read `study.yaml`, checked as a run
reads it, with a data table or saved state copied into the study.
`simulation_study(spec)` gives the settings of a plain simulation (the plant
at steady state, saving the streams that leave it with their flow, COD, TKN,
NOx, TN and TSS), and `study_results(path)` reads a study's results back as
plain data with their units (temperature in °C; the files keep kelvin).
