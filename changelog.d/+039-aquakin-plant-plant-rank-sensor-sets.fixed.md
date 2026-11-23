`aquakin.plant.plant_rank_sensor_sets` ranks candidate sensor sets on a plant
by a design criterion of the Fisher information each adds to the sensors
already in place. It is the plant version of `rank_experiments`. A candidate
is a `SensorSet`: its observables, with an absolute or relative `sigma` and
an optional `mask`. Every candidate is scored from one shared sensitivity
solve, and `param_draws=` averages the criterion over uncertain parameters.
