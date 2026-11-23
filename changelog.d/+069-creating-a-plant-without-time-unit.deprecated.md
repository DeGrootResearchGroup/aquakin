Creating a `Plant` without `time_unit` gives a `FutureWarning`. The plant
then uses the time unit its models share, as before. Pass it explicitly,
e.g. `Plant("my_plant", time_unit="d")`; it will be required in a future
release.
