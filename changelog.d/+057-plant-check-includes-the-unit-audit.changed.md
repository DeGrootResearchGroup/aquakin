`plant.check()` includes the unit audit: `PlantCheck.ok` is now also false
when `Plant.check_units` finds an error, and `check(raise_on_error=True)`
raises `UnitsError` for one.
