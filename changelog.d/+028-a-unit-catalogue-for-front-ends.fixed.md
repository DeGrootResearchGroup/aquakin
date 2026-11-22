A unit catalogue for front ends, `aquakin.plant.catalog`. It describes each
plant unit type as plain JSON-compatible data: its settings, with types,
defaults and the help text from the unit's docstring; how its ports follow
from those settings; and which settings must be given one at a time. Settings
are read from each unit's constructor, so new arguments appear without
editing the catalogue. `check_unit(kind, name, settings)` builds the real unit
and reports its ports or the constructor's error.
