`C_named`, `list_species` and `calibrate(free_ic=...)` now accept any unit
whose state starts with its concentrations, followed by a volume or other
scalars. This covers the filtered primary clarifier, `StorageTank`,
`MBRUnit` and `SBRUnit`.
