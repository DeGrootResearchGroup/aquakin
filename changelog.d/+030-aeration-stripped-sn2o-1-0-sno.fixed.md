`Aeration(stripped={"SN2O": 1.0, "SNO": 1.0})` strips dissolved gases other
than oxygen in an aerated tank, inside the solve. Each gas transfers at
`ratio × kLa_O2 × (sat − C)`, with the saturation 0 by default, and follows the
controller's kLa under DO control. It works in CSTR, IFAS/MBBR, and MBR tanks.
`direct_n2o_emission` then reports the N₂O the solve removed, using each
tank's own ratio, and `plant.mass_balance` counts the stripped gases in its gas
term. `stripped_n2o` accepts a per-reactor ratio and saturation.
