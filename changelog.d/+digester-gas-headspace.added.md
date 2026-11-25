`plant.digester_gas(solution)` now reports the biogas of a `HeadspaceDigesterUnit`,
not only of an ADM1 digester: the flow normalized to the vent pressure, the CH₄,
CO₂ and H₂ partial pressures and the methane mass flow, in the BSM2 convention
and per day whatever the plant's time unit. So `methane_production()` and the
biogas line of `plant.mass_balance()` work for a digester on any model that
declares its gases.
