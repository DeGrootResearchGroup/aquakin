Under a heat-balance temperature model (`HeatBalanceTemperature`), an MBR and
an IFAS/MBBR tank now run their kinetics and aeration at their own lagged
temperature state. They used the inlet temperature, so the plant tracked a
temperature for them that their kinetics ignored. Plants on the default
algebraic temperature are unchanged.
