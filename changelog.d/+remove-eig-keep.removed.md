`CalibrationResult.predictive_band` no longer takes `eig_keep`, which was
deprecated and has had no effect: the band is always drawn from the
calibration's `posterior_cov`. A call that still passes `eig_keep=` now raises
a `TypeError`; delete the argument. `aquakin.identifiability(..., eig_keep=)` is
unchanged.
