The membrane bioreactor (`MBRUnit`) and sequencing batch reactor (`SBRUnit`)
now apply a model's positivity limiter, as the CSTR, IFAS/MBBR tank and
digester already did. They computed the raw reaction term, so for a model
that declares `positivity_limiter:` (ASM2d, for example) a nearly
exhausted species such as oxygen or nitrate kept its full consumption rate and
could go negative. Results change only for such models, and only where a
species is near zero. `plant.mass_balance()` now integrates the same limited
reaction term the reactors do.
