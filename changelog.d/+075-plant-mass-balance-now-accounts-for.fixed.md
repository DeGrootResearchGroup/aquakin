`plant.mass_balance()` now accounts for an IFAS/MBBR tank. Its state is a depth
profile rather than a single concentration vector, so the balance skipped the
unit outright -- its bulk liquid as well as its attached biomass -- and the
shortfall appeared as an imbalance the size of the whole tank.
