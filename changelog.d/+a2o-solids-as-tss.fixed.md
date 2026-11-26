The A²O plant now counts its suspended solids as TSS. Its influent
(`a2o_influent`) and warm start (`a2o_warm_start`) carried no ASM2d `XTSS`, so
the reactors' `XTSS` stayed near 20 g/m³ beside some 5000 g/m³ of solids; both
now set it from their particulates with the model's `iTSS` factors (193.5 g
TSS/m³ in the influent). The Takács settler of `build_a2o`, whose settling
velocity depends on the solids concentration, added COD, phosphorus and `XTSS`
together; it now weighs each particulate by its `iTSS` factor. `plant.sludge_age`
on an ASM2d or ASM3 plant counted only the ASM1-named particulates in the tanks
and streams and now weighs every solid the same way. A²O results with the
Takács settler change: the steady effluent phosphate moves from 4.10 to 3.89
g P/m³ and the effluent suspended solids from 48 to 54 g TSS/m³, which the
effluent `XTSS` now reports (it read 0.2 g/m³). Results with
`use_takacs=False` are unchanged apart from `XTSS`.
