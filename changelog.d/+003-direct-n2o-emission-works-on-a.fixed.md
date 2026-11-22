`direct_n2o_emission` works on a plant whose model names dissolved oxygen
other than `SO`, such as the shipped `asm3_2step_n2o` (`SO2`); it raised
`KeyError: 'SO'`. It, and the aeration and mixing energy terms of the BSM
evaluations, now read each tank's oxygen kLa from the species its `Aeration`
names. Aerated units (CSTR, IFAS/MBBR, MBR) gain the
public readers `oxygen_kla()`, `kla_vector()`, `saturation_vector` and
`stripped_gases`.
