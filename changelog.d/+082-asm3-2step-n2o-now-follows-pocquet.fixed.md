`asm3_2step_n2o` now follows Pocquet et al. (2016) in two places, which
changes its N₂O and ammonia-oxidation rates. The ND-pathway dissolved-oxygen
factor had an inverted constant ratio and peaked near 5.3 instead of 1,
overstating ND-pathway N₂O; it now peaks at exactly 1 at
DO = sqrt(K_O2_AOB_ND * KI_O2_AOB) (about 0.63 g O₂/m³). Ammonia oxidation
(AMO) was Monod in total ammonium; it is now Monod in free ammonia, the base
fraction of ammonium at the operating pH and temperature, so it depends on pH
and is slower at typical pH for the same affinity constant.
