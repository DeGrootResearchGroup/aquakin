`direct_n2o_emission` now reports kg N/d whatever the plant's time unit, as its
documentation says and as `carbon_footprint` labels it. On a plant whose clock
is not days it returned kg N per unit of that clock (on an hourly plant, 24 times
too small). Plants in days, such as the BSM1 and BSM2 builders, are unchanged.
