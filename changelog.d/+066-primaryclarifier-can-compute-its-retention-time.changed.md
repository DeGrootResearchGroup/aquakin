`PrimaryClarifier` can compute its retention time from a first-order filtered
inflow, as the BSM2 reference does: pass `t_m` (the filter time constant, in
days) and `initial_flow`. The filtered flow becomes one extra state entry
after the concentrations, and only the particulate removal uses it; the
underflow is still `f_PS` times the instantaneous inflow. The default
`t_m=0` keeps the old behaviour and state layout.
