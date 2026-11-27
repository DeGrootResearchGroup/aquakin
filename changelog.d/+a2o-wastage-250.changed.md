The A²O plant (`build_a2o`) now wastes 250 m³/d of sludge by default, up from
120 m³/d (`A2O_WASTAGE_FLOW`). At 120 m³/d the plant grew more sludge than it
wasted, so the clarifier's sludge blanket filled eight of its ten layers and
the effluent carried about 54 g TSS/m³; the sludge age was about 14.5 d rather
than the intended 25 d. At 250 m³/d the sludge age is about 11 d, the MLSS
about 4000 g TSS/m³, the blanket stays in the bottom three layers and the
effluent carries about 14 g TSS/m³. Biological phosphorus removal improves
(effluent phosphate 3.9 → 3.0 g P/m³) with nitrification unchanged (ammonia
about 0.2 g N/m³). `A2O_WARM_REACTOR_COMPOSITION` is re-derived from the new
operating point. Pass `wastage_flow=120.0` to recover the earlier design.
