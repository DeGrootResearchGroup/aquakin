`effluent_averages`, `effluent_quality_index`, `derived_TSS` and `derived_TKN`
work on ASM2d, ASM3 and other ASM-family models, not only ASM1, where they
raised a `KeyError`. TSS comes from the model's suspended-solids factors, TKN
from its composition, and the `"SNH"` and `"SNO"` averages are the model's
ammonium and its nitrate plus nitrite. `derived_TKN` gains `params=`. ASM1
results are unchanged.
