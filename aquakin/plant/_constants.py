"""Shared ASM1 particulate / TSS constants for the plant units.

Single source of truth for the ASM1 particulate-species sets that the
clarifiers (settling) and the effluent metrics (TSS) would otherwise each
hardcode. The two sets are deliberately different:

- :data:`ASM1_SETTLING_SPECIES` are the particulates that *settle* in a
  secondary clarifier, including ``XND`` (organic nitrogen attached to the
  settling ``XS``).
- :data:`ASM1_TSS_SPECIES` are the particulates that contribute to *total
  suspended solids*. ``XND`` is excluded: it is nitrogen carried on ``XS``,
  not a separate solid (Copp 2002, ``TSS = 0.75 * (XS + XI + XBH + XBA + XP)``).
"""

from __future__ import annotations

# ASM1 particulates that settle in a secondary clarifier (includes XND).
ASM1_SETTLING_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP", "XND")

# ASM1 particulates that contribute to TSS (excludes XND), and the COD->TSS
# conversion factor (Copp 2002).
ASM1_TSS_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP")
ASM1_TSS_FACTOR: float = 0.75
