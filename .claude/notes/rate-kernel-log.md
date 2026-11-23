# Vectorized rate kernel: compile measurements

Moved out of `.claude/rules/core-rate-functions.md` (rule audit #824). The
invariant the numbers support stays in the rule: keep the batched kernel,
because the rate jaxpr dominates the compile of a differentiated stiff solve.

- Share of a differentiated stiff-solve compile taken by the rate jaxpr:
  60% (asm1), 85% (adm1), 94% (wats_sewer_extended).
- Rate jaxpr reduction from the kernel: asm1 2.9x, adm1 2.2x, asm2d 10x,
  wats 5.6x.
- End-to-end differentiated-compile speedup: ~1.15x (asm1), 1.5x (adm1),
  3.4x (wats). Runtime unchanged.
