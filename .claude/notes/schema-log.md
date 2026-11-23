# Schema: measurements moved out of the rule

Moved from `.claude/rules/schema.md` on 2026-11-23 during the #824 condensing.

- **`asm1` temperature correction.** The six BSM temperature-dependent rate
  constants (`muH`, `muA`, `bH`, `bA`, `ka`, `kh`, `ref_T = 293.15 K`) carry
  thetas derived from the standard BSM 15 °C / 10 °C parameter pairs.
  Nitrification, the most temperature-sensitive process, drops to about 36% of
  its 20 °C rate at 10 °C; at the default 20 °C the model is identical to the
  uncorrected one.
- **`asm1` prior widths** (Hauduc 2011 PhD thesis, Table 3-1, per-parameter
  reported CV): most are narrow (`muH` 6%, `bH` 2%); the half-saturations `KS` /
  `KNO` and autotroph decay `bA` are the widest (50–80%).
- **Why the Mo et al. 2023 ADM1 ranges were rejected for the `adm1` priors**:
  their spans reflect substrate, e.g. `k_hyd_pr` in [0.0014, 10] and
  `K_I_h2_c4` down to 5e-8 (grass silage, Koch 2010). Mo's own remark that
  hydrolysis varies considerably with the substrate supports using the
  within-context Brun CV instead.
- **`extends:` size saving**: `asm1_ammonia_limitation` (`asm1` plus the
  `KNH_H` nutrient switch) is about 30 lines as an `extends` file against a
  200-line full copy.
