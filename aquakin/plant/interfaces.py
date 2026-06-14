"""ASM1 <-> ADM1 state interfaces for the BSM2 anaerobic digester loop.

Two :class:`~aquakin.plant.translators.StateTranslator` implementations that
convert a concentration vector between the activated-sludge model (ASM1, 13
states) and the anaerobic digestion model (ADM1, 26 liquid states), following
the continuity-based interfaces of Nopens et al. (2009) / Rosen & Jeppsson
(2006) as distributed with BSM2:

- :class:`ASM1toADM1` (``asm2adm``) — sludge fed to the digester. COD held by
  electron acceptors (O2, NO3) is removed first, then the remaining ASM COD is
  partitioned into ADM substrates (sugars, amino acids, proteins, carbohydrates,
  lipids, particulate inerts) under a nitrogen budget drawn greedily from a
  priority-ordered list of N pools. Inorganic carbon (``S_IC``) and the strong
  ion difference (``S_cat``/``S_an``) come from a charge balance at the digester
  pH.
- :class:`ADM1toASM1` (``adm2asm``) — digester effluent returned to the water
  line.

Both conserve total nitrogen, and conserve total COD whenever the ``asm2adm``
electron-acceptor (O₂ + NO₃) demand does not exceed the degradable COD it draws
from (SS + XS + XB_H + XB_A) — always true for a real, near-anoxic digester feed
(the only intended use), where SO ≈ SNO ≈ 0 and the demand is ≈ 0. In the
pathological case of a demand larger than the degradable COD (e.g. recycled
nitrate far exceeding the substrate), the leftover demand is dropped rather than
carried as a negative pool, so COD is then over-conserved; this mirrors the
reference BSM2 implementation and never arises for an anoxic feed. They are
pure, AD-clean functions of the concentration vector — the nested ``if/else`` N
cascades of the reference C are written here as branch-free greedy draws
(``jnp.minimum``), which are mathematically identical to the unrolled
conditionals.

The digester pH used by the charge balance is a fixed parameter
(``pH_adm``, default 7.0). In the full benchmark it is fed back from the
digester; for the open-loop steady state the digester's own charge-balance
speciation solver sets the actual pH from the ``S_IC``/``S_cat``/``S_an``/VFA
state, so a representative fixed value here is sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


# ADM1 nitrogen contents (kmolN/kgCOD) and derived gN/gCOD fractions, BSM2.
_N_xc = 0.0376 / 14.0
_N_I = 0.06 / 14.0
_N_aa = 0.007
_N_bac = 0.08 / 14.0


@dataclass
class ASM1toADM1:
    """ASM1 -> ADM1 interface (``asm2adm``).

    Parameters
    ----------
    source_network : CompiledNetwork
        ASM1 network (13 states).
    target_network : CompiledNetwork
        ADM1 network (BSM2 form, 26 liquid + 3 gas states).
    pH_adm : float
        Digester pH used in the inorganic-carbon / charge balance (default 7.0).
    T_op : float
        Operating temperature (K) of the digester/interface (default 308.15).
    """

    source_network: "CompiledNetwork"
    target_network: "CompiledNetwork"
    pH_adm: float = 7.0
    T_op: float = 308.15

    # Interface stoichiometric parameters (BSM2 defaults).
    CODequiv: float = 40.0 / 14.0
    fnaa: float = _N_aa * 14.0          # 0.098  gN/gCOD in amino acids / Xpr
    fnxc: float = _N_xc * 14.0          # 0.0376 gN/gCOD in composites
    fnbac: float = _N_bac * 14.0        # 0.08   gN/gCOD in biomass
    fxni: float = _N_I * 14.0           # 0.06   gN/gCOD in particulate inerts
    fsni: float = 0.0                   # gN/gCOD in SI (ASM1: 0)
    fsni_adm: float = _N_I * 14.0       # 0.06   gN/gCOD in SI (ADM1)
    frlixs: float = 0.7                 # lipid fraction of non-N XS
    frlibac: float = 0.4                # lipid fraction of non-N biomass
    frxs_adm: float = 0.68              # anaerobically degradable fraction of biomass
    fdegrade_adm: float = 0.0           # AS XI/XP degradable in AD (BSM2: 0)

    R: float = 0.083145
    T_base: float = 298.15
    pK_w_base: float = 14.0
    pK_a_va_base: float = 4.86
    pK_a_bu_base: float = 4.82
    pK_a_pro_base: float = 4.88
    pK_a_ac_base: float = 4.76
    pK_a_co2_base: float = 6.35
    pK_a_IN_base: float = 9.25

    def __post_init__(self) -> None:
        if self.fdegrade_adm != 0.0:
            raise NotImplementedError(
                "ASM1toADM1 only implements fdegrade_adm = 0 (the BSM2 value); "
                "the biodegradable-inert composite cascade is not ported."
            )
        si = self.source_network.species_index
        self._si = si
        # Temperature-corrected pK's (van't Hoff), precomputed constants.
        factor = (1.0 / self.T_base - 1.0 / self.T_op) / (100.0 * self.R)
        self._pK_a_co2 = self.pK_a_co2_base - math.log10(math.exp(7646.0 * factor))
        self._pK_a_IN = self.pK_a_IN_base - math.log10(math.exp(51965.0 * factor))
        self._pK_w = self.pK_w_base - math.log10(math.exp(55900.0 * factor))
        # Charge-contribution factors at the digester pH.
        pH = self.pH_adm
        self._alfa_va = 1.0 / 208.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_va_base - pH)))
        self._alfa_bu = 1.0 / 160.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_bu_base - pH)))
        self._alfa_pro = 1.0 / 112.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_pro_base - pH)))
        self._alfa_ac = 1.0 / 64.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_ac_base - pH)))
        self._alfa_co2 = -1.0 / (1.0 + 10 ** (self._pK_a_co2 - pH))
        self._alfa_IN = (10 ** (self._pK_a_IN - pH)) / (1.0 + 10 ** (self._pK_a_IN - pH))
        self._alfa_NH = 1.0 / 14000.0
        self._alfa_alk = -0.001
        self._alfa_NO = -1.0 / 14000.0
        # Target indices for assembling the ADM1 output vector by name.
        self._ti = self.target_network.species_index

    def translate(self, C_source: jnp.ndarray) -> jnp.ndarray:
        si = self._si
        # ASM1 inputs (gCOD/m3 or gN/m3).
        SI = C_source[si["SI"]]
        SS = C_source[si["SS"]]
        XI = C_source[si["XI"]]
        XS = C_source[si["XS"]]
        XBH = C_source[si["XB_H"]]
        XBA = C_source[si["XB_A"]]
        XP = C_source[si["XP"]]
        SO = C_source[si["SO"]]
        SNO = C_source[si["SNO"]]
        SNH = C_source[si["SNH"]]
        SND = C_source[si["SND"]]
        XND = C_source[si["XND"]]
        SALK = C_source[si["SALK"]]

        fnaa, fnbac, fxni, fsni, fsni_adm = (
            self.fnaa, self.fnbac, self.fxni, self.fsni, self.fsni_adm)
        frlixs, frlibac, frxs_adm = self.frlixs, self.frlibac, self.frxs_adm

        # --- 1) Remove COD demand of O2 + NO3, hierarchically SS, XS, XBH, XBA.
        d = SO + self.CODequiv * SNO
        take = jnp.minimum(d, SS);  ut_SS = SS - take;  d = d - take
        take = jnp.minimum(d, XS);  ut_XS = XS - take;  d = d - take
        take = jnp.minimum(d, XBH); ut_XBH = XBH - take; nrel = take * fnbac; d = d - take
        take = jnp.minimum(d, XBA); ut_XBA = XBA - take; nrel = nrel + take * fnbac; d = d - take
        ut_SNH = SNH + nrel  # N released from consumed biomass

        ut_SND = SND
        ut_XND = XND

        # --- 2) SS -> amino acids (Saa), limited by SND-N.
        saa = jnp.minimum(ut_SS, ut_SND / fnaa)
        ut_SS = ut_SS - saa
        ut_SND = ut_SND - saa * fnaa

        # --- 3) XS -> proteins (Xpr), limited by XND-N; remainder -> Xli/Xch.
        xpr1 = jnp.minimum(ut_XS, ut_XND / fnaa)
        rem = ut_XS - xpr1
        xli1 = frlixs * rem
        xch1 = (1.0 - frlixs) * rem
        ut_XND = ut_XND - xpr1 * fnaa
        ut_XS = jnp.zeros(())

        # --- 4) Biomass (XBH+XBA) -> Xpr + XI; remainder -> Xli/Xch.
        biomass = ut_XBH + ut_XBA
        biomass_nobio = biomass * (1.0 - frxs_adm)          # -> ADM XI
        biomass_bioN = biomass * fnbac - biomass_nobio * fxni
        xpr2_base = biomass_bioN / fnaa
        remCOD0 = biomass - biomass_nobio - xpr2_base
        condA = xpr2_base <= (biomass - biomass_nobio)
        # Branch A: draw extra protein-N from leftover XND.
        condB = (ut_XND / fnaa) > remCOD0
        xpr2_A = xpr2_base + jnp.where(condB, remCOD0, ut_XND / fnaa)
        remCOD_A = jnp.where(condB, 0.0, remCOD0 - ut_XND / fnaa)
        utXND_A = jnp.where(condB, ut_XND - remCOD0 * fnaa, 0.0)
        xli2_A = frlibac * remCOD_A
        xch2_A = (1.0 - frlibac) * remCOD_A
        # Branch not-A: all biomass COD is protein; surplus N back to XND.
        xpr2_nA = biomass - biomass_nobio
        utXND_nA = ut_XND + biomass * fnbac - biomass_nobio * fxni - xpr2_nA * fnaa
        xpr2 = jnp.where(condA, xpr2_A, xpr2_nA)
        xli2 = jnp.where(condA, xli2_A, 0.0)
        xch2 = jnp.where(condA, xch2_A, 0.0)
        ut_XND = jnp.where(condA, utXND_A, utXND_nA)
        ut_XBH = jnp.zeros(())
        ut_XBA = jnp.zeros(())

        # --- 5) Particulate inerts XI, XP -> ADM XI (fdegrade_adm = 0).
        # XI and XP are untouched by the COD-demand removal, so use originals.
        inertX = XI + XP

        # --- 6) ASM SI -> ADM SI inert; N drawn from SND, XND, SNH; remainder -> sugar.
        inertS = SI * (fsni / fsni_adm)
        ut_SI = SI - SI * (fsni / fsni_adm)
        take = jnp.minimum(ut_SI, ut_SND / fsni_adm)
        inertS = inertS + take; ut_SI = ut_SI - take; ut_SND = ut_SND - take * fsni_adm
        take = jnp.minimum(ut_SI, ut_XND / fsni_adm)
        inertS = inertS + take; ut_SI = ut_SI - take; ut_XND = ut_XND - take * fsni_adm
        take = jnp.minimum(ut_SI, ut_SNH / fsni_adm)
        inertS = inertS + take; ut_SI = ut_SI - take; ut_SNH = ut_SNH - take * fsni_adm
        ut_SS = ut_SS + ut_SI  # leftover SI COD -> monosaccharides
        ut_SI = jnp.zeros(())

        # --- 7) Assemble ADM1 outputs (kgCOD/m3, kmol/m3).
        S_su = ut_SS / 1000.0
        S_aa = saa / 1000.0
        S_IN = (ut_SNH + ut_SND + ut_XND) / 14000.0
        S_I = inertS / 1000.0
        X_ch = (xch1 + xch2) / 1000.0
        X_pr = (xpr1 + xpr2) / 1000.0
        X_li = (xli1 + xli2) / 1000.0
        X_I = (biomass_nobio + inertX) / 1000.0

        # Charge balance for inorganic carbon (VFA outputs are zero here).
        S_IC = (
            (SNO * self._alfa_NO + SNH * self._alfa_NH + SALK * self._alfa_alk)
            - (S_IN * self._alfa_IN)
        ) / self._alfa_co2
        ScatminusSan = (
            S_IN * self._alfa_IN + S_IC * self._alfa_co2
            + 10 ** (-self._pK_w + self.pH_adm) - 10 ** (-self.pH_adm)
        )
        S_cat = jnp.maximum(ScatminusSan, 0.0)
        S_an = jnp.maximum(-ScatminusSan, 0.0)

        ti = self._ti
        out = jnp.zeros((self.target_network.n_species,))
        out = out.at[ti["S_su"]].set(S_su)
        out = out.at[ti["S_aa"]].set(S_aa)
        out = out.at[ti["S_IC"]].set(S_IC)
        out = out.at[ti["S_IN"]].set(S_IN)
        out = out.at[ti["S_I"]].set(S_I)
        out = out.at[ti["X_ch"]].set(X_ch)
        out = out.at[ti["X_pr"]].set(X_pr)
        out = out.at[ti["X_li"]].set(X_li)
        out = out.at[ti["X_I"]].set(X_I)
        out = out.at[ti["S_cat"]].set(S_cat)
        out = out.at[ti["S_an"]].set(S_an)
        return out


@dataclass
class ADM1toASM1:
    """ADM1 -> ASM1 interface (``adm2asm``).

    Maps the digester effluent back to the water line. ADM biomass becomes
    slowly-biodegradable ``XS`` (degradable fraction ``frxs_as``) plus inert
    ``XP``; the soluble organics collapse to ``SS`` (H2 and CH4 are stripped to
    gas and so leave the COD balance); ADM inerts map to ``XI``/``SI``;
    inorganic N becomes ``SNH``; and ``SALK`` comes from a charge balance at the
    digester pH. Nitrogen is conserved; COD is conserved apart from the stripped
    ``S_h2`` + ``S_ch4``.

    Parameters
    ----------
    source_network : CompiledNetwork
        ADM1 network (BSM2 form).
    target_network : CompiledNetwork
        ASM1 network.
    pH_adm, T_op : float
        Digester pH and operating temperature for the charge balance.
    """

    source_network: "CompiledNetwork"
    target_network: "CompiledNetwork"
    pH_adm: float = 7.0
    T_op: float = 308.15

    fnaa: float = _N_aa * 14.0
    fnxc: float = _N_xc * 14.0
    fnbac: float = _N_bac * 14.0
    fxni: float = _N_I * 14.0
    fsni: float = 0.0
    fsni_adm: float = _N_I * 14.0
    frxs_as: float = 0.79               # aerobically degradable fraction of AD biomass
    fdegrade_as: float = 0.0

    R: float = 0.083145
    T_base: float = 298.15
    pK_w_base: float = 14.0
    pK_a_va_base: float = 4.86
    pK_a_bu_base: float = 4.82
    pK_a_pro_base: float = 4.88
    pK_a_ac_base: float = 4.76
    pK_a_co2_base: float = 6.35
    pK_a_IN_base: float = 9.25

    def __post_init__(self) -> None:
        if self.fdegrade_as != 0.0:
            raise NotImplementedError(
                "ADM1toASM1 only implements fdegrade_as = 0 (the BSM2 value)."
            )
        self._si = self.source_network.species_index
        self._ti = self.target_network.species_index
        factor = (1.0 / self.T_base - 1.0 / self.T_op) / (100.0 * self.R)
        self._pK_a_co2 = self.pK_a_co2_base - math.log10(math.exp(7646.0 * factor))
        self._pK_a_IN = self.pK_a_IN_base - math.log10(math.exp(51965.0 * factor))
        pH = self.pH_adm
        self._alfa_va = 1.0 / 208.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_va_base - pH)))
        self._alfa_bu = 1.0 / 160.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_bu_base - pH)))
        self._alfa_pro = 1.0 / 112.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_pro_base - pH)))
        self._alfa_ac = 1.0 / 64.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_ac_base - pH)))
        self._alfa_co2 = -1.0 / (1.0 + 10 ** (self._pK_a_co2 - pH))
        self._alfa_IN = (10 ** (self._pK_a_IN - pH)) / (1.0 + 10 ** (self._pK_a_IN - pH))
        self._alfa_NH = 1.0 / 14000.0
        self._alfa_alk = -0.001
        self._alfa_NO = -1.0 / 14000.0

    def translate(self, C_source: jnp.ndarray) -> jnp.ndarray:
        si = self._si
        g = lambda name: C_source[si[name]]
        S_su, S_aa, S_fa = g("S_su"), g("S_aa"), g("S_fa")
        S_va, S_bu, S_pro, S_ac = g("S_va"), g("S_bu"), g("S_pro"), g("S_ac")
        S_IC, S_IN, S_I = g("S_IC"), g("S_IN"), g("S_I")
        X_c, X_ch, X_pr, X_li = g("X_c"), g("X_ch"), g("X_pr"), g("X_li")
        X_su, X_aa, X_fa = g("X_su"), g("X_aa"), g("X_fa")
        X_c4, X_pro, X_ac, X_h2 = g("X_c4"), g("X_pro"), g("X_ac"), g("X_h2")
        X_I = g("X_I")

        fnaa, fnxc, fnbac, fxni = self.fnaa, self.fnxc, self.fnbac, self.fxni
        fsni, fsni_adm = self.fsni, self.fsni_adm

        # Biomass -> XS (degradable) + XP (inert); N bookkeeping to S_IN.
        biomass = 1000.0 * (X_su + X_aa + X_fa + X_c4 + X_pro + X_ac + X_h2)  # gCOD/m3
        XPtemp = biomass * (1.0 - self.frxs_as)
        XStemp = biomass - XPtemp                                            # = biomass*frxs_as
        S_IN_adj = (
            S_IN + biomass * fnbac / 14000.0
            - XPtemp * fxni / 14000.0 - XStemp * fnxc / 14000.0
        )

        # Inert XI (AD) -> XI (ASM); inert SI (AD) -> SI (ASM), N freed to S_IN.
        inertX = X_I * 1000.0
        inertS = S_I
        S_IN_adj = S_IN_adj + S_I * (fsni_adm - fsni) / 14.0

        # ASM outputs (gCOD/m3, gN/m3). Biomass/SO/SNO are zero.
        SI = inertS * 1000.0
        XS = (X_c + X_ch + X_pr + X_li) * 1000.0 + XStemp
        XP = XPtemp
        XI = inertX
        SS = (S_su + S_aa + S_fa + S_va + S_bu + S_pro + S_ac) * 1000.0  # S_h2/S_ch4 stripped
        SND = fnaa * 1000.0 * S_aa
        XND = fnxc * XStemp + fnxc * 1000.0 * X_c + fnaa * 1000.0 * X_pr
        SNH = S_IN_adj * 14000.0

        SALK = (
            S_va * self._alfa_va + S_bu * self._alfa_bu + S_pro * self._alfa_pro
            + S_ac * self._alfa_ac + S_IC * self._alfa_co2 + S_IN * self._alfa_IN
            - SNH * self._alfa_NH
        ) / self._alfa_alk

        ti = self._ti
        out = jnp.zeros((self.target_network.n_species,))
        out = out.at[ti["SI"]].set(SI)
        out = out.at[ti["SS"]].set(SS)
        out = out.at[ti["XI"]].set(XI)
        out = out.at[ti["XS"]].set(XS)
        out = out.at[ti["XP"]].set(XP)
        out = out.at[ti["SNH"]].set(SNH)
        out = out.at[ti["SND"]].set(SND)
        out = out.at[ti["XND"]].set(XND)
        out = out.at[ti["SALK"]].set(SALK)
        return out
