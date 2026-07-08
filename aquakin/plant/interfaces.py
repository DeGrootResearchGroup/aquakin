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
reference BSM2 implementation and never arises for an anoxic feed. Construct
``ASM1toADM1(strict=True)`` to instead raise (jit/AD-safe) when the demand is not
fully absorbed, asserting the feed stays in the intended regime. They are
pure, AD-clean functions of the concentration vector — the nested ``if/else`` N
cascades of the reference C are written here as branch-free greedy draws
(``jnp.minimum``), which are mathematically identical to the unrolled
conditionals.

The charge balances (inorganic carbon in ``asm2adm``, alkalinity in ``adm2asm``)
are evaluated at the **digester pH**, as in BSM2. Inside a plant the
digester's instantaneous, state-derived pH is fed in via
``translate(..., digester_pH=...)``: ``asm2adm`` (whose destination is the
digester) reads it from the destination unit, ``adm2asm`` (whose source is the
digester) from the source unit, both wired by ``Plant._collect_inputs`` through
the ``needs_dest_pH`` / ``needs_src_pH`` flags. The ``pH_adm`` parameter (default
7.0) is only the fallback for a standalone ``translate`` call with no plant to
supply the pH.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


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
    source_model : CompiledModel
        ASM1 model (13 states).
    target_model : CompiledModel
        ADM1 model (BSM2 form, 26 liquid + 3 gas states).
    pH_adm : float
        Digester pH used in the inorganic-carbon / charge balance (default 7.0).
    T_op : float
        Operating temperature (K) of the digester/interface (default 308.15).
    """

    source_model: CompiledModel
    target_model: CompiledModel
    pH_adm: float = 7.0
    T_op: float = 308.15
    # The inorganic-carbon charge balance is evaluated at the digester pH. When
    # the plant can supply the digester's instantaneous (state-derived) pH it is
    # fed in via ``translate(..., digester_pH=...)``; ``pH_adm`` is the fallback
    # for a standalone call. This flag tells the plant to read the pH from the
    # destination (digester) unit's state each step.
    needs_dest_pH: bool = True

    # When True, raise at runtime (via ``eqx.error_if``, so it is jit/AD-safe) if
    # the electron-acceptor (O2 + NO3) COD demand exceeds the degradable COD it
    # draws from -- the pathological regime where the surplus is silently dropped
    # and COD is over-conserved (see the module docstring). Default False
    # reproduces the reference BSM2 behaviour exactly; opt in to assert that a feed
    # stays within the interface's intended near-anoxic regime.
    strict: bool = False
    strict_tol: float = 1e-6

    # Interface stoichiometric parameters (BSM2 defaults).
    CODequiv: float = 40.0 / 14.0
    fnaa: float = _N_aa * 14.0  # 0.098  gN/gCOD in amino acids / Xpr
    fnxc: float = _N_xc * 14.0  # 0.0376 gN/gCOD in composites
    fnbac: float = _N_bac * 14.0  # 0.08   gN/gCOD in biomass
    fxni: float = _N_I * 14.0  # 0.06   gN/gCOD in particulate inerts
    fsni: float = 0.0  # gN/gCOD in SI (ASM1: 0)
    fsni_adm: float = _N_I * 14.0  # 0.06   gN/gCOD in SI (ADM1)
    frlixs: float = 0.7  # lipid fraction of non-N XS
    frlibac: float = 0.4  # lipid fraction of non-N biomass
    frxs_adm: float = 0.68  # anaerobically degradable fraction of biomass
    fdegrade_adm: float = 0.0  # AS XI/XP degradable in AD (BSM2: 0)

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
        si = self.source_model.species_index
        self._si = si
        # Temperature-corrected pK's (van't Hoff), precomputed constants.
        factor = (1.0 / self.T_base - 1.0 / self.T_op) / (100.0 * self.R)
        self._pK_a_co2 = self.pK_a_co2_base - math.log10(math.exp(7646.0 * factor))
        self._pK_a_IN = self.pK_a_IN_base - math.log10(math.exp(51965.0 * factor))
        self._pK_w = self.pK_w_base - math.log10(math.exp(55900.0 * factor))
        # pH-independent charge-contribution factors read by translate. The
        # pH-dependent CO2 / inorganic-N / VFA factors are recomputed locally in
        # translate at the digester's instantaneous pH, so they do not live here.
        self._alfa_NH = 1.0 / 14000.0
        self._alfa_alk = -0.001
        self._alfa_NO = -1.0 / 14000.0
        # Target indices for assembling the ADM1 output vector by name.
        self._ti = self.target_model.species_index

    def _remove_electron_acceptor_demand(self, SO, SNO, SS, XS, XBH, XBA, SNH):
        """Strip the O2 + NO3 electron-acceptor COD demand hierarchically from the
        degradable pools (SS, then XS, XBH, XBA).

        Returns ``(ut_SS, ut_XS, ut_XBH, ut_XBA, ut_SNH)`` -- the residual pools,
        with ``ut_SNH`` the influent ammonia augmented by the nitrogen released
        from the consumed biomass.
        """
        demand = SO + self.CODequiv * SNO
        taken = jnp.minimum(demand, SS)
        ut_SS = SS - taken
        demand = demand - taken
        taken = jnp.minimum(demand, XS)
        ut_XS = XS - taken
        demand = demand - taken
        taken = jnp.minimum(demand, XBH)
        ut_XBH = XBH - taken
        n_released = taken * self.fnbac
        demand = demand - taken
        taken = jnp.minimum(demand, XBA)
        ut_XBA = XBA - taken
        n_released = n_released + taken * self.fnbac
        demand = demand - taken
        ut_SNH = SNH + n_released  # N released from consumed biomass
        if self.strict:
            # `demand` > 0 here means the electron-acceptor demand outran every
            # degradable COD pool, so the surplus is about to be dropped (COD
            # over-conserved). Flag it; AD-/jit-safe, and ut_SNH flows to the
            # output so the check is not eliminated.
            ut_SNH = eqx.error_if(
                ut_SNH,
                demand > self.strict_tol,
                "ASM1toADM1(strict=True): electron-acceptor (O2+NO3) COD demand "
                "exceeds the degradable COD (SS+XS+XB_H+XB_A); the surplus is "
                "dropped and total COD is over-conserved. This is a non-anoxic / "
                "nitrate-heavy feed outside the interface's intended regime.",
            )
        return ut_SS, ut_XS, ut_XBH, ut_XBA, ut_SNH

    def _inorganic_carbon_charge(self, SNO, SNH, SALK, S_IN, pH):
        """Inorganic-carbon and strong-ion charge balance at the digester ``pH``.

        The only pH-dependent part of the mapping; everything else is a
        pH-independent COD/N partition. Returns ``(S_IC, S_cat, S_an)``.
        """
        alfa_co2 = -1.0 / (1.0 + 10 ** (self._pK_a_co2 - pH))
        alfa_IN = (10 ** (self._pK_a_IN - pH)) / (1.0 + 10 ** (self._pK_a_IN - pH))
        S_IC = (
            (SNO * self._alfa_NO + SNH * self._alfa_NH + SALK * self._alfa_alk) - (S_IN * alfa_IN)
        ) / alfa_co2
        ScatminusSan = S_IN * alfa_IN + S_IC * alfa_co2 + 10 ** (-self._pK_w + pH) - 10 ** (-pH)
        S_cat = jnp.maximum(ScatminusSan, 0.0)
        S_an = jnp.maximum(-ScatminusSan, 0.0)
        return S_IC, S_cat, S_an

    def translate(self, C_source: jnp.ndarray, digester_pH=None) -> jnp.ndarray:
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
            self.fnaa,
            self.fnbac,
            self.fxni,
            self.fsni,
            self.fsni_adm,
        )
        frlixs, frlibac, frxs_adm = self.frlixs, self.frlibac, self.frxs_adm

        # --- 1) Remove COD demand of O2 + NO3, hierarchically SS, XS, XBH, XBA.
        ut_SS, ut_XS, ut_XBH, ut_XBA, ut_SNH = self._remove_electron_acceptor_demand(
            SO, SNO, SS, XS, XBH, XBA, SNH
        )
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
        biomass_nobio = biomass * (1.0 - frxs_adm)  # -> ADM XI
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
        taken = jnp.minimum(ut_SI, ut_SND / fsni_adm)
        inertS = inertS + taken
        ut_SI = ut_SI - taken
        ut_SND = ut_SND - taken * fsni_adm
        taken = jnp.minimum(ut_SI, ut_XND / fsni_adm)
        inertS = inertS + taken
        ut_SI = ut_SI - taken
        ut_XND = ut_XND - taken * fsni_adm
        taken = jnp.minimum(ut_SI, ut_SNH / fsni_adm)
        inertS = inertS + taken
        ut_SI = ut_SI - taken
        ut_SNH = ut_SNH - taken * fsni_adm
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

        # Charge balance for inorganic carbon, evaluated at the digester pH. BSM2
        # feeds the digester's own pH into this balance, so ``digester_pH`` (the
        # digester's instantaneous state-derived pH, supplied by the plant) is
        # used when available; the fixed ``pH_adm`` is the standalone fallback.
        # (VFA outputs are zero here.)
        pH = self.pH_adm if digester_pH is None else digester_pH
        S_IC, S_cat, S_an = self._inorganic_carbon_charge(SNO, SNH, SALK, S_IN, pH)

        ti = self._ti
        out = jnp.zeros((self.target_model.n_species,))
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
    source_model : CompiledModel
        ADM1 model (BSM2 form).
    target_model : CompiledModel
        ASM1 model.
    pH_adm, T_op : float
        Digester pH and operating temperature for the charge balance.
    """

    source_model: CompiledModel
    target_model: CompiledModel
    pH_adm: float = 7.0
    T_op: float = 308.15
    # The alkalinity (SALK) charge balance is evaluated at the digester pH. The
    # source of this map IS the digester, so the plant feeds the source unit's
    # state-derived pH via ``translate(..., digester_pH=...)``; ``pH_adm`` is the
    # standalone fallback.
    needs_src_pH: bool = True

    fnaa: float = _N_aa * 14.0
    fnxc: float = _N_xc * 14.0
    fnbac: float = _N_bac * 14.0
    fxni: float = _N_I * 14.0
    fsni: float = 0.0
    fsni_adm: float = _N_I * 14.0
    frxs_as: float = 0.79  # aerobically degradable fraction of AD biomass
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
        self._si = self.source_model.species_index
        self._ti = self.target_model.species_index
        factor = (1.0 / self.T_base - 1.0 / self.T_op) / (100.0 * self.R)
        self._pK_a_co2 = self.pK_a_co2_base - math.log10(math.exp(7646.0 * factor))
        self._pK_a_IN = self.pK_a_IN_base - math.log10(math.exp(51965.0 * factor))
        # pH-independent charge-contribution factors read by translate. The
        # pH-dependent CO2 / inorganic-N / VFA factors are recomputed locally in
        # translate at the digester's instantaneous pH, so they do not live here.
        self._alfa_NH = 1.0 / 14000.0
        self._alfa_alk = -0.001

    def translate(self, C_source: jnp.ndarray, digester_pH=None) -> jnp.ndarray:
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
        XStemp = biomass - XPtemp  # = biomass*frxs_as
        S_IN_adj = (
            S_IN + biomass * fnbac / 14000.0 - XPtemp * fxni / 14000.0 - XStemp * fnxc / 14000.0
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

        # Alkalinity charge balance, evaluated at the digester pH (fed back from
        # the digester state when available, else the fixed pH_adm). These are
        # the only pH-dependent terms; everything above is a pH-independent
        # COD/N partition.
        pH = self.pH_adm if digester_pH is None else digester_pH
        alfa_va = 1.0 / 208.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_va_base - pH)))
        alfa_bu = 1.0 / 160.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_bu_base - pH)))
        alfa_pro = 1.0 / 112.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_pro_base - pH)))
        alfa_ac = 1.0 / 64.0 * (-1.0 / (1.0 + 10 ** (self.pK_a_ac_base - pH)))
        alfa_co2 = -1.0 / (1.0 + 10 ** (self._pK_a_co2 - pH))
        alfa_IN = (10 ** (self._pK_a_IN - pH)) / (1.0 + 10 ** (self._pK_a_IN - pH))
        SALK = (
            S_va * alfa_va
            + S_bu * alfa_bu
            + S_pro * alfa_pro
            + S_ac * alfa_ac
            + S_IC * alfa_co2
            + S_IN * alfa_IN
            - SNH * self._alfa_NH
        ) / self._alfa_alk

        ti = self._ti
        out = jnp.zeros((self.target_model.n_species,))
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
