"""Per-species composition tables (COD / N / P content) for the shipped models.

A *composition table* maps each species to its content of a conserved quantity
per unit of the species' own measure -- ``{"COD": 1.0}`` for an organic (1 g COD
per g COD), ``{"COD": -1.0}`` for dissolved oxygen (an electron acceptor, i.e.
negative COD), ``{"N": 1.0}`` for ammonia-N, and so on. These are the vectors a
conservation check dots against the stoichiometry
(:mod:`aquakin.utils.balance`) and that a results-level balance
(:meth:`aquakin.plant.Plant.mass_balance`) dots against concentrations.

The tables here are *shipped* so the engineer never hand-authors them: they read
each model's own composition parameters (``iN_BM`` / ``iN_SF`` / ``iP_*`` /
``N_bac`` ...), so a calibrated N- or P-fraction flows straight through. They are
pure content *ratios* in the species' native measure; :func:`canonical_content`
folds in the unit conversion (kg COD -> g COD, kmol N -> g N) so a balance can
sum inventories across models of different units (the ASM water line in
g/m³, the ADM digester in kg/m³ and kmol/m³) on one canonical g basis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from aquakin.core.hints import did_you_mean

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel

# species name -> {component -> content per unit of the species' own measure}
Composition = dict[str, dict[str, float]]

# NH4-referenced COD of nitrate (g COD / g N) for models without an explicit
# parameter (ASM1): 8 electrons for NH4 -> NO3, 32/7 g O2 per g N.
_ICOD_NO3 = -32.0 / 7.0  # = -4.571428...


def _p(net: CompiledModel, name: str, default: float = 0.0, params=None) -> float:
    """A model parameter value by name, or ``default`` if it has no such
    parameter (so one table serves a family with different parameter sets).

    ``params`` (a parameter vector) overrides the model defaults -- so a
    composition tracks the *calibrated / run* fractions (e.g. a BSM2 i_XB) rather
    than the YAML defaults."""
    if name in net.param_index:
        vec = net.default_parameters() if params is None else params
        return float(vec[net.param_index[name]])
    return default


# --- per-family species roles (names differ across ASM1/2d/3) ----------------
_BIOMASS = {"XB_H", "XB_A", "XH", "XPAO", "XAUT", "XA", "XAOB", "XNOB", "XAMX", "XCMX"}
_STORAGE = {"SA", "XPHA", "XSTO", "XGLY"}  # COD = 1, no N / P
_NPOOL = {"SNH", "SNH4", "SND", "XND"}  # N = 1
_PPOOL = {"SPO4", "SPO", "XPP"}  # P = 1
_OXYGEN = {"SO", "SO2"}  # COD = -1 (electron acceptor)
_NITRATE = {"SNO", "SNO3", "SNOX"}  # COD = iCOD_NO3, N = 1
# Recognised carriers of no conserved quantity (alkalinity, TSS/SS, metal
# hydroxide, the ADM inorganic-carbon / charge / gas-CO2 states): these
# legitimately map to ``{}``. A species that is neither a role carrier nor in
# these sets fell through unrecognised -- see ``_warn_unmapped``.
_ASM_NO_CONTENT = {"SALK", "XSS", "XTSS", "XMeOH", "SHCO"}
_ADM1_NO_CONTENT = {"S_IC", "S_cat", "S_an", "S_gas_co2"}


def _warn_unmapped(net: CompiledModel, unmapped: list[str]) -> None:
    """Warn that ``unmapped`` species got no role-based composition content.

    The shipped role-based table assigns an unrecognised species zero
    COD / N / P, which silently treats it as inert mass -- so a conservation
    check would validate the stoichiometry against wrong reference data. Flag it
    loudly instead of inventing content."""
    warnings.warn(
        f"Model {net.name!r}: species {sorted(unmapped)} are not recognised by "
        f"the shipped role-based composition table, so they were assigned no "
        f"COD/N/P content. A conservation check would then treat them as inert "
        f"mass and could mask a real imbalance. Declare a `composition:` block "
        f"for these species (or confirm they carry no conserved quantity).",
        stacklevel=3,
    )


def _asm_composition(
    net: CompiledModel, electron_acceptor_cod: bool = True, params=None
) -> Composition:
    """COD / N / P content for an ASM-family model, from its own composition
    parameters. Mirrors the Gujer-matrix continuity convention: organic COD
    carriers carry ``COD = 1``, oxygen ``COD = -1``, nitrate the NH4-referenced
    electron COD, and N / P from the model's ``i*`` fractions.

    With ``electron_acceptor_cod=False`` the **lab-COD** convention is used
    instead: nitrate and N₂ carry no COD (a COD test does not oxidise them), so a
    reported COD is the organic oxygen demand an analyst would measure. The
    choice does not affect a *closure* (the bookkeeping is self-consistent either
    way) -- only what the reported COD numbers mean; the electron convention is
    what makes the per-reaction stoichiometry conserve, so it is the default."""
    P = lambda name, default=0.0: _p(net, name, default, params)
    iN_BM = P("iN_BM", P("i_XB"))
    iN_SF, iN_SS = P("iN_SF"), P("iN_SS")
    # ASM1 has no per-pool inert N parameters: its particulate inert XI carries
    # the product N fraction i_XP (the ASM1 TKN convention), the same as XP.
    # XI is reaction-inert, so the stoichiometry continuity check never
    # constrains this -- but the ASM↔ADM interface converts XI, so a results
    # balance needs its real N content.
    iN_SI = P("iN_SI")
    iN_XI = P("iN_XI", P("i_XP"))
    iN_XS = P("iN_XS")
    iP_BM, iP_SF, iP_SI = P("iP_BM"), P("iP_SF"), P("iP_SI")
    iP_XI, iP_XS = P("iP_XI"), P("iP_XS")
    iXP = P("i_XP")
    icod_no3 = P("iCOD_NO3", _ICOD_NO3) if electron_acceptor_cod else 0.0
    n2_cod = (icod_no3 + P("iNO3_N2")) if electron_acceptor_cod else 0.0
    fMeP = P("fMeP_PO4_MW")

    comp: Composition = {}
    unmapped: list[str] = []
    for sp in net.species:
        c: dict[str, float] = {}
        recognized = True
        if sp in _BIOMASS:
            c = {"COD": 1.0, "N": iN_BM, "P": iP_BM}
        elif sp == "SF":
            c = {"COD": 1.0, "N": iN_SF, "P": iP_SF}
        elif sp == "SS":
            c = {"COD": 1.0, "N": iN_SS}
        elif sp == "SI":
            c = {"COD": 1.0, "N": iN_SI, "P": iP_SI}
        elif sp == "XI":
            c = {"COD": 1.0, "N": iN_XI, "P": iP_XI}
        elif sp == "XS":
            c = {"COD": 1.0, "N": iN_XS, "P": iP_XS}
        elif sp == "XP":
            c = {"COD": 1.0, "N": iXP}
        elif sp in _STORAGE:
            c = {"COD": 1.0}
        elif sp in _OXYGEN:
            c = {"COD": -1.0}
        elif sp in _NITRATE and not (sp == "SNO" and "SNO3" in net.species):
            # ``SNO`` means nitrate in the ASM1 family but NITRIC OXIDE in a
            # two-step model (which names nitrate ``SNO3``); defer the latter
            # to the dedicated nitric-oxide case below.
            c = {"COD": icod_no3, "N": 1.0}
        elif sp == "SNO2":
            # Nitrite: the 6-electron NH4-referenced electron COD, vs nitrate's
            # 8-electron value (the NO3->NO2 step is the 2-electron difference).
            no2_cod = (icod_no3 + P("iCOD_NO3NO2")) if electron_acceptor_cod else 0.0
            c = {"COD": no2_cod, "N": 1.0}
        elif sp == "SNH2OH":  # hydroxylamine: 2 e- above NH4 (icod_no3 is 8)
            c = {"COD": icod_no3 * 0.25, "N": 1.0}
        elif sp == "SNO":  # nitric oxide: 5 e- above NH4
            c = {"COD": icod_no3 * 0.625, "N": 1.0}
        elif sp == "SN2O":  # nitrous oxide: 4 e- above NH4 (per N)
            c = {"COD": icod_no3 * 0.5, "N": 1.0}
        elif sp == "SN2":
            c = {"COD": n2_cod, "N": 1.0}
        elif sp in _NPOOL:
            c = {"N": 1.0}
        elif sp in _PPOOL:
            c = {"P": 1.0}
        elif sp == "XMeP":  # precipitated phosphate
            c = {"P": 1.0 / fMeP} if fMeP else {}
        elif sp not in _ASM_NO_CONTENT:
            recognized = False
        if not recognized:
            unmapped.append(sp)
        comp[sp] = {k: v for k, v in c.items() if v != 0.0}
    if unmapped:
        _warn_unmapped(net, unmapped)
    return comp


# --- ADM1 (BSM2 form) --------------------------------------------------------
# COD carriers (all kg COD / m³): substrates, the four VFAs, H2, CH4, the soluble
# inert, the composite/particulates, the biomass, and the H2/CH4 gas headspace.
_ADM1_COD = {
    "S_su",
    "S_aa",
    "S_fa",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_h2",
    "S_ch4",
    "S_I",
    "X_c",
    "X_ch",
    "X_pr",
    "X_li",
    "X_su",
    "X_aa",
    "X_fa",
    "X_c4",
    "X_pro",
    "X_ac",
    "X_h2",
    "X_I",
    "S_gas_h2",
    "S_gas_ch4",
}
_ADM1_BIOMASS = {"X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2"}


def _adm1_composition(net: CompiledModel, params=None) -> Composition:
    """COD / N content for ADM1 (BSM2 form). N (kmol N / m³ for ``S_IN``, else
    the model's ``N_*`` fractions in kmol N / kg COD) is converted to the
    canonical g basis later; here it is in the species' native measure. ADM1
    tracks no phosphorus."""
    N_aa, N_bac = _p(net, "N_aa", params=params), _p(net, "N_bac", params=params)
    N_I, N_xc = _p(net, "N_I", params=params), _p(net, "N_xc", params=params)

    comp: Composition = {}
    unmapped: list[str] = []
    for sp in net.species:
        c: dict[str, float] = {}
        recognized = sp in _ADM1_COD
        if sp in _ADM1_COD:
            c = {"COD": 1.0}
        if sp in _ADM1_BIOMASS:
            c["N"] = N_bac
            recognized = True
        elif sp in ("S_aa", "X_pr"):
            c["N"] = N_aa
            recognized = True
        elif sp in ("S_I", "X_I"):
            c["N"] = N_I
            recognized = True
        elif sp == "X_c":
            c["N"] = N_xc
            recognized = True
        elif sp == "S_IN":
            c["N"] = 1.0
            recognized = True
        # S_IC / S_gas_co2 carry carbon only; S_cat / S_an are charge only
        if not recognized and sp not in _ADM1_NO_CONTENT:
            unmapped.append(sp)
        comp[sp] = {k: v for k, v in c.items() if v != 0.0}
    if unmapped:
        _warn_unmapped(net, unmapped)
    return comp


_BUILDERS = {
    "asm1": _asm_composition,
    "asm1_ammonia_limitation": _asm_composition,
    "asm3_2step": _asm_composition,
    "asm3_2step_n2o": _asm_composition,
    "asm3_2step_anammox": _asm_composition,
    "asm3_2step_comammox": _asm_composition,
    "asm2d": _asm_composition,
    "asm2d_tud": _asm_composition,
    "asm3": _asm_composition,
    "asm3_biop": _asm_composition,
    "adm1": _adm1_composition,
}

# Conversion of one unit of a model's content *ratio* into canonical grams of
# the component -- i.e. the gram value of the ratio's numerator unit. It is a
# property of the model's content convention, not of the per-species
# concentration unit: the currency in the species' concentration (g COD, kg COD)
# cancels with the per-currency in the content (the ASM ``g N / g COD`` and the
# ADM ``kmol N / kg COD`` are both *per the species' own COD*), so only the
# numerator's unit converts. The ASM family states grams (factor 1); ADM1 states
# kg COD (×1000) and kmol N (×14000 = 14 g/mol × 1000).
_CONTENT_FACTOR = {
    "adm1": {"COD": 1000.0, "N": 14000.0, "P": 1000.0},
}
_ASM_CONTENT_FACTOR = {"COD": 1.0, "N": 1.0, "P": 1.0}


def composition_table(
    model: CompiledModel, *, electron_acceptor_cod: bool = True, params=None
) -> Composition:
    """The shipped COD / N / P composition table for a model.

    Parameters
    ----------
    model : CompiledModel
        A shipped ASM-family model (``asm1`` / ``asm1_ammonia_limitation`` /
        ``asm2d`` / ``asm2d_tud`` / ``asm3`` / ``asm3_biop``) or ``adm1``.
    electron_acceptor_cod : bool, optional
        ASM family only. If ``True`` (default) nitrate / N₂ carry their
        NH₄-referenced electron-equivalent COD -- the convention under which the
        Gujer stoichiometry conserves COD (use this for
        :func:`aquakin.check_conservation`). If ``False`` they carry no COD
        (**lab COD**: a reported COD is then the organic oxygen demand an analyst
        measures), the right choice for reporting a results balance.
    params : array-like, optional
        A parameter vector whose composition fractions (``i_XB`` / ``iN_*`` /
        ``N_bac`` ...) override the YAML defaults, so the table tracks a
        calibrated / run-specific composition.

    Returns
    -------
    dict
        ``{species: {component: content}}`` in the species' native measure, for
        the components the model carries (``COD``, ``N``, and ``P`` where
        modelled). Species with no COD / N / P content (alkalinity, TSS,
        metal-hydroxide) map to an empty dict.

    Raises
    ------
    KeyError
        If there is no shipped table for ``model.name`` (author one by passing
        an explicit composition to the :mod:`aquakin.utils.balance` /
        :meth:`Plant.mass_balance` functions).
    """
    builder = _BUILDERS.get(model.name)
    if builder is None:
        suffix = did_you_mean(model.name, list(_BUILDERS))
        raise KeyError(
            f"No shipped composition table for model '{model.name}'. "
            f"Tables: {sorted(_BUILDERS)}.{suffix}"
        )
    if builder is _asm_composition:
        return _asm_composition(model, electron_acceptor_cod, params)
    return builder(model, params)


def canonical_content(
    model: CompiledModel,
    component: str,
    composition: Composition | None = None,
    *,
    electron_acceptor_cod: bool = True,
    params=None,
) -> np.ndarray:
    """Per-species canonical content vector for one component, shape ``(n_species,)``.

    Entry ``j`` is ``composition[species_j][component] × unit_factor`` -- the
    grams of ``component`` (COD / N / P) per cubic metre of bulk per unit of the
    species' native concentration. Dotting it with a concentration vector ``C``
    gives that volume's areal content in canonical g/m³, so inventories and
    fluxes are summable across models whose species use different units (the
    ASM water line in g/m³, the ADM digester in kg/m³ and kmol/m³).

    Parameters
    ----------
    model : CompiledModel
    component : str
        ``"COD"``, ``"N"`` or ``"P"``.
    composition : dict, optional
        Override the shipped :func:`composition_table` (e.g. a hand-authored
        table for an unshipped model).
    electron_acceptor_cod : bool, optional
        Passed to :func:`composition_table` when ``composition`` is not given
        (``False`` selects the lab-COD convention; see there).
    """
    comp = (
        composition_table(model, electron_acceptor_cod=electron_acceptor_cod, params=params)
        if composition is None
        else composition
    )
    factor = _CONTENT_FACTOR.get(model.name, _ASM_CONTENT_FACTOR).get(component, 1.0)
    vec = np.zeros(model.n_species)
    for sp, content in comp.items():
        if sp in model.species_index and component in content:
            vec[model.species_index[sp]] = content[component] * factor
    return vec
