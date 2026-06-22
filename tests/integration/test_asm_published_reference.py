"""Literature-anchored correctness gate for the bio-P / nitrification ASM models.

The ASM2d / ASM2d-TUD / ASM3 / ASM3-BioP networks were imported from a vendor
tool and carried six latent bugs (a heterotroph-decay biomass swap, autotroph
and PAO Monod terms collapsed onto the heterotroph half-saturation constants,
and dropped chemical-precipitation metals). Every one of them **conserves COD /
N / P**, so the continuity suite (``test_asm_continuity.py``) passed while
nitrification and bio-P were broken -- conservation cannot catch a wrong
half-saturation constant or a swapped biomass.

This module is the regression gate for exactly those classes of error. It is
anchored to the **published** models, not to any vendor implementation:

    Henze, M., Gujer, W., Mino, T. & van Loosdrecht, M. (2000). Activated Sludge
    Models ASM1, ASM2, ASM2d and ASM3. IWA Scientific & Technical Report No. 9.
    (ASM2d: Henze et al. 1999, Wat. Sci. Tech. 39(1), 165-182; ASM3: Gujer et
    al. 1999, Wat. Sci. Tech. 39(1), 183-193.)

Two of the three checks are deliberately **unit-basis-agnostic**, so they hold
regardless of the mol/L-vs-mol/m3 conventions a particular port happens to use:

1. ``group_kinetics`` -- a nitrifier / PAO process must *depend on* its own
   group's half-saturation constant and be *independent of* the heterotroph
   counterpart. This is tested by perturbing the parameter and watching the
   single reaction's rate, so it asserts the published *structure* of the
   kinetics (the autotroph term is the autotroph's, not a collapsed copy of the
   heterotroph's) without pinning a value in any particular unit basis.
2. ``structure`` -- the sign of each species' stoichiometric coefficient in a
   process (consumed / produced / absent). Catches the lysis biomass swap and
   the dropped precipitation metals.
3. ``value_constants`` -- a value check, used only for the constants whose
   published value is unambiguous in g/m3 (the nitrifier ammonia affinity
   ~1.0 gN/m3 and the PAO maximum poly-P ratio K_MAX = 0.34 gP/gCOD).

The vendor cross-check (``scripts/verify_sumo_asm.py``) is a separate,
spreadsheet-dependent tool run by hand; this gate needs nothing but the shipped
networks and runs in the PR fast gate.
"""

import jax.numpy as jnp
import pytest

import aquakin


# Per network: the published facts the conservation suite cannot see. See the
# module docstring for the structure and the literature anchor.
#
#   value_constants : {param: published value}                 (unambiguous g/m3)
#   group_kinetics  : [(process, [depends_on...], [independent_of...])]
#   structure       : [(process, {species: "+"|"-"|"0"})]
ASM_REFERENCE = {
    "asm2d": {
        "value_constants": {
            "KNH4_AUT": 1.0,   # nitrifier NH4 affinity, gN/m3
            "KMAX": 0.34,      # PAO max poly-P ratio, gP/gCOD
        },
        "group_kinetics": [
            # Autotroph growth uses the nitrifier half-saturations, not the
            # heterotroph ones (the collapse bug pointed it at KNH4_H / KO2_H).
            ("Aerobic_growth_of_XAUT",
             ["KO2_AUT", "KNH4_AUT", "KALK_AUT"],
             ["KNH4_H", "KO2_H", "KALK_H"]),
            # Poly-P storage carries the maximum-ratio inhibition (K_MAX); the
            # import had dropped it, capping stored poly-P far too low.
            ("Aerobic_storage_of_XPP", ["KMAX"], []),
            ("Anoxic_storage_of_XPP", ["KMAX"], []),
        ],
        "structure": [
            # Heterotroph lysis decrements XH (not XAUT -- the swapped bug).
            ("Lysis_1", {"XH": "-", "XAUT": "0"}),
            ("Lysis_2", {"XAUT": "-", "XH": "0"}),
            # Chemical precipitation consumes phosphate + metal-hydroxide and
            # forms metal-phosphate (the metals were dropped, leaving only TSS).
            ("Precipitation", {"SPO4": "-", "XMeOH": "-", "XMeP": "+"}),
            ("Redissolution", {"SPO4": "+", "XMeOH": "+", "XMeP": "-"}),
        ],
    },
    "asm3": {
        "value_constants": {"KA_NH4": 1.0},   # nitrifier NH4 affinity, gN/m3
        "group_kinetics": [
            ("Growth_of_autotrophic_nitrifying_organisms",
             ["KA_O2", "KA_NH4", "KA_ALK"],
             ["KO2", "KNH4", "KALK"]),
        ],
        "structure": [
            ("Growth_of_autotrophic_nitrifying_organisms", {"XA": "+"}),
            ("Aerobic_endogenous_respiration_of_XA", {"XA": "-"}),
        ],
    },
    "asm3_biop": {
        "value_constants": {"KNH_A": 1.0},
        "group_kinetics": [
            ("Growth_of_autotrophic_nitrifying_organisms",
             ["KO_A", "KNH_A", "KHCO_A"],
             ["KO_H", "KNH_H", "KHCO_H"]),
            ("Aerobic_storage_of_XPP", ["Kmax_PAO"], []),
            ("Anoxic_storage_of_XPP", ["Kmax_PAO"], []),
        ],
        "structure": [
            ("Growth_of_autotrophic_nitrifying_organisms", {"XA": "+"}),
        ],
    },
    "asm2d_tud": {
        "value_constants": {"KNH_A": 1.0},
        "group_kinetics": [
            ("Aerobic_growth_of_XA",
             ["KO_A", "KNH_A", "KHCO_A"],
             ["KN_H", "KO_HYD"]),
            ("Aerobic_storage_of_XPP", ["fPP_max"], []),
            ("Anoxic_storage_of_XPP", ["fPP_max"], []),
        ],
        "structure": [
            ("Lysis_1", {"XH": "-", "XA": "0"}),
            ("Lysis_2", {"XA": "-", "XH": "0"}),
        ],
    },
}


def _probe_state(net):
    """A physical mixed-liquor probe state where every process's Monod terms are
    active and sensitive. Defaults are 1.0 g/m3 for each species; the stored
    poly-P / glycogen pools are lowered relative to the PAO biomass so the
    maximum-ratio inhibition term sits in its physical (positive) range."""
    over = {}
    if {"XPP", "XPAO"} <= set(net.species_index):
        over.update(XPP=0.05, XPAO=1.0)
    if "XGLY" in net.species_index:
        over["XGLY"] = 0.05
    return net.concentrations(over)


def _reaction_rate(net, process, params, state, cond):
    ri = net.reaction_names.index(process)
    return float(net.rates(state, params, cond, 0)[ri])


def _perturbed(params, idx):
    """Return params with entry idx changed by a detectable amount (x2, or +1
    when it is zero), for the dependency probe."""
    v = float(params[idx])
    new = 2.0 * v if v != 0.0 else 1.0
    return params.at[idx].set(new)


_NETWORKS = list(ASM_REFERENCE)


@pytest.mark.parametrize("name", _NETWORKS)
def test_value_constants_match_published(name):
    """The unambiguous-basis published constants (nitrifier NH4 affinity, PAO
    max poly-P ratio) are at their literature values -- not a heterotroph value
    a collapsed term inherited."""
    net = aquakin.load_network(name)
    pv = net.parameter_values({})
    for param, expected in ASM_REFERENCE[name]["value_constants"].items():
        assert param in net.param_index, f"{name}: missing parameter {param}"
        got = float(pv[net.param_index[param]])
        assert got == pytest.approx(expected), (
            f"{name}.{param} = {got}, published {expected}")


@pytest.mark.parametrize("name", _NETWORKS)
def test_group_kinetics_use_their_own_constants(name):
    """Each nitrifier / PAO process depends on its own group's half-saturation
    constants and is independent of the heterotroph counterparts -- the
    published kinetic structure the import collapsed. Unit-basis-agnostic: it
    perturbs a parameter and checks whether the single reaction's rate moves."""
    net = aquakin.load_network(name)
    params = net.default_parameters()
    cond = net.default_conditions().fields
    state = _probe_state(net)

    for process, depends_on, independent_of in ASM_REFERENCE[name]["group_kinetics"]:
        base = _reaction_rate(net, process, params, state, cond)
        for param in depends_on:
            idx = net.param_index[param]
            moved = _reaction_rate(net, process, _perturbed(params, idx), state, cond)
            assert moved != pytest.approx(base, rel=1e-9, abs=1e-30), (
                f"{name}: {process} should depend on {param} but its rate did "
                f"not change when {param} was perturbed (collapsed/dropped term?)")
        for param in independent_of:
            idx = net.param_index[param]
            same = _reaction_rate(net, process, _perturbed(params, idx), state, cond)
            assert same == pytest.approx(base, rel=1e-9, abs=1e-30), (
                f"{name}: {process} must NOT depend on the heterotroph constant "
                f"{param}, but its rate changed when {param} was perturbed "
                f"(autotroph term collapsed onto the heterotroph value?)")


@pytest.mark.parametrize("name", _NETWORKS)
def test_structural_participation_matches_published(name):
    """The sign of each species' stoichiometric coefficient in a process matches
    the published process matrix -- catching a swapped biomass (heterotroph
    decay applied to nitrifiers) or a dropped species (precipitation metals)."""
    net = aquakin.load_network(name)
    S = net.compute_stoich(net.default_parameters())
    ridx = {n: i for i, n in enumerate(net.reaction_names)}
    sidx = net.species_index

    for process, signs in ASM_REFERENCE[name]["structure"]:
        ri = ridx[process]
        for species, sign in signs.items():
            coeff = float(S[ri, sidx[species]])
            if sign == "+":
                assert coeff > 0.0, f"{name}.{process}: {species} should be produced (>0), got {coeff}"
            elif sign == "-":
                assert coeff < 0.0, f"{name}.{process}: {species} should be consumed (<0), got {coeff}"
            else:  # "0" -- species must not participate
                assert coeff == 0.0, f"{name}.{process}: {species} should not participate, got {coeff}"
