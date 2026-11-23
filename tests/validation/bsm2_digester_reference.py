"""The published BSM2 anaerobic-digester operating point, shared by the digester
validation tests.

The IWA Benchmark Simulation Model No. 2 (BSM2) specifies a continuously fed
mesophilic (35 degC) anaerobic digester: a 3400 m3 liquid reactor with a 300 m3
headspace, fed a constant particulate-rich sludge at 178.47 m3/d (hydraulic
retention time ~19 d).

The feed (``INFLUENT``) and the steady state (``REFERENCE_SS``) are the exact
published figures: the "ADM1 influent (post ASM2ADM interface)" and "ADM1
effluent (prior ADM2ASM interface)" tables of the BSM2 open-loop steady-state
report (``Results/BSM2_steady_state.pdf`` in the official BSM2 distribution).
Units are kgCOD/m3 except S_IC (kmolC/m3), S_IN (kmolN/m3), S_cat / S_an
(kmol/m3) and the headspace gases (kgCOD/m3, S_gas_co2 kmolC/m3).

References
----------
Rosen, C. & Jeppsson, U. (2006). Aspects on ADM1 Implementation within the BSM2
Framework. Dept. IEA, Lund University.
Gernaey, K.V. et al. (2014). Benchmarking of Control Strategies for Wastewater
Treatment Plants. IWA Scientific and Technical Report No. 23.
"""

import numpy as np

Q_FEED = 178.4674  # m3/d -- benchmark digester feed flow
V_LIQ = 3400.0  # m3
T_OP = 308.15  # K (35 degC)

# The feed is dominated by particulate protein / lipid / carbohydrate and inert
# solids; chloride (S_an) is the only strong ion fed.
INFLUENT = {
    "S_aa": 0.04388,
    "S_IC": 0.0079326,
    "S_IN": 0.0019721,
    "S_I": 0.028067,
    "X_ch": 3.7236,
    "X_pr": 15.9235,
    "X_li": 8.047,
    "X_I": 17.0106,
    "S_an": 0.0052101,
}

REFERENCE_SS = {
    "S_su": 0.012394, "S_aa": 0.0055432, "S_fa": 0.10741, "S_va": 0.012333,
    "S_bu": 0.014003, "S_pro": 0.017584, "S_ac": 0.089315, "S_h2": 2.5055e-7,
    "S_ch4": 0.05549, "S_IC": 0.095149, "S_IN": 0.094468, "S_I": 0.13087,
    "X_c": 0.10792, "X_ch": 0.020517, "X_pr": 0.08422, "X_li": 0.043629,
    "X_su": 0.31222, "X_aa": 0.93167, "X_fa": 0.33839, "X_c4": 0.33577,
    "X_pro": 0.10112, "X_ac": 0.67724, "X_h2": 0.28484, "X_I": 17.2162,
    "S_gas_h2": 1.1032e-5, "S_gas_ch4": 1.6535, "S_gas_co2": 0.01354,
    "S_cat": 0.0, "S_an": 0.0052101,
}

GAS = ("S_gas_h2", "S_gas_ch4", "S_gas_co2")


def state_vector(model, values):
    """A full ``model`` state vector holding ``values`` (name -> value), zero elsewhere."""
    C = np.zeros(model.n_species)
    for name, val in values.items():
        C[model.species_index[name]] = val
    return C


def perturbed_start(model):
    """``REFERENCE_SS`` with every state alternately halved and raised by half.

    Starting the steady-state run here, not at the reference itself, makes
    "settles onto the published steady state" a claim about where the dynamics
    go: from this start the 600-day run converges to the same state as from
    ``REFERENCE_SS`` to about 1e-11.
    """
    factor = np.where(np.arange(model.n_species) % 2 == 0, 0.5, 1.5)
    return state_vector(model, REFERENCE_SS) * factor


def worst_relative_error(model, C):
    """The largest relative deviation of ``C`` from ``REFERENCE_SS`` over the
    states the reference reports above 1e-4 (the ~zero pools S_h2, S_gas_h2 and
    S_cat are on an absolute scale), as ``(error, species name)``."""
    worst, worst_name = 0.0, ""
    for name, ref in REFERENCE_SS.items():
        if ref < 1e-4:
            continue
        rel = abs(float(C[model.species_index[name]]) - ref) / ref
        if rel > worst:
            worst, worst_name = rel, name
    return worst, worst_name
