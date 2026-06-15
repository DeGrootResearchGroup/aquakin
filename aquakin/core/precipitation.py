"""Mineral precipitation: saturation indices and SI-driven kinetic rates.

Implements the generalised chemical-precipitation framework of Kazadi Mbamba et
al. (2015): minerals precipitate (or dissolve) at a rate

    R = k_cryst * X_cryst * sign(sigma) * |sigma|^n

driven by the relative supersaturation of the aqueous phase,

    sigma = (IAP / Ksp)^(1/nu) - 1,   SI = log10(IAP / Ksp),

where ``IAP = prod_i (a_i)^(count_i)`` is the ion-activity product over the
mineral's constituent ions, ``Ksp`` its solubility product, ``nu = sum count_i``
the number of ions, and ``a_i`` the free-ion *activities* at the system pH. A
small non-zero ``X_cryst`` seed lets precipitation self-nucleate; ``sign(sigma)``
makes the same rate law describe dissolution when the phase is undersaturated
(``sigma < 0``).

The aqueous chemistry -- the temperature-corrected dissociation constants, the
free-ion fractions of the carbonate/phosphate/ammonia/sulfide acid-base systems,
and the Davies / Debye-Hückel activity coefficients -- is shared with the
charge-balance pH solver (:mod:`aquakin.core.ph_solver`). This module turns a
``precipitation:`` network block into a derived-condition callable that, given
the state and the system pH (supplied as a condition, e.g. by a ``speciation:``
block), exposes each mineral's saturation index ``SI_<name>`` and supersaturation
rate factor ``R_<name>`` as condition fields. A precipitation reaction then reads
``{R_<name>}`` in its rate and consumes the constituent ions / produces the solid
through ordinary stoichiometry.

NOTE: free Ca/Mg are taken as the full declared total (ion-pairing with
carbonate/phosphate is not yet subtracted -- a documented simplification of the
source aqueous model); the ionic strength for the activity coefficients is the
declared ``ionic_strength_offset`` plus the mineral ions' contribution, unless an
``ionic_strength_field`` is given (then it is read from that condition -- e.g.
the ionic strength a ``speciation:`` block solved the pH at, so the pH and the
saturation indices share one ionic strength).
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from aquakin.core.ph_solver import (
    _R_SI,
    _T_BASE,
    _log10_gamma,
    debye_huckel_A,
    equilibrium_constants,
)

# Free-ion fraction of a total acid/base system as a function of h = [H+].
# Each returns the fraction present as the fully de/protonated ion the minerals
# use: CO3^2-, PO4^3-, NH4+, S^2-.
_FRACTIONS = {
    "carbonate": lambda h, K: (K["co3_1"] * K["co3_2"])
    / (h * h + K["co3_1"] * h + K["co3_1"] * K["co3_2"]),
    "phosphate": lambda h, K: (K["po4_1"] * K["po4_2"] * K["po4_3"])
    / (h ** 3 + K["po4_1"] * h * h + K["po4_1"] * K["po4_2"] * h
       + K["po4_1"] * K["po4_2"] * K["po4_3"]),
    "ammonia": lambda h, K: h / (h + K["nh"]),
    "sulfide": lambda h, K: (K["s_1"] * K["s_2"])
    / (h * h + K["s_1"] * h + K["s_1"] * K["s_2"]),
}

# Specials computed from pH / water alone (no species total): "proton" is the
# H+ activity (10^-pH) and "hydroxide" the OH- activity (Kw / [H+]). Used by
# minerals that incorporate a proton (DCPD, OCP) or a hydroxyl (the metal
# hydroxides Fe(OH)3 / Al(OH)3, hydroxylapatite).
_PH_SPECIALS = ("proton", "hydroxide")

# An ion's ``fraction`` selects how its free activity is obtained: an acid/base
# system key (the species total times its de/protonated fraction at pH), a
# pH/water special ("proton" or "hydroxide"), or omitted -- a fully-free cation
# (the species total taken as the free ion). Single source of truth shared with
# the Pydantic schema.
VALID_PRECIP_FRACTIONS = tuple(_FRACTIONS) + _PH_SPECIALS


def build_precipitation_derived_fn(
    config: dict,
    species_index: dict[str, int],
) -> tuple[Callable, list[str], set[str]]:
    """Compile a ``precipitation:`` block into a derived-condition callable.

    Parameters
    ----------
    config : dict
        The validated ``precipitation:`` declaration (see the module docstring /
        the network schema): ``pH_field``, ``temperature_field``,
        ``temperature_units``, ``activity_model``, ``ionic_strength_offset`` and
        a list of ``minerals``, each ``{name, pKsp, order, dH_sp, ions: [...]}``
        (``pKsp`` at the reference temperature; ``dH_sp`` the enthalpy of
        dissolution in J/mol, van't Hoff-correcting ``Ksp`` with temperature --
        0, the default, leaves ``Ksp`` temperature-independent)
        with ions ``{species, molar_mass, count, charge, fraction?}``
        (``fraction`` is one of ``carbonate``/``phosphate``/``ammonia``/``sulfide``
        for an acid-base ion, ``proton`` for H+, ``hydroxide`` for OH-, or omitted
        for a free cation).
    species_index : dict[str, int]
        Map from species name to its index in the state vector.

    Returns
    -------
    (derived_fn, produced_fields, required_fields)
        ``derived_fn(C, params, condition_arrays, loc_idx) -> dict`` produces
        ``SI_<name>`` and ``R_<name>`` per mineral; ``produced_fields`` lists
        them; ``required_fields`` are the condition fields it reads (pH, T).
    """
    pH_field = config.get("pH_field", "pH")
    temp_field = config.get("temperature_field", "T")
    temp_units = config.get("temperature_units", "celsius")
    model = config.get("activity_model", "none")
    I_offset = float(config.get("ionic_strength_offset", 0.0))
    # If set, read the ionic strength from this condition field (e.g. the one a
    # speciation block produces) instead of building it from ionic_strength_offset
    # + the mineral ions -- so the activity coefficients here match the ionic
    # strength the pH was solved at. The mineral ions are NOT added on top, since
    # a shared field already reflects the full solution composition.
    ic_field = config.get("ionic_strength_field")

    # Pre-resolve each mineral to plain numbers / indices (no per-call lookups).
    # Equilibrium-mode minerals are handled by the algebraic equilibrium engine
    # (core/precipitation_equilibrium.py), not the kinetic SI/R factor here.
    minerals = []
    produced: list[str] = []
    for m in config["minerals"]:
        if m.get("mode") == "equilibrium":
            continue
        name = m["name"]
        Ksp_ref = 10.0 ** (-float(m["pKsp"]))   # at the reference temperature _T_BASE
        dH_sp = float(m.get("dH_sp", 0.0))       # enthalpy of dissolution (J/mol), van't Hoff
        order = float(m["order"])
        form = m.get("supersaturation_form", "power")
        nu = sum(int(ion["count"]) for ion in m["ions"])
        ions = []
        for ion in m["ions"]:
            frac = ion.get("fraction")
            if frac is not None and frac not in VALID_PRECIP_FRACTIONS:
                raise ValueError(
                    f"mineral {name!r} ion has unknown fraction {frac!r}; valid: "
                    f"{VALID_PRECIP_FRACTIONS} (or omit for a free cation).")
            sp = ion.get("species")
            if frac not in _PH_SPECIALS:
                if sp is None:
                    raise ValueError(
                        f"mineral {name!r} ion needs a 'species' (only the "
                        f"{_PH_SPECIALS} fractions may omit it).")
                if sp not in species_index:
                    raise KeyError(
                        f"mineral {name!r} references undeclared species {sp!r}; "
                        f"declared: {sorted(species_index)}")
            idx = species_index[sp] if sp is not None else -1
            ions.append((
                idx,
                float(ion.get("molar_mass", 1.0)),
                int(ion["count"]),
                float(ion["charge"]) ** 2,   # z^2 for the activity coefficient
                frac,
            ))
        minerals.append((name, Ksp_ref, dH_sp, order, nu, ions, form))
        produced += [f"SI_{name}", f"R_{name}"]

    def derived(C, params, condition_arrays, loc_idx) -> dict:
        T = condition_arrays[temp_field][loc_idx]
        T_kelvin = T + 273.15 if temp_units == "celsius" else T
        pH = condition_arrays[pH_field][loc_idx]
        h = jnp.power(10.0, -pH)              # H+ activity (measurable-pH basis)
        K = equilibrium_constants(T_kelvin)

        # Ionic strength + activity coefficients (Davies / Debye-Hückel) once.
        use_activity = model != "none"
        if use_activity:
            A = debye_huckel_A(T_kelvin)
            if ic_field is not None:
                # Share the ionic strength the pH solver converged at.
                I = condition_arrays[ic_field][loc_idx]
            else:
                # Self-contained: background electrolyte + the mineral ions.
                I = I_offset
                for _, _Ksp_ref, _dH, _order, _nu, ions, _form in minerals:
                    for idx, mm, count, z2, frac in ions:
                        if idx < 0:
                            continue
                        tot = jnp.maximum(C[idx], 0.0) / mm
                        free = tot * (_FRACTIONS[frac](h, K) if frac in _FRACTIONS else 1.0)
                        I = I + 0.5 * z2 * free
            sqrt_I = jnp.sqrt(jnp.maximum(I, 0.0))

        def gamma(z2):
            if not use_activity:
                return 1.0
            return jnp.power(10.0, _log10_gamma(z2, sqrt_I, I, A, model))

        # van't Hoff temperature correction of each Ksp (unity when dH_sp = 0):
        # Ksp(T) = Ksp(T_ref) * exp(dH_sp/R * (1/T_ref - 1/T)). Same form and
        # reference temperature as the dissociation constants above.
        vant_hoff = (1.0 / _T_BASE - 1.0 / T_kelvin) / _R_SI

        out = {}
        for name, Ksp_ref, dH_sp, order, nu, ions, form in minerals:
            Ksp = Ksp_ref * jnp.exp(dH_sp * vant_hoff)
            log_iap = 0.0
            for idx, mm, count, z2, frac in ions:
                if frac == "proton":               # H+ activity is h directly
                    a = h
                elif frac == "hydroxide":          # OH- activity = Kw / [H+]
                    a = K["w"] / h
                else:
                    tot = jnp.maximum(C[idx], 0.0) / mm
                    a = gamma(z2) * tot * (_FRACTIONS[frac](h, K) if frac in _FRACTIONS else 1.0)
                # guard the log against a depleted ion (a -> 0)
                log_iap = log_iap + count * jnp.log(jnp.maximum(a, 1e-300))
            si = (log_iap - jnp.log(Ksp)) / jnp.log(10.0)         # log10(IAP/Ksp)
            if form == "bounded":
                # Thermodynamic driver bounded in (-1, 1): tanh(SI/(2 nu) ln10) =
                # (Omega^(1/nu) - 1)/(Omega^(1/nu) + 1). R -> +-1 far from
                # saturation (so the rate Jacobian stays ~k, differentiable) and
                # is 0 at SI = 0 (the same equilibrium as the power law).
                r_factor = jnp.tanh(si * jnp.log(10.0) / (2.0 * nu))
            else:
                ratio = jnp.power(10.0, si)
                sigma = jnp.power(ratio, 1.0 / nu) - 1.0
                r_factor = jnp.sign(sigma) * jnp.power(jnp.abs(sigma), order)
            out[f"SI_{name}"] = si
            out[f"R_{name}"] = r_factor
        return out

    required = {pH_field, temp_field}
    if ic_field is not None and model != "none":
        required.add(ic_field)
    return derived, produced, required
