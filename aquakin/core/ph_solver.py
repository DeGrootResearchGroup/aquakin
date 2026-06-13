"""Differentiable charge-balance pH solver.

Many environmental kinetic models couple reaction rates to pH, but pH itself
is not an independent state — it is fixed by the instantaneous acid/base
speciation of the dissolved species through electroneutrality. This module
solves that algebraic constraint for ``[H+]`` (and hence pH) given the total
molar concentrations of the relevant acid/base systems plus the strong-ion
charge balance.

The solver performs a proton (charge) balance over the carbonate, ammonia,
phosphate and sulfide acid/base systems and the monoprotic volatile fatty acids
(acetate, propionate, butyrate, valerate — the ADM1 set) together with water
self-ionisation, strong anions (e.g. sulfate, nitrate) and a net fixed cation
charge.

The solver is a fixed-iteration *safeguarded* Newton-bisection on ``u = ln[H+]``
(log space keeps ``[H+] > 0``). The residual is strictly monotone with a unique,
trivially bracketable root, so each step takes a Newton step but falls back to
bisection whenever Newton would leave the bracket — making the iteration
*globally* convergent in a fixed step count, where a bare Newton step can
overshoot to ``exp(u) = inf`` (NaN) from a poor start when the buffering is weak
relative to the strong-ion charge. Because the step count is fixed, the body uses
only smooth JAX primitives, and near the root the step is pure Newton, the whole
routine is differentiable end-to-end with ``jax.grad`` / ``jax.jacobian`` (the
exact implicit-function-theorem pH sensitivity) — so it composes inside a Diffrax
RHS and survives a charge balance far outside the buffered regime without NaNs.

All inputs and outputs are plain JAX scalars (or broadcastable arrays); there
is no Pydantic or dataclass dependency, keeping this usable from the core
runtime hot path.

Equilibrium-constant provenance
-------------------------------
The base (25 degC) dissociation constants and van't Hoff reaction enthalpies in
``_PK_BASE`` are the standard aquatic acid/base set:

- Water, carbonate (Ka1/Ka2), ammonium, and phosphate (Ka1-Ka3) pK and
  reaction-enthalpy values follow the standard aquatic-chemistry tabulations
  (Stumm & Morgan, *Aquatic Chemistry*, 3rd ed., 1996).
- The four monoprotic volatile-fatty-acid systems (acetate, propionate,
  butyrate, valerate) and the temperature-correction form
  ``K(T) = K_base * exp(dH/R * (1/T_base - 1/T))`` follow the ADM1 anaerobic
  digestion model and its BSM2 implementation, which use this same constant set
  (Batstone et al., *Anaerobic Digestion Model No. 1 (ADM1)*, IWA STR No. 13,
  2002; Rosen & Jeppsson, *Aspects on ADM1 Implementation within the BSM2
  Framework*, Lund University, 2006). The VFA acids are treated as
  temperature-independent (dH = 0).
- The second sulfide dissociation constant (``s_2``, pKa2 = 13.9) is **contested
  in the literature** (reported values span roughly 12-19); 13.9 is the
  commonly tabulated mid-range value adopted here. Treat sulfide speciation
  above pH ~12 as correspondingly uncertain.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

# Universal gas constant in J / (mol K), used in the van't Hoff correction.
_R_SI = 8.314462618
_T_BASE = 298.15  # reference temperature for the tabulated pK values (K)
_LN10 = jnp.log(10.0)

# Convergence bracket for the charge balance, in ``u = ln[H+]``. The residual
# ``f([H+])`` runs from ``+inf`` (as h->0) to ``-inf`` (as h->inf) -- the water
# self-ionisation term dominates at both extremes whatever the buffering or
# strong-ion charge -- so it changes sign exactly once inside any sufficiently
# wide bracket. ``pH in [-20, 40]`` spans every physical charge balance and far
# beyond (a charge imbalance of ~1e18 eq/L would be needed to push the root
# outside), with ``exp(u)`` nowhere near overflow.
_U_LO = -40.0 * math.log(10.0)   # pH 40  (basic extreme):  f(u) > 0
_U_HI = 20.0 * math.log(10.0)    # pH -20 (acidic extreme): f(u) < 0

# Base (25 degC) pK values and van't Hoff reaction enthalpies (J/mol) for the
# temperature correction K(T) = K_base * exp(dH/R * (1/T_base - 1/T)).
# Acetate is treated as temperature-independent (dH = 0).
_PK_BASE = {
    "w": (14.0, 55900.0),        # water self-ionisation
    "ac": (4.76, 0.0),           # CH3COOH    <-> H+ + CH3COO-
    "pro": (4.88, 0.0),          # propionate <-> H+ + propionate-
    "bu": (4.82, 0.0),           # butyrate   <-> H+ + butyrate-
    "va": (4.86, 0.0),           # valerate   <-> H+ + valerate-
    "nh": (9.25, 51965.0),       # NH4+     <-> H+ + NH3
    "co3_1": (6.35, 5200.0),     # H2CO3*   <-> H+ + HCO3-   (Ka1)
    "co3_2": (10.33, 14900.0),   # HCO3-    <-> H+ + CO3 2-  (Ka2)
    "s_1": (7.0, 21000.0),       # H2S      <-> H+ + HS-     (Ka1)
    "s_2": (13.9, 50700.0),      # HS-      <-> H+ + S 2-    (Ka2)
    "po4_1": (2.16, -24600.0),   # H3PO4    <-> H+ + H2PO4-  (Ka1)
    "po4_2": (7.21, 4200.0),     # H2PO4-   <-> H+ + HPO4 2- (Ka2)
    "po4_3": (12.32, 14700.0),   # HPO4 2-  <-> H+ + PO4 3-  (Ka3)
}


def equilibrium_constants(T_kelvin):
    """Return the temperature-corrected dissociation constants.

    Parameters
    ----------
    T_kelvin : scalar
        Absolute temperature in kelvin.

    Returns
    -------
    dict[str, jnp.ndarray]
        Mapping from system key (see :data:`_PK_BASE`) to the dissociation
        constant at ``T_kelvin``.
    """
    factor = (1.0 / _T_BASE - 1.0 / T_kelvin) / _R_SI
    out = {}
    for key, (pk, dH) in _PK_BASE.items():
        out[key] = jnp.power(10.0, -pk) * jnp.exp(dH * factor)
    return out


_ACTIVITY_MODELS = ("none", "davies", "debye_huckel")


def water_dielectric(T_kelvin):
    """Static relative permittivity of liquid water (Malmberg & Maryott 1956).

    ``eps_r(t) = 87.74 - 0.4008 t + 9.398e-4 t^2 - 1.410e-6 t^3`` with ``t`` in
    degrees Celsius; ~78.4 at 25 degC. Used only by :func:`debye_huckel_A`.
    """
    t = jnp.asarray(T_kelvin, dtype=float) - 273.15
    return 87.74 - 0.4008 * t + 9.398e-4 * t * t - 1.410e-6 * t * t * t


def debye_huckel_A(T_kelvin):
    """Debye-Hückel ``A`` parameter (base-10, mol^-1/2 L^1/2) vs temperature.

    ``A = 1.8246e6 / (eps_r * T)^1.5`` from the water dielectric constant
    (:func:`water_dielectric`); ~0.509 at 25 degC, rising slowly with
    temperature. This is the slope factor in the Davies / extended Debye-Hückel
    activity-coefficient expressions.
    """
    T = jnp.asarray(T_kelvin, dtype=float)
    eps = water_dielectric(T)
    return 1.8246e6 / jnp.power(eps * T, 1.5)


def _log10_gamma(z2, sqrt_I, I, A, model: str):
    """``log10`` of the activity coefficient of an ion of charge-squared ``z2``.

    ``model`` is a *static* string (branched at trace time):

    - ``"davies"``: ``log10 g = -A z^2 (sqrt(I)/(1+sqrt(I)) - 0.3 I)`` -- valid to
      ``I ~ 0.5 M``, the form SUMO/BioWin use.
    - ``"debye_huckel"``: the extended Debye-Hückel / Güntelberg form
      ``log10 g = -A z^2 sqrt(I)/(1+sqrt(I))`` (no ``0.3 I`` term) -- valid to
      ``I ~ 0.1 M``.
    """
    base = sqrt_I / (1.0 + sqrt_I)
    if model == "davies":
        base = base - 0.3 * I
    return -A * z2 * base


def _conditional_constants(K, I, A, model: str):
    """Activity-corrected (conditional, concentration-basis) dissociation constants.

    Replaces each thermodynamic ``K`` with ``Kc = K * g_acid / (g_H * g_base)``
    so the fraction expressions -- which are written in *concentrations* -- obey
    the activity-based equilibrium at ionic strength ``I``. With neutral acids
    ``g=1`` this reduces to dividing each proton release ``z_acid -> z_base+H+`` by
    the appropriate ``g`` product. Charge-symmetric ``NH4+ -> NH3 + H+`` is
    unchanged (the ``g`` of the +1 ammonium and the +1 proton cancel).
    """
    sqrt_I = jnp.sqrt(jnp.maximum(I, 0.0))
    g1 = jnp.power(10.0, _log10_gamma(1.0, sqrt_I, I, A, model))   # |z| = 1
    g2 = jnp.power(10.0, _log10_gamma(4.0, sqrt_I, I, A, model))   # |z| = 2
    g3 = jnp.power(10.0, _log10_gamma(9.0, sqrt_I, I, A, model))   # |z| = 3
    g1_sq = g1 * g1
    Kc = dict(K)
    # Single proton release between a neutral/-1 acid and a -1/-2 base, plus
    # water self-ionisation: divide by g1^2.
    for key in ("w", "ac", "pro", "bu", "va", "co3_1", "s_1", "po4_1"):
        Kc[key] = K[key] / g1_sq
    # -1 -> -2 and -2 -> -3 second/third releases: g1*g2 / g1 = g2, etc.
    Kc["co3_2"] = K["co3_2"] / g2
    Kc["s_2"] = K["s_2"] / g2
    Kc["po4_2"] = K["po4_2"] / g2
    Kc["po4_3"] = K["po4_3"] * g2 / (g1 * g3)
    # nh ( +1 -> 0 + H+ ) is activity-symmetric: Kc == K.
    return Kc, g1


def _ionic_strength_total(h, K, I_strong, *, totals):
    """Ionic strength ``I = 1/2 sum c_i z_i^2`` at trial ``[H+] = h``.

    Sums the (concentration-basis, so ``K`` must be the *conditional* constants)
    weak-acid ion concentrations -- the same speciation fractions the charge
    residual uses, re-weighted by ``z^2`` -- plus ``H+`` and ``OH-``, on top of
    the pH-independent strong-ion contribution ``I_strong`` (passed in because the
    speciation layer holds each strong ion's charge).
    """
    Kw = K["w"]
    Iw = 0.5 * (h + Kw / h)                       # H+, OH- (z = 1)

    for key, tot in (
        ("ac", totals["tot_acetate"]),
        ("pro", totals["tot_propionate"]),
        ("bu", totals["tot_butyrate"]),
        ("va", totals["tot_valerate"]),
    ):
        Ka = K[key]
        Iw = Iw + 0.5 * tot * Ka / (h + Ka)       # A- (z = 1)

    Ka_nh = K["nh"]
    Iw = Iw + 0.5 * totals["tot_ammonia"] * h / (h + Ka_nh)   # NH4+ (z = 1)

    k1, k2 = K["co3_1"], K["co3_2"]
    Dc = h * h + k1 * h + k1 * k2
    tc = totals["tot_carbonate"]
    Iw = Iw + 0.5 * tc * (k1 * h + 4.0 * k1 * k2) / Dc        # HCO3- z1, CO3-- z2

    s1, s2 = K["s_1"], K["s_2"]
    Ds = h * h + s1 * h + s1 * s2
    ts = totals["tot_sulfide"]
    Iw = Iw + 0.5 * ts * (s1 * h + 4.0 * s1 * s2) / Ds        # HS- z1, S-- z2

    p1, p2, p3 = K["po4_1"], K["po4_2"], K["po4_3"]
    Dp = h ** 3 + p1 * h * h + p1 * p2 * h + p1 * p2 * p3
    tp = totals["tot_phosphate"]
    Iw = Iw + 0.5 * tp * (
        p1 * h * h + 4.0 * p1 * p2 * h + 9.0 * p1 * p2 * p3
    ) / Dp                                                    # z1, z2, z3

    return I_strong + Iw


def charge_balance_residual(
    h,
    *,
    tot_carbonate=0.0,
    tot_acetate=0.0,
    tot_propionate=0.0,
    tot_butyrate=0.0,
    tot_valerate=0.0,
    tot_ammonia=0.0,
    tot_phosphate=0.0,
    tot_sulfide=0.0,
    strong_anion_eq=0.0,
    z_cation_eq=0.0,
    K,
):
    """Electroneutrality residual ``f([H+])`` (eq/L).

    ``f(h) = 0`` at the physical proton concentration. The residual is the net
    anionic charge: anions and OH- count positive, cations (H+, NH4+, fixed
    cations) count negative. It is strictly decreasing in ``h`` so the root is
    unique.

    Parameters
    ----------
    h : scalar
        Trial hydrogen-ion concentration ``[H+]`` (mol/L).
    tot_carbonate, tot_acetate, tot_propionate, tot_butyrate, tot_valerate, tot_ammonia, tot_phosphate, tot_sulfide : scalar
        Total molar concentrations (mol/L) of each acid/base system. Propionate,
        butyrate and valerate are monoprotic weak acids treated exactly like
        acetate (the ADM1 volatile-fatty-acid set).
    strong_anion_eq : scalar
        Charge equivalents per litre from fully dissociated strong anions
        (e.g. ``2*[SO4] + [NO3]``), positive.
    z_cation_eq : scalar
        Net fixed cation charge (eq/L), positive for a cation excess. Lumps
        the strong base cations not modelled explicitly (Na+, K+, ...).
    K : dict
        Dissociation constants from :func:`equilibrium_constants`.

    Returns
    -------
    scalar
        Net charge balance in eq/L.
    """
    Kw = K["w"]

    # Water.
    f = Kw / h - h

    # Strong ions and fixed charge.
    f = f + strong_anion_eq - z_cation_eq

    # Monoprotic volatile fatty acids: anionic fraction A- = Ka / (h + Ka).
    Ka_ac = K["ac"]
    f = f + tot_acetate * Ka_ac / (h + Ka_ac)
    Ka_pro = K["pro"]
    f = f + tot_propionate * Ka_pro / (h + Ka_pro)
    Ka_bu = K["bu"]
    f = f + tot_butyrate * Ka_bu / (h + Ka_bu)
    Ka_va = K["va"]
    f = f + tot_valerate * Ka_va / (h + Ka_va)

    # Ammonia: cationic fraction NH4+ = h / (h + Ka).
    Ka_nh = K["nh"]
    f = f - tot_ammonia * h / (h + Ka_nh)

    # Carbonate (diprotic): charge = (HCO3- + 2 CO3 2-) fractions.
    k1, k2 = K["co3_1"], K["co3_2"]
    Dc = h * h + k1 * h + k1 * k2
    f = f + tot_carbonate * (k1 * h + 2.0 * k1 * k2) / Dc

    # Sulfide (diprotic): charge = (HS- + 2 S 2-) fractions.
    s1, s2 = K["s_1"], K["s_2"]
    Ds = h * h + s1 * h + s1 * s2
    f = f + tot_sulfide * (s1 * h + 2.0 * s1 * s2) / Ds

    # Phosphate (triprotic): charge = (H2PO4- + 2 HPO4 2- + 3 PO4 3-) fractions.
    p1, p2, p3 = K["po4_1"], K["po4_2"], K["po4_3"]
    Dp = h ** 3 + p1 * h * h + p1 * p2 * h + p1 * p2 * p3
    f = f + tot_phosphate * (
        p1 * h * h + 2.0 * p1 * p2 * h + 3.0 * p1 * p2 * p3
    ) / Dp

    return f


def charge_balance_residual_deriv(
    h,
    *,
    tot_carbonate=0.0,
    tot_acetate=0.0,
    tot_propionate=0.0,
    tot_butyrate=0.0,
    tot_valerate=0.0,
    tot_ammonia=0.0,
    tot_phosphate=0.0,
    tot_sulfide=0.0,
    K,
):
    """Analytic derivative ``df/dh`` of :func:`charge_balance_residual`.

    Supplied in closed form (rather than via ``jax.grad``) so the Newton
    iteration contains no nested autodiff. This keeps the solver cleanly
    differentiable when it is itself embedded in an outer ``jax.grad`` /
    ODE-adjoint computation. Strong-ion and fixed-charge terms are constant in
    ``h`` and drop out.
    """
    Kw = K["w"]
    df = -Kw / (h * h) - 1.0

    Ka_ac = K["ac"]
    df = df - tot_acetate * Ka_ac / (h + Ka_ac) ** 2
    Ka_pro = K["pro"]
    df = df - tot_propionate * Ka_pro / (h + Ka_pro) ** 2
    Ka_bu = K["bu"]
    df = df - tot_butyrate * Ka_bu / (h + Ka_bu) ** 2
    Ka_va = K["va"]
    df = df - tot_valerate * Ka_va / (h + Ka_va) ** 2

    Ka_nh = K["nh"]
    df = df - tot_ammonia * Ka_nh / (h + Ka_nh) ** 2

    k1, k2 = K["co3_1"], K["co3_2"]
    Dc = h * h + k1 * h + k1 * k2
    numc = k1 * h + 2.0 * k1 * k2
    df = df + tot_carbonate * (k1 * Dc - numc * (2.0 * h + k1)) / (Dc * Dc)

    s1, s2 = K["s_1"], K["s_2"]
    Ds = h * h + s1 * h + s1 * s2
    nums = s1 * h + 2.0 * s1 * s2
    df = df + tot_sulfide * (s1 * Ds - nums * (2.0 * h + s1)) / (Ds * Ds)

    p1, p2, p3 = K["po4_1"], K["po4_2"], K["po4_3"]
    Dp = h ** 3 + p1 * h * h + p1 * p2 * h + p1 * p2 * p3
    nump = p1 * h * h + 2.0 * p1 * p2 * h + 3.0 * p1 * p2 * p3
    nump_d = 2.0 * p1 * h + 2.0 * p1 * p2
    Dp_d = 3.0 * h * h + 2.0 * p1 * h + p1 * p2
    df = df + tot_phosphate * (nump_d * Dp - nump * Dp_d) / (Dp * Dp)

    return df


def solve_ph(
    *,
    tot_carbonate=0.0,
    tot_acetate=0.0,
    tot_propionate=0.0,
    tot_butyrate=0.0,
    tot_valerate=0.0,
    tot_ammonia=0.0,
    tot_phosphate=0.0,
    tot_sulfide=0.0,
    strong_anion_eq=0.0,
    z_cation_eq=0.0,
    T_kelvin=293.15,
    n_iter: int = 40,
    h_init: float = 1e-7,
    activity_model: str = "none",
    ionic_strength_strong=0.0,
):
    """Solve the charge balance for pH.

    Performs ``n_iter`` safeguarded Newton-bisection steps in log space on the
    electroneutrality residual (Newton near the root, bisection when a Newton
    step would leave the root bracket). The fixed iteration count makes the
    routine ``jax.jit`` / ``vmap`` / ``grad`` friendly with no data-dependent
    control flow, and the bracketing makes it globally convergent — it cannot
    overshoot to ``NaN`` even when the strong-ion charge far exceeds the
    buffering.

    All concentration arguments are in mol/L; charge arguments in eq/L.

    Parameters
    ----------
    tot_carbonate, tot_acetate, tot_propionate, tot_butyrate, tot_valerate, tot_ammonia, tot_phosphate, tot_sulfide : scalar, optional
        Total molar concentrations of each acid/base system.
    strong_anion_eq : scalar, optional
        Strong-anion charge equivalents (e.g. ``2*[SO4]+[NO3]``).
    z_cation_eq : scalar, optional
        Net fixed cation charge (eq/L).
    T_kelvin : scalar, optional
        Absolute temperature (K). Default 293.15 (20 degC).
    n_iter : int, optional
        Number of safeguarded Newton-bisection iterations. 40 reaches machine
        precision well inside the buffered regime (Newton converges in a handful
        of steps) and still guarantees the bisection fallback pins the pH to
        ``bracket_width / 2**40`` for any charge balance.
    h_init : float, optional
        Initial guess for ``[H+]`` (mol/L). Default 1e-7 (pH 7). With the
        bracketed iteration the result no longer depends on a good initial
        guess, but a guess near the operating pH saves a few bisection steps.

    activity_model : str, optional
        Ionic-strength activity-coefficient model (a *static* choice, branched at
        trace time): ``"none"`` (default) uses molar concentrations directly (all
        activity coefficients = 1, the ADM1/BSM2 convention -- bit-identical to the
        historic behaviour); ``"davies"`` and ``"debye_huckel"`` apply the Davies
        or extended Debye-Hückel correction. With a non-``none`` model the
        equilibrium constants become *conditional* (concentration-basis) constants
        at the self-consistent ionic strength, and the returned pH is the
        **measurable** ``-log10(a_H) = -log10(g_H [H+])`` (it reduces to
        ``-log10([H+])`` when all ``g = 1``).
    ionic_strength_strong : scalar, optional
        The pH-independent strong-ion contribution to ionic strength,
        ``1/2 sum c_i z_i^2`` over the strong anions/cations (and the lumped fixed
        cation charge, taken monovalent). Supplied by the caller because only it
        knows each strong ion's charge. Used only when ``activity_model`` is not
        ``"none"``.

    Returns
    -------
    jnp.ndarray
        Solution pH -- ``-log10([H+])`` for ``activity_model="none"``, else the
        activity-based ``-log10(a_H)``.

    Examples
    --------
    >>> import aquakin
    >>> from aquakin.core.ph_solver import solve_ph
    >>> # Pure carbonate buffer, 1 mM bicarbonate-equivalent alkalinity.
    >>> float(solve_ph(tot_carbonate=1e-3, z_cation_eq=1e-3))  # doctest: +SKIP
    8.3
    """
    if activity_model not in _ACTIVITY_MODELS:
        raise ValueError(
            f"activity_model must be one of {_ACTIVITY_MODELS}; got "
            f"{activity_model!r}"
        )
    K = equilibrium_constants(jnp.asarray(T_kelvin, dtype=float))

    totals = dict(
        tot_carbonate=tot_carbonate,
        tot_acetate=tot_acetate,
        tot_propionate=tot_propionate,
        tot_butyrate=tot_butyrate,
        tot_valerate=tot_valerate,
        tot_ammonia=tot_ammonia,
        tot_phosphate=tot_phosphate,
        tot_sulfide=tot_sulfide,
    )

    def residual(h):
        return charge_balance_residual(
            h, strong_anion_eq=strong_anion_eq, z_cation_eq=z_cation_eq, K=K, **totals
        )

    def dresidual_dh(h):
        return charge_balance_residual_deriv(h, K=K, **totals)

    # The Newton state ``u = ln[H+]`` broadcasts to the shape of the residual,
    # which is the broadcast of every input it depends on (the totals, the
    # strong-ion / fixed-cation charges, and -- via the temperature-corrected
    # equilibrium constants -- ``T_kelvin``). Compute that shape directly rather
    # than evaluating the full residual once just to read its shape.
    out_shape = jnp.broadcast_shapes(
        jnp.shape(jnp.asarray(h_init)),
        jnp.shape(strong_anion_eq),
        jnp.shape(z_cation_eq),
        jnp.shape(T_kelvin),
        *(jnp.shape(v) for v in totals.values()),
    )
    # Safeguarded Newton-bisection. A bare Newton step (the old scheme) can
    # overshoot to ``exp(u) = inf -> NaN`` from the fixed pH-7 start when the
    # buffering is weak relative to the strong-ion charge -- the residual is then
    # flat (water-only derivative) but large, so the step is enormous. Bracketing
    # the (unique, monotone) root and falling back to bisection whenever Newton
    # would leave the bracket makes the iteration *globally* convergent in a FIXED
    # step count: bisection halves the bracket every fallback, so n_iter steps pin
    # the pH to ``bracket_width / 2**n_iter``. Near the root the step is pure
    # Newton, so convergence stays quadratic and AD through the iteration yields
    # the exact implicit-function-theorem pH sensitivity (a bisection-only scheme
    # would, by contrast, have a near-zero AD gradient through its sign tests).
    u_lo = jnp.broadcast_to(jnp.asarray(_U_LO), out_shape).astype(float)
    u_hi = jnp.broadcast_to(jnp.asarray(_U_HI), out_shape).astype(float)
    u = jnp.clip(
        jnp.broadcast_to(jnp.asarray(jnp.log(h_init)), out_shape).astype(float),
        _U_LO, _U_HI,
    )

    def rtsafe_update(u_lo, u_hi, u, f, dfdu):
        """One safeguarded Newton-bisection step (Newton, bisection fallback).

        ``dfdu`` need only have the right sign and order of magnitude -- the
        bracketing guarantees convergence regardless, and near the root the
        Newton step is tiny and stays in-bracket (so the iteration is pure Newton
        there and AD yields the exact implicit-function-theorem pH sensitivity).
        """
        # Tighten the bracket using the sign of f at u. f is decreasing, so f > 0
        # means the root lies at a larger u; the endpoints keep f(u_lo) >= 0 >=
        # f(u_hi) and the bracket only shrinks.
        pos = f > 0.0
        u_lo = jnp.where(pos, u, u_lo)
        u_hi = jnp.where(pos, u_hi, u)
        # Newton candidate; fall back to bisection if it leaves the bracket. The
        # acceptance test is the non-strict rtsafe product test (``<= 0`` is
        # essential -- at convergence the bracket collapses an endpoint onto ``u``
        # and the Newton step lands *on* it; a strict test would bisect it away).
        u_newton = u - f / dfdu
        u_bisect = 0.5 * (u_lo + u_hi)
        in_bracket = (u_newton - u_lo) * (u_newton - u_hi) <= 0.0
        u_next = jnp.where(in_bracket, u_newton, u_bisect)
        return u_lo, u_hi, u_next

    if activity_model == "none":
        # Ideal (g = 1): the historic path, bit-for-bit. No ionic strength.
        def body(carry, _):
            u_lo, u_hi, u = carry
            h = jnp.exp(u)
            f = residual(h)
            # df/du = f'(h) * h (analytic, no nested AD); f'(h) <= -1, h > 0, so
            # dfdu < 0 always -- the Newton step never divides by zero.
            dfdu = dresidual_dh(h) * h
            u_lo, u_hi, u_next = rtsafe_update(u_lo, u_hi, u, f, dfdu)
            return (u_lo, u_hi, u_next), None

        (_, _, u), _ = jax.lax.scan(body, (u_lo, u_hi, u), None, length=n_iter)
        h = jnp.exp(u)
        return -jnp.log(h) / _LN10

    # Activity-corrected path. The conditional constants depend on the ionic
    # strength, which depends on the speciation, which depends on [H+] -- a
    # coupled fixed point. We resolve it *inside* the same bracketed scan by
    # carrying the ionic strength ``I``: each step forms the conditional constants
    # at the carried ``I``, takes one safeguarded step on the resulting residual,
    # and recomputes ``I`` from the new speciation. ``I`` and ``[H+]`` converge
    # together, so at the root ``I`` is self-consistent with the speciation.
    A = debye_huckel_A(jnp.asarray(T_kelvin, dtype=float))
    I_strong = jnp.broadcast_to(
        jnp.asarray(ionic_strength_strong, dtype=float), out_shape)
    I0 = jnp.maximum(I_strong, 0.0)

    def act_body(carry, _):
        u_lo, u_hi, u, I = carry
        h = jnp.exp(u)
        Kc, _g1 = _conditional_constants(K, I, A, activity_model)
        f = charge_balance_residual(
            h, strong_anion_eq=strong_anion_eq, z_cation_eq=z_cation_eq,
            K=Kc, **totals)
        # Newton derivative holding Kc fixed (the d Kc / dh coupling is a small
        # perturbation; bracketing covers it). f'(h) <= -1 still, so dfdu < 0.
        dfdu = charge_balance_residual_deriv(h, K=Kc, **totals) * h
        u_lo, u_hi, u_next = rtsafe_update(u_lo, u_hi, u, f, dfdu)
        I_next = _ionic_strength_total(
            jnp.exp(u_next), Kc, I_strong, totals=totals)
        return (u_lo, u_hi, u_next, I_next), None

    (_, _, u, I), _ = jax.lax.scan(
        act_body, (u_lo, u_hi, u, I0), None, length=n_iter)
    h = jnp.exp(u)
    # Report the measurable pH = -log10(a_H) = -log10(g_H [H+]).
    _, g1 = _conditional_constants(K, I, A, activity_model)
    return -jnp.log(g1 * h) / _LN10
