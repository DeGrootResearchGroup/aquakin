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

The solver is a fixed-iteration Newton method on ``u = ln[H+]`` (log space
keeps ``[H+] > 0`` without clipping). Because the iteration count is fixed and
the body uses only smooth JAX primitives, the whole routine is differentiable
end-to-end with ``jax.grad`` / ``jax.jacobian`` — so pH sensitivities flow
through automatically and the solver composes inside a Diffrax RHS.

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

import jax
import jax.numpy as jnp

# Universal gas constant in J / (mol K), used in the van't Hoff correction.
_R_SI = 8.314462618
_T_BASE = 298.15  # reference temperature for the tabulated pK values (K)
_LN10 = jnp.log(10.0)

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
):
    """Solve the charge balance for pH.

    Performs ``n_iter`` Newton steps in log space on the electroneutrality
    residual. The fixed iteration count makes the routine ``jax.jit`` / ``vmap``
    / ``grad`` friendly with no data-dependent control flow.

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
        Number of Newton iterations. 40 is comfortably enough for convergence
        to machine precision across the environmentally relevant range.
    h_init : float, optional
        Initial guess for ``[H+]`` (mol/L). Default 1e-7 (pH 7).

    Returns
    -------
    jnp.ndarray
        Solution pH = ``-log10([H+])``.

    Examples
    --------
    >>> import aquakin
    >>> from aquakin.core.ph_solver import solve_ph
    >>> # Pure carbonate buffer, 1 mM bicarbonate-equivalent alkalinity.
    >>> float(solve_ph(tot_carbonate=1e-3, z_cation_eq=1e-3))  # doctest: +SKIP
    8.3
    """
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

    u = jnp.broadcast_to(
        jnp.asarray(jnp.log(h_init)),
        jnp.shape(residual(jnp.asarray(h_init))),
    ).astype(float)

    def body(u, _):
        h = jnp.exp(u)
        f = residual(h)
        # df/du = df/dh * dh/du = f'(h) * h (analytic derivative, no nested AD).
        dfdu = dresidual_dh(h) * h
        u_new = u - f / dfdu
        return u_new, None

    u, _ = jax.lax.scan(body, u, None, length=n_iter)
    h = jnp.exp(u)
    return -jnp.log(h) / _LN10
