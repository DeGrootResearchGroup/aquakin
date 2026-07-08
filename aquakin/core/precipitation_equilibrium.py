"""Algebraic mineral-equilibrium reduction for fast / ultra-insoluble precipitates.

The kinetic precipitation law (:mod:`aquakin.core.precipitation`) drives each
mineral at ``k * X * sign(sigma) * |sigma|^n``. For a *very* insoluble mineral
operated far from saturation (a metal phosphate at ``SI ~ 14``, ``IAP/Ksp ~
1e14``) the supersaturation rate factor and its Jacobian are enormous
(``~1e13``). The forward solve is fine -- the implicit solver damps the fast
mode -- but the rate Jacobian is so large that no sensitivity method survives
the transient: the per-step linear operator is near-singular and every adjoint
overflows.

This module provides the principled alternative: solve the precipitation
*equilibrium* algebraically instead of integrating the stiff kinetics. For a set
of equilibrium-mode minerals it finds the phase amounts and dissolved ion
concentrations that satisfy, simultaneously,

  * **mass balance** -- the total of each constituent ion (dissolved plus that
    bound in the equilibrium solids) is conserved, and
  * **mineral equilibrium with complementarity** -- every precipitated mineral
    sits exactly on its solubility (``SI = 0``) and every absent mineral is
    undersaturated (``SI < 0``): ``X_m >= 0``, ``-SI_m >= 0``, ``X_m * SI_m = 0``.

This is the classical chemical-equilibrium (equilibrium-phase) problem. It is
solved as a smooth nonlinear system -- log-free-ion + phase-amount unknowns, the
complementarity written with a smoothed Fischer--Burmeister function -- by a
fixed-iteration safeguarded Newton scan (the pattern of
:func:`aquakin.core.ph_solver.solve_ph`): epsilon-continuation on the
complementarity smoothing, residual-adaptive Levenberg--Marquardt damping, and a
bounded step on the log-free-ion unknowns make it globally convergent even when a
component is driven to near-complete consumption (free ion ``~1e-18`` mol/L).
Because the iteration count is fixed and every step is a smooth JAX primitive,
the converged solution is differentiable end-to-end (the exact
implicit-function-theorem sensitivity), and the algebraic Jacobian it inverts is
well conditioned -- so the ``1e13`` stiffness of the kinetic transient is gone.

The equilibrium phase amounts are exposed per mineral as a derived condition
field ``Xeq_<name>``. An equilibrium-mode precipitation reaction relaxes its
solid toward that target, ``rate = k_relax * ({Xeq_<name>} - [X_<name>])``: a
first-order pull whose Jacobian is ``~k_relax`` (non-stiff, differentiable) and
whose steady state is exactly the algebraic equilibrium.

The aqueous chemistry -- dissociation constants, acid/base free-ion fractions and
activity coefficients -- is shared with the kinetic engine and the pH solver
(:mod:`aquakin.core.ph_solver`, :mod:`aquakin.core.precipitation`).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from aquakin.core.ph_solver import (
    _log10_gamma,
    debye_huckel_A,
    equilibrium_constants,
)
from aquakin.core.precipitation import _FRACTIONS
from aquakin.core.temperature import van_t_hoff_factor

_LN10 = jnp.log(10.0)

# Solver controls. The defaults converge the worked metal-phosphate equilibrium
# (four competing minerals, free ion driven to ~1e-18 mol/L) to SI = 0 on the
# present phases with the residual at machine zero; see the module tests.
_N_ITER = 80
_EPS_START = 1.0  # complementarity smoothing at the first iteration
_EPS_END = 1.0e-9  # ... annealed geometrically to here (near-exact NCP)
_LM_FLOOR = 1.0e-8  # Levenberg-Marquardt damping floor
_LM_SCALE = 1.0e-2  # ... plus this times the residual norm (robust far out)
_MAX_DLOG = 2.0  # max change in log10(free ion) per iteration


def _build_equilibrium_system(minerals_cfg, species_index):
    """Resolve the equilibrium-mode minerals to plain numbers / indices.

    Returns ``(produced, solid_idx, Cmat, ion_states, ion_mm, ion_states_list,
    mineral_ions, log_ksp_ref, dH_sp)`` describing the conserved-ion mass-balance
    system and the per-mineral ion-activity products (``log_ksp_ref`` the reference
    ``ln Ksp``, ``dH_sp`` its enthalpy of dissolution for the van't Hoff
    correction). ``None`` when there are no equilibrium-mode minerals.
    """
    eq_minerals = [m for m in minerals_cfg if m.get("mode") == "equilibrium"]
    if not eq_minerals:
        return None

    # Conserved dissolved ions: the distinct state-backed (non pH-special) ions
    # across the equilibrium minerals. pH specials (proton / hydroxide) are set
    # by pH / water, not conserved here.
    ion_pos: dict[int, int] = {}
    ion_mm: list[float] = []
    ion_states: list[int] = []
    for m in eq_minerals:
        for ion in m["ions"]:
            frac = ion.get("fraction")
            if frac in ("proton", "hydroxide"):
                continue
            sp = ion["species"]
            idx = species_index[sp]
            if idx not in ion_pos:
                ion_pos[idx] = len(ion_states)
                ion_states.append(idx)
                ion_mm.append(float(ion.get("molar_mass", 1.0)))

    M = len(eq_minerals)
    N = len(ion_states)
    Cmat = [[0.0] * N for _ in range(M)]  # conserved-ion count per mineral
    solid_idx: list[int] = []
    log_ksp_ref: list[float] = []  # ln(Ksp) at the reference T
    dH_sp: list[float] = []  # enthalpy of dissolution (J/mol)
    mineral_ions: list[list[tuple]] = []  # (kind, count, z2, frac, mm, ion_pos)
    produced: list[str] = []
    for mi, m in enumerate(eq_minerals):
        if m.get("solid") is None:
            raise ValueError(
                f"equilibrium-mode mineral {m['name']!r} needs a 'solid:' species "
                f"(the precipitated phase the equilibrium amount is reported for)."
            )
        if m["solid"] not in species_index:
            raise KeyError(
                f"mineral {m['name']!r} solid {m['solid']!r} is not a declared "
                f"species; declared: {sorted(species_index)}"
            )
        solid_idx.append(species_index[m["solid"]])
        log_ksp_ref.append(-float(m["pKsp"]) * float(_LN10))
        dH_sp.append(float(m.get("dH_sp", 0.0)))
        ions = []
        for ion in m["ions"]:
            frac = ion.get("fraction")
            count = int(ion["count"])
            z2 = float(ion["charge"]) ** 2
            mm = float(ion.get("molar_mass", 1.0))
            if frac in ("proton", "hydroxide"):
                ions.append((frac, count, z2, frac, mm, -1))
            else:
                pos = ion_pos[species_index[ion["species"]]]
                Cmat[mi][pos] += count
                kind = frac if frac in _FRACTIONS else "free"
                ions.append((kind, count, z2, frac, mm, pos))
        mineral_ions.append(ions)
        produced.append(f"Xeq_{m['name']}")

    return (
        produced,
        jnp.asarray(solid_idx),
        jnp.asarray(Cmat),
        jnp.asarray(ion_states),
        jnp.asarray(ion_mm),
        ion_states,
        mineral_ions,
        jnp.asarray(log_ksp_ref),
        jnp.asarray(dH_sp),
    )


def _make_si_fn(mineral_ions, log_ksp_ref, dH_sp):
    """Build ``SI(u, h, K, gamma, vant_hoff) -> (M,)`` for the equilibrium minerals.

    ``u`` is ``log10`` of each conserved ion's *total dissolved* concentration in
    the state's own units; the per-ion ``molar_mass`` converts to the mol/L the
    ``Ksp`` / activities use. ``Ksp`` is van't Hoff-corrected with temperature,
    ``ln(Ksp(T)) = ln(Ksp_ref) + dH_sp * vant_hoff`` (``vant_hoff = (1/T_ref -
    1/T)/R``), the same form and reference temperature as the kinetic engine.
    Returns each mineral's ``log10(IAP/Ksp)``.
    """

    def si(u, h, K, gamma, vant_hoff):
        out = []
        for ions, lk_ref, dH in zip(mineral_ions, log_ksp_ref, dH_sp):
            log_iap = 0.0
            for kind, count, z2, frac, mm, pos in ions:
                if kind == "proton":
                    a = h
                elif kind == "hydroxide":
                    a = K["w"] / h
                else:
                    tot = jnp.power(10.0, u[pos]) / mm  # mol/L
                    fr = _FRACTIONS[frac](h, K) if kind in _FRACTIONS else 1.0
                    a = gamma(z2) * tot * fr
                log_iap = log_iap + count * jnp.log(jnp.maximum(a, 1e-300))
            out.append((log_iap - (lk_ref + dH * vant_hoff)) / _LN10)
        return jnp.stack(out)

    return si


def solve_equilibrium_amounts(totals, h, T_kelvin, *, si_fn, Cmat, model, ionic_strength_offset):
    """Solve the coupled mineral equilibrium for the phase amounts.

    Parameters
    ----------
    totals : jnp.ndarray, shape (N,)
        Total conserved-ion inventories (dissolved + bound in the equilibrium
        solids), in the state's own concentration units.
    h : scalar
        Hydrogen-ion activity ``10^-pH``.
    T_kelvin : scalar
        Absolute temperature (K).
    si_fn : callable
        ``(u, h, K, gamma, vant_hoff) -> (M,)`` saturation indices (see
        :func:`_make_si_fn`).
    Cmat : jnp.ndarray, shape (M, N)
        Conserved-ion count of each mineral.
    model : str
        Activity model (``"none"`` / ``"davies"`` / ``"debye_huckel"``).
    ionic_strength_offset : float
        Background ionic strength (mol/L) for the activity coefficients.

    Returns
    -------
    jnp.ndarray, shape (M,)
        Equilibrium phase amount of each mineral, in the state's units
        (non-negative). Differentiable w.r.t. ``totals`` / ``h`` / ``T`` via the
        implicit function theorem.
    """
    M, N = Cmat.shape
    K = equilibrium_constants(T_kelvin)
    # van't Hoff factor for the Ksp(T) correction inside si_fn (unity at T_ref).
    vant_hoff = van_t_hoff_factor(T_kelvin)
    use_activity = model != "none"
    if use_activity:
        A = debye_huckel_A(T_kelvin)
        I = jnp.maximum(jnp.asarray(ionic_strength_offset, dtype=float), 0.0)
        sqrt_I = jnp.sqrt(I)

        def gamma(z2):
            return jnp.power(10.0, _log10_gamma(z2, sqrt_I, I, A, model))
    else:

        def gamma(z2):
            return 1.0

    def residual(w, eps):
        u = w[:N]
        X = w[N:]
        mass = jnp.power(10.0, u) + X @ Cmat - totals  # (N,) feasibility
        si = si_fn(u, h, K, gamma, vant_hoff)  # (M,)
        # Smoothed Fischer-Burmeister: phi(X, -SI) = 0  <=>  X>=0, SI<=0, X*SI=0.
        fb = X + (-si) - jnp.sqrt(X * X + si * si + eps * eps)
        return jnp.concatenate([mass, fb])

    # Start fully dissolved (u = log10 totals), no solid.
    u0 = jnp.log10(jnp.maximum(totals, 1e-12))
    w = jnp.concatenate([u0, jnp.full((M,), 1e-2)])
    eps_sched = jnp.geomspace(_EPS_START, _EPS_END, _N_ITER)
    eye = jnp.eye(N + M)

    def step(w, eps):
        r = residual(w, eps)
        J = jax.jacobian(lambda ww: residual(ww, eps))(w)
        nrm = jnp.sqrt(jnp.sum(r * r))
        lam = _LM_FLOOR + _LM_SCALE * nrm  # residual-adaptive damping
        dw = jnp.linalg.solve(J.T @ J + lam * eye, -(J.T @ r))
        du = jnp.clip(dw[:N], -_MAX_DLOG, _MAX_DLOG)  # bounded log-free-ion step
        return w + jnp.concatenate([du, dw[N:]]), None

    # Forward: run the robust globalized iteration WITHOUT differentiating through
    # it (backprop through the unrolled scan + its per-step Jacobians is huge and
    # unnecessary). Detach, then attach the *exact* gradient via the implicit
    # function theorem with a single Newton step on the converged residual: with
    # ``w`` detached and ``G(w*) ~ 0``, ``w - (dG/dw)^-1 G(w, theta)`` leaves the
    # value unchanged but carries the derivative ``-(dG/dw)^-1 dG/dtheta`` -- the
    # IFT sensitivity. Backward is then one (N+M) linear solve, not a 200-step
    # backprop.
    def solve_scan(w):
        w, _ = jax.lax.scan(step, w, eps_sched)
        return w

    w_star = jax.lax.stop_gradient(solve_scan(w))
    eps_exact = _EPS_END
    G = residual(w_star, eps_exact)
    JG = jax.jacobian(lambda ww: residual(ww, eps_exact))(w_star)
    w_star = w_star - jnp.linalg.solve(JG + _LM_FLOOR * eye, G)
    return jnp.maximum(w_star[N:], 0.0)


def build_precipitation_equilibrium_derived_fn(
    config: dict,
    species_index: dict[str, int],
) -> tuple[Callable, list[str], set[str], Callable] | None:
    """Compile the equilibrium-mode minerals of a ``precipitation:`` block.

    Parameters
    ----------
    config : dict
        The validated ``precipitation:`` declaration. Minerals with
        ``mode: equilibrium`` are solved algebraically here; ``mode: kinetic``
        minerals (the default) are handled by
        :func:`aquakin.core.precipitation.build_precipitation_derived_fn`.
    species_index : dict[str, int]
        Map from species name to state-vector index.

    Returns
    -------
    (derived_fn, produced_fields, required_fields, project_fn) or None
        ``None`` when the block has no equilibrium-mode minerals. Otherwise
        ``derived_fn(C, params, condition_arrays, loc_idx) -> dict`` produces
        ``Xeq_<name>`` (the equilibrium phase amount, in state units) for each
        equilibrium mineral; ``produced_fields`` lists them; ``required_fields``
        are the conditions it reads (pH, T); ``project_fn(C, condition_arrays,
        loc_idx)`` snaps a composition onto the precipitation equilibrium
        (each equilibrium solid set to its equilibrium amount, dissolved ions
        rebalanced, mass-conserving).
    """
    built = _build_equilibrium_system(config["minerals"], species_index)
    if built is None:
        return None
    (
        produced,
        solid_idx,
        Cmat,
        ion_states_arr,
        _ion_mm,
        ion_states,
        mineral_ions,
        log_ksp_ref,
        dH_sp,
    ) = built

    pH_field = config.get("pH_field", "pH")
    temp_field = config.get("temperature_field", "T")
    temp_units = config.get("temperature_units", "celsius")
    model = config.get("activity_model", "none")
    I_offset = float(config.get("ionic_strength_offset", 0.0))
    si_fn = _make_si_fn(mineral_ions, log_ksp_ref, dH_sp)

    def _solve(C, condition_arrays, loc_idx):
        """Shared core: total inventory -> equilibrium phase amounts ``Xeq`` (M,)
        and the conserved-ion totals (N,)."""
        T = condition_arrays[temp_field][loc_idx]
        T_kelvin = T + 273.15 if temp_units == "celsius" else T
        pH = condition_arrays[pH_field][loc_idx]
        h = jnp.power(10.0, -pH)
        # Total conserved-ion inventory = dissolved + that bound in the
        # equilibrium solids (so the solve captures redissolution / ripening).
        dissolved = jnp.maximum(C[ion_states_arr], 0.0)  # (N,)
        solids = jnp.maximum(C[solid_idx], 0.0)  # (M,)
        totals = dissolved + solids @ Cmat  # (N,)
        Xeq = solve_equilibrium_amounts(
            totals, h, T_kelvin, si_fn=si_fn, Cmat=Cmat, model=model, ionic_strength_offset=I_offset
        )
        return Xeq, totals

    def derived(C, params, condition_arrays, loc_idx) -> dict:
        Xeq, _ = _solve(C, condition_arrays, loc_idx)
        return {name: Xeq[i] for i, name in enumerate(produced)}

    def project(C, condition_arrays, loc_idx):
        """Project a composition onto the precipitation equilibrium: set each
        equilibrium solid to its equilibrium amount and rebalance the conserved
        dissolved ions (mass-conserving)."""
        Xeq, totals = _solve(C, condition_arrays, loc_idx)
        new_dissolved = totals - Xeq @ Cmat  # (N,)
        out = C
        out = out.at[ion_states_arr].set(new_dissolved)
        out = out.at[solid_idx].set(Xeq)
        return out

    required = {pH_field, temp_field}
    return derived, produced, required, project
