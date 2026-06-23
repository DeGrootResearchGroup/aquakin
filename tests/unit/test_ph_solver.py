"""Unit tests for the differentiable charge-balance pH solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin  # noqa: F401  (enables x64)
from aquakin.core.ph_solver import (
    charge_balance_residual,
    charge_balance_residual_deriv,
    equilibrium_constants,
    solve_ph,
)


def test_analytic_derivative_matches_autodiff():
    """The hand-written charge_balance_residual_deriv is kept in closed form so
    the Newton iteration carries no nested autodiff, but it must stay in sync
    with the residual it differentiates. Pin it to jax.grad of the residual
    across a range of [H+] and buffer loadings so the two cannot silently
    desync (the duplication risk in the audit)."""
    totals = dict(
        tot_carbonate=2e-3, tot_acetate=1e-3, tot_propionate=5e-4,
        tot_butyrate=3e-4, tot_valerate=2e-4, tot_ammonia=1.5e-3,
        tot_phosphate=8e-4, tot_sulfide=4e-4,
    )
    # Vectorise the (T, pH) grid with one vmap (one trace/compile of the residual
    # and its grad over the whole batch) instead of a Python double loop, which
    # re-traced+recompiled jax.grad each of the 18 cells. Same coverage.
    Ts = jnp.array([283.15, 293.15, 308.15])
    pHs = jnp.array([4.0, 6.0, 7.0, 8.5, 11.0, 13.0])
    Tg, pHg = jnp.meshgrid(Ts, pHs, indexing="ij")
    K = jax.vmap(equilibrium_constants)(Tg.ravel())     # batched constants over T
    h = 10.0 ** (-pHg.ravel())

    def residual(hh, KK):
        return charge_balance_residual(
            hh, strong_anion_eq=0.0, z_cation_eq=0.0, K=KK, **totals)

    analytic = jax.vmap(lambda hh, KK: charge_balance_residual_deriv(
        hh, K=KK, **totals))(h, K)
    auto = jax.vmap(jax.grad(residual))(h, K)
    np.testing.assert_allclose(np.asarray(analytic), np.asarray(auto),
                               rtol=1e-6, atol=1e-9)


def _bisection_reference(totals, T_kelvin=293.15, n=200):
    """Independent NumPy bisection root of the charge-balance residual.

    Uses the same residual function as the solver but a derivative-free root
    finder, so agreement is a genuine cross-check of the Newton solver.
    """
    K = equilibrium_constants(jnp.asarray(T_kelvin))

    def f(h):
        return float(charge_balance_residual(jnp.asarray(h), K=K, **totals))

    lo, hi = 1e-14, 1.0  # f is strictly decreasing in h
    flo, fhi = f(lo), f(hi)
    assert flo > 0 > fhi, "root not bracketed"
    for _ in range(n):
        mid = np.sqrt(lo * hi)  # geometric bisection (log space)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    h = np.sqrt(lo * hi)
    return -np.log10(h)


CASES = [
    # Carbonate buffer with matching cation alkalinity -> mildly alkaline.
    dict(tot_carbonate=1.0e-3, z_cation_eq=1.0e-3),
    # Ammonia + carbonate, with a strong-anion deficit.
    dict(tot_carbonate=2.0e-3, tot_ammonia=3.0e-3, z_cation_eq=2.0e-3),
    # Sulfide + phosphate + acetate, representative sewer water.
    dict(
        tot_carbonate=1.5e-3,
        tot_acetate=5.0e-4,
        tot_ammonia=2.0e-3,
        tot_phosphate=3.0e-4,
        tot_sulfide=3.0e-4,
        strong_anion_eq=5.0e-4,
        z_cation_eq=3.28e-3,
    ),
    # Acidic case: strong-anion excess.
    dict(tot_carbonate=1.0e-3, strong_anion_eq=2.0e-3, z_cation_eq=5.0e-4),
    # ADM1 digester: carbonate + ammonia buffer with the full VFA set,
    # representative of an anaerobic digester at the BSM2 operating point.
    dict(
        tot_carbonate=0.0951,
        tot_acetate=0.0893 / 64.0,
        tot_propionate=0.0176 / 112.0,
        tot_butyrate=0.0140 / 160.0,
        tot_valerate=0.0123 / 208.0,
        tot_ammonia=0.0945,
        z_cation_eq=-9.330944e-4,
    ),
]


@pytest.mark.parametrize("totals", CASES)
def test_matches_bisection(totals):
    pH = float(solve_ph(**totals))
    pH_ref = _bisection_reference(totals)
    assert pH == pytest.approx(pH_ref, abs=1e-6)


@pytest.mark.parametrize("totals", CASES)
def test_residual_is_zero_at_solution(totals):
    K = equilibrium_constants(jnp.asarray(293.15))
    pH = solve_ph(**totals)
    h = 10.0 ** (-pH)
    res = float(charge_balance_residual(h, K=K, **totals))
    assert abs(res) < 1e-12


def test_temperature_shifts_pH():
    totals = dict(tot_carbonate=2.0e-3, z_cation_eq=2.0e-3)
    pH_20 = float(solve_ph(**totals, T_kelvin=293.15))
    pH_10 = float(solve_ph(**totals, T_kelvin=283.15))
    # Equilibrium constants are temperature dependent, so pH must move.
    assert abs(pH_20 - pH_10) > 1e-3


def test_gradient_is_finite_and_matches_fd():
    # d(pH)/d(total ammonia): adding base (NH3 buffer) should be smooth.
    def f(nh):
        return solve_ph(tot_carbonate=2.0e-3, tot_ammonia=nh, z_cation_eq=2.0e-3)

    nh0 = 3.0e-3
    g = float(jax.grad(f)(nh0))
    assert np.isfinite(g)
    eps = 1e-7
    fd = (float(f(nh0 + eps)) - float(f(nh0 - eps))) / (2 * eps)
    assert g == pytest.approx(fd, rel=1e-4, abs=1e-4)


def test_vmap_over_states():
    carb = jnp.array([1.0e-3, 2.0e-3, 3.0e-3])
    zcat = jnp.array([1.0e-3, 2.0e-3, 3.0e-3])
    pH = jax.vmap(lambda c, z: solve_ph(tot_carbonate=c, z_cation_eq=z))(carb, zcat)
    assert pH.shape == (3,)
    assert jnp.all(jnp.isfinite(pH))


# --- Implicit-function-theorem (custom_root) path: the ideal solve uses an
# adaptive while_loop wrapped in jax.lax.custom_root, so the pH sensitivity is
# the analytic IFT tangent (one scalar solve at the root) rather than a
# differentiate-through of the iteration. These pin that behaviour.

def test_pH_independent_of_iteration_cap():
    """The adaptive loop stops at convergence, so the result is independent of the
    n_iter *cap* once the cap exceeds the (small) iteration count needed -- a tiny
    cap, the default, and a huge cap all agree to machine precision."""
    # A buffered state: Newton converges in ~6 steps, so any cap >= ~8 agrees.
    kw = dict(tot_carbonate=2.0e-3, tot_ammonia=3.0e-3, z_cation_eq=2.0e-3)
    ref = float(solve_ph(**kw, n_iter=200))
    for cap in (8, 12, 40, 200):
        assert float(solve_ph(**kw, n_iter=cap)) == pytest.approx(ref, abs=1e-12)
    # A hard, weakly-buffered state far from the pH-7 start is bisection-dominated
    # and genuinely needs the worst-case count, so a tiny cap (12) legitimately
    # under-converges -- only the default cap (40, the validated production count)
    # agrees with the fully-converged reference. This also confirms the loop does
    # NOT falsely declare convergence on a bisection step.
    hard = dict(tot_carbonate=1e-2, strong_anion_eq=0.3)
    ref_h = float(solve_ph(**hard, n_iter=200))
    assert float(solve_ph(**hard, n_iter=40)) == pytest.approx(ref_h, abs=1e-9)
    assert float(solve_ph(**hard, n_iter=12)) != pytest.approx(ref_h, abs=1e-6)


def test_forward_and_reverse_ad_agree_and_match_fd():
    """The IFT path must differentiate in BOTH modes: forward-mode (jax.jvp /
    jacfwd -- the per-step Jacobian-materialisation path) and reverse-mode
    (jax.grad). They must agree with each other and with central finite
    differences (the IFT tangent is exact)."""
    def f(z):
        return solve_ph(tot_carbonate=2.0e-3, tot_ammonia=3.0e-3, z_cation_eq=z)

    z0 = 2.0e-3
    g_rev = float(jax.grad(f)(z0))
    _, g_fwd = jax.jvp(f, (z0,), (1.0,))      # forward-mode directional derivative
    g_fwd = float(g_fwd)
    eps = 1e-8
    fd = (float(f(z0 + eps)) - float(f(z0 - eps))) / (2 * eps)
    assert np.isfinite(g_rev) and np.isfinite(g_fwd)
    assert g_fwd == pytest.approx(g_rev, rel=1e-9)      # both are the IFT tangent
    assert g_rev == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_jit_and_grad_under_jit():
    """solve_ph (adaptive while_loop + custom_root) is jit-clean, and grad
    composes with jit."""
    f = lambda z: solve_ph(tot_carbonate=2.0e-3, z_cation_eq=z)
    z0 = 2.0e-3
    assert float(jax.jit(f)(z0)) == pytest.approx(float(f(z0)), abs=1e-12)
    assert float(jax.jit(jax.grad(f))(z0)) == pytest.approx(float(jax.grad(f)(z0)), rel=1e-9)


# --- Global convergence: the charge balance far outside the buffered regime ---
# A bare Newton step overshoots to exp(u)=inf (NaN) -- or silently to an absurd
# pH that saturates the rate terms -- when the strong-ion charge exceeds the
# buffering. The safeguarded Newton-bisection brackets the (monotone) root and
# stays finite and correct. These cases all returned NaN / nonsense before.

# (description, kwargs, expected pH from an independent NumPy reference)
_WEAK_BUFFER_CASES = [
    ("strong acid, no buffer", dict(strong_anion_eq=1.0e-2), 2.0),
    ("strong acid + trace carbonate", dict(tot_carbonate=1e-3, strong_anion_eq=2e-2), 1.699),
    ("strong base, weak buffer", dict(tot_carbonate=1e-3, z_cation_eq=5e-2), 12.848),
    ("large acid excess", dict(tot_carbonate=1e-2, strong_anion_eq=0.3), 0.523),
]


@pytest.mark.parametrize("desc, totals, pH_expected", _WEAK_BUFFER_CASES)
def test_weak_buffer_regime_converges_finite(desc, totals, pH_expected):
    K = equilibrium_constants(jnp.asarray(293.15))
    pH = float(solve_ph(**totals))
    assert np.isfinite(pH), f"{desc}: pH is non-finite"
    assert pH == pytest.approx(pH_expected, abs=1e-2), desc
    # The root is actually solved, not just bounded.
    h = 10.0 ** (-pH)
    res = float(charge_balance_residual(jnp.asarray(h), K=K, **totals))
    assert abs(res) < 1e-10, f"{desc}: residual {res} not ~0"


def test_no_nan_over_extreme_inputs():
    """Sweep wide strong-ion imbalances against weak buffers; the solver must
    stay finite everywhere (it returned NaN here before the safeguard).

    Vectorised with ``jax.vmap`` over the whole batch -- one trace/compile of the
    pH solver applied to all cases -- rather than a Python loop, which re-traced
    and recompiled the solver's ``while_loop`` per case (~150 s for 400 cases vs
    ~0.3 s vmapped). Same coverage, fast enough to stay in the fast gate."""
    rng = np.random.default_rng(0)
    n = 400
    cases = dict(
        tot_carbonate=jnp.asarray(10.0 ** rng.uniform(-6, 0, n)),
        tot_ammonia=jnp.asarray(10.0 ** rng.uniform(-6, 0, n)),
        tot_sulfide=jnp.asarray(10.0 ** rng.uniform(-6, -1, n)),
        strong_anion_eq=jnp.asarray(rng.uniform(-1.0, 1.0, n)),
        z_cation_eq=jnp.asarray(rng.uniform(-1.0, 1.0, n)),
    )
    pH = jax.vmap(
        lambda c, a, s, sa, zc: solve_ph(
            tot_carbonate=c, tot_ammonia=a, tot_sulfide=s,
            strong_anion_eq=sa, z_cation_eq=zc)
    )(cases["tot_carbonate"], cases["tot_ammonia"], cases["tot_sulfide"],
      cases["strong_anion_eq"], cases["z_cation_eq"])
    assert bool(jnp.all(jnp.isfinite(pH)))


def test_gradient_correct_in_weak_buffer_regime():
    """AD through the safeguarded iteration is the exact implicit-function-theorem
    sensitivity even where a bare Newton step would diverge."""
    def f(san):
        return solve_ph(tot_carbonate=1e-3, strong_anion_eq=san)

    s0 = 2e-2
    g = float(jax.grad(f)(s0))
    assert np.isfinite(g)
    eps = 1e-8
    fd = (float(f(s0 + eps)) - float(f(s0 - eps))) / (2 * eps)
    assert g == pytest.approx(fd, rel=1e-5)


# ----- Ionic-strength activity corrections (issue #205) --------------------

from aquakin.core.ph_solver import debye_huckel_A  # noqa: E402


def test_activity_none_is_bit_identical_to_default():
    """activity_model='none' must reproduce the historic concentration-based
    solver exactly -- this is what keeps every validated BSM2/WATS result
    recoverable bit-for-bit."""
    kw = dict(tot_carbonate=3e-3, tot_ammonia=4e-3, tot_phosphate=1e-3,
              strong_anion_eq=2e-3, z_cation_eq=5e-3, T_kelvin=308.15)
    assert float(solve_ph(**kw)) == float(solve_ph(**kw, activity_model="none"))


def test_debye_huckel_A_value():
    assert float(debye_huckel_A(298.15)) == pytest.approx(0.51, abs=0.01)
    # Rises slowly with temperature.
    assert float(debye_huckel_A(308.15)) > float(debye_huckel_A(298.15))


@pytest.mark.parametrize("model", ["davies", "debye_huckel"])
@pytest.mark.parametrize("I", [0.01, 0.1, 0.25])
def test_pure_water_in_inert_salt_stays_neutral(model, I):
    """An inert (neutral) salt cannot change the pH of pure water: the
    *measurable* activity pH stays at the neutral value at any ionic strength.
    This is the decisive check on the activity-pH formulation -log10(g_H [H+])."""
    T = 298.15
    neutral = float(solve_ph(T_kelvin=T))                      # no salt
    salted = float(solve_ph(T_kelvin=T, activity_model=model,
                            ionic_strength_strong=I))
    assert salted == pytest.approx(neutral, abs=1e-4)


@pytest.mark.parametrize("model", ["davies", "debye_huckel"])
def test_activity_lowers_carbonate_buffer_ph(model):
    """Raising ionic strength increases dissociation (conditional pKa drop), so a
    carbonate buffer's pH falls -- by a sensible ~0.1-0.3 units at I~0.1."""
    buf = dict(tot_carbonate=5e-3, z_cation_eq=5e-3, T_kelvin=308.15)
    p0 = float(solve_ph(**buf))
    pI = float(solve_ph(**buf, activity_model=model, ionic_strength_strong=0.1))
    assert -0.4 < pI - p0 < -0.05


@pytest.mark.parametrize("model", ["davies", "debye_huckel"])
def test_activity_reduces_to_none_at_low_strength(model):
    """As the total ion content -> 0 the activity coefficients -> 1, so the
    activity path collapses onto the ideal one. (At finite ionic strength even
    the buffer's own dissolved ions contribute, so the two coincide only in the
    dilute limit.)"""
    kw = dict(tot_carbonate=1e-6, z_cation_eq=1e-6, T_kelvin=298.15)
    assert float(solve_ph(**kw, activity_model=model,
                          ionic_strength_strong=0.0)) == pytest.approx(
        float(solve_ph(**kw)), abs=2e-3)


def test_activity_solve_is_converged():
    """The coupled ionic-strength / [H+] fixed point is converged at n_iter=40
    (more iterations do not move it)."""
    kw = dict(tot_carbonate=5e-3, tot_ammonia=3e-3, z_cation_eq=5e-3,
              T_kelvin=308.15, activity_model="davies",
              ionic_strength_strong=0.1)
    assert float(solve_ph(**kw, n_iter=40)) == pytest.approx(
        float(solve_ph(**kw, n_iter=80)), abs=1e-8)


def test_activity_gradient_matches_fd():
    """AD through the activity-coupled solve is finite and matches central FD."""
    def f(z):
        return solve_ph(tot_carbonate=5e-3, z_cation_eq=z, T_kelvin=308.15,
                        activity_model="davies", ionic_strength_strong=0.1)
    z0 = 5e-3
    g = float(jax.grad(f)(z0))
    assert np.isfinite(g)
    eps = 1e-7
    fd = (float(f(z0 + eps)) - float(f(z0 - eps))) / (2 * eps)
    assert g == pytest.approx(fd, rel=1e-4)


def test_activity_forward_and_reverse_ad_agree():
    """The coupled (h, I) activity path is also wrapped in custom_root, so its
    gradient is the 2x2 implicit-function-theorem tangent. Forward-mode (jvp) and
    reverse-mode (grad) must agree -- and both match FD."""
    def f(z):
        return solve_ph(tot_carbonate=5e-3, z_cation_eq=z, T_kelvin=308.15,
                        activity_model="davies", ionic_strength_strong=0.1)
    z0 = 5e-3
    g_rev = float(jax.grad(f)(z0))
    _, g_fwd = jax.jvp(f, (z0,), (1.0,))
    g_fwd = float(g_fwd)
    eps = 1e-7
    fd = (float(f(z0 + eps)) - float(f(z0 - eps))) / (2 * eps)
    assert g_fwd == pytest.approx(g_rev, rel=1e-7)
    assert g_rev == pytest.approx(fd, rel=1e-4)


@pytest.mark.parametrize("model", ["davies", "debye_huckel"])
def test_activity_no_nan_over_extreme_inputs(model):
    """The activity-corrected path must stay finite over wide strong-ion / weak-
    buffer extremes, just like the ideal path (#382). Before the ionic-strength
    clamp a far-overshoot trial [H+] in the bracketed iteration blew the water
    self-ionisation term up to I ~ 1e20-1e25, where the activity coefficients
    overflow to inf and the conditional constants become NaN -- which the
    bracketing could then never recover from (most extreme cases returned NaN)."""
    rng = np.random.default_rng(0)
    n = 400
    sa = rng.uniform(0.0, 0.5, n)            # strong-anion charge / ionic strength
    cb = 10.0 ** rng.uniform(-6, -1, n)
    pH = jax.vmap(
        lambda c, s: solve_ph(
            tot_carbonate=c, tot_ammonia=2e-3, strong_anion_eq=s,
            activity_model=model, ionic_strength_strong=s)
    )(jnp.asarray(cb), jnp.asarray(sa))
    assert bool(jnp.all(jnp.isfinite(pH)))


@pytest.mark.parametrize("model", ["davies", "debye_huckel"])
def test_activity_gradient_finite_at_degenerate_point(model):
    """The activity-path IFT tangent is a 2x2 determinant solve; at a degenerate
    input -- all-zero totals with zero ionic strength -- the 2x2 can go near-
    singular, so the determinant is floored to keep the gradient finite (#382).
    Both the forward value and its gradient must stay finite there."""
    def f(s):
        return solve_ph(strong_anion_eq=s, activity_model=model,
                        ionic_strength_strong=s)
    assert np.isfinite(float(f(0.0)))
    assert np.isfinite(float(jax.grad(f)(0.0)))
    # And finite at the issue's extreme buffered point.
    g = jax.grad(lambda s: solve_ph(
        tot_carbonate=1e-3, strong_anion_eq=s, activity_model=model,
        ionic_strength_strong=s))
    assert np.isfinite(float(g(0.01)))


def test_activity_clamp_inactive_at_physical_root():
    """The ionic-strength clamp only tames the transient overshoot: the converged
    self-consistent ionic strength at any routine root sits far below the physical
    ceiling _I_MAX, so jnp.clip(I, 0, _I_MAX) is exactly identity there and the
    #382 safeguard cannot perturb the converged pH. (A digester/sewer I is
    ~0.01-0.2 M; _I_MAX is 10 M.)"""
    from aquakin.core.ph_solver import _I_MAX
    for I_strong in (1e-3, 0.05, 0.25):
        pH, I = solve_ph(tot_carbonate=5e-3, tot_ammonia=3e-3, z_cation_eq=5e-3,
                         T_kelvin=308.15, activity_model="davies",
                         ionic_strength_strong=I_strong, return_ionic_strength=True)
        assert np.isfinite(float(pH))
        assert float(I) < 0.1 * _I_MAX


def test_activity_pH_independent_of_iteration_cap():
    """The adaptive coupled loop stops once both [H+] and the ionic strength have
    settled, so the activity-path pH is independent of the n_iter cap once it
    exceeds the (small) count needed."""
    kw = dict(tot_carbonate=5e-3, tot_ammonia=3e-3, z_cation_eq=5e-3,
              T_kelvin=308.15, activity_model="davies", ionic_strength_strong=0.1)
    ref = float(solve_ph(**kw, n_iter=200))
    for cap in (15, 25, 40, 200):
        assert float(solve_ph(**kw, n_iter=cap)) == pytest.approx(ref, abs=1e-10)


def test_invalid_activity_model_raises():
    with pytest.raises(ValueError, match="activity_model"):
        solve_ph(tot_carbonate=1e-3, activity_model="bogus")


def test_activity_ionic_strength_does_not_double_count_strong_cations():
    """Regression: the speciation-layer ionic strength must count each explicit
    strong cation once. A monovalent strong-cation *state* and the equivalent
    fixed-charge *offset* describe identical chemistry, so under activity
    correction they must give the same pH. (Previously the I_strong seed used the
    post-fold net cation charge, which already contained the explicit cations, so
    an explicit ``strong_cations`` state was counted twice and shifted the pH.)"""
    from aquakin.core.speciation import build_ph_derived_fn

    species_index = {"S_cat": 0, "S_CO2": 1}
    base = dict(field="pH", temperature_field="T", temperature_units="kelvin",
                activity_model="davies", n_iter=60,
                totals={"carbonate": {"species": "S_CO2", "molar_mass": 12000}})
    c = 5.0e-3                # eq/L of monovalent cation charge (buffer scale)
    co2 = 5.0e-3 * 12000      # gC/m3 so tot_carbonate == 5e-3 mol/L

    # (A) the cation as an explicit strong_cations state (molar_mass 1 -> eq == C)
    fn_state, _, _ = build_ph_derived_fn(
        {**base, "z_cation_eq": 0.0,
         "strong_cations": [{"species": "S_cat", "molar_mass": 1.0, "charge": 1}]},
        species_index)
    # (B) the same cation charge as the fixed monovalent offset, no strong state
    fn_offset, _, _ = build_ph_derived_fn(
        {**base, "z_cation_eq": c, "strong_cations": []}, species_index)

    cond = {"T": jnp.array([308.15])}
    pH_state = float(fn_state(jnp.array([c, co2]), None, cond, 0)["pH"])
    pH_offset = float(fn_offset(jnp.array([0.0, co2]), None, cond, 0)["pH"])
    assert np.isfinite(pH_state)
    assert pH_state == pytest.approx(pH_offset, abs=1e-9)
