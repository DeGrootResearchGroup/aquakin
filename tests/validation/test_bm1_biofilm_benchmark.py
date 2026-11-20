"""Validation: the IFAS/MBBR unit reproduces IWA biofilm Benchmark Problem 1.

BM1 is the first of the three benchmark problems the IWA Task Group on Biofilm
Modeling defined to compare biofilm modelling approaches side by side. It is a
single-species biofilm on a flat substratum in a completely mixed reactor: one
particulate component (active heterotrophic biomass) growing on an organic
substrate with oxygen as the electron acceptor, at **fixed biofilm thickness and
fixed biomass density**, with no conversion in the bulk phase. Solutions were
published for models ranging from a closed-form analytical one to 3-D numerical
ones, and the flat-biofilm models agree closely with each other -- which is what
makes the published table usable as a reference.

That specification is exactly aquakin's mature-biofilm model: ``IFASUnit`` holds
the attached biomass as a sustained reservoir (``biofilm_fixed_mask``) at a fixed
``biofilm_thickness``, and resolves the solubles over depth. So BM1 exercises the
unit as shipped, rather than a special mode built for the test.

**Which cases are reproduced.** The benchmark defines five. Three are checked
here -- the standard case, biomass limitation (a thin, fully penetrated film) and
reduced in-biofilm diffusivity. The remaining two (low bulk oxygen, and an
external mass-transfer boundary layer) report a *mean biofilm thickness below the
specified value*, because there lysis exceeds growth and the thickness becomes a
result rather than an input. ``IFASUnit`` takes thickness as a fixed parameter, so
those two cases are outside what this unit models and are deliberately left out.

**What is compared against what.** Following the practice used for the settler,
the published reference model is transcribed independently rather than trusted
from the table alone. Both consumption terms are proportional to the same growth
rate, so ``D_O2*O'' = (1-Y_H)*D_S*S''``; integrating twice from the no-flux wall
gives the exact coupling ``O = O_base + (1-Y_H)*(D_S/D_O2)*(S - S_base)``, which
reduces the biofilm to a single second-order ODE in the substrate. That is
integrated from the wall outward as an initial-value problem and closed by two
nested root finds -- one on the wall concentration to meet the surface condition,
one on the bulk concentration to meet the reactor balance. The resulting solution
is checked against the published table, and the unit is then checked against it.

References
----------
Morgenroth, E., Eberl, H.J., van Loosdrecht, M.C.M., Noguera, D.R., Pizarro,
G.E., Picioreanu, C., Rittmann, B.E., Schwarz, A.O. & Wanner, O. (2004).
Comparing biofilm models for a single species biofilm system. Water Science and
Technology 49(11-12), 145-154.
Noguera, D.R. & Morgenroth, E. (2004). Introduction to the IWA Task Group on
Biofilm Modeling. Water Science and Technology 49(11-12), 131-136.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

import aquakin
from aquakin.plant import Aeration, IFASUnit, Plant

pytestmark = pytest.mark.validation

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "bm1_biofilm.yaml")

# --- the benchmark specification (physical, kinetic and stoichiometric) ------
Q = 0.02  # flow rate, m3/d
V_R = 1.25e-3  # reactor volume, m3
A_F = 0.1  # biofilm surface area, m2
L_F = 500e-6  # average biofilm thickness, m
L_L = 0.0  # boundary-layer thickness, m
X_F = 1.0e4  # biofilm biomass density, g_COD/m3
C_S_IN = 30.0  # substrate influent concentration, g_COD/m3
C_O2 = 10.0  # bulk oxygen concentration, g_O2/m3 (held fixed)

MU_MAX = 6.0  # maximum specific growth rate, 1/d
K_H = 0.4  # lysis rate, 1/d
K_S = 4.0  # substrate half-maximum-rate concentration, g_COD/m3
K_O2 = 0.2  # oxygen half-maximum-rate concentration, g_O2/m3
Y_H = 0.63  # true yield, g_CODX/g_CODS
D_S = 1.0e-4  # substrate diffusion coefficient in water, m2/d
D_O2 = 2.0e-4  # oxygen diffusion coefficient in water, m2/d

#: Carrier area per unit bulk volume (1/m).
A_V = A_F / V_R

#: Published results for the cases reproduced here, taken from the numerical
#: 1-D column of the benchmark's result tables. Fluxes in g/(m2 d),
#: concentrations in g/m3. ``None`` marks a value the table reports as a bare
#: zero, which carries no significant figures to compare against.
PUBLISHED = {
    "standard": dict(
        case=dict(),
        cs_bulk=4.4,
        cs_base=0.008,
        co2_base=9.4,
        flux_s=5.1,
        flux_o2=1.9,
    ),
    "biomass_limited": dict(
        case=dict(thickness=20e-6),
        cs_bulk=22.0,
        cs_base=22.0,
        co2_base=10.0,
        flux_s=1.6,
        flux_o2=0.6,
    ),
    "reduced_diffusivity": dict(
        case=dict(diffusivity_ratio=0.2),
        cs_bulk=9.6,
        cs_base=None,
        co2_base=8.8,
        flux_s=4.0,
        flux_o2=1.5,
    ),
}


# --- independent reference solution ------------------------------------------


def _reference(thickness=L_F, co2_bulk=C_O2, diffusivity_ratio=1.0):
    """Solve BM1 independently of aquakin.

    Parameters
    ----------
    thickness : float
        Biofilm thickness ``L_F`` (m).
    co2_bulk : float
        Bulk oxygen concentration, held fixed (g_O2/m3).
    diffusivity_ratio : float
        In-biofilm diffusivity as a fraction of the pure-water value.

    Returns
    -------
    dict
        ``cs_bulk``, ``cs_base``, ``co2_base``, ``flux_s`` and ``flux_o2``.
    """
    d_s, d_o2 = D_S * diffusivity_ratio, D_O2 * diffusivity_ratio
    # S'' = scale * monod(S); oxygen follows the substrate affinely (see the
    # module docstring), so only the substrate is integrated.
    scale = (MU_MAX / Y_H) * X_F / d_s
    slope = (1.0 - Y_H) * d_s / d_o2

    def shoot(cs_base, cs_surface):
        """Integrate wall -> surface from a trial wall concentration."""
        o_base = co2_bulk - slope * (cs_surface - cs_base)

        def rhs(_z, y):
            substrate = max(y[0], 0.0)
            oxygen = max(o_base + slope * (substrate - cs_base), 0.0)
            growth = (substrate / (K_S + substrate)) * (oxygen / (K_O2 + oxygen))
            return [y[1], scale * growth]

        sol = solve_ivp(
            rhs, (0.0, thickness), [cs_base, 0.0], method="Radau", rtol=1e-12, atol=1e-14
        )
        return sol.y[0, -1], sol.y[1, -1], o_base

    def residuals(log_unknowns):
        """Surface-concentration and reactor-balance residuals.

        Solved jointly rather than as nested root finds: in a deep biofilm the
        wall concentration is many orders of magnitude below the surface one, so
        a bracketing search on it costs hundreds of stiff integrations. Working
        in log coordinates keeps both unknowns well scaled and lets one Newton
        solve do the whole job.
        """
        cs_base, cs_surface = np.exp(log_unknowns)
        surface, gradient, _ = shoot(cs_base, cs_surface)
        return [
            surface / cs_surface - 1.0,
            (Q * (C_S_IN - cs_surface) - d_s * gradient * A_F) / (Q * C_S_IN),
        ]

    guess = np.log([min(0.1, 0.01 * C_S_IN), 0.15 * C_S_IN])
    solution = fsolve(residuals, guess, xtol=1e-13)
    assert np.max(np.abs(residuals(solution))) < 1e-9, "reference solve did not converge"

    cs_base, cs_bulk = np.exp(solution)
    _, gradient, co2_base = shoot(cs_base, cs_bulk)
    flux_s = d_s * gradient
    return dict(
        cs_bulk=cs_bulk,
        cs_base=cs_base,
        co2_base=co2_base,
        flux_s=flux_s,
        flux_o2=flux_s * (1.0 - Y_H),
    )


# --- the unit under test ------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return aquakin.load_model_from_file(FIXTURE)


def _ifas_solution(model, n_layers, thickness=L_F, co2_bulk=C_O2, diffusivity_ratio=1.0):
    """Run the benchmark reactor as a one-unit plant and read the indices back.

    The bulk oxygen is held at the benchmark value by a very fast aeration term,
    which is how a completely mixed reactor whose oxygen is specified rather than
    fed is expressed in the flowsheet.

    The default steady-state criterion is a **norm** test on ``dstate/dt``, which
    this problem defeats: the frozen biofilm biomass contributes ``n_layers *
    1e4`` to the state norm, so the tolerance it implies is far larger than the
    bulk substrate's own residual and the solve stops with the bulk still moving.
    The componentwise ``scaled_residual`` criterion is the one to use here.
    """
    i_s, i_o2 = model.species_index["SS"], model.species_index["SO2"]
    diffusivity = jnp.array([D_S, D_O2, 0.0])
    unit = IFASUnit(
        "reactor",
        model,
        volume=V_R,
        input_port_names=["in"],
        specific_surface_area=A_V,
        fill_fraction=1.0,
        biofilm_thickness=thickness,
        n_layers=n_layers,
        diffusivity=diffusivity * diffusivity_ratio,
        # The external film is liquid, so it keeps the pure-water diffusivity
        # even when the in-biofilm value is reduced.
        boundary_diffusivity=diffusivity,
        # The benchmark specifies no boundary layer; a negligible one stands in,
        # since the mass-transfer coefficient is D/L_L.
        boundary_layer=1e-9,
        conditions={"T": 293.15},
        aeration=Aeration(kla=1.0e6, do_sat=co2_bulk, species="SO2"),
        biofilm_initial=model.concentrations({"XH": X_F, "SS": C_S_IN, "SO2": co2_bulk}),
    )
    plant = Plant("bm1", time_unit="d")
    plant.add_unit(unit)
    plant.add_influent(
        "feed", model.influent({"SS": C_S_IN, "SO2": co2_bulk}, Q=Q), to="reactor.in"
    )
    result = plant.run_to_steady_state(
        criterion="scaled_residual", tol=1e-7, scale_floor=1e-3, max_time=200.0
    )
    state = np.asarray(result.state).reshape(n_layers + 1, model.n_species)
    residual = np.asarray(plant.derivative(result.state)).reshape(n_layers + 1, model.n_species)
    cs_bulk = float(state[0, i_s])
    flux_s = Q * (C_S_IN - cs_bulk) / A_F
    return dict(
        cs_bulk=cs_bulk,
        # Layers run surface-first, so the last one sits against the carrier.
        cs_base=float(state[-1, i_s]),
        co2_base=float(state[-1, i_o2]),
        flux_s=flux_s,
        flux_o2=flux_s * (1.0 - Y_H),
        bulk_residual=float(residual[0, i_s]),
    )


def _extrapolated(model, coarse=32, fine=64, **case):
    """First-order Richardson extrapolation of the unit's result in ``n_layers``."""
    lo = _ifas_solution(model, coarse, **case)
    hi = _ifas_solution(model, fine, **case)
    keys = ("cs_bulk", "cs_base", "co2_base", "flux_s", "flux_o2")
    return {k: 2.0 * hi[k] - lo[k] for k in keys}, lo, hi


# --- tests --------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_reference_solution_reproduces_the_published_table(name):
    """The independently transcribed reference matches the published results.

    The table reports two significant figures, so agreement is checked at the
    precision the publication carries.
    """
    published = PUBLISHED[name]
    got = _reference(**published["case"])
    for key in ("cs_bulk", "cs_base", "co2_base", "flux_s", "flux_o2"):
        expected = published[key]
        if expected is None:  # reported as a bare zero
            assert got[key] < 1e-3
            continue
        # The absolute floor is for the concentration at the base of a deep
        # biofilm: the table reports it as a single significant figure of a
        # number within a hair of zero, so a relative comparison there is
        # comparing rounding, not physics.
        assert got[key] == pytest.approx(expected, rel=0.08, abs=5e-3), (
            f"{name}.{key}: reference {got[key]:.4g} vs published {expected:.4g}"
        )


def test_ifas_unit_reproduces_the_standard_case(model):
    """The IFAS tank reproduces the benchmark's standard case.

    The flux and the effluent substrate concentration are the benchmark's own
    headline performance indices; the concentration at the base of the biofilm
    is what shows the film is deep (substrate exhausted before the carrier).
    """
    reference = _reference()
    extrapolated, _, fine = _extrapolated(model)

    assert abs(fine["bulk_residual"]) < 1e-3, "the bulk is not at steady state"
    assert extrapolated["flux_s"] == pytest.approx(reference["flux_s"], rel=5e-3)
    assert extrapolated["cs_bulk"] == pytest.approx(reference["cs_bulk"], rel=5e-3)
    assert extrapolated["co2_base"] == pytest.approx(reference["co2_base"], rel=5e-3)
    # ... and therefore against the published table itself.
    assert extrapolated["flux_s"] == pytest.approx(PUBLISHED["standard"]["flux_s"], rel=0.02)
    assert extrapolated["cs_bulk"] == pytest.approx(PUBLISHED["standard"]["cs_bulk"], rel=0.02)
    # Deep biofilm: the substrate is gone long before the carrier wall.
    assert fine["cs_base"] < 0.05 * fine["cs_bulk"]


def test_ifas_unit_reproduces_the_biomass_limited_case(model):
    """A thin, fully penetrated film: removal is set by how much biomass there is.

    With the thickness cut to 20 um neither diffusion nor oxygen limits the rate,
    so the profile is flat and even a coarse grid is accurate -- the opposite
    regime to the standard case.
    """
    case = PUBLISHED["biomass_limited"]
    reference = _reference(**case["case"])
    got = _ifas_solution(model, 16, **case["case"])

    assert abs(got["bulk_residual"]) < 1e-3
    assert got["flux_s"] == pytest.approx(reference["flux_s"], rel=0.01)
    assert got["cs_bulk"] == pytest.approx(reference["cs_bulk"], rel=0.01)
    # Fully penetrated: the wall sees essentially the surface concentration.
    assert got["cs_base"] == pytest.approx(got["cs_bulk"], rel=0.02)


def test_ifas_unit_reproduces_the_reduced_diffusivity_case(model):
    """Cutting the in-biofilm diffusivity by 80% cuts the flux by only ~20%.

    Deep-biofilm kinetics make the flux scale roughly as the square root of the
    diffusivity, and the reactor balance absorbs part of what is left -- a lower
    flux raises the bulk concentration, which raises the flux again. Reproducing
    that muted response is a sharper test of the transport than the flux alone.
    """
    case = PUBLISHED["reduced_diffusivity"]
    reference = _reference(**case["case"])
    extrapolated, _, fine = _extrapolated(model, **case["case"])

    assert abs(fine["bulk_residual"]) < 1e-3
    assert extrapolated["flux_s"] == pytest.approx(reference["flux_s"], rel=0.01)
    assert extrapolated["cs_bulk"] == pytest.approx(reference["cs_bulk"], rel=0.01)

    standard = _reference()
    assert reference["flux_s"] / standard["flux_s"] == pytest.approx(0.80, abs=0.03)


def test_coarse_layer_counts_are_inadequate_for_this_biofilm(model):
    """Four layers -- the ``IFASUnit`` default -- are far off on this biofilm.

    BM1's standard case is deeply penetration-limited (the film is several times
    the penetration depth), which is the regime where the first-order layer
    discretization needs many layers. Pinning the size of the error keeps the
    default honest: a user who takes it on a biofilm like this gets a flux that
    is high by close to ten per cent and an effluent that is wrong by half.
    """
    reference = _reference()
    coarse = _ifas_solution(model, 4)

    assert coarse["flux_s"] > 1.05 * reference["flux_s"]
    assert coarse["cs_bulk"] < 0.6 * reference["cs_bulk"]
