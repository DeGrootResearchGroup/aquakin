"""Validation: the layered biofilm reactor reproduces the analytical deep-biofilm flux.

A biofilm thick enough that the rate-limiting substrate is consumed before it
reaches the wall ("deep", or fully penetration-limited) has a closed-form
steady-state flux. Integrating the 1-D diffusion--reaction balance
``D*S'' = k*Xf*S/(Ks+S)`` once, using the no-flux condition deep inside the
film, gives

    J = sqrt( 2*D*k*Xf*( Ss - Ks*ln(1 + Ss/Ks) ) )

where ``Ss`` is the substrate concentration at the biofilm surface, ``k`` the
maximum specific utilization rate and ``Xf`` the (uniform, fixed) biomass
density. This is exact, so it is the cleanest possible check on the
diffusion--reaction kernel: no reference implementation to transcribe and no
published table to round-trip.

The tests pin three things:

1. The converged flux matches the closed form.
2. Convergence in ``n_layers`` is **first order**, so halving the layer
   thickness halves the error. That is a property of the scheme worth pinning:
   it sets how many layers a given biofilm needs.
3. The accuracy at a given ``n_layers`` is governed by the penetration ratio
   ``L_f/delta``, where ``delta = sqrt(D*Ks/(k*Xf))``. A fully penetrated film
   is accurate with very few layers; a deeply penetration-limited one is not.

References
----------
Rittmann, B.E. & McCarty, P.L. (2001). Environmental Biotechnology: Principles
and Applications. McGraw-Hill. (The steady-state biofilm model and its deep-
biofilm limit.)
Saez, P.B. & Rittmann, B.E. (1992). Accurate pseudoanalytical solution for
steady-state biofilms. Biotechnology and Bioengineering 39(7), 790-793.
"""

import math
import os

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

pytestmark = pytest.mark.validation

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "monod_biofilm.yaml")

# Benchmark physical parameters (days, metres, g/m3).
D = 1.0e-4  # effective diffusivity in the biofilm, m2/d
K = 10.0  # maximum specific utilization rate, 1/d
KS = 10.0  # half-saturation concentration, g/m3
XF = 1.0e4  # biofilm biomass density, g/m3
SB = 20.0  # bulk (and, with no boundary layer, surface) concentration, g/m3
AV = 50.0  # biofilm area per bulk volume, 1/m (immaterial: the bulk is clamped)
LBL = 1.0e-9  # external boundary layer, m -- negligible, so Ss == bulk

#: Penetration length scale. ``L_f / DELTA`` is what sets the layer count needed.
DELTA = math.sqrt(D * KS / (K * XF))


def _exact_flux(ss=SB):
    """Closed-form deep-biofilm substrate flux, g/(m2 d)."""
    return math.sqrt(2.0 * D * K * XF * (ss - KS * math.log(1.0 + ss / KS)))


@pytest.fixture(scope="module")
def model():
    return aquakin.load_model_from_file(FIXTURE)


def _flux(model, thickness, n_layers):
    """Steady-state substrate flux into a biofilm held at a fixed surface value.

    Clamps the bulk at ``SB`` so the surface boundary condition is exactly the
    analytical one, integrates to steady state, and reads the flux back as the
    total consumption per unit biofilm area.
    """
    reactor = aquakin.BiofilmReactor(
        model,
        aquakin.SpatialConditions.uniform(T=293.15),
        n_layers=n_layers,
        thickness=thickness,
        area_per_volume=AV,
        diffusivity=D,
        boundary_layer=LBL,
        clamp_bulk=True,
    )
    i_s = model.species_index["SS"]
    i_x = model.species_index["XB"]
    y0 = jnp.zeros((n_layers + 1, model.n_species))
    y0 = y0.at[:, i_s].set(SB).at[1:, i_x].set(XF)
    solution = reactor.solve(y0, params=model.default_parameters(), t_span=(0.0, 20.0))
    profile = np.asarray(solution.profile[-1])[1:, i_s]
    rate = K * XF * profile / (KS + profile)
    return float(np.sum(rate) * (thickness / n_layers))


def test_deep_biofilm_flux_matches_the_closed_form(model):
    """The converged flux reproduces the analytical deep-biofilm solution.

    The scheme is first order, so neither grid is converged on its own; the
    Richardson extrapolation of the two is what the scheme converges to, and
    that is what must equal the exact flux.
    """
    thickness = 20.0 * DELTA  # unambiguously deep: substrate never reaches the wall
    coarse = _flux(model, thickness, 64)
    fine = _flux(model, thickness, 128)
    extrapolated = 2.0 * fine - coarse  # first-order Richardson

    exact = _exact_flux()
    assert extrapolated == pytest.approx(exact, rel=2e-3)
    # Both grids must also bracket sensibly: the scheme over-predicts the flux
    # on a coarse grid (the surface cell resolves the steep gradient poorly).
    assert exact < fine < coarse


def test_layer_convergence_is_first_order(model):
    """Halving the layer thickness halves the flux error.

    Pins the convergence order, which is what turns "how thick is the biofilm"
    into "how many layers do I need".
    """
    thickness = 20.0 * DELTA
    exact = _exact_flux()
    errors = [abs(_flux(model, thickness, n) / exact - 1.0) for n in (16, 32, 64, 128)]
    ratios = [b / a for a, b in zip(errors, errors[1:])]
    assert all(0.4 < r < 0.6 for r in ratios), f"not first order: {errors} -> {ratios}"


@pytest.mark.parametrize(
    ("penetration_ratio", "tolerance"),
    [
        (0.5, 0.01),  # fully penetrated: flat profile, four layers resolve it
        (1.0, 0.02),
        (2.0, 0.08),  # partially penetrated: four layers are already poor
    ],
)
def test_default_layer_count_accuracy_tracks_penetration(model, penetration_ratio, tolerance):
    """Four layers -- the ``IFASUnit`` default -- are only accurate while the
    biofilm is close to fully penetrated.

    The error at a fixed layer count is set by ``L_f/delta``, not by ``L_f``
    alone. This pins the band so a change to the discretization that quietly
    degrades a thick, partially penetrated biofilm fails here.
    """
    thickness = penetration_ratio * DELTA
    reference = _flux(model, thickness, 512)
    coarse = _flux(model, thickness, 4)
    assert coarse == pytest.approx(reference, rel=tolerance)
