"""Unit tests for the moving-boundary biofilm kernel (advection + thickness).

Pure functions, no solve: the displacement velocity implied by
incompressibility, the arbitrary Lagrangian--Eulerian flux on the stretching
grid, and the thickness balance. Checked against cases whose answer is known in
closed form, so the kernel is pinned independently of any reactor built on it.

References
----------
Wanner, O. & Gujer, W. (1986). A multispecies biofilm model. Biotechnology and
Bioengineering 28(3), 314-328.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.integrate.biofilm_growth import (
    BALANCED_DETACHMENT,
    DETACHMENT_LAWS,
    DETACHMENT_POWERS,
    advection_transport,
    advection_velocity,
    detachment_velocity,
    layer_thickness,
    surface_detachment,
    thickness_derivative,
)

RHO = 1.0e4  # biofilm density, g/m3
N = 8


def test_detachment_velocity_laws():
    assert float(detachment_velocity(3.0, 5e-4, "constant")) == pytest.approx(3.0)
    assert float(detachment_velocity(3.0, 5e-4, "linear")) == pytest.approx(3.0 * 5e-4)
    assert float(detachment_velocity(3.0, 5e-4, "quadratic")) == pytest.approx(3.0 * 5e-4**2)
    with pytest.raises(ValueError, match="law must be one of"):
        detachment_velocity(1.0, 1.0, "exponential")


def test_velocity_is_zero_at_the_wall_and_integrates_production_outward():
    """``u(z)`` is the net production below ``z``, divided by the density.

    With uniform production the velocity must rise linearly from zero at the
    wall to ``mu * L_f`` at the surface.
    """
    thickness = 4.0e-4
    dz = thickness / N
    growth = 0.5  # net specific production, 1/d
    production = jnp.full((N,), growth * RHO)  # g/(m3 d), uniform

    faces = advection_velocity(production, RHO, dz)
    assert faces.shape == (N + 1,)
    assert float(faces[-1]) == pytest.approx(0.0)  # wall: impermeable
    assert float(faces[0]) == pytest.approx(growth * thickness)  # surface

    # Linear in depth: face f (surface-first) sits at depth (N - f) * dz.
    depth_from_wall = (N - np.arange(N + 1)) * dz
    np.testing.assert_allclose(np.asarray(faces), growth * depth_from_wall, rtol=1e-12)


def test_no_production_means_no_displacement():
    faces = advection_velocity(jnp.zeros((N,)), RHO, 1e-5)
    np.testing.assert_allclose(np.asarray(faces), 0.0, atol=0.0)


def test_steady_thickness_matches_the_closed_form():
    """Uniform growth against quadratic detachment settles at ``L_f = mu / lambda``.

    ``dL_f/dt = mu*L_f - lambda*L_f**2`` is zero at ``L_f = mu/lambda``, which is
    the standard closure for an eroding biofilm and the sharpest available check
    that the surface velocity and the detachment law are consistent.
    """
    growth, lam = 0.5, 800.0  # 1/d and 1/(m d)
    expected = growth / lam

    def dLdt(thickness):
        dz = layer_thickness(thickness, N)
        faces = advection_velocity(jnp.full((N,), growth * RHO), RHO, dz)
        return float(thickness_derivative(faces, detachment_velocity(lam, thickness, "quadratic")))

    assert dLdt(expected) == pytest.approx(0.0, abs=1e-15)
    assert dLdt(0.5 * expected) > 0.0  # thin film grows
    assert dLdt(2.0 * expected) < 0.0  # thick film erodes


def test_advection_conserves_what_it_moves():
    """Nothing is created: the change in inventory equals what crossed the surface.

    The wall face is impermeable, so the only exit is the biofilm surface. The
    inventory per unit area is ``sum(X) * dz`` and *both* factors move -- the
    concentrations through the fluxes and ``dz`` because the film is growing --
    so the balance has to be taken on the product, which is the mass.
    """
    thickness, growth = 4.0e-4, 0.5
    dz = layer_thickness(thickness, N)
    rng = np.random.default_rng(0)
    layers = jnp.asarray(rng.uniform(100.0, 5000.0, size=(N, 2)))
    bulk = jnp.asarray([10.0, 2.0])
    mask = jnp.asarray([True, True])

    faces = advection_velocity(jnp.full((N,), growth * RHO), RHO, dz)
    rate = thickness_derivative(faces, detachment_velocity(300.0, thickness, "quadratic"))
    assert float(rate) != 0.0  # the film must actually be moving for this to bite
    d_layers, surface_flux = advection_transport(layers, bulk, faces, rate, dz, mask)

    # d/dt sum(X*dz) = sum(dX/dt)*dz + sum(X)*d(dz)/dt, with d(dz)/dt = dL/dt / N.
    inventory_rate = np.asarray(jnp.sum(d_layers, axis=0) * dz + jnp.sum(layers, axis=0) * rate / N)
    np.testing.assert_allclose(inventory_rate, -np.asarray(surface_flux), rtol=1e-10)


def test_a_growing_uniform_film_holds_its_density():
    """Advection plus stretching must leave a film at its density exactly.

    This is the premise the displacement velocity is derived from: production
    anywhere inside is pushed outward precisely so that the total particulate
    concentration never moves. Growth and stretching cancel only if the cells'
    changing volume is accounted for, so this is what catches a missing
    dilation term.
    """
    growth, lam = 0.5, 300.0
    thickness = 4.0e-4  # deliberately away from the steady thickness, so dL/dt != 0
    dz = layer_thickness(thickness, N)
    layers = jnp.full((N, 1), RHO)

    faces = advection_velocity(jnp.full((N,), growth * RHO), RHO, dz)
    rate = thickness_derivative(faces, detachment_velocity(lam, thickness, "quadratic"))
    assert float(rate) != 0.0
    d_layers, _ = advection_transport(layers, jnp.zeros((1,)), faces, rate, dz, jnp.asarray([True]))
    # Production is uniform at growth*RHO; the transport must cancel it exactly.
    np.testing.assert_allclose(np.asarray(d_layers) + growth * RHO, 0.0, atol=1e-9 * RHO)


def test_only_masked_species_are_carried():
    """Solubles move by diffusion, not with the biofilm matrix."""
    dz = layer_thickness(4.0e-4, N)
    layers = jnp.tile(jnp.asarray([1000.0, 50.0]), (N, 1))
    faces = advection_velocity(jnp.full((N,), 0.5 * RHO), RHO, dz)
    d_layers, surface_flux = advection_transport(
        layers, jnp.zeros((2,)), faces, 0.0, dz, jnp.asarray([True, False])
    )
    assert np.all(np.asarray(d_layers)[:, 1] == 0.0)
    assert float(surface_flux[1]) == 0.0


def test_a_uniform_film_in_steady_state_does_not_restratify():
    """A single species at the biofilm density is already at its steady profile.

    Uniform growth, uniform composition and a thickness that is holding steady:
    advection must then leave every layer alone, because what each layer passes
    outward is exactly what it receives from within.
    """
    growth, lam = 0.5, 800.0
    thickness = growth / lam
    dz = layer_thickness(thickness, N)
    layers = jnp.full((N, 1), RHO)

    faces = advection_velocity(jnp.full((N,), growth * RHO), RHO, dz)
    rate = thickness_derivative(faces, detachment_velocity(lam, thickness, "quadratic"))
    d_layers, surface_flux = advection_transport(
        layers, jnp.zeros((1,)), faces, rate, dz, jnp.asarray([True])
    )
    # Production exactly balances the divergence, so the net is the growth term.
    np.testing.assert_allclose(np.asarray(d_layers), -growth * RHO, rtol=1e-10)
    assert float(surface_flux[0]) == pytest.approx(
        detachment_velocity(lam, thickness, "quadratic") * RHO, rel=1e-10
    )


def test_inward_velocity_draws_from_the_bulk():
    """A film eroding faster than it grows takes its surface flux from the bulk.

    The upwind switch has to pick the value outside the face when flow is
    inward, or a shrinking film would manufacture material at its surface.
    """
    dz = layer_thickness(4.0e-4, N)
    layers = jnp.full((N, 1), RHO)
    bulk = jnp.asarray([7.0])
    faces = advection_velocity(jnp.zeros((N,)), RHO, dz)  # no growth at all
    # Positive dL/dt with no production means the grid is stretching faster than
    # material arrives, so the surface face draws inward.
    d_layers, surface_flux = advection_transport(
        layers, bulk, faces, 1.0e-4, dz, jnp.asarray([True])
    )
    assert float(surface_flux[0]) < 0.0  # into the film
    assert float(surface_flux[0]) == pytest.approx(-1.0e-4 * 7.0, rel=1e-10)


def test_layer_thickness_floors_a_vanishing_film():
    assert float(layer_thickness(0.0, N)) > 0.0
    assert float(layer_thickness(8.0e-4, N)) == pytest.approx(1.0e-4)


# --- a specified total thickness ---------------------------------------------


def test_balanced_detachment_holds_the_thickness_exactly():
    """``"balance"`` sets detachment to the velocity arriving at the surface.

    A thickness that is specified rather than predicted has to stay put to the
    last bit, not to a tolerance, or a long solve drifts off the value the
    problem was posed with.
    """
    dz = layer_thickness(5.0e-4, N)
    faces = advection_velocity(jnp.full((N,), 0.5 * RHO), RHO, dz)
    u_det = surface_detachment(faces, 0.0, 5.0e-4, "balance")

    assert float(u_det) == float(faces[0])
    assert float(thickness_derivative(faces, u_det)) == 0.0


def test_balanced_detachment_still_sheds_biomass_at_the_surface():
    """Pinning the thickness must not pin the film: growth still leaves.

    With no grid motion the whole displacement crosses the outer face, so what
    detaches is the surface concentration carried at the arriving velocity.
    """
    dz = layer_thickness(5.0e-4, N)
    layers = jnp.full((N, 1), RHO)
    faces = advection_velocity(jnp.full((N,), 0.5 * RHO), RHO, dz)
    u_det = surface_detachment(faces, 0.0, 5.0e-4, "balance")
    d_layers, surface_flux = advection_transport(
        layers,
        jnp.zeros((1,)),
        faces,
        thickness_derivative(faces, u_det),
        dz,
        jnp.asarray([True]),
    )
    assert float(surface_flux[0]) == pytest.approx(float(faces[0]) * RHO, rel=1e-12)
    # Uniform production is exactly cancelled, so the film holds its density.
    np.testing.assert_allclose(np.asarray(d_layers), -0.5 * RHO, rtol=1e-10)


def test_surface_detachment_dispatches_the_rate_driven_laws():
    """The shared resolver must agree with the plain law for every other case."""
    faces = advection_velocity(jnp.full((N,), 0.5 * RHO), RHO, layer_thickness(4e-4, N))
    for law in ("constant", "linear", "quadratic"):
        assert float(surface_detachment(faces, 3.0, 4.0e-4, law)) == pytest.approx(
            float(detachment_velocity(3.0, 4.0e-4, law))
        )


def test_every_accepted_law_is_listed():
    """``DETACHMENT_LAWS`` is what the unit validates against, so it must cover
    the rate-driven powers and the balanced case."""
    assert set(DETACHMENT_LAWS) == set(DETACHMENT_POWERS) | {BALANCED_DETACHMENT}
