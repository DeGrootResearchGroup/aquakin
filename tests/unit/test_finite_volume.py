"""The 1-D finite-volume building blocks in ``aquakin.integrate.finite_volume``.

Each check is against a hand calculation or a conservation law, on a grid of
unequal cells so a width / spacing mix-up cannot pass.
"""

import jax.numpy as jnp
import numpy as np

from aquakin.integrate.finite_volume import (
    diffusive_flux,
    divergence,
    upwind_flux,
)

# Cells [0,1], [1,3], [3,6]: unequal widths, with their centres and the spacing
# between neighbouring centres worked out by hand.
WIDTHS = jnp.array([1.0, 2.0, 3.0])
CENTRES = jnp.array([0.5, 2.0, 4.5])
SPACING = jnp.array([1.5, 2.5])


def test_divergence_matches_hand_values_and_conserves_the_weighted_total():
    widths = WIDTHS
    fluxes = jnp.array([[2.0, 0.0], [5.0, 1.0], [-1.0, 4.0], [0.5, -3.0]])
    d = divergence(fluxes, widths)
    # (F_i - F_{i+1}) / w_i
    np.testing.assert_allclose(
        d, [[-3.0, -1.0], [3.0, -1.5], [-0.5, 7.0 / 3.0]], rtol=1e-15, atol=0
    )
    # Only the boundary faces change sum(w * dC/dt).
    np.testing.assert_allclose(
        jnp.sum(widths[:, None] * d, axis=0), fluxes[0] - fluxes[-1], rtol=1e-14, atol=0
    )


def test_divergence_with_a_scalar_width_equals_the_constant_array():
    fluxes = jnp.array([[1.0], [0.3], [-2.0]])
    np.testing.assert_array_equal(
        divergence(fluxes, 0.25), divergence(fluxes, jnp.full((2,), 0.25))
    )


def test_diffusive_flux_on_an_uneven_grid_matches_a_hand_calculation():
    cells = jnp.array([[4.0, 0.0], [1.0, 3.0], [2.0, 3.0]])
    D = jnp.array([0.3, 0.6])
    f = diffusive_flux(cells, D, SPACING)
    # Down the gradient: species 0 moves right at 0.3*(4-1)/1.5 = 0.6, then left
    # at 0.3*(1-2)/2.5 = -0.12; species 1 moves left at 0.6*(0-3)/1.5 = -1.2 and
    # is flat across the second face.
    np.testing.assert_allclose(f, [[0.6, -1.2], [-0.12, 0.0]], rtol=1e-15, atol=0)


def test_a_linear_profile_with_matching_boundary_fluxes_is_steady_on_an_uneven_grid():
    """A profile linear in ``x`` carries the same flux across every face, so with
    that flux imposed at both boundaries every cell is at steady state."""
    D = jnp.array([0.4])
    slope = -2.0
    cells = (3.0 + slope * CENTRES)[:, None]
    interior = diffusive_flux(cells, D, SPACING)
    boundary = jnp.array([[-D[0] * slope]])
    faces = jnp.concatenate([boundary, interior, boundary], axis=0)
    np.testing.assert_allclose(interior, -D[0] * slope, rtol=1e-14, atol=0)
    np.testing.assert_allclose(divergence(faces, WIDTHS), 0.0, rtol=0, atol=1e-14)


def test_upwind_flux_takes_the_value_the_velocity_comes_from():
    cells = jnp.array([[1.0], [2.0], [3.0]])
    left, right = jnp.array([10.0]), jnp.array([30.0])
    v = jnp.array([1.0, -1.0, 2.0, -0.5])
    # face 0: the left boundary value; face 1: cell 1 (on its right);
    # face 2: cell 1 (on its left); face 3: the right boundary value.
    np.testing.assert_array_equal(
        upwind_flux(cells, v, left, right), [[10.0], [-2.0], [4.0], [-15.0]]
    )
    # Reversing the flow takes every face's value from the other side.
    np.testing.assert_array_equal(
        upwind_flux(cells, -v, left, right), [[-1.0], [1.0], [-6.0], [1.5]]
    )


def test_upwind_advection_of_a_uniform_field_is_steady():
    """A uniform field in a uniform flow, fed the same value, does not change."""
    cells = jnp.full((3, 2), 5.0)
    flux = upwind_flux(cells, jnp.full((4,), 0.7), cells[0], cells[-1])
    np.testing.assert_array_equal(divergence(flux, WIDTHS), jnp.zeros((3, 2)))


def test_upwind_flux_carries_an_inflow_into_the_first_cell():
    """Inflow at the left boundary fills the first cell at ``v (C_in - C_0) / w_0``."""
    cells = jnp.array([[1.0], [1.0], [1.0]])
    flux = upwind_flux(cells, jnp.full((4,), 0.5), jnp.array([3.0]), cells[-1])
    np.testing.assert_allclose(
        divergence(flux, WIDTHS), [[0.5 * (3.0 - 1.0) / 1.0], [0.0], [0.0]], rtol=1e-15, atol=0
    )
