"""The 1-D finite-volume building blocks in ``aquakin.integrate.finite_volume``.

Each check is against a hand calculation or a conservation law, on a grid of
unequal cells so a width / spacing mix-up cannot pass.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.integrate.finite_volume import (
    Grid1D,
    diffusive_flux,
    divergence,
    green_gauss_gradient,
    limited_upwind_flux,
    upwind_flux,
    van_albada_slope,
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


GRID = Grid1D(np.array([0.0, 1.0, 3.0, 6.0]))


def test_grid_geometry_matches_the_hand_values():
    """Widths, centres, nodes and face spacing of the cells [0,1], [1,3], [3,6]."""
    np.testing.assert_array_equal(GRID.widths, WIDTHS)
    np.testing.assert_array_equal(GRID.centres, CENTRES)
    np.testing.assert_array_equal(GRID.nodes, [0.0, 0.5, 2.0, 4.5, 6.0])
    np.testing.assert_array_equal(GRID.face_spacing, [0.5, 1.5, 2.5, 1.5])
    assert GRID.n_cells == 3


def test_uniform_grid_has_equal_cells_over_its_span():
    grid = Grid1D.uniform(2.0, 4, start=1.0)
    np.testing.assert_allclose(grid.edges, [1.0, 1.5, 2.0, 2.5, 3.0], rtol=1e-15, atol=0)
    np.testing.assert_allclose(grid.widths, 0.5, rtol=1e-15, atol=0)


@pytest.mark.parametrize(
    "edges, message",
    [
        (np.array([1.0]), "at least two values"),
        (np.zeros((2, 2)), "at least two values"),
        (np.array([0.0, 2.0, 1.0]), "strictly increasing"),
        (np.array([0.0, np.nan]), "strictly increasing"),
    ],
)
def test_grid_rejects_bad_edges(edges, message):
    with pytest.raises(ValueError, match=message):
        Grid1D(edges)


@pytest.mark.parametrize(
    "length, n_cells, message",
    [(1.0, 0, "positive integer"), (1.0, 2.5, "positive integer"), (0.0, 3, "length")],
)
def test_uniform_grid_rejects_bad_arguments(length, n_cells, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        Grid1D.uniform(length, n_cells)


def _linear(x):
    """Two species, linear in x: the profile a second-order scheme reproduces exactly."""
    x = jnp.asarray(x)
    return jnp.stack([2.0 + 0.5 * x, 7.0 - 1.5 * x], axis=-1)


def test_green_gauss_gradient_is_exact_for_a_linear_profile_on_uneven_cells():
    cells = _linear(GRID.centres)
    gradient = green_gauss_gradient(cells, GRID, _linear(0.0), _linear(6.0))
    np.testing.assert_allclose(gradient, jnp.tile(jnp.array([0.5, -1.5]), (3, 1)), rtol=1e-14)


def test_green_gauss_gradient_on_uniform_cells_is_the_central_difference():
    grid = Grid1D.uniform(4.0, 4)
    cells = jnp.array([[1.0], [4.0], [2.0], [8.0]])
    gradient = green_gauss_gradient(cells, grid, jnp.array([0.0]), jnp.array([10.0]))
    # Boundary faces carry the given values; interior faces the mean of neighbours.
    expected = [[2.5 - 0.0], [(2.0 - 1.0) / 2.0], [(8.0 - 4.0) / 2.0], [10.0 - 5.0]]
    np.testing.assert_allclose(gradient, expected, rtol=1e-15, atol=0)


def test_van_albada_slope_matches_hand_values():
    """``[a (b^2 + e^2) + b (a^2 + e^2)] / (a^2 + b^2 + 2 e^2)`` at ``e = 0.5``:
    equal gradients pass through, opposite ones cancel."""
    a = jnp.array([1.0, 2.0, 1.0, 0.0, 0.0, -3.0])
    b = jnp.array([1.0, 1.0, -1.0, 2.0, 0.0, -3.0])
    expected = [1.0, (2.0 * 1.25 + 4.25) / 5.5, 0.0, 0.5 / 4.5, 0.0, -3.0]
    np.testing.assert_allclose(van_albada_slope(a, b, 0.5), expected, rtol=1e-15, atol=0)


def test_van_albada_slope_is_the_central_mean_near_a_flat_region():
    """At ``a = b = 0`` the slope is differentiable, with the central-mean derivative
    ``(1/2, 1/2)`` -- the property an implicit solver's Newton iteration needs there."""
    grads = jax.grad(lambda a, b: van_albada_slope(a, b, 1e-9), argnums=(0, 1))(0.0, 0.0)
    np.testing.assert_allclose([float(g) for g in grads], [0.5, 0.5], rtol=1e-15)


@pytest.mark.parametrize("velocity", [0.8, -0.8])
def test_limited_flux_reproduces_a_linear_profile_exactly_on_uneven_cells(velocity):
    """Face values of a linear profile are exact (second order), in either flow direction."""
    cells = _linear(GRID.centres)
    left, right = _linear(0.0), _linear(6.0)
    gradient = green_gauss_gradient(cells, GRID, left, right)
    flux = limited_upwind_flux(cells, jnp.full(4, velocity), GRID, left, right, gradient, 0.3)
    np.testing.assert_allclose(flux, velocity * _linear(GRID.edges), rtol=1e-14)


def test_limited_flux_limits_to_the_cell_value_at_an_extremum():
    """At a local maximum the slope is limited to zero: face values equal the cell."""
    grid = Grid1D.uniform(3.0, 3)
    cells = jnp.array([[0.0], [1.0], [0.0]])
    left = right = jnp.array([0.0])
    gradient = green_gauss_gradient(cells, grid, left, right)
    flux = limited_upwind_flux(cells, jnp.ones(4), grid, left, right, gradient, 0.3)
    np.testing.assert_allclose(flux[2], [1.0], rtol=1e-15, atol=0)


def test_limited_flux_matches_the_classical_muscl_face_value_on_uniform_cells():
    """``C_f = C_i + phi(r) (C_(i+1) - C_i) / 2`` with van Albada ``phi`` and ``r`` the
    ratio of successive differences -- the textbook form of the same scheme, which
    the limited slope equals once the gradients are far above ``epsilon`` (here
    their ratio to it is 1e12, so the difference is ~1e-24)."""
    grid = Grid1D.uniform(5.0, 5)
    values = jnp.array([1.0, 2.0, 4.0, 5.0, 3.0])
    cells = values[:, None]
    left, right = jnp.array([0.5]), jnp.array([3.0])
    gradient = green_gauss_gradient(cells, grid, left, right)
    flux = limited_upwind_flux(cells, jnp.ones(6), grid, left, right, gradient, 1e-12)
    # Interior faces 1..3 (upwind cells 0..2, which have an interior downwind cell).
    for i in range(1, 4):
        upwind = values[i - 1] - (values[i - 2] if i >= 2 else 2 * 0.5 - values[0])
        downwind = values[i] - values[i - 1]
        r = upwind / downwind
        phi = (r * r + r) / (r * r + 1)
        expected = values[i - 1] + 0.5 * phi * downwind
        np.testing.assert_allclose(flux[i, 0], expected, rtol=1e-14)
