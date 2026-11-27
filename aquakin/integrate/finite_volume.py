"""One-dimensional finite-volume building blocks: face fluxes and divergence.

A conservative finite-volume operator on a line of cells is three pieces: a
flux at every face (the interior faces from the neighbouring cells, the two
boundary faces from whatever lies outside), and the divergence that turns those
face fluxes into a rate of change in each cell. Writing the update as a
difference of face fluxes is what makes it conservative: every interior flux
leaves one cell and enters its neighbour, so the volume-weighted total changes
only by what crosses the two boundary faces.

The functions here supply the interior diffusive flux, the upwind advective flux
and the divergence. Each operator built on them keeps its own boundary physics -- a
mass-transfer film, an impermeable wall, an inflow -- and builds the boundary
face fluxes itself.

Conventions
-----------
- Cells are indexed ``0 .. n_cells - 1`` along increasing ``x``; face ``f``
  sits between cell ``f - 1`` and cell ``f``, so face 0 is the left boundary and
  face ``n_cells`` the right boundary.
- A face flux is **positive in the direction of increasing index** (from cell
  ``f - 1`` into cell ``f``). An operator whose natural direction is the other
  way negates its flux before calling :func:`divergence`.
- Concentration arrays have shape ``(n_cells, n_species)``; face arrays have
  shape ``(n_cells + 1, n_species)`` (all faces) or ``(n_cells - 1,
  n_species)`` (interior faces only).
- Cell widths and centre spacings may be scalars, for a uniform grid, or arrays
  with one entry per cell or interior face, so unequal cells are handled. A
  scalar gives exactly the same arithmetic as a constant array.
"""

from __future__ import annotations

import jax.numpy as jnp


def _per_row(values):
    """Broadcast a scalar or a ``(n,)`` array against ``(n, n_species)`` rows."""
    values = jnp.asarray(values)
    return values[:, None] if values.ndim == 1 else values


def divergence(face_fluxes, widths):
    """Conservative rate of change in each cell from its two face fluxes.

    ``dC_i/dt = (F_i - F_(i+1)) / dx_i``: what enters across the left face less
    what leaves across the right one, over the cell's width. Summing
    ``widths * dC/dt`` over the cells telescopes to ``F_0 - F_n``, so only the
    boundary faces change the total.

    Parameters
    ----------
    face_fluxes : jnp.ndarray
        Flux at every face, boundary faces included, shape
        ``(n_cells + 1, n_species)``, positive toward increasing index.
    widths : float or jnp.ndarray
        Cell widths: a scalar, or shape ``(n_cells,)``.

    Returns
    -------
    jnp.ndarray
        Rate of change of each cell's concentration, shape
        ``(n_cells, n_species)``.
    """
    return (face_fluxes[:-1] - face_fluxes[1:]) / _per_row(widths)


def diffusive_flux(cells, diffusivity, spacing):
    """Fickian flux across each interior face, ``D (C_L - C_R) / h``.

    Parameters
    ----------
    cells : jnp.ndarray
        Cell concentrations, shape ``(n_cells, n_species)``.
    diffusivity : jnp.ndarray
        Diffusivity per species, shape ``(n_species,)``; zero for a species
        that does not diffuse.
    spacing : float or jnp.ndarray
        Distance between neighbouring centres: a scalar, or shape
        ``(n_cells - 1,)`` with entry ``f - 1`` belonging to interior face ``f``.

    Returns
    -------
    jnp.ndarray
        Interior face fluxes, shape ``(n_cells - 1, n_species)``, positive
        toward increasing index (down the gradient when ``C_L > C_R``).
    """
    return diffusivity[None, :] * (cells[:-1] - cells[1:]) / _per_row(spacing)


def upwind_flux(cells, velocity, left, right):
    """First-order upwind advective flux at every face.

    A face carries its velocity times the value on the side the velocity comes
    from: the cell (or boundary value) on its left when the velocity is
    positive, the one on its right otherwise. Upwinding keeps an advective
    scheme positive.

    Parameters
    ----------
    cells : jnp.ndarray
        Cell concentrations, shape ``(n_cells, n_species)``.
    velocity : jnp.ndarray
        Velocity at every face, shape ``(n_cells + 1,)``, positive toward
        increasing index.
    left : jnp.ndarray
        Value outside the left boundary (face 0), shape ``(n_species,)``.
    right : jnp.ndarray
        Value outside the right boundary (face ``n_cells``), shape
        ``(n_species,)``.

    Returns
    -------
    jnp.ndarray
        Face fluxes, shape ``(n_cells + 1, n_species)``, positive toward
        increasing index, ready for :func:`divergence`.

    Notes
    -----
    The velocity multiplies each candidate before the selection, and the
    right-hand candidates are built first. Both are equal in exact arithmetic to
    the alternatives, but under ``jax.jit`` the reverse-mode cotangents are then
    summed (and fused) differently and change in the last bit, so the operators
    built on this function would no longer reproduce their gradients exactly.
    """
    from_right = jnp.concatenate([cells, right[None, :]], axis=0)
    from_left = jnp.concatenate([left[None, :], cells], axis=0)
    v = velocity[:, None]
    return jnp.where(v > 0, v * from_left, v * from_right)
