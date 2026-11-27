"""One-dimensional finite-volume building blocks: grid, face fluxes, divergence.

A conservative finite-volume operator on a line of cells is three pieces: a
flux at every face (the interior faces from the neighbouring cells, the two
boundary faces from whatever lies outside), and the divergence that turns those
face fluxes into a rate of change in each cell. Writing the update as a
difference of face fluxes is what makes it conservative: every interior flux
leaves one cell and enters its neighbour, so the volume-weighted total changes
only by what crosses the two boundary faces.

The functions here supply the grid, the interior diffusive flux, the
first-order upwind advective flux, the limited second-order upwind flux (with
its Green--Gauss cell gradient and van Albada slope) and the divergence. Each
operator built on them keeps its own boundary physics -- a mass-transfer film,
an impermeable wall, an inflow -- and builds the boundary face fluxes itself.

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

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, eq=False)
class Grid1D:
    """A line of cells given by its cell edges.

    The geometry is fixed when the operator is built, so it is held as NumPy
    arrays: constants of any program that closes over it.

    Parameters
    ----------
    edges : array-like
        Cell edges, shape ``(n_cells + 1,)``, strictly increasing.

    Raises
    ------
    ValueError
        If ``edges`` is not a 1-D array of at least two strictly increasing,
        finite values.
    """

    edges: np.ndarray

    def __post_init__(self):
        edges = np.asarray(self.edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                f"edges must be a 1-D array of at least two values, got shape {edges.shape}."
            )
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be finite and strictly increasing.")
        object.__setattr__(self, "edges", edges)

    @classmethod
    def uniform(cls, length: float, n_cells: int, start: float = 0.0) -> Grid1D:
        """``n_cells`` equal cells over ``[start, start + length]``.

        Parameters
        ----------
        length : float
            Total length, positive.
        n_cells : int
            Number of cells, at least 1.
        start : float, optional
            Position of the left edge.

        Returns
        -------
        Grid1D

        Raises
        ------
        ValueError
            If ``length`` is not positive or ``n_cells`` is below 1.
        """
        if int(n_cells) != n_cells or n_cells < 1:
            raise ValueError(f"n_cells must be a positive integer, got {n_cells}.")
        if not length > 0:
            raise ValueError(f"length must be positive, got {length}.")
        return cls(np.linspace(start, start + length, int(n_cells) + 1))

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.edges.size - 1

    @property
    def widths(self) -> np.ndarray:
        """Cell widths, shape ``(n_cells,)``."""
        return np.diff(self.edges)

    @property
    def centres(self) -> np.ndarray:
        """Cell centres, shape ``(n_cells,)``."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def nodes(self) -> np.ndarray:
        """The left edge, the cell centres and the right edge, shape ``(n_cells + 2,)``.

        The points either side of every face: a boundary face's outer
        neighbour is the boundary itself, where a boundary value is given.
        """
        return np.concatenate([self.edges[:1], self.centres, self.edges[-1:]])

    @property
    def face_spacing(self) -> np.ndarray:
        """Distance between the two nodes either side of each face, shape ``(n_cells + 1,)``.

        Interior faces span two cell centres; a boundary face spans half its
        cell, from the centre to the boundary.
        """
        return np.diff(self.nodes)


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


def green_gauss_gradient(cells, grid: Grid1D, left, right):
    """Green--Gauss gradient in each cell.

    The gradient is the difference of the two face values over the cell width,
    ``g_i = (C_(i+1/2) - C_(i-1/2)) / dx_i``. An interior face value is
    interpolated linearly between the two cell centres; a boundary face takes
    the given boundary value. On a uniform grid this is the central difference
    ``(C_(i+1) - C_(i-1)) / (2 dx)``.

    Parameters
    ----------
    cells : jnp.ndarray
        Cell concentrations, shape ``(n_cells, n_species)``.
    grid : Grid1D
        The cells.
    left, right : jnp.ndarray
        Values on the left (face 0) and right (face ``n_cells``) boundary
        faces, shape ``(n_species,)``.

    Returns
    -------
    jnp.ndarray
        Gradient ``dC/dx`` in each cell, shape ``(n_cells, n_species)``.
    """
    centres = grid.centres
    # Weight of the left cell at each interior face: its distance to the right centre.
    weight = (centres[1:] - grid.edges[1:-1]) / np.diff(centres)
    interior = _per_row(weight) * cells[:-1] + _per_row(1.0 - weight) * cells[1:]
    faces = jnp.concatenate([left[None, :], interior, right[None, :]], axis=0)
    return (faces[1:] - faces[:-1]) / _per_row(grid.widths)


def van_albada_slope(a, b):
    """The van Albada limited slope of two one-sided gradients.

    ``s = a b (a + b) / (a^2 + b^2)``: the harmonic-like mean of ``a`` and
    ``b``, equal to both where they agree, falling to zero where either does,
    and with no constant to tune. It is a smooth function of ``a`` and ``b``
    away from ``a = b = 0`` (where ``s = 0``), so a gradient through a solve
    that uses it stays smooth. Written as the van Albada limiter ``phi(r) =
    (r^2 + r) / (r^2 + 1)``, ``r = a / b``, applied to ``b``.

    Parameters
    ----------
    a, b : jnp.ndarray
        The upwind-side and downwind-side gradients, the same shape.

    Returns
    -------
    jnp.ndarray
        The limited slope, the shape of ``a``.
    """
    denominator = a * a + b * b
    nonzero = denominator > 0
    # The inner where keeps the division's gradient finite where both vanish.
    return jnp.where(nonzero, a * b * (a + b) / jnp.where(nonzero, denominator, 1.0), 0.0)


def limited_upwind_flux(cells, velocity, grid: Grid1D, left, right, gradient):
    """Second-order upwind advective flux at every face, van Albada limited.

    The face value is reconstructed from the upwind cell ``C`` as ``C_f = C_C
    + s (x_f - x_C)``, where ``s`` is the :func:`van_albada_slope` of the
    gradient toward the downwind node, ``b = (C_D - C_C) / (x_D - x_C)``, and
    the upwind-side gradient recovered from the cell's Green--Gauss gradient,
    ``a = 2 g_C - b``. On a uniform grid ``a`` is the backward difference
    ``(C_C - C_U) / dx``, and the scheme is the classical limited
    second-order upwind (MUSCL) scheme: second order where the solution is
    smooth and monotone, falling toward first order at extrema and steep
    fronts, so it does not oscillate. A face whose upwind side is a boundary
    carries the boundary value.

    Parameters
    ----------
    cells : jnp.ndarray
        Cell concentrations, shape ``(n_cells, n_species)``.
    velocity : jnp.ndarray
        Velocity at every face, shape ``(n_cells + 1,)``, positive toward
        increasing index.
    grid : Grid1D
        The cells.
    left, right : jnp.ndarray
        Values on the left and right boundary faces, shape ``(n_species,)``.
    gradient : jnp.ndarray
        Cell gradients, shape ``(n_cells, n_species)``, normally
        :func:`green_gauss_gradient` with the same boundary values.

    Returns
    -------
    jnp.ndarray
        Face fluxes, shape ``(n_cells + 1, n_species)``, positive toward
        increasing index, ready for :func:`divergence`.
    """
    nodes = jnp.concatenate([left[None, :], cells, right[None, :]], axis=0)
    # Gradient across each face, from the node on its left to the one on its right.
    across = (nodes[1:] - nodes[:-1]) / _per_row(grid.face_spacing)
    # Reconstructed from the cell on the left of faces 1..n (flow toward +x) ...
    toward_right = van_albada_slope(2.0 * gradient - across[1:], across[1:])
    from_left = cells + toward_right * _per_row(grid.edges[1:] - grid.centres)
    # ... and from the cell on the right of faces 0..n-1 (flow toward -x).
    toward_left = van_albada_slope(2.0 * gradient - across[:-1], across[:-1])
    from_right = cells + toward_left * _per_row(grid.edges[:-1] - grid.centres)
    from_left = jnp.concatenate([left[None, :], from_left], axis=0)
    from_right = jnp.concatenate([from_right, right[None, :]], axis=0)
    v = velocity[:, None]
    return jnp.where(v > 0, v * from_left, v * from_right)
