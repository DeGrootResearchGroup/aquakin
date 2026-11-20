"""Biofilm growth with a moving boundary: advection, detachment and thickness.

A biofilm whose thickness is a fixed parameter can only redistribute solubles;
it cannot represent a film that thickens as it grows, thins when decay outruns
growth, or sorts its species by depth. All three need the mechanism this module
supplies, the one-dimensional multispecies biofilm of Wanner & Gujer (1986):

- The biofilm is **incompressible** -- the particulate species fill it at a
  constant total density ``rho``. Net production anywhere inside therefore has
  to push material outward, since the carrier wall is impermeable.
- That displacement is an **advective velocity** ``u(z)``, fixed by
  incompressibility alone: ``du/dz = (sum_i r_i) / rho`` with ``u = 0`` at the
  wall, so ``u(z)`` is the production integrated from the wall out to ``z``.
  Species are carried along at that velocity, which is what makes them
  stratify -- a fast grower near the surface stays near the surface, while
  whatever is produced deep down is conveyed outward through its neighbours.
- The **thickness** then moves at the difference between the velocity arriving
  at the surface and the velocity at which detachment strips it:
  ``dL_f/dt = u(L_f) - u_det``.

Discretization. The film is divided into a fixed number of layers spanning the
*current* thickness, so the grid stretches and shrinks with ``L_f``. Writing the
balance on that moving grid (an arbitrary Lagrangian--Eulerian form) adds the
grid's own velocity to the flux: material crosses a face at ``w = u - sigma *
dL_f/dt``, where ``sigma`` is the face's fractional depth. The two ends are then
exactly what the physics requires and need no special-casing -- ``w = 0`` at the
wall (nothing crosses an impermeable surface) and ``w = u_det`` at the biofilm
surface (what leaves is exactly what detaches).

Layer arrays here follow the convention of
:mod:`aquakin.integrate.biofilm`: index 0 is the **surface** layer and the last
index sits against the carrier wall.

References
----------
Wanner, O. & Gujer, W. (1986). A multispecies biofilm model. Biotechnology and
Bioengineering 28(3), 314-328.
"""

from __future__ import annotations

import jax.numpy as jnp

#: Detachment laws, as the exponent of ``L_f`` in ``u_det = rate * L_f**power``.
#: Constant erosion (0) strips a fixed velocity; linear (1) is proportional to
#: thickness; quadratic (2) is the common erosion closure, whose restoring force
#: grows fastest and so pins the thickness most tightly.
DETACHMENT_POWERS = {"constant": 0.0, "linear": 1.0, "quadratic": 2.0}

#: The law that holds the thickness at its specified value instead of letting a
#: rate decide it: detachment is set to exactly the velocity arriving at the
#: surface, so ``dL_f/dt`` is zero by construction. Species still grow, advect
#: and stratify, and biomass still leaves at the surface -- only the total
#: thickness is pinned. This is what a benchmark means by a specified average
#: thickness "maintained by detachment", and it takes no detachment rate.
BALANCED_DETACHMENT = "balance"

#: Every accepted ``detachment_law``.
DETACHMENT_LAWS = (*sorted(DETACHMENT_POWERS), BALANCED_DETACHMENT)


def detachment_velocity(rate, thickness, law: str = "quadratic"):
    """Velocity at which detachment strips the biofilm surface (m / time).

    Parameters
    ----------
    rate : float or jnp.ndarray
        Detachment coefficient. Its units follow ``law`` so that the product is
        a velocity: m/d for ``"constant"``, 1/d for ``"linear"``, 1/(m d) for
        ``"quadratic"``.
    thickness : float or jnp.ndarray
        Current biofilm thickness ``L_f`` (m).
    law : {"quadratic", "linear", "constant"}
        Dependence on thickness. ``"balance"`` is not handled here -- it has no
        rate of its own, being defined by the arriving velocity, so a caller
        resolves it with :func:`balanced_detachment_velocity`.

    Returns
    -------
    jnp.ndarray
        The detachment velocity.
    """
    try:
        power = DETACHMENT_POWERS[law]
    except KeyError:
        raise ValueError(f"law must be one of {sorted(DETACHMENT_POWERS)}; got {law!r}.") from None
    return rate * thickness**power


def advection_velocity(production, density, dz):
    """Displacement velocity at each layer face, from incompressibility.

    Integrates ``du/dz = (sum_i r_i)/rho`` outward from the wall, where nothing
    crosses, so the velocity at a face is the net particulate production below
    it. Returns the velocity at every face, surface first, matching the layer
    ordering.

    Parameters
    ----------
    production : jnp.ndarray
        Net particulate production per unit biofilm volume in each layer,
        summed over species, shape ``(n_layers,)``, surface first.
    density : float
        Total particulate density ``rho`` of the biofilm.
    dz : float or jnp.ndarray
        Layer thickness.

    Returns
    -------
    jnp.ndarray
        Face velocities, shape ``(n_layers + 1,)``. Entry 0 is the biofilm
        surface and the last entry is the wall, which is always zero.
    """
    # Accumulate from the wall outward, then restore the surface-first order.
    from_wall = jnp.flip(production) * (dz / density)
    faces_from_wall = jnp.concatenate(
        [jnp.zeros((1,), dtype=from_wall.dtype), jnp.cumsum(from_wall)]
    )
    return jnp.flip(faces_from_wall)


def balanced_detachment_velocity(face_velocities):
    """Detachment that exactly absorbs the velocity arriving at the surface.

    Holds the thickness at whatever it was specified to be: growth inside the
    film still displaces material outward and still carries the species with it,
    but everything that reaches the surface detaches, so ``dL_f/dt`` is zero to
    the last bit rather than to a tolerance.
    """
    return face_velocities[0]


def surface_detachment(face_velocities, rate, thickness, law: str = "quadratic"):
    """The detachment velocity under any accepted law, balanced or rate-driven."""
    if law == BALANCED_DETACHMENT:
        return balanced_detachment_velocity(face_velocities)
    return detachment_velocity(rate, thickness, law)


def thickness_derivative(face_velocities, detach_velocity):
    """``dL_f/dt``: what arrives at the surface, less what detaches."""
    return face_velocities[0] - detach_velocity


def advection_transport(layers, bulk, face_velocities, dLdt, dz, advect_mask):
    """Particulate advection on the stretching grid, plus what reaches the bulk.

    Material crosses face ``f`` at ``w_f = u_f - sigma_f * dL_f/dt`` where
    ``sigma_f`` is the face's fractional depth, so the grid's own motion is
    taken out of the flux. Fluxes are upwinded, which keeps the scheme positive:
    an outward velocity carries the layer beneath the face, an inward one (a
    shrinking film, or attachment at the surface) carries what lies outside it.

    Parameters
    ----------
    layers : jnp.ndarray
        Layer concentrations, shape ``(n_layers, n_species)``, surface first.
    bulk : jnp.ndarray
        Bulk concentrations, shape ``(n_species,)`` -- the upwind value for flux
        entering the biofilm from outside.
    face_velocities : jnp.ndarray
        Displacement velocity at each face, shape ``(n_layers + 1,)``, from
        :func:`advection_velocity`.
    dLdt : float or jnp.ndarray
        Rate of change of the thickness.
    dz : float or jnp.ndarray
        Layer thickness.
    advect_mask : jnp.ndarray
        Boolean ``(n_species,)``: which species the biofilm matrix carries.
        Solubles are excluded -- they move by diffusion, not with the matrix.

    Returns
    -------
    tuple
        ``(d_layers, surface_flux)``. ``d_layers`` is the advective contribution
        to each layer's rate, shape ``(n_layers, n_species)``. ``surface_flux``
        is what crosses the outer face per unit biofilm area (positive leaving
        the film), shape ``(n_species,)``; scale it by the area-to-volume ratio
        to add it to the bulk.
    """
    n_layers = layers.shape[0]
    # Fractional depth of each face, surface first: 1 at the surface, 0 at the wall.
    sigma = jnp.flip(jnp.arange(n_layers + 1, dtype=layers.dtype)) / n_layers
    w = face_velocities - sigma * dLdt  # velocity relative to the moving grid

    # Upwind concentration at each face. Outward flow (w > 0) carries the layer
    # on the wall side of the face; inward flow carries the one outside it, which
    # at the surface is the bulk.
    inner = jnp.concatenate([layers, layers[-1:]], axis=0)  # wall side of each face
    outer = jnp.concatenate([bulk[None, :], layers], axis=0)  # surface side
    w_col = w[:, None]
    flux = jnp.where(w_col > 0, w_col * inner, w_col * outer) * advect_mask[None, :]

    # Divergence over each layer: flux in at its wall-side face, out at its
    # surface-side face. Face f bounds layer f on its surface side.
    divergence = (flux[1:] - flux[:-1]) / dz

    # The cells themselves stretch with the film, and it is the mass in a cell
    # that the fluxes govern, not its concentration: ``d(X*dz)/dt`` carries a
    # ``X * d(dz)/dt`` term that has to come back out. With the layers spanning
    # the thickness, ``d(dz)/dt / dz`` is ``(dL_f/dt) / L_f``. Without this a
    # uniform film drifts off its own density as it grows -- the incompressibility
    # the displacement velocity was derived from.
    dilation = dLdt / (n_layers * dz)
    d_layers = divergence - layers * dilation * advect_mask[None, :]
    return d_layers, flux[0]


def layer_thickness(thickness, n_layers, floor: float = 1e-9):
    """Layer thickness ``L_f / n``, floored so a vanishing film cannot divide by zero.

    A film that detaches faster than it grows drives ``L_f`` toward zero. The
    floor keeps the right-hand side finite there; the biofilm simply carries
    negligible mass, and the advective fluxes vanish with it.
    """
    return jnp.maximum(thickness, floor) / n_layers
