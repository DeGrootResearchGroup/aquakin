"""Layered (1-D) biofilm reactor: depth-resolved diffusion--reaction.

A lumped reactor (:class:`~aquakin.integrate.batch.BatchReactor`) applies the
biofilm chemistry to a single well-mixed bulk concentration scaled by an
area-to-volume ratio. That cannot represent processes that are controlled by how
far a solute *penetrates* the biofilm --- e.g. an electron acceptor that is
consumed in the outer layers and never reaches organisms deeper in, while those
deep organisms keep turning over a substrate that diffuses in freely. Resolving
the biofilm in the direction normal to the wall is required to capture this
(Wanner & Gujer 1986; the 1-D multispecies sewer-biofilm model of Jiang et al.
2009; the stratified sulfide/methane sewer biofilms of Sun et al. 2014).

:class:`BiofilmReactor` discretises the biofilm into ``n_layers`` layers between
a well-mixed bulk compartment and the (no-flux) wall, and integrates a
diffusion--reaction system:

- **Solubles** diffuse between adjacent layers (Fick's law, effective diffusivity
  ``D_eff`` per species) and exchange with the bulk across an external boundary
  layer (mass-transfer coefficient ``k_L``). They also react in every
  compartment.
- **Particulates** (biomass and particulate substrate) do not diffuse. Whether
  each one is *held fixed* (net rate zeroed -- a sustained, non-depleting
  "mature biofilm" source/sink) or *evolves* (grows, decays, drains) is a
  separate choice governed by ``fixed_mask`` (see below), decoupled from
  diffusion. The default holds every particulate fixed; but a reactive
  particulate (a draining substrate pool, a growing biomass, elemental sulfur,
  precipitated FeS) must be left out of ``fixed_mask`` so it reacts -- otherwise
  it becomes a non-depleting source/sink and silently breaks mass balance.

The same :class:`~aquakin.core.network.CompiledNetwork` runs in every
compartment --- the point is that identical chemistry behaves differently once
depth is resolved. A network intended for this reactor should express its rates
per unit volume with the local biomass as an explicit reactant (so a compartment
with little biomass carries little rate), rather than lumping the biofilm into an
area-to-volume multiplier.

The diffusion operator conserves the volume-weighted total exactly. Element
(COD/S/N) conservation across reactions is exact only when the network's
``positivity_limiter`` is *off*: the limiter throttles a species' net rate near
zero independently of its reaction partners, so it trades a small (~1e-3)
mass-balance residual for guaranteed positivity. Per-reaction stoichiometric
balance (checked by ``aquakin.utils.balance``) is unaffected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import diffrax
import jax
import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions
from aquakin.core.network import CompiledNetwork
from aquakin.integrate._common import _HasNamedSpecies, _run_diffeqsolve


@dataclass
class BiofilmSolution(_HasNamedSpecies):
    """Solution returned by :meth:`BiofilmReactor.solve`.

    Attributes
    ----------
    t : jnp.ndarray
        Times at which the solution was recorded, shape ``(n_t,)``.
    C : jnp.ndarray
        **Bulk** concentration trajectory, shape ``(n_t, n_species)``. This is
        the measurable (well-mixed liquid) signal; :meth:`C_named` reads it.
    profile : jnp.ndarray
        Full depth-resolved trajectory, shape ``(n_t, n_compartments,
        n_species)``, where compartment 0 is the bulk and 1..``n_layers`` run
        from the biofilm surface (adjacent to the bulk) to the wall.
    depth : jnp.ndarray
        Mid-point depth of each biofilm layer from the surface, shape
        ``(n_layers,)`` (metres). Bulk has no depth and is omitted.
    network : CompiledNetwork
        The network that produced this solution.
    """

    t: jnp.ndarray
    C: jnp.ndarray
    profile: jnp.ndarray
    depth: jnp.ndarray
    network: CompiledNetwork

    def profile_named(self, species: str) -> jnp.ndarray:
        """Depth profile of one species over time, shape ``(n_t, n_compartments)``."""
        if species not in self.network.species_index:
            raise KeyError(
                f"Unknown species '{species}'. Available: {self.network.species}"
            )
        return self.profile[:, :, self.network.species_index[species]]


def _diffusion_transport(C, D, kL, dz, area_per_volume, n_species):
    """Diffusive transport for the whole state, shape ``(n_comp, n_species)``.

    Pure function (no reactor ``self``) so it can be closed over by a jit-compiled
    solve without leaking trace-created state. ``C[0]`` is the bulk; ``C[1:]`` the
    layers (surface..wall). Particulate columns are zero where ``D == 0``. Uses
    the conservative finite-volume (face-flux) form: only first-order face
    differences, so the volume-weighted total is conserved exactly.
    """
    bulk = C[0]                       # (n_species,)
    layers = C[1:]                    # (n_layers, n_species)
    # Internal interface fluxes between layer j and j+1 (positive toward the wall).
    f_internal = D[None, :] * (layers[:-1] - layers[1:]) / dz   # (n_layers-1, n_species)
    # Bulk <-> surface flux across the boundary layer (positive into the film).
    f_bs = kL * (bulk - layers[0])                              # (n_species,)
    flux_in = jnp.concatenate([f_bs[None, :], f_internal], axis=0)
    zero = jnp.zeros((1, n_species))
    flux_out = jnp.concatenate([f_internal, zero], axis=0)      # wall: no flux out
    d_layers = (flux_in - flux_out) / dz                        # (n_layers, n_species)
    d_bulk = -area_per_volume * f_bs                            # bulk loses surface flux
    return jnp.concatenate([d_bulk[None, :], d_layers], axis=0)


def _default_soluble_mask(network: CompiledNetwork) -> jnp.ndarray:
    """Classify species as soluble (diffuses, evolves) vs particulate (fixed).

    Heuristic for the WATS/ASM naming convention: soluble names start with
    ``S`` (``S_*``, ``sumS``), particulate names start with ``X``. Callers can
    override with an explicit mask.
    """
    return jnp.asarray(
        [not s.startswith("X") for s in network.species], dtype=bool
    )


class BiofilmReactor:
    """Stateless 1-D layered biofilm reactor (diffusion--reaction over depth).

    Parameters
    ----------
    network : CompiledNetwork
        Compiled reaction network, run in every compartment.
    conditions : SpatialConditions
        Condition fields. The same conditions are used in every compartment
        (location ``loc_idx=0``).
    n_layers : int
        Number of biofilm layers between the bulk and the wall.
    thickness : float
        Total biofilm thickness ``L_f`` (metres). Layer thickness is
        ``thickness / n_layers``.
    area_per_volume : float
        Biofilm-area-to-bulk-volume ratio ``A_V`` (m^2 biofilm per m^3 bulk,
        i.e. 1/m). Sets how strongly the (small) biofilm exchange moves the
        (large) bulk pool.
    diffusivity : float or jnp.ndarray
        Effective diffusivity ``D_eff`` of solubles inside the biofilm
        (m^2/day). A scalar applies to every soluble; an array of shape
        ``(n_species,)`` gives a per-species value (entries for particulates are
        ignored). Particulates never diffuse.
    boundary_layer : float
        External boundary-layer thickness ``L_bl`` (metres) across which the
        bulk exchanges with the surface layer. The per-species mass-transfer
        coefficient is ``k_L = D_bl / L_bl``.
    boundary_diffusivity : float or jnp.ndarray, optional
        Diffusivity used in the external boundary layer (m^2/day). The boundary
        layer is *liquid* (outside the biofilm matrix), so it should carry the
        free-water diffusivity ``D_w``, not the density-reduced in-biofilm
        ``D_eff``. Same shape rules as ``diffusivity``. ``None`` (default) reuses
        ``diffusivity`` for backward compatibility, which understates the
        bulk<->film transfer by the biofilm reduction factor; pass the water
        values to model the boundary layer correctly.
    soluble_mask : jnp.ndarray, optional
        Boolean ``(n_species,)`` mask: ``True`` for solubles (which diffuse),
        ``False`` for particulates (which do not). Defaults to the ``S``/``X``
        name heuristic. This controls **diffusion only** --- whether a species is
        held fixed is a separate question governed by ``fixed_mask``.
    fixed_mask : jnp.ndarray, optional
        Boolean ``(n_species,)`` mask: ``True`` for species **held fixed** (net
        rate zeroed everywhere --- the "mature biofilm" sustained source/sink),
        ``False`` for species that evolve. Defaults to ``~soluble_mask`` (every
        particulate held fixed), which is correct for *inert* particulates
        (biomass, inert solids) but **wrong for reactive particulates** that do
        not diffuse yet must still react --- e.g. elemental sulfur or precipitated
        FeS, whose inventory genuinely drains and fills. For such species pass a
        ``fixed_mask`` that holds only the inert biomass/solids fixed and lets the
        reactive particulates evolve; otherwise a held-fixed reactive particulate
        becomes a non-depleting source/sink and silently breaks mass balance.
    biofilm_reactions : list of str or jnp.ndarray, optional
        Which reactions are biofilm processes --- they run in the biofilm
        **layers** only, never in the well-mixed bulk. Given as a list of
        reaction names or a boolean ``(n_reactions,)`` mask. The remaining
        reactions (bulk-suspended and bulk-chemical) run in the **bulk** only.
        ``None`` (default) runs every reaction in every compartment, appropriate
        for a single-phase network. For a WATS-style network the biofilm
        reactions are those carrying the ``A_V`` area factor; bulk reactions
        carry the suspended biomass ``[X_BH]`` or are abiotic. This separation,
        not a zeroed biomass state, is what keeps the two phases from leaking
        into each other.
    rtol, atol : float
        Solver tolerances (scalar; applied across the whole layered state).
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint strategy (see :class:`~aquakin.integrate.batch.BatchReactor`).
    dtmax : float, optional
        Maximum integrator step (see :class:`~aquakin.integrate.batch.BatchReactor`).
    max_steps : int, optional
        Maximum number of solver steps (default 100000). A stiff per-layer-biomass
        network with a tight ``dtmax`` can exceed the default; raise this if the
        solve raises a max-steps error.

    Notes
    -----
    The state is a ``(n_layers + 1, n_species)`` array: row 0 is the bulk, rows
    1..``n_layers`` are the biofilm layers from surface to wall. The wall is a
    no-flux boundary. Species in ``fixed_mask`` have their net rate zeroed
    everywhere (held fixed); all others evolve. Diffusion is governed separately
    by ``soluble_mask``.
    """

    def __init__(
        self,
        network: CompiledNetwork,
        conditions: SpatialConditions,
        *,
        n_layers: int,
        thickness: float,
        area_per_volume: float,
        diffusivity,
        boundary_layer: float,
        boundary_diffusivity=None,
        soluble_mask: Optional[jnp.ndarray] = None,
        fixed_mask: Optional[jnp.ndarray] = None,
        biofilm_reactions=None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        adjoint: Optional[diffrax.AbstractAdjoint] = None,
        dtmax: Optional[float] = None,
        max_steps: int = 100_000,
    ) -> None:
        conditions.validate_required(network.conditions_required)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {n_layers}.")
        if not (thickness > 0 and boundary_layer > 0 and area_per_volume > 0):
            raise ValueError(
                "thickness, boundary_layer and area_per_volume must be positive."
            )
        self.network = network
        self.conditions = conditions
        self.n_layers = int(n_layers)
        self.thickness = float(thickness)
        self.area_per_volume = float(area_per_volume)
        self.boundary_layer = float(boundary_layer)
        self.rtol = rtol
        self.atol = float(atol)
        self.adjoint = adjoint
        self.dtmax = dtmax
        self.max_steps = int(max_steps)

        n = network.n_species
        if soluble_mask is None:
            soluble_mask = _default_soluble_mask(network)
        self.soluble_mask = jnp.asarray(soluble_mask, dtype=bool)
        if self.soluble_mask.shape != (n,):
            raise ValueError(
                f"soluble_mask must have shape ({n},); got {self.soluble_mask.shape}"
            )

        # Which species are held fixed (net rate zeroed). Decoupled from
        # diffusion: a reactive particulate diffuses (soluble_mask False) yet must
        # still react (fixed_mask False). Default: every particulate fixed.
        fixed_defaulted = fixed_mask is None
        if fixed_mask is None:
            fixed_mask = ~self.soluble_mask
        self.fixed_mask = jnp.asarray(fixed_mask, dtype=bool)
        if self.fixed_mask.shape != (n,):
            raise ValueError(
                f"fixed_mask must have shape ({n},); got {self.fixed_mask.shape}"
            )
        # Footgun guard: the default freezes every particulate, which is wrong for
        # a REACTIVE particulate (one that some reaction produces/consumes) -- a
        # frozen reactive particulate becomes a non-depleting source/sink and
        # silently breaks mass balance (e.g. elemental sulfur feeding a sulfate
        # source). Only warn for the default; an explicit fixed_mask is a
        # deliberate choice (a particulate may be frozen on purpose as a sustained
        # "mature biofilm" source).
        if fixed_defaulted:
            stoich = network.compute_stoich(network.default_parameters())
            reactive = jnp.any(stoich != 0.0, axis=0)            # (n_species,)
            frozen_reactive = reactive & self.fixed_mask
            offenders = [s for s, f in zip(network.species,
                                           list(map(bool, frozen_reactive))) if f]
            if offenders:
                warnings.warn(
                    "BiofilmReactor is holding reactive particulate(s) "
                    f"{offenders} fixed by the default fixed_mask: their reactions "
                    "are zeroed everywhere, so they act as non-depleting "
                    "source/sinks and can break mass balance. Pass an explicit "
                    "fixed_mask that leaves reactive particulates free (only inert "
                    "biomass/solids held fixed).",
                    stacklevel=2,
                )

        # Per-reaction phase mask: which reactions are biofilm processes (run in
        # the layers only) vs bulk/chemical (run in the bulk only). ``None`` ->
        # every reaction runs in every compartment (a single-phase network).
        # Accepts a list of reaction names or a boolean ``(n_reactions,)`` array.
        if biofilm_reactions is None:
            self._biofilm_mask = None
        elif all(isinstance(x, str) for x in biofilm_reactions):
            names = set(biofilm_reactions)
            unknown = names - set(network.reaction_names)
            if unknown:
                raise ValueError(f"Unknown biofilm reaction names: {sorted(unknown)}")
            self._biofilm_mask = jnp.asarray(
                [rn in names for rn in network.reaction_names], dtype=bool
            )
        else:
            bm = jnp.asarray(biofilm_reactions, dtype=bool)
            if bm.shape != (network.n_reactions,):
                raise ValueError(
                    f"biofilm_reactions mask must have shape ({network.n_reactions},);"
                    f" got {bm.shape}"
                )
            self._biofilm_mask = bm

        D = jnp.broadcast_to(jnp.asarray(diffusivity, dtype=float), (n,))
        # Particulates do not diffuse: zero their diffusivity regardless.
        self._D = jnp.where(self.soluble_mask, D, 0.0)          # (n_species,)
        self._dz = self.thickness / self.n_layers
        # Boundary layer is liquid: use the free-water diffusivity if supplied,
        # else fall back to the in-biofilm value (backward compatible).
        if boundary_diffusivity is None:
            D_bl = self._D
        else:
            D_bl = jnp.broadcast_to(jnp.asarray(boundary_diffusivity, dtype=float), (n,))
            D_bl = jnp.where(self.soluble_mask, D_bl, 0.0)
        self._kL = D_bl / self.boundary_layer                   # (n_species,)
        # Mid-point depth of each layer from the surface (for reporting).
        self._depth = (jnp.arange(self.n_layers) + 0.5) * self._dz
        self._jit_cache: dict = {}

    def solve(
        self,
        C0: jnp.ndarray,
        params: jnp.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[jnp.ndarray] = None,
        *,
        conditions: Optional[SpatialConditions] = None,
    ) -> BiofilmSolution:
        """Integrate the layered biofilm over a time span.

        Parameters
        ----------
        C0 : jnp.ndarray
            Initial state. Either ``(n_species,)`` --- the same composition in
            the bulk and every layer --- or ``(n_layers + 1, n_species)`` to set
            the bulk and each layer explicitly (row 0 bulk, rows 1.. surface to
            wall). The latter sets the stratified particulate (biomass) profile.
        params : jnp.ndarray
            Rate constant vector, shape ``(n_params,)``.
        t_span : tuple of float
            ``(t_start, t_end)`` integration interval.
        t_eval : jnp.ndarray, optional
            Times at which to record the solution. If ``None`` only the endpoint.
        conditions : SpatialConditions, optional
            Override the reactor conditions for this call.

        Returns
        -------
        BiofilmSolution
        """
        params = jnp.asarray(params)
        n = self.network.n_species
        if params.shape != (self.network.n_params,):
            raise ValueError(
                f"params has shape {params.shape}, expected ({self.network.n_params},)"
            )
        C0 = jnp.asarray(C0)
        n_comp = self.n_layers + 1
        if C0.shape == (n,):
            y0 = jnp.broadcast_to(C0, (n_comp, n))
        elif C0.shape == (n_comp, n):
            y0 = C0
        else:
            raise ValueError(
                f"C0 has shape {C0.shape}, expected ({n},) or ({n_comp}, {n})."
            )

        t0, t1 = float(t_span[0]), float(t_span[1])
        if not (t1 > t0):
            raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")
        active = conditions if conditions is not None else self.conditions
        condition_arrays = active.fields

        if t_eval is None:
            t_eval_arr, cache_key = None, (t0, t1, None)
        else:
            t_eval_arr = jnp.asarray(t_eval)
            cache_key = (t0, t1, tuple(t_eval_arr.shape))

        jitted = self._jit_cache.get(cache_key)
        if jitted is None:
            jitted = self._build_jitted_solve(t0, t1, t_eval_arr is not None)
            self._jit_cache[cache_key] = jitted

        if t_eval_arr is None:
            ts, ys = jitted(y0, params, condition_arrays)
        else:
            ts, ys = jitted(y0, params, condition_arrays, t_eval_arr)
        # ys: (n_t, n_comp, n_species). Bulk is compartment 0.
        return BiofilmSolution(
            t=ts, C=ys[:, 0, :], profile=ys, depth=self._depth, network=self.network
        )

    def _build_jitted_solve(self, t0: float, t1: float, has_t_eval: bool):
        # Capture only pre-existing concrete arrays/scalars (set in __init__),
        # never values created in this scope: solve() builds and CACHES the jitted
        # function lazily, possibly inside a caller's trace (e.g. calibrate), so
        # anything created here would escape that trace. All new arrays are formed
        # inside the jit below.
        network = self.network
        rtol, atol, adjoint, dtmax = self.rtol, self.atol, self.adjoint, self.dtmax
        max_steps = self.max_steps
        fixed_mask = self.fixed_mask            # (n_species,)
        biofilm_mask = self._biofilm_mask       # (n_reactions,) or None
        D, kL, dz = self._D, self._kL, self._dz
        area_per_volume = self.area_per_volume
        n_species = network.n_species

        def make_rhs(condition_arrays, params):
            stoich = network.compute_stoich(params)
            # Phase split: zeroing a reaction's stoichiometry row removes its
            # contribution to dCdt (which is stoich.T @ rates), so the positivity
            # limiter still sees the correct per-compartment net term. Bulk
            # reactions run only in the bulk; biofilm reactions only in the layers.
            if biofilm_mask is None:
                stoich_bulk = stoich_film = stoich
            else:
                stoich_bulk = stoich * (~biofilm_mask)[:, None]   # bulk + chemical
                stoich_film = stoich * biofilm_mask[:, None]      # biofilm only

            def rhs(t, y, args):
                bulk = network.dCdt(y[0], args, condition_arrays, 0, stoich=stoich_bulk)
                layers = jax.vmap(
                    lambda c: network.dCdt(c, args, condition_arrays, 0, stoich=stoich_film)
                )(y[1:])
                react = jnp.concatenate([bulk[None, :], layers], axis=0)
                dydt = react + _diffusion_transport(
                    y, D, kL, dz, area_per_volume, n_species
                )
                # Held-fixed species (mature-biofilm sustained sources/sinks)
                # have their net rate zeroed; everything else evolves. Reactive
                # particulates (D==0, not in fixed_mask) react but do not diffuse.
                return jnp.where(fixed_mask[None, :], 0.0, dydt)

            return rhs

        if has_t_eval:
            @jax.jit
            def _solve(y0, params, condition_arrays, t_eval):
                sol = _run_diffeqsolve(
                    make_rhs(condition_arrays, params),
                    t0=t0, t1=t1, y0=y0, args=params,
                    saveat=diffrax.SaveAt(ts=t_eval),
                    rtol=rtol, atol=atol, adjoint=adjoint, dtmax=dtmax,
                    max_steps=max_steps,
                )
                return sol.ts, sol.ys
            return _solve

        @jax.jit
        def _solve(y0, params, condition_arrays):
            sol = _run_diffeqsolve(
                make_rhs(condition_arrays, params),
                t0=t0, t1=t1, y0=y0, args=params,
                saveat=diffrax.SaveAt(t1=True),
                rtol=rtol, atol=atol, adjoint=adjoint, dtmax=dtmax,
                max_steps=max_steps,
            )
            return sol.ts, sol.ys
        return _solve
