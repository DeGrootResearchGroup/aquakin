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

The same :class:`~aquakin.core.model.CompiledModel` runs in every
compartment --- the point is that identical chemistry behaves differently once
depth is resolved. A model intended for this reactor should express its rates
per unit volume with the local biomass as an explicit reactant (so a compartment
with little biomass carries little rate), rather than lumping the biofilm into an
area-to-volume multiplier.

The diffusion operator conserves the volume-weighted total exactly. Element
(COD/S/N) conservation across reactions is exact only when the model's
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
from aquakin.core.model import CompiledModel
from aquakin.integrate._common import (
    DifferentiationConfig,
    GradientCheckMixin,
    IntegratorConfig,
    _HasNamedSpecies,
    _run_diffeqsolve,
    friendly_solve_errors,
    init_solver_settings,
    to_native_time,
    validate_t_eval,
)


@dataclass
class BiofilmSolution(_HasNamedSpecies):
    """Solution returned by :meth:`BiofilmReactor.solve`.

    Attributes
    ----------
    t : jnp.ndarray
        Times at which the solution was recorded, shape ``(n_t,)``.
    C : jnp.ndarray
        **Bulk** concentration trajectory, shape ``(n_t, n_species)``. This is
        the measurable (well-mixed liquid) signal; :meth:`C_named` reads it. It
        is the raw integrated state (likewise ``profile``): if the model sets
        ``clip_negative_states``, entries may be **small transient negatives**,
        because the ``max(C, 0)`` clamp is applied only when evaluating the
        reaction rates, not to the saved state. These are a normal numerical
        transient, not an error; clip with ``jnp.maximum(sol.C, 0.0)`` for
        display if needed.
    profile : jnp.ndarray
        Full depth-resolved trajectory, shape ``(n_t, n_compartments,
        n_species)``, where compartment 0 is the bulk and 1..``n_layers`` run
        from the biofilm surface (adjacent to the bulk) to the wall.
    depth : jnp.ndarray
        Mid-point depth of each biofilm layer from the surface, shape
        ``(n_layers,)`` (metres). Bulk has no depth and is omitted.
    model : CompiledModel
        The model that produced this solution.
    """

    t: jnp.ndarray
    C: jnp.ndarray
    profile: jnp.ndarray
    depth: jnp.ndarray
    model: CompiledModel

    def profile_named(self, species: str) -> jnp.ndarray:
        """Depth profile of one species over time, shape ``(n_t, n_compartments)``."""
        if species not in self.model.species_index:
            raise KeyError(f"Unknown species '{species}'. Available: {self.model.species}")
        return self.profile[:, :, self.model.species_index[species]]

    def to_dataframe(self, *, profile: bool = False, units_in_columns: bool = False):
        """Return the solution as a pandas ``DataFrame``.

        By default this is the **bulk** (measurable) trajectory: one row per
        time, one column per species, indexed by time -- identical to the other
        solutions.

        Parameters
        ----------
        profile : bool, optional
            If ``True``, return the full depth-resolved trajectory instead: a
            ``MultiIndex`` of ``(t, compartment)`` rows (compartment 0 is the
            bulk, 1..``n_layers`` run surface->wall), a ``depth`` column (NaN
            for the bulk row), and one column per species.
        units_in_columns : bool, optional
            Append ``" [unit]"`` to each species column label.
        """
        if not profile:
            return super().to_dataframe(units_in_columns=units_in_columns)

        import numpy as np

        from aquakin.integrate._common import build_dataframe, require_pandas

        pd = require_pandas()
        prof = np.asarray(self.profile)  # (n_t, n_comp, n_species)
        n_t, n_comp, _ = prof.shape
        t = np.asarray(self.t)
        index = pd.MultiIndex.from_arrays(
            [np.repeat(t, n_comp), np.tile(np.arange(n_comp), n_t)],
            names=["t", "compartment"],
        )
        flat = prof.reshape(n_t * n_comp, prof.shape[2])
        # depth aligned to compartments: NaN for the bulk (compartment 0).
        depth_per_comp = np.concatenate([[np.nan], np.asarray(self.depth)])
        depth_col = np.tile(depth_per_comp, n_t)
        columns = [(sp, flat[:, j]) for j, sp in enumerate(self.model.species)]
        units = {sp: self.model.units_of(sp) for sp in self.model.species}
        return build_dataframe(
            index,
            columns,
            units=units,
            units_in_columns=units_in_columns,
            extra=[("depth", depth_col)],
        )


def _diffusion_transport(C, D, kL, dz, area_per_volume, n_species):
    """Diffusive transport for the whole state, shape ``(n_comp, n_species)``.

    Pure function (no reactor ``self``) so it can be closed over by a jit-compiled
    solve without leaking trace-created state. ``C[0]`` is the bulk; ``C[1:]`` the
    layers (surface..wall). Particulate columns are zero where ``D == 0``. Uses
    the conservative finite-volume (face-flux) form: only first-order face
    differences, so the volume-weighted total is conserved exactly.
    """
    bulk = C[0]  # (n_species,)
    layers = C[1:]  # (n_layers, n_species)
    # Internal interface fluxes between layer j and j+1 (positive toward the wall).
    f_internal = D[None, :] * (layers[:-1] - layers[1:]) / dz  # (n_layers-1, n_species)
    # Bulk <-> surface flux across the boundary layer (positive into the film).
    f_bs = kL * (bulk - layers[0])  # (n_species,)
    flux_in = jnp.concatenate([f_bs[None, :], f_internal], axis=0)
    zero = jnp.zeros((1, n_species))
    flux_out = jnp.concatenate([f_internal, zero], axis=0)  # wall: no flux out
    d_layers = (flux_in - flux_out) / dz  # (n_layers, n_species)
    d_bulk = -area_per_volume * f_bs  # bulk loses surface flux
    return jnp.concatenate([d_bulk[None, :], d_layers], axis=0)


def _attachment_transport(C, k_att, attach_mask, dz, area_per_volume, n_species):
    """Particulate attachment from the bulk onto the biofilm surface (Eq 1).

    ``r_att = k_att * X_i^bulk`` (per bulk volume, 1/d * conc) removes particulate
    from the bulk and deposits it in the surface layer. Mass-conserving: the bulk
    loss ``k_att*X_bulk`` over the bulk volume equals the surface-layer gain
    ``k_att*X_bulk/(A_V*dz)`` over the layer volume (A*dz). Returns a
    ``(n_comp, n_species)`` rate; zero everywhere if ``k_att == 0``.
    """
    bulk = C[0]  # (n_species,)
    r = (k_att * bulk) * attach_mask  # (n_species,) bulk loss rate
    d = jnp.zeros((C.shape[0], n_species))
    d = d.at[0].add(-r)  # bulk loses attached mass
    d = d.at[1].add(r / (area_per_volume * dz))  # surface layer gains it
    return d


def _detachment_transport(C, k_det, detach_mask, dz, area_per_volume, n_species):
    """Biofilm particulate detachment back to the bulk (Eqs 2-3, first order).

    Each biofilm layer loses ``k_det * X_i,j``; the eroded mass enters the bulk
    (where it then washes out with the feed). Mass-conserving: the per-layer loss
    over the layer volume (A*dz) equals the bulk gain over the bulk volume. Zero
    if ``k_det == 0``.
    """
    layers = C[1:]  # (n_layers, n_species)
    r = (k_det * layers) * detach_mask[None, :]  # per-layer loss rate
    d = jnp.zeros((C.shape[0], n_species))
    d = d.at[1:].add(-r)  # layers lose biomass
    d = d.at[0].add(r.sum(axis=0) * dz * area_per_volume)  # bulk gains it
    return d


def _default_soluble_mask(model: CompiledModel) -> jnp.ndarray:
    """Classify species as soluble (diffuses, evolves) vs particulate (fixed).

    Heuristic for the WATS/ASM naming convention: soluble names start with
    ``S`` (``S_*``, ``sumS``), particulate names start with ``X``. Callers can
    override with an explicit mask.
    """
    return jnp.asarray([not s.startswith("X") for s in model.species], dtype=bool)


class BiofilmReactor(GradientCheckMixin):
    """Stateless 1-D layered biofilm reactor (diffusion--reaction over depth).

    Parameters
    ----------
    model : CompiledModel
        Compiled reaction model, run in every compartment.
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
        for a single-phase model. For a WATS-style model the biofilm
        reactions are those carrying the ``A_V`` area factor; bulk reactions
        carry the suspended biomass ``[X_BH]`` or are abiotic. This separation,
        not a zeroed biomass state, is what keeps the two phases from leaking
        into each other.
    rtol, atol : float
        Solver tolerances (scalar; applied across the whole layered state).
    integrator : IntegratorConfig, optional
        Integrator / step-size configuration (ESDIRK ``order``, ``factormax``,
        ``dtmax``, ``max_steps``, an explicit ``solver``); see
        :class:`~aquakin.integrate.batch.BatchReactor`. A stiff per-layer-biomass
        model with a tight ``dtmax`` can exceed the default ``max_steps``; raise
        it if the solve raises a max-steps error.
    diff : DifferentiationConfig, optional
        Autodiff configuration (``mode``, ``method``); see
        :class:`~aquakin.integrate.batch.BatchReactor`.

    Notes
    -----
    The state is a ``(n_layers + 1, n_species)`` array: row 0 is the bulk, rows
    1..``n_layers`` are the biofilm layers from surface to wall. The wall is a
    no-flux boundary. Species in ``fixed_mask`` have their net rate zeroed
    everywhere (held fixed); all others evolve. Diffusion is governed separately
    by ``soluble_mask``.

    Examples
    --------
    Sulfur biofilm models carry *reactive* non-diffusing particulates ---
    elemental sulfur ``X_S0`` and precipitated ``X_FeS`` --- whose inventory
    genuinely drains and fills. The default ``fixed_mask`` would freeze them and
    silently break mass balance (the reactor warns when it would). Build a mask
    that holds **only the inert solids** fixed and lets every reactive pool
    evolve:

    >>> net = aquakin.load_model("wats_sewer_khalil_paper_balanced_biofilm_multispecies")
    >>> inert = {"X_I"}   # the only genuinely inert, non-depleting solid
    >>> fixed_mask = jnp.array([s in inert for s in net.species])
    >>> reactor = aquakin.BiofilmReactor(
    ...     net, conditions, n_layers=6, thickness=8e-4, area_per_volume=50.0,
    ...     diffusivity=1e-4, boundary_layer=1e-4, fixed_mask=fixed_mask)

    Everything not named in ``inert`` --- the heterotrophs and functional-group
    biomass, the stored-substrate reservoirs ``X_S1``/``X_S2``, and the reactive
    sulfur pools ``X_S0``/``X_FeS`` --- then evolves and conserves mass. (The
    areal ``*_biofilm`` variant, by contrast, deliberately freezes its biomass as
    a sustained "mature biofilm" reservoir, so its mask holds the biomass fixed
    too; the rule is always "freeze only what is genuinely non-depleting".)
    """

    def __init__(
        self,
        model: CompiledModel,
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
        max_density=None,
        packing_fraction: float = 0.8,
        k_att: float = 0.0,
        attach_mask: Optional[jnp.ndarray] = None,
        k_det: float = 0.0,
        detach_mask: Optional[jnp.ndarray] = None,
        clamp_bulk: bool = False,
        feed=None,
        dilution_rate: float = 0.0,
        biofilm_reactions=None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        integrator: IntegratorConfig = IntegratorConfig(),
        diff: DifferentiationConfig = DifferentiationConfig(),
    ) -> None:
        conditions.validate_required(model.conditions_required)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {n_layers}.")
        if not (thickness > 0 and boundary_layer > 0 and area_per_volume > 0):
            raise ValueError("thickness, boundary_layer and area_per_volume must be positive.")
        init_solver_settings(self, model, rtol=rtol, integrator=integrator, diff=diff)
        self.conditions = conditions
        self.n_layers = int(n_layers)
        self.thickness = float(thickness)
        self.area_per_volume = float(area_per_volume)
        self.boundary_layer = float(boundary_layer)
        # Scalar atol over the multi-compartment (n_layers+1, n_species) state --
        # the per-species default_atol of the single-vector reactors does not
        # apply to this state shape, so the biofilm keeps an explicit scalar.
        self.atol = float(atol)

        n = model.n_species
        if soluble_mask is None:
            soluble_mask = _default_soluble_mask(model)
        self.soluble_mask = jnp.asarray(soluble_mask, dtype=bool)
        if self.soluble_mask.shape != (n,):
            raise ValueError(f"soluble_mask must have shape ({n},); got {self.soluble_mask.shape}")

        # Which species are held fixed (net rate zeroed). Decoupled from
        # diffusion: a reactive particulate diffuses (soluble_mask False) yet must
        # still react (fixed_mask False). Default: every particulate fixed.
        fixed_defaulted = fixed_mask is None
        if fixed_mask is None:
            fixed_mask = ~self.soluble_mask
        self.fixed_mask = jnp.asarray(fixed_mask, dtype=bool)
        if self.fixed_mask.shape != (n,):
            raise ValueError(f"fixed_mask must have shape ({n},); got {self.fixed_mask.shape}")
        # Footgun guard: the default freezes every particulate, which is wrong for
        # a REACTIVE particulate (one that some reaction produces/consumes) -- a
        # frozen reactive particulate becomes a non-depleting source/sink and
        # silently breaks mass balance (e.g. elemental sulfur feeding a sulfate
        # source). Only warn for the default; an explicit fixed_mask is a
        # deliberate choice (a particulate may be frozen on purpose as a sustained
        # "mature biofilm" source).
        #
        # "Reactive" is ANY nonzero stoichiometry, not "appears with both signs".
        # A precipitation sink like X_FeS is only ever PRODUCED (consumed > 0
        # never), so a both-signs test would miss it -- yet freezing it is exactly
        # the mass-balance break this guards against. Any-nonzero catches it; the
        # cost is also flagging inert solids (X_I) and stored reservoirs, but under
        # the default mask freezing those is wrong too, so the broad warning is the
        # right "pass an explicit mask here" signal.
        if fixed_defaulted:
            stoich = model.compute_stoich(model.default_parameters())
            reactive = jnp.any(stoich != 0.0, axis=0)  # (n_species,)
            frozen_reactive = reactive & self.fixed_mask
            offenders = [s for s, f in zip(model.species, list(map(bool, frozen_reactive))) if f]
            if offenders:
                warnings.warn(
                    "BiofilmReactor is holding reactive particulate(s) "
                    f"{offenders} fixed by the default fixed_mask: their reactions "
                    "are zeroed everywhere, so they act as non-depleting "
                    "source/sinks and can break mass balance (e.g. a frozen "
                    "elemental-sulfur or FeS pool). Pass an explicit fixed_mask "
                    "holding only the genuinely inert solids fixed (e.g. "
                    "jnp.array([s in {'X_I'} for s in model.species])) and "
                    "letting every reactive particulate evolve.",
                    stacklevel=2,
                )

        # Per-reaction phase mask: which reactions are biofilm processes (run in
        # the layers only) vs bulk/chemical (run in the bulk only). ``None`` ->
        # every reaction runs in every compartment (a single-phase model).
        # Accepts a list of reaction names or a boolean ``(n_reactions,)`` array.
        if biofilm_reactions is None:
            self._biofilm_mask = None
        elif all(isinstance(x, str) for x in biofilm_reactions):
            names = set(biofilm_reactions)
            unknown = names - set(model.reaction_names)
            if unknown:
                raise ValueError(f"Unknown biofilm reaction names: {sorted(unknown)}")
            self._biofilm_mask = jnp.asarray(
                [rn in names for rn in model.reaction_names], dtype=bool
            )
        else:
            bm = jnp.asarray(biofilm_reactions, dtype=bool)
            if bm.shape != (model.n_reactions,):
                raise ValueError(
                    f"biofilm_reactions mask must have shape ({model.n_reactions},); got {bm.shape}"
                )
            self._biofilm_mask = bm

        D = jnp.broadcast_to(jnp.asarray(diffusivity, dtype=float), (n,))
        # Particulates do not diffuse: zero their diffusivity regardless.
        self._D = jnp.where(self.soluble_mask, D, 0.0)  # (n_species,)
        self._dz = self.thickness / self.n_layers
        # Boundary layer is liquid: use the free-water diffusivity if supplied,
        # else fall back to the in-biofilm value (backward compatible).
        if boundary_diffusivity is None:
            D_bl = self._D
        else:
            D_bl = jnp.broadcast_to(jnp.asarray(boundary_diffusivity, dtype=float), (n,))
            D_bl = jnp.where(self.soluble_mask, D_bl, 0.0)
        self._kL = D_bl / self.boundary_layer  # (n_species,)
        # Mid-point depth of each layer from the surface (for reporting).
        self._depth = (jnp.arange(self.n_layers) + 0.5) * self._dz

        # Biofilm-growth closure (Jiang 2009 Eqs 8-10): a per-species maximum
        # density rho_i^f caps the solid volume fraction. The inverse density
        # 1/rho_i (0 where uncapped) gives the volume fraction X_i/rho_i; biomass
        # GROWTH is throttled as the layer's total solid fraction approaches the
        # packing limit ``packing_fraction`` (= 1 - eps_l, taken constant since
        # Jiang Eq 9 varies only at cm scale). Without this closure a growing
        # biofilm has no unique steady state. ``max_density=None`` -> no cap.
        self.packing_fraction = float(packing_fraction)
        self._has_cap = max_density is not None
        if max_density is None:
            self._inv_rho = jnp.zeros((n,))
        else:
            rho = jnp.broadcast_to(jnp.asarray(max_density, dtype=float), (n,))
            # Uncapped (rho<=0 or inf) contribute no volume fraction.
            self._inv_rho = jnp.where(jnp.isfinite(rho) & (rho > 0), 1.0 / rho, 0.0)

        # Attachment (Jiang 2009 Eq 1): bulk particulates attach to the biofilm
        # surface at rate k_att * X_i^bulk, seeding the layers. ``k_att=0`` -> off.
        self._k_att = float(k_att)
        self._has_att = self._k_att != 0.0

        # Detachment (Jiang 2009 Eqs 2-3, lumped to first order): biofilm
        # particulates erode back to the bulk at rate k_det * X_i,j, where they
        # then wash out with the feed. This -- not the density cap -- is what sets
        # the steady state (growth = decay + detachment, a chemostat-like fixed
        # point), gives the weeks-to-months maturation timescale, and (as a
        # -k_det Jacobian-diagonal term) regularizes the steady-state solve. With
        # no sewer shear data, k_det is a calibration knob (low shear -> low k_det
        # -> thicker, denser biofilm). ``k_det=0`` -> off.
        self._k_det = float(k_det)
        self._has_det = self._k_det != 0.0
        if detach_mask is None:
            detach_mask = ~self.soluble_mask  # particulates detach
        self._detach_mask = jnp.asarray(detach_mask, dtype=bool)
        if self._detach_mask.shape != (n,):
            raise ValueError(f"detach_mask must have shape ({n},); got {self._detach_mask.shape}")

        # Hold the bulk (compartment 0) fixed as a reservoir at its initial value
        # -- a Dirichlet boundary representing a sustained operating condition
        # against which the biofilm matures. The biofilm still exchanges with it.
        self.clamp_bulk = bool(clamp_bulk)

        # Continuous feed into the bulk: d_bulk += dilution_rate*(feed - bulk),
        # a CSTR mass balance (dilution_rate = Q/V, 1/d). The steady bulk is then
        # the predicted effluent. ``feed=None`` or dilution_rate=0 -> off.
        self.dilution_rate = float(dilution_rate)
        self._has_feed = feed is not None and self.dilution_rate != 0.0
        if feed is None:
            self._feed = jnp.zeros((n,))
        else:
            self._feed = jnp.broadcast_to(jnp.asarray(feed, dtype=float), (n,))
        if attach_mask is None:
            attach_mask = ~self.soluble_mask  # particulates attach
        self._attach_mask = jnp.asarray(attach_mask, dtype=bool)
        if self._attach_mask.shape != (n,):
            raise ValueError(f"attach_mask must have shape ({n},); got {self._attach_mask.shape}")

        self._jit_cache: dict = {}
        self._sens_jit_cache: dict = {}

    def _check_params(self, params: jnp.ndarray) -> jnp.ndarray:
        """Coerce and shape-check the parameter vector."""
        params = jnp.asarray(params)
        if params.shape != (self.model.n_params,):
            raise ValueError(f"params has shape {params.shape}, expected ({self.model.n_params},)")
        return params

    def _coerce_y0(self, C0: jnp.ndarray) -> jnp.ndarray:
        """Validate and broadcast the initial state to ``(n_layers+1, n_species)``.

        Accepts ``(n_species,)`` --- the same composition in the bulk and every
        layer --- or the full ``(n_layers+1, n_species)`` bulk-plus-per-layer
        profile.
        """
        n = self.model.n_species
        n_comp = self.n_layers + 1
        C0 = jnp.asarray(C0)
        if C0.shape == (n,):
            return jnp.broadcast_to(C0, (n_comp, n))
        if C0.shape == (n_comp, n):
            return C0
        raise ValueError(f"C0 has shape {C0.shape}, expected ({n},) or ({n_comp}, {n}).")

    def solve(
        self,
        C0: jnp.ndarray,
        t_span: tuple[float, float] = None,
        t_eval: Optional[jnp.ndarray] = None,
        *,
        params: Optional[jnp.ndarray] = None,
        conditions: Optional[SpatialConditions] = None,
        time_unit: Optional[str] = None,
    ) -> BiofilmSolution:
        """Integrate the layered biofilm over a time span.

        Parameters
        ----------
        C0 : jnp.ndarray
            Initial state. Either ``(n_species,)`` --- the same composition in
            the bulk and every layer --- or ``(n_layers + 1, n_species)`` to set
            the bulk and each layer explicitly (row 0 bulk, rows 1.. surface to
            wall). The latter sets the stratified particulate (biomass) profile.
        t_span : tuple of float
            ``(t_start, t_end)`` integration interval, in the model's time unit
            unless ``time_unit`` is given. The required second positional argument.
        t_eval : jnp.ndarray, optional
            Times at which to record the solution. If ``None`` only the endpoint.
        params : jnp.ndarray, optional, keyword-only
            Rate constant vector, shape ``(n_params,)``. Defaults to
            ``model.default_parameters()``. Keyword-only so a positional
            ``t_span`` can never land in it.
        conditions : SpatialConditions, optional
            Override the reactor conditions for this call.
        time_unit : str, optional
            The time unit ``t_span`` / ``t_eval`` are in (``"s"``/``"min"``/
            ``"h"``/``"d"``); see :meth:`BatchReactor.solve`. Default ``None``
            uses the model's native unit.

        Returns
        -------
        BiofilmSolution
        """
        if params is None:
            params = self.model.default_parameters()
        params = self._check_params(params)
        y0 = self._coerce_y0(C0)

        if t_span is None:
            raise ValueError("t_span=(t_start, t_end) is required.")
        t_span, t_eval, _time_factor = to_native_time(
            self.model.time_unit, time_unit, t_span, t_eval
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
            validate_t_eval(t_eval_arr, t0, t1)
            cache_key = (t0, t1, tuple(t_eval_arr.shape))

        jitted = self._jit_cache.get(cache_key)
        if jitted is None:
            jitted = self._build_jitted_solve(t0, t1, t_eval_arr is not None)
            self._jit_cache[cache_key] = jitted

        with friendly_solve_errors(self.max_steps, what="biofilm reactor solve"):
            if t_eval_arr is None:
                ts, ys = jitted(y0, params, condition_arrays)
            else:
                ts, ys = jitted(y0, params, condition_arrays, t_eval_arr)
        # ys: (n_t, n_comp, n_species). Bulk is compartment 0.
        if _time_factor != 1.0:
            ts = ts / _time_factor  # native -> requested unit
        sol = BiofilmSolution(t=ts, C=ys[:, 0, :], profile=ys, depth=self._depth, model=self.model)
        if time_unit is not None:
            sol._requested_time_unit = time_unit
        return sol

    def solve_sensitivity(
        self,
        C0: jnp.ndarray,
        params: jnp.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[jnp.ndarray] = None,
        *,
        sens_params,
        conditions: Optional[SpatialConditions] = None,
        sens_rtol: Optional[float] = None,
        sens_atol=None,
        param_scale=None,
        shared_factor: Optional[bool] = None,
    ) -> tuple["BiofilmSolution", jnp.ndarray]:
        """Solve and return the forward sensitivity of the **bulk** ``dC/dtheta``.

        Integrates the augmented ``[y; S]`` system over the full layered state
        with adaptive control over both, so the sensitivity is exact and finite
        without the ``dtmax`` cap that ordinary AD through this stiff
        diffusion--reaction solve needs (see
        :mod:`aquakin.integrate.forward_sensitivity`). This is the canonical use
        case: the biofilm models are stiff enough that capping ``dtmax`` for a
        reverse-mode gradient is an ~10x penalty, and this removes it.

        Parameters
        ----------
        C0, params, t_span, t_eval, conditions
            As for :meth:`solve` (``C0`` may be ``(n_species,)`` or
            ``(n_layers + 1, n_species)``).
        sens_params : list of str or int
            Namespaced parameter names (or integer indices into ``params``).
        sens_rtol, sens_atol, param_scale
            Sensitivity error-control tolerances (CVODES defaults; see
            :func:`~aquakin.integrate.forward_sensitivity.augmented_forward_sensitivity`).
        shared_factor : bool, optional
            CVODES simultaneous-corrector linear solve. ``None`` (default)
            auto-selects ``True`` for more than one sensitivity parameter --
            the regime where factorising the shared diagonal block once is
            markedly cheaper than the dense augmented solve -- else ``False``.

        Returns
        -------
        sol : BiofilmSolution
            The usual solution (bulk ``C`` and full ``profile``).
        S : jnp.ndarray
            Sensitivity of the **bulk** (measurable) concentration,
            ``dC_bulk/dtheta``, shape ``(n_t, n_species, n_sens_params)`` --
            aligned with ``sol.C``.
        """
        from aquakin.integrate.forward_sensitivity import (
            resolve_sens_indices,
            run_forward_sensitivity,
        )

        params = self._check_params(params)
        y0 = self._coerce_y0(C0)
        n = self.model.n_species
        n_comp = self.n_layers + 1
        t0, t1 = float(t_span[0]), float(t_span[1])
        if not (t1 > t0):
            raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")

        free_idx = resolve_sens_indices(self.model, sens_params)
        if shared_factor is None:
            shared_factor = free_idx.shape[0] > 1
        active = conditions if conditions is not None else self.conditions
        cond = active.fields
        ndof = n_comp * n
        k = free_idx.shape[0]
        atol_y = jnp.full((ndof,), float(self.atol))
        y0_flat = y0.reshape(-1)
        t_eval_arr = None if t_eval is None else jnp.asarray(t_eval)
        if t_eval_arr is not None:
            validate_t_eval(t_eval_arr, t0, t1)

        def _finish(ts, y_traj, S_traj):
            n_t = ts.shape[0]
            profile = y_traj.reshape(n_t, n_comp, n)
            S_full = S_traj.reshape(n_t, n_comp, n, k)
            sol = BiofilmSolution(
                t=ts,
                C=profile[:, 0, :],
                profile=profile,
                depth=self._depth,
                model=self.model,
            )
            return sol, S_full[:, 0, :, :]

        def make_f_flat(condition_arrays):
            def f_flat(t, y_flat, p):
                return self._make_rhs(condition_arrays, p)(
                    0.0, y_flat.reshape(n_comp, n), p
                ).reshape(-1)

            return f_flat

        cache_key = (
            t0,
            t1,
            None if t_eval_arr is None else tuple(t_eval_arr.shape),
            tuple(int(i) for i in free_idx),
            bool(shared_factor),
            None if sens_rtol is None else float(sens_rtol),
        )
        return _finish(
            *run_forward_sensitivity(
                make_f_flat,
                y0_flat,
                params,
                free_idx,
                cond,
                t0=t0,
                t1=t1,
                t_eval=t_eval_arr,
                rtol=self.rtol,
                atol_y=atol_y,
                sens_rtol=sens_rtol,
                sens_atol=sens_atol,
                param_scale=param_scale,
                dtmax=self.dtmax,
                max_steps=self.max_steps,
                shared_factor=shared_factor,
                cache=self._sens_jit_cache,
                cache_key=cache_key,
            )
        )

    def _make_rhs(self, condition_arrays, params):
        """Build the depth-resolved diffusion--reaction RHS ``f(t, y, args)``.

        Shared by the time-stepping solve and :meth:`steady_state`. Reads only
        concrete ``__init__`` arrays on ``self``, so it bakes as constants when
        traced/jitted (no trace leak). ``args`` carries the (possibly traced)
        parameter vector for the rate evaluation; the stoichiometry is built from
        ``params`` so the whole RHS depends on the parameters through both.
        """
        model = self.model
        fixed_mask = self.fixed_mask
        biofilm_mask = self._biofilm_mask
        D, kL, dz = self._D, self._kL, self._dz
        area_per_volume = self.area_per_volume
        n_species = model.n_species
        inv_rho, packing = self._inv_rho, self.packing_fraction
        has_cap, has_att = self._has_cap, self._has_att
        k_att, attach_mask = self._k_att, self._attach_mask
        has_det, k_det, detach_mask = self._has_det, self._k_det, self._detach_mask
        clamp_bulk = self.clamp_bulk
        has_feed, feed, dilution = self._has_feed, self._feed, self.dilution_rate

        stoich = model.compute_stoich(params)
        # Phase split: zeroing a reaction's stoichiometry row removes its
        # contribution to dCdt (= stoich.T @ rates), so the positivity limiter
        # still sees the correct per-compartment net term. Bulk reactions run only
        # in the bulk; biofilm reactions only in the layers.
        if biofilm_mask is None:
            stoich_bulk = stoich_film = stoich
        else:
            stoich_bulk = stoich * (~biofilm_mask)[:, None]  # bulk + chemical
            stoich_film = stoich * biofilm_mask[:, None]  # biofilm only
        # Growth reactions: those producing a density-capped species. The cap
        # throttles the WHOLE reaction (not the net per-species rate), so substrate
        # uptake and biomass production scale together -- mass-conserving.
        if has_cap:
            capped = inv_rho > 0
            growth_rxn = jnp.any((stoich > 0) & capped[None, :], axis=1)

        def cell(c, st, args):
            # chemistry in one compartment via the canonical dCdt (clip inputs +
            # positivity limiter), with the optional growth throttle as a
            # per-reaction rate_scale so uptake and production scale together
            rate_scale = None
            if has_cap:
                # space availability: 1 when empty, 0 at the packing limit
                s = jnp.clip(1.0 - (c @ inv_rho) / packing, 0.0, 1.0)
                rate_scale = jnp.where(growth_rxn, s, 1.0)
            return model.dCdt(c, args, condition_arrays, 0, stoich=st, rate_scale=rate_scale)

        def rhs(t, y, args):
            bulk = cell(y[0], stoich_bulk, args)
            layers = jax.vmap(lambda c: cell(c, stoich_film, args))(y[1:])
            react = jnp.concatenate([bulk[None, :], layers], axis=0)
            transport = _diffusion_transport(y, D, kL, dz, area_per_volume, n_species)
            if has_att:
                transport = transport + _attachment_transport(
                    y, k_att, attach_mask, dz, area_per_volume, n_species
                )
            if has_det:
                transport = transport + _detachment_transport(
                    y, k_det, detach_mask, dz, area_per_volume, n_species
                )
            if has_feed:
                transport = transport.at[0].add(dilution * (feed - y[0]))
            dydt = react + transport
            # Held-fixed species (mature-biofilm sustained sources/sinks) have
            # their net rate zeroed; everything else evolves. Reactive particulates
            # (D==0, not in fixed_mask) react but do not diffuse.
            dydt = jnp.where(fixed_mask[None, :], 0.0, dydt)
            if clamp_bulk:
                dydt = dydt.at[0].set(0.0)  # bulk held as a fixed reservoir
            return dydt

        return rhs

    def _build_jitted_solve(self, t0: float, t1: float, has_t_eval: bool):
        make_rhs = self._make_rhs
        rtol, atol, adjoint, dtmax = self.rtol, self.atol, self.adjoint, self.dtmax
        max_steps = self.max_steps
        order, factormax, solver = self.order, self.factormax, self.solver

        if has_t_eval:

            @jax.jit
            def _solve(y0, params, condition_arrays, t_eval):
                sol = _run_diffeqsolve(
                    make_rhs(condition_arrays, params),
                    t0=t0,
                    t1=t1,
                    y0=y0,
                    args=params,
                    saveat=diffrax.SaveAt(ts=t_eval),
                    rtol=rtol,
                    atol=atol,
                    adjoint=adjoint,
                    dtmax=dtmax,
                    max_steps=max_steps,
                    order=order,
                    factormax=factormax,
                    solver=solver,
                )
                return sol.ts, sol.ys

            return _solve

        @jax.jit
        def _solve(y0, params, condition_arrays):
            sol = _run_diffeqsolve(
                make_rhs(condition_arrays, params),
                t0=t0,
                t1=t1,
                y0=y0,
                args=params,
                saveat=diffrax.SaveAt(t1=True),
                rtol=rtol,
                atol=atol,
                adjoint=adjoint,
                dtmax=dtmax,
                max_steps=max_steps,
                order=order,
                factormax=factormax,
                solver=solver,
            )
            return sol.ts, sol.ys

        return _solve

    def steady_state(
        self,
        C0: jnp.ndarray,
        params: jnp.ndarray,
        *,
        conditions: Optional[SpatialConditions] = None,
        warmup: float = 20.0,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        newton_steps: int = 200,
    ) -> BiofilmSolution:
        """Solve for the steady-state profile by pseudo-transient continuation.

        Instead of integrating to steady state (slow for a maturing biofilm), this
        finds ``y*`` such that ``f(y*, params) = 0`` directly, by pseudo-transient
        continuation (PTC) -- damped-Newton steps that ramp from a stable
        time-stepping move to a full Newton step as the residual falls, robust on
        the stiff/slow biofilm where a plain Newton / Levenberg--Marquardt
        root-find stalls (see :func:`aquakin.plant.steady.solve_steady_state`). A
        short forward integration to ``warmup`` seeds the iteration (the seed is
        detached from the gradient). The result is differentiable w.r.t. ``params``
        via the implicit function theorem, so it composes
        with :func:`~aquakin.calibrate` -- the intended use is the continuous-feed
        maturation (``feed=...``, ``dilution_rate>0``) whose steady bulk is the
        predicted effluent and whose biofilm profile is a downstream batch IC.

        Parameters
        ----------
        C0 : jnp.ndarray
            Initial guess, ``(n_species,)`` (broadcast) or ``(n_layers+1,
            n_species)``.
        params : jnp.ndarray
            Rate constant vector.
        conditions : SpatialConditions, optional
            Override the reactor conditions for this call.
        warmup : float
            Forward-integration time used to seed the iteration. ``0`` uses
            ``C0`` directly.
        rtol : float
            Convergence tolerance on the scaled steady-state residual
            ``max_i |f_i| / max(|y_i|, 1)``.
        atol : float
            Retained for backward compatibility; unused (PTC converges on the
            relative residual ``rtol``).
        newton_steps : int
            Maximum PTC iterations.

        Returns
        -------
        BiofilmSolution
            With ``t = [inf]`` and a single time slice holding the steady profile.

        Notes
        -----
        Not compatible with ``clamp_bulk`` or held-fixed species (their RHS rows
        are identically zero, making the residual Jacobian singular). Use the
        continuous feed to drive the bulk instead.
        """
        from aquakin.plant.steady import solve_steady_state

        params = self._check_params(params)
        y0 = self._coerce_y0(C0)
        n = self.model.n_species
        n_comp = self.n_layers + 1
        active = conditions if conditions is not None else self.conditions
        condition_arrays = active.fields

        if warmup and warmup > 0.0:
            seed = self.solve(
                y0, params=params, t_span=(0.0, float(warmup)), conditions=active
            ).profile[-1]
        else:
            seed = y0
        seed = jax.lax.stop_gradient(seed)  # root-find seed: no path gradient

        # Solve RHS=0 on the flattened ``(n_comp, n)`` profile by pseudo-transient
        # continuation. PTC damps each step by a per-state pseudo-time that ramps
        # from a stable time-stepping move (far from the root) to a full Newton
        # step (near it), so it is robust to the difficulties of the stiff/slow
        # biofilm steady state that defeat a plain Newton or Levenberg--Marquardt
        # root-find: a generically singular reaction Jacobian (dormant species
        # under an anaerobic feed, or a density-capped layer, give zero rows) and
        # Newton overshoot into non-physical states. The per-state scaling weighs
        # the O(1e3) biomass and O(1) soluble modes comparably. The result is
        # differentiable w.r.t. ``params`` through the implicit function theorem
        # (see :func:`aquakin.plant.steady.solve_steady_state`).
        def rhs_flat(y_flat, p):
            rhs = self._make_rhs(condition_arrays, p)
            return rhs(0.0, y_flat.reshape(n_comp, n), p).reshape(-1)

        sol = solve_steady_state(
            rhs_flat,
            params,
            seed.reshape(-1),
            tol=rtol,
            max_iter=newton_steps,
            scale_floor=1.0,
            nonneg=True,
        )
        y_star = sol.state.reshape(n_comp, n)
        return BiofilmSolution(
            t=jnp.asarray([jnp.inf]),
            C=y_star[0][None, :],
            profile=y_star[None, :, :],
            depth=self._depth,
            model=self.model,
        )
