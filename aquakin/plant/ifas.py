"""IFAS / MBBR plant unit: a CSTR bulk coupled to an attached biofilm.

An Integrated Fixed-film Activated Sludge (IFAS) or Moving-Bed Biofilm Reactor
(MBBR) tank hosts carrier media whose surface grows a biofilm, alongside the
suspended sludge of an ordinary activated-sludge tank. The two fractions share
the bulk liquid: solubles (substrate, oxygen, ...) exchange between the bulk and
the depth-resolved biofilm, so the biofilm adds removal capacity beyond the
suspended biomass -- the whole point of an intensification retrofit.

This unit wires the existing depth-resolved :class:`~aquakin.integrate.biofilm.BiofilmReactor`
(1-D diffusion--reaction over biofilm depth) into the flowsheet. Its state is the
bulk concentration plus the biofilm layer profile; its RHS is the biofilm
reactor's diffusion--reaction core with the **bulk convection and aeration of a
plant CSTR** added on the bulk compartment (in place of the biofilm reactor's own
stand-alone CSTR-feed model). The effluent is the well-mixed bulk -- the biofilm
stays on the carrier.

The biofilm in this unit is a **mature, fixed attached-biomass** model: the
attached biomass is held as a sustained reservoir (its initial inventory) while
solubles diffuse and react through the depth and the suspended (bulk) fraction
evolves fully. This is stable by construction and is the natural first model for
placing an MBBR/IFAS tank in a plant; a fully dynamic biofilm (growth with
attachment / detachment / a density cap) is available on the underlying
:class:`~aquakin.integrate.biofilm.BiofilmReactor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions
from aquakin.integrate.biofilm import BiofilmReactor, _default_soluble_mask
from aquakin.plant.coupling import CouplingAware
from aquakin.plant.cstr import (
    Aeration,
    AerationUnit,
    aeration_transfer,
)
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


def _default_biofilm_fixed_mask(model, soluble_mask) -> jnp.ndarray:
    """Which particulates are the sustained mature-biofilm structure (held fixed).

    Freeze every particulate **except** a hydrolysis substrate -- one that some
    reaction consumes while producing a soluble (``XS -> SS``, ``XND -> SND``).
    Freezing such a pool would make it a non-depleting soluble source (the biofilm
    footgun); biomass and inert solids, by contrast, are not hydrolysed into
    solubles, so holding them as the attached-biomass reservoir is the intended
    mature-biofilm assumption. Derived from the stoichiometry, so it adapts to the
    model (for ASM1 it freezes ``XI``/``XB_H``/``XB_A``/``XP`` and leaves
    ``XS``/``XND`` dynamic).
    """
    stoich = model.compute_stoich(model.default_parameters())  # (n_rxn, n_sp)
    particulate = ~soluble_mask
    produces_soluble = jnp.any((stoich > 0) & soluble_mask[None, :], axis=1)
    consumed_into_soluble = jnp.any((stoich < 0) & produces_soluble[:, None], axis=0)
    return particulate & ~consumed_into_soluble


@dataclass
class IFASUnit(AerationUnit, CouplingAware):
    """An IFAS / MBBR tank: a CSTR bulk coupled to a depth-resolved biofilm.

    Parameters
    ----------
    name : str
        Unit identifier.
    model : CompiledModel
        Kinetic model, run in the bulk and in every biofilm layer.
    volume : float
        Bulk liquid volume (m^3).
    input_port_names : list[str]
        Incoming-stream ports (summed as a built-in mixer, like a CSTR).
    specific_surface_area : float
        Protected surface area of the carrier media (m^2 of biofilm area per m^3
        of media) -- the manufacturer's SSA.
    fill_fraction : float
        Media fill: the volume fraction of the tank occupied by carriers (0-1).
        The effective biofilm-area-to-bulk-volume ratio is
        ``A_V = specific_surface_area * fill_fraction`` (1/m).
    biofilm_thickness : float
        Total biofilm thickness ``L_f`` (m).
    n_layers : int
        Number of biofilm layers resolved between the bulk and the carrier wall
        (default 4).
    diffusivity : float or jnp.ndarray
        Effective soluble diffusivity inside the biofilm ``D_eff`` (m^2/day).
        Scalar or per-species; particulates never diffuse.
    boundary_layer : float
        External boundary-layer thickness ``L_bl`` (m); the bulk<->surface
        mass-transfer coefficient is ``D_bl / L_bl``.
    boundary_diffusivity : float or jnp.ndarray, optional
        Diffusivity in the (liquid) external boundary layer; ``None`` reuses
        ``diffusivity`` (see :class:`~aquakin.integrate.biofilm.BiofilmReactor`).
    conditions : dict[str, float]
        Spatially-uniform condition values (e.g. ``{"T": 293.15}``); one per
        condition the model declares. Used in the bulk and every layer.
    aeration : Aeration, optional
        Bulk aeration (open- or closed-loop), exactly as for a
        :class:`~aquakin.plant.cstr.CSTRUnit`. Oxygen enters the **bulk**; it
        reaches the biofilm by diffusion (so deep layers can be oxygen-limited --
        the reason for depth resolution). ``None`` (default) is unaerated.
    soluble_mask : jnp.ndarray, optional
        Boolean ``(n_species,)``: which species diffuse. Defaults to the ``S``/
        ``X`` name heuristic.
    biofilm_fixed_mask : jnp.ndarray, optional
        Boolean ``(n_species,)``: which species are held as a fixed mature-biofilm
        reservoir **in the layers** (their layer rate is zeroed). The bulk is
        always fully dynamic. Defaults to the biomass + inert structure (the
        particulates that are *not* hydrolysis substrates -- see
        :func:`_default_biofilm_fixed_mask`), so the attached biomass is sustained
        while substrate pools and solubles react and diffuse. Pass an all-``False``
        mask for a fully dynamic biofilm (then configure detachment / a density
        cap on a :class:`BiofilmReactor` for a self-limiting inventory).
    biofilm_reactions : list[str] or jnp.ndarray, optional
        Reactions that run in the biofilm layers only (the rest run in the bulk
        only). ``None`` (default) runs every reaction in every compartment -- the
        right choice for a single-phase model (ASM1/2/3), where the same
        kinetics act on whatever biomass is locally present.
    biofilm_initial : jnp.ndarray, optional
        Initial biofilm-layer concentrations ``(n_species,)``, tiled across the
        layers -- the attached-biomass inventory of the mature biofilm. Defaults
        to the model's default concentrations.
    output_port : str
        Name of the single output port (default ``"out"``).
    """

    name: str
    model: "CompiledModel"
    volume: float
    input_port_names: list[str]
    specific_surface_area: float
    fill_fraction: float
    biofilm_thickness: float
    n_layers: int = 4
    diffusivity: object = 1e-4
    boundary_layer: float = 1e-4
    boundary_diffusivity: object = None
    conditions: dict = field(default_factory=dict)
    aeration: "Aeration | None" = None
    soluble_mask: Optional[jnp.ndarray] = None
    biofilm_fixed_mask: Optional[jnp.ndarray] = None
    biofilm_reactions: object = None
    biofilm_initial: Optional[jnp.ndarray] = None
    output_port: str = "out"

    def __post_init__(self) -> None:
        missing = set(self.model.conditions_required) - set(self.conditions)
        if missing:
            raise ValueError(
                f"IFASUnit '{self.name}' is missing required condition values "
                f"for: {sorted(missing)}. Provided: {sorted(self.conditions)}"
            )
        if not (0.0 < self.fill_fraction <= 1.0):
            raise ValueError(
                f"IFASUnit '{self.name}' fill_fraction must be in (0, 1]; got {self.fill_fraction}."
            )
        if self.specific_surface_area <= 0 or self.volume <= 0:
            raise ValueError(
                f"IFASUnit '{self.name}' specific_surface_area and volume must be positive."
            )

        n = self.model.n_species
        soluble = (
            _default_soluble_mask(self.model)
            if self.soluble_mask is None
            else jnp.asarray(self.soluble_mask, dtype=bool)
        )

        # Carrier area per bulk volume (1/m): SSA of the media times the fill.
        area_per_volume = float(self.specific_surface_area) * float(self.fill_fraction)

        # Reuse the depth-resolved biofilm core for the diffusion + per-compartment
        # reaction. fixed_mask=all-False so its RHS evolves every compartment
        # (including the bulk); the mature-biofilm freeze is applied to the LAYERS
        # only, by this unit, leaving the suspended (bulk) fraction fully dynamic.
        # feed/dilution are off -- the plant supplies the bulk convection.
        nothing_fixed = jnp.zeros((n,), dtype=bool)
        self._biofilm = BiofilmReactor(
            self.model,
            SpatialConditions.uniform(**{k: float(v) for k, v in self.conditions.items()}),
            n_layers=self.n_layers,
            thickness=self.biofilm_thickness,
            area_per_volume=area_per_volume,
            diffusivity=self.diffusivity,
            boundary_layer=self.boundary_layer,
            boundary_diffusivity=self.boundary_diffusivity,
            soluble_mask=soluble,
            fixed_mask=nothing_fixed,
            biofilm_reactions=self.biofilm_reactions,
        )
        self._area_per_volume = area_per_volume

        # Which species are held fixed in the LAYERS (mature attached biomass):
        # the biomass + inert structure, but NOT hydrolysis substrates (freezing
        # those would make them spurious soluble sources). Derived from the
        # stoichiometry; override with an explicit mask.
        if self.biofilm_fixed_mask is None:
            layer_fixed = _default_biofilm_fixed_mask(self.model, soluble)
        else:
            layer_fixed = jnp.asarray(self.biofilm_fixed_mask, dtype=bool)
        if layer_fixed.shape != (n,):
            raise ValueError(f"biofilm_fixed_mask must have shape ({n},); got {layer_fixed.shape}")
        self._layer_fixed = layer_fixed

        # Bulk aeration vectors (the AerationUnit mixin, shared with CSTRUnit).
        self._setup_aeration()

        self._condition_arrays = {
            cname: jnp.asarray([float(self.conditions[cname])])
            for cname in self.model.conditions_required
        }
        self._n = n
        self._n_comp = self.n_layers + 1

    # --- Unit protocol surface ------------------------------------------------

    @property
    def state_size(self) -> int:
        return self._n_comp * self._n

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        """Flat ``(n_comp * n_species,)`` state: bulk row + the biofilm layers.

        The bulk seeds at the model defaults; the layers seed at
        ``biofilm_initial`` (the mature attached-biomass inventory), defaulting to
        the model defaults.
        """
        bulk = self.model.default_concentrations()
        layer = bulk if self.biofilm_initial is None else jnp.asarray(self.biofilm_initial)
        rows = jnp.concatenate(
            [bulk[None, :], jnp.tile(layer[None, :], (self.n_layers, 1))], axis=0
        )
        return rows.reshape(-1)

    def set_temperature(self, temperature_K: float) -> None:
        """Set the static operating temperature (Kelvin) for the bulk + biofilm."""
        if "T" not in self.model.conditions_required:
            return
        self.conditions = {**self.conditions, "T": float(temperature_K)}
        self._condition_arrays = {
            **self._condition_arrays,
            "T": jnp.asarray([float(temperature_K)]),
        }
        self._biofilm.conditions = SpatialConditions.uniform(
            **{k: float(v) for k, v in self.conditions.items()}
        )

    def coupling_pattern(self):
        """Structural Jacobian sparsity (issue #388).

        State is the flat ``(n_comp * n_species,)`` profile (bulk row + biofilm
        layers). Two structures feed the ``self`` block: the soluble diffusion
        between adjacent compartments and the bulk convection/aeration are *linear*
        (their Jacobian is state-independent, so AD over diverse states captures
        them exactly -- :func:`ad_union`), while the per-compartment reaction
        kinetics carry saturated Monod terms that are numerically invisible to a
        probe, so the model's syntactic AST pattern is unioned into each
        compartment's diagonal sub-block. In the biofilm layers the fixed
        attached-biomass species have their rate zeroed, so their rows are dropped
        from the layer kinetics blocks. ``inlet``: the bulk dilution diagonal --
        the inflow enters the bulk row only; the biofilm couples to it solely by
        diffusion (captured in ``self``).
        """
        import jax
        import numpy as np

        from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
        from aquakin.plant.coupling import CouplingPattern, ad_union

        net = self.model
        n, n_comp = self._n, self._n_comp
        m = self.state_size
        params = net.default_parameters()
        state0 = np.asarray(self.initial_state())
        base_C = jnp.asarray(np.maximum(np.abs(np.asarray(net.default_concentrations())), 1e-3))
        Q = jnp.asarray(self.volume)  # representative positive inflow
        inputs = {nm: Stream(Q=Q, C=base_C, model=net) for nm in self.input_port_names}

        # AD over diverse states: the (linear) diffusion + convection + aeration
        # couplings are captured exactly; the reaction kinetics are unioned from the
        # AST below (saturated Monod is invisible to AD).
        jac = lambda s: jax.jacfwd(lambda x: self.rhs(jnp.asarray(0.0), x, inputs, params))(s)
        self_pat = ad_union(jac, state0)

        kin = structural_sparsity_pattern(net)  # (n, n) reaction couplings
        layer_kin = kin.copy()
        layer_kin[np.asarray(self._layer_fixed), :] = False  # frozen layer rates
        for c in range(n_comp):
            block = kin if c == 0 else layer_kin  # bulk fully dynamic
            self_pat[c * n : (c + 1) * n, c * n : (c + 1) * n] |= block
        np.fill_diagonal(self_pat, True)

        inlet_pat = np.zeros((m, n), dtype=bool)
        inlet_pat[:n, :] = np.eye(n, dtype=bool)  # inflow dilutes the bulk row
        return CouplingPattern(self_pattern=self_pat, inlet_pattern=inlet_pat)

    def _bulk(self, state: jnp.ndarray) -> jnp.ndarray:
        return state.reshape(self._n_comp, self._n)[0]

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        Q_total = jnp.zeros(())
        for nm in self.input_port_names:
            Q_total = Q_total + inputs[nm].Q
        return {
            self.output_port: Stream(
                Q=Q_total,
                C=self._bulk(state),
                model=self.model,
                T=self._mixed_inlet_T(inputs),
            )
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Outflow equals total inflow (constant bulk volume)."""
        Q_total = jnp.zeros(())
        for nm in self.input_port_names:
            Q_total = Q_total + input_flows[nm]
        return {self.output_port: Q_total}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        y = state.reshape(self._n_comp, self._n)  # row 0 bulk, 1.. layers

        # Inlet temperature drives the (bulk + biofilm) kinetics through the 'T'
        # condition, exactly as for a CSTR.
        T_in = self._mixed_inlet_T(inputs)
        conditions = self._condition_arrays
        if T_in is not None and "T" in self._condition_arrays:
            conditions = {**self._condition_arrays, "T": jnp.reshape(T_in, (1,))}

        # Biofilm diffusion + per-compartment reaction (no internal feed): this is
        # the BiofilmReactor RHS, finite-volume face fluxes between bulk<->surface
        # <->...<->wall plus reaction in every cell. Reused, not reimplemented.
        bio_rhs = self._biofilm._make_rhs(conditions, params)
        dydt = bio_rhs(t, y, params)  # (n_comp, n_species)

        # Hold the mature attached biomass fixed in the LAYERS only (the suspended
        # bulk biomass stays dynamic): zero the fixed species' layer rates.
        layer_fix = jnp.zeros((self._n_comp, self._n), dtype=bool)
        layer_fix = layer_fix.at[1:].set(self._layer_fixed[None, :])
        dydt = jnp.where(layer_fix, 0.0, dydt)

        # Bulk convection (plant inflow) + bulk aeration, added to the bulk row.
        Q_total = jnp.zeros(())
        mass_total = jnp.zeros((self._n,))
        for nm in self.input_port_names:
            s = inputs[nm]
            Q_total = Q_total + s.Q
            mass_total = mass_total + s.Q * s.C
        C_in = mass_total / (Q_total + 1e-12)
        bulk = y[0]
        convection = (Q_total / self.volume) * (C_in - bulk)
        T_eff = T_in if T_in is not None else self.conditions.get("T")
        aeration = aeration_transfer(self._av, bulk, T_eff, signals, self.model)
        dydt = dydt.at[0].add(convection + aeration)

        return dydt.reshape(-1)


# An MBBR is an IFAS tank operated without significant suspended sludge (the bulk
# carries little biomass and the biofilm does the work); the same unit models
# both, so this is a discoverable alias rather than a separate class.
MBBRUnit = IFASUnit
