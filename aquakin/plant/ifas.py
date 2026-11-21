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

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquakin.core.conditions import SpatialConditions
from aquakin.integrate.biofilm import BiofilmReactor, _default_soluble_mask
from aquakin.integrate.biofilm_growth import (
    BALANCED_DETACHMENT,
    DETACHMENT_LAWS,
    advection_transport,
    advection_velocity,
    layer_thickness,
    surface_detachment,
    thickness_derivative,
)
from aquakin.plant._constants import check_positive
from aquakin.plant.coupling import CouplingAware
from aquakin.plant.cstr import (
    Aeration,
    AerationUnit,
    aeration_transfer,
)
from aquakin.plant.kinetic import AdjustableTemperatureUnit
from aquakin.plant.streams import Stream, mixed_feed, mixed_scalars, total_flow

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
class IFASUnit(AdjustableTemperatureUnit, AerationUnit, CouplingAware):
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
        Protected surface area of the carrier media (m²/m³: biofilm area per
        volume of media) -- the manufacturer's SSA.
    fill_fraction : float
        Media fill: the volume fraction of the tank occupied by carriers (0-1).
        The effective biofilm-area-to-bulk-volume ratio is
        ``A_V = specific_surface_area * fill_fraction`` (1/m).
    biofilm_thickness : float
        Total biofilm thickness ``L_f`` (m).
    n_layers : int
        Number of biofilm layers resolved between the bulk and the carrier wall
        (default 16).

        The layer discretization is **first order**, so halving the layer
        thickness halves the error in the substrate flux, and the error at a
        given layer count is set by how far the rate-limiting substrate
        penetrates rather than by ``L_f`` alone. The scale that matters is the
        penetration depth ``delta = sqrt(D_eff * K_s / (k * X_f))`` (``k`` the
        maximum specific utilization rate, ``X_f`` the biofilm biomass density),
        and the error tracks ``L_f / delta``:

        =============  =====  =====  ======  ======
        ``L_f/delta``  n=4    n=8    n=16    n=32
        =============  =====  =====  ======  ======
        1 (penetrated) 1.4%   0.7%   0.4%    0.2%
        2              6.9%   3.6%   1.8%    0.9%
        4              25.7%  12.4%  6.0%    2.9%
        =============  =====  =====  ======  ======

        So a thin, fully penetrated film is accurate with very few layers, while
        a thick partially penetrated one needs many. The default of 16 keeps a
        typical carrier biofilm within a couple of per cent; raise it for a
        deeply penetration-limited film, and lower it to make a well-penetrated
        one cheaper. :meth:`layer_resolution` reads the achieved resolution back
        off a solved state.
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
        to the model's default concentrations. With a growing biofilm this is the
        starting composition of the film, and its particulate total should be
        ``biofilm_density`` -- the film is modelled as incompressible at that
        density, so a seed far from it starts the solve off its own constraint.
    dynamic_thickness : bool
        ``False`` (default) is the mature biofilm: the attached biomass is held
        as a sustained reservoir and nothing moves through the film. ``True``
        makes the film **grow**: net production anywhere inside displaces
        material outward (the film is incompressible and the wall impermeable),
        that displacement carries the species with it, so they stratify by
        growth rate, and biomass leaves at the surface by detachment. ``L_f``
        becomes the final state entry. See
        :mod:`aquakin.integrate.biofilm_growth`.
    biofilm_density : float, optional
        Total particulate density of the film (e.g. ``1e4`` g_COD/m^3). Required
        when ``dynamic_thickness=True``: it is what converts production into a
        displacement velocity. Ignored otherwise.
    detachment_law : {"quadratic", "linear", "constant", "balance"}
        What sets the detachment velocity at the surface, and so what the
        thickness does:

        - ``"balance"`` -- **a specified total thickness.** Detachment is set to
          exactly the velocity arriving at the surface, so ``dL_f/dt`` is zero by
          construction and the film stays at ``biofilm_thickness``. The species
          still grow, advect and stratify, and biomass still leaves at the
          surface; only the total is pinned. Takes no ``detachment_rate``. This
          is what a specified average thickness "maintained by detachment" means.
        - ``"quadratic"`` (default), ``"linear"``, ``"constant"`` -- **a
          thickness that follows from the physics.** ``u_det = rate * L_f**p``
          with ``p = 2, 1, 0``, so the film settles where growth and detachment
          balance, and thins when decay outruns growth. Use
          ``detachment_rate=0`` to let it settle wherever net production
          integrates to zero.
    detachment_rate : float
        Detachment coefficient, in the units its law implies: m/time for
        ``"constant"``, 1/time for ``"linear"``, 1/(m time) for ``"quadratic"``.
        Must be left at zero for ``"balance"``, which has no rate of its own.
    output_port : str
        Name of the single output port (default ``"out"``).
    """

    name: str
    model: CompiledModel
    volume: float
    input_port_names: list[str]
    specific_surface_area: float
    fill_fraction: float
    biofilm_thickness: float
    n_layers: int = 16
    diffusivity: object = 1e-4
    boundary_layer: float = 1e-4
    boundary_diffusivity: object = None
    conditions: dict = field(default_factory=dict)
    aeration: Aeration | None = None
    soluble_mask: jnp.ndarray | None = None
    biofilm_fixed_mask: jnp.ndarray | None = None
    biofilm_reactions: object = None
    biofilm_initial: jnp.ndarray | None = None
    dynamic_thickness: bool = False
    biofilm_density: float | None = None
    detachment_rate: float = 0.0
    detachment_law: str = "quadratic"
    output_port: str = "out"

    def __post_init__(self) -> None:
        check_positive(self.name, "volume", self.volume)
        self._init_conditions(self.conditions)
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
        if self.dynamic_thickness:
            # A growing biofilm freezes nothing: its inventory is precisely what
            # the advection and detachment balance is solving for, so a freeze
            # would both pin the thing being computed and break the
            # incompressibility the displacement velocity rests on.
            if self.biofilm_fixed_mask is not None and bool(
                jnp.any(jnp.asarray(self.biofilm_fixed_mask, dtype=bool))
            ):
                raise ValueError(
                    f"IFASUnit '{self.name}' cannot combine dynamic_thickness=True with a "
                    f"non-empty biofilm_fixed_mask: a growing biofilm's attached biomass "
                    f"must evolve. Drop one or the other."
                )
            layer_fixed = jnp.zeros((n,), dtype=bool)
        elif self.biofilm_fixed_mask is None:
            layer_fixed = _default_biofilm_fixed_mask(self.model, soluble)
        else:
            layer_fixed = jnp.asarray(self.biofilm_fixed_mask, dtype=bool)
        if layer_fixed.shape != (n,):
            raise ValueError(f"biofilm_fixed_mask must have shape ({n},); got {layer_fixed.shape}")
        self._layer_fixed = layer_fixed

        # Species the biofilm matrix carries with it: the particulates. Solubles
        # move by diffusion instead, so advecting them would double-count.
        self._advect_mask = ~soluble
        if self.dynamic_thickness:
            if self.biofilm_density is None or self.biofilm_density <= 0:
                raise ValueError(
                    f"IFASUnit '{self.name}' needs a positive biofilm_density (the total "
                    f"particulate density of the film, e.g. 1e4 g_COD/m3) when "
                    f"dynamic_thickness=True: it sets the displacement velocity."
                )
            if self.detachment_rate < 0:
                raise ValueError(f"IFASUnit '{self.name}' detachment_rate must not be negative.")
            if self.detachment_law not in DETACHMENT_LAWS:
                raise ValueError(
                    f"IFASUnit '{self.name}' detachment_law must be one of "
                    f"{sorted(DETACHMENT_LAWS)}; got {self.detachment_law!r}."
                )
            if self.detachment_law == BALANCED_DETACHMENT and self.detachment_rate:
                raise ValueError(
                    f"IFASUnit '{self.name}' takes no detachment_rate with "
                    f"detachment_law='{BALANCED_DETACHMENT}': the detachment velocity is "
                    f"whatever holds the thickness, not something set independently. "
                    f"Got detachment_rate={self.detachment_rate!r}."
                )
            seeded = self.biofilm_initial
            if seeded is not None:
                total = float(jnp.sum(jnp.asarray(seeded) * self._advect_mask))
                if total > 0 and not 0.5 <= total / float(self.biofilm_density) <= 2.0:
                    warnings.warn(
                        f"IFASUnit '{self.name}': the seeded attached biomass sums to "
                        f"{total:.4g} but biofilm_density is {self.biofilm_density:.4g}. "
                        f"The film is modelled as incompressible at that density, so a "
                        f"seed far from it starts the solve off its own constraint.",
                        RuntimeWarning,
                        stacklevel=3,
                    )

        # Bulk aeration vectors (the AerationUnit mixin, shared with CSTRUnit).
        self._setup_aeration()

        self._n = n
        self._n_comp = self.n_layers + 1

    # --- Unit protocol surface ------------------------------------------------

    @property
    def state_size(self) -> int:
        return self._n_comp * self._n + int(self.dynamic_thickness)

    def state_names(self) -> tuple[str, ...]:
        """Names of the state entries, in order: the bulk concentrations (by
        species), then each biofilm layer's, ``film1.<species>`` at the surface
        to ``film<n_layers>.<species>`` at the carrier, and -- when the biofilm
        grows -- the thickness ``L_f`` as the final entry."""
        species = tuple(self.model.species)
        films = (f"film{k}.{sp}" for k in range(1, self.n_layers + 1) for sp in species)
        tail = ("L_f",) if self.dynamic_thickness else ()
        return species + tuple(films) + tail

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
        flat = rows.reshape(-1)
        if self.dynamic_thickness:
            flat = jnp.concatenate([flat, jnp.asarray([float(self.biofilm_thickness)])])
        return flat

    def _split(self, state: jnp.ndarray):
        """Split a flat state into the ``(n_comp, n_species)`` rows and ``L_f``.

        ``L_f`` is the fixed ``biofilm_thickness`` unless the biofilm grows, in
        which case it is the final state entry.
        """
        state = jnp.asarray(state)
        if not self.dynamic_thickness:
            return state.reshape(self._n_comp, self._n), self.biofilm_thickness
        return state[:-1].reshape(self._n_comp, self._n), state[-1]

    def layer_resolution(self, state: jnp.ndarray) -> dict[str, float]:
        """How well ``n_layers`` resolves each soluble's gradient in a solved state.

        The layer discretization is first order, so its error is governed by how
        much of a soluble's profile falls across a single layer. This reports,
        per soluble species, the largest fractional drop over one layer step,
        measured from the bulk inward:

            ``max_j (C_j - C_{j+1}) / C_surface``

        A well-resolved profile changes gradually (values of a few per cent); a
        value approaching 1 means the substrate is consumed almost entirely
        inside one layer, so the flux is being read off a gradient the grid
        cannot represent and ``n_layers`` should be raised. See the ``n_layers``
        note in the class docstring for the error this implies.

        Parameters
        ----------
        state : jnp.ndarray
            This unit's state, as ``(n_comp * n_species,)`` or
            ``(n_comp, n_species)`` -- e.g.
            ``plant.states_by_unit(result.state)[unit.name]``.

        Returns
        -------
        dict
            ``{species: largest fractional drop across one layer}``, for the
            soluble species only, in model species order. A species that is flat
            or absent reports ``0.0``.

        Examples
        --------
        >>> worst = max(unit.layer_resolution(states[unit.name]).values())
        >>> worst < 0.1
        True
        """
        rows = self._split(jnp.asarray(state).reshape(-1))[0]
        soluble = np.asarray(self._biofilm.soluble_mask)
        profile = np.asarray(rows)
        out: dict[str, float] = {}
        for index, name in enumerate(self.model.species):
            if not soluble[index]:
                continue
            column = profile[:, index]
            surface = column[0]
            if not np.isfinite(surface) or abs(surface) <= 0.0:
                out[name] = 0.0
                continue
            out[name] = float(np.max(np.abs(np.diff(column))) / abs(surface))
        return out

    def component_inventory(self, state, content, params) -> dict[str, float]:
        """Component mass held in the tank: the bulk liquid plus the biofilm.

        The unit's state is a profile rather than a single concentration vector,
        so without this the plant's mass balance skips the tank entirely -- not
        only its attached biomass but its bulk liquid too, which shows up as an
        imbalance the size of the whole unit. The biofilm holds
        ``A_V * L_f * mean(C)`` per unit bulk volume, and when the film grows
        that inventory moves with ``L_f``, so it has to be read from the state
        rather than from the configured thickness.

        Parameters
        ----------
        state : jnp.ndarray
            This unit's flat state.
        content : dict
            ``{component: (n_species,) array}`` of per-species content.
        params : jnp.ndarray
            This unit's parameter slice (unused -- the geometry is fixed).

        Returns
        -------
        dict
            ``{component: grams}``.
        """
        rows, thickness = self._split(jnp.asarray(state).reshape(-1))
        rows = np.asarray(rows)
        bulk, layers = rows[0], rows[1:]
        # Biofilm volume per unit bulk volume: area ratio times thickness, split
        # evenly across the layers.
        film_volume = self._area_per_volume * float(thickness) / self.n_layers
        out: dict[str, float] = {}
        for component, vec in content.items():
            vec = np.asarray(vec)
            out[component] = self.volume * (
                float(np.dot(bulk, vec)) + film_volume * float(np.sum(layers @ vec))
            )
        return out

    def set_temperature(self, temperature_K: float) -> None:
        """Set the static operating temperature (Kelvin) for the bulk + biofilm.

        A no-op for a model without a ``T`` condition: the conditions are then
        unchanged, and the biofilm's are rebuilt from them as at construction.
        """
        super().set_temperature(temperature_K)
        self._biofilm.conditions = SpatialConditions.uniform(
            **{k: float(v) for k, v in self.conditions.items()}
        )

    def coupling_pattern(self):
        """Structural Jacobian sparsity.

        State is the flat ``(n_comp * n_species,)`` profile (bulk row + biofilm
        layers). Two structures feed the ``self`` block: the soluble diffusion
        between adjacent compartments and the bulk convection/aeration are *linear*
        (their Jacobian is state-independent, so AD over diverse states captures
        them exactly -- :func:`ad_union`), while the per-compartment reaction
        kinetics carry saturated Monod terms that are numerically invisible to a
        probe, so the model's syntactic AST pattern is unioned into each
        compartment's diagonal sub-block. In the biofilm layers the fixed
        attached-biomass species have their rate zeroed, so their rows are dropped
        from the layer kinetics blocks. A growing biofilm's thickness entry is
        declared coupled to everything, since it scales every flux and its own
        rate integrates the whole film. ``inlet``: the bulk dilution diagonal --
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

        if self.dynamic_thickness:
            # The thickness couples to everything and everything to it: it sets
            # the layer spacing every flux is divided by, while its own rate is
            # the production integrated over the whole film. Those couplings run
            # through saturated Monod terms a probe cannot see, so they are
            # declared dense rather than discovered -- one row and one column, so
            # the colouring barely notices, and a miss here would corrupt
            # gradients.
            self_pat[-1, :] = True
            self_pat[:, -1] = True

        inlet_pat = np.zeros((m, n), dtype=bool)
        inlet_pat[:n, :] = np.eye(n, dtype=bool)  # inflow dilutes the bulk row
        output_pat = np.zeros((n, m), dtype=bool)
        output_pat[:, :n] = np.eye(n, dtype=bool)  # the effluent is the bulk row
        return CouplingPattern(
            self_pattern=self_pat, inlet_pattern=inlet_pat, output_pattern=output_pat
        )

    def _bulk(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._split(state)[0][0]

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        Q_total = total_flow(inputs[nm].Q for nm in self.input_port_names)
        return {
            self.output_port: Stream(
                Q=Q_total,
                C=self._bulk(state),
                model=self.model,
                scalars=mixed_scalars(inputs, self.input_port_names),
            )
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Outflow equals total inflow (constant bulk volume)."""
        Q_total = total_flow(input_flows[nm] for nm in self.input_port_names)
        return {self.output_port: Q_total}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        y, thickness = self._split(state)  # row 0 bulk, 1.. layers; plus L_f
        dz = layer_thickness(thickness, self.n_layers)

        # The operating temperature (under a heat-balance model the tank's own
        # temperature state) drives the bulk + biofilm kinetics, as for a CSTR.
        # The reaction itself runs per compartment inside the biofilm RHS, which
        # calls the same canonical model.dCdt.
        conditions, T_op = self._operating_conditions(inputs, signals)

        # Biofilm diffusion + per-compartment reaction (no internal feed): this is
        # the BiofilmReactor RHS, finite-volume face fluxes between bulk<->surface
        # <->...<->wall plus reaction in every cell. Reused, not reimplemented.
        # The layer thickness is passed explicitly so a growing biofilm's
        # diffusion operator follows its moving boundary.
        bio_rhs = self._biofilm._make_rhs(conditions, params, dz=dz)
        # Reaction and diffusion are both per the model's time unit.
        dydt = self._scale_rates(bio_rhs(t, y, params))  # (n_comp, n_species)

        # Hold the mature attached biomass fixed in the LAYERS only (the suspended
        # bulk biomass stays dynamic): zero the fixed species' layer rates. A
        # growing biofilm freezes nothing -- its inventory is the thing evolving.
        layer_fix = jnp.zeros((self._n_comp, self._n), dtype=bool)
        layer_fix = layer_fix.at[1:].set(self._layer_fixed[None, :])
        dydt = jnp.where(layer_fix, 0.0, dydt)

        # Bulk convection (plant inflow) + bulk aeration, added to the bulk row.
        Q_total, C_in = mixed_feed(inputs, self.input_port_names)
        bulk = y[0]
        convection = (Q_total / self.volume) * (C_in - bulk)
        aeration = aeration_transfer(self._av, bulk, T_op, signals, self.model)
        dydt = dydt.at[0].add(convection + aeration)

        if not self.dynamic_thickness:
            return dydt.reshape(-1)

        # --- the biofilm grows ------------------------------------------------
        # Net particulate production displaces material outward (the film is
        # incompressible and the wall impermeable). Particulates neither diffuse
        # nor exchange with the bulk here, so their layer rates are exactly the
        # reaction term the velocity integral needs.
        production = jnp.sum(dydt[1:] * self._advect_mask[None, :], axis=1)
        faces = advection_velocity(production, self.biofilm_density, dz)
        u_det = surface_detachment(faces, self.detachment_rate, thickness, self.detachment_law)
        dLdt = thickness_derivative(faces, u_det)
        d_layers, surface_flux = advection_transport(
            y[1:], bulk, faces, dLdt, dz, self._advect_mask
        )
        dydt = dydt.at[1:].add(d_layers)
        # What detaches enters the bulk, over the bulk volume it is shed into.
        dydt = dydt.at[0].add(surface_flux * self._area_per_volume)
        return jnp.concatenate([dydt.reshape(-1), jnp.reshape(dLdt, (1,))])


# An MBBR is an IFAS tank operated without significant suspended sludge (the bulk
# carries little biomass and the biofilm does the work); the same unit models
# both, so this is a discoverable alias rather than a separate class.
MBBRUnit = IFASUnit
