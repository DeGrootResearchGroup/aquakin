"""ADM1 anaerobic digester unit for plant-wide (BSM2) simulation.

A continuously-fed, constant-volume mesophilic digester running the ``adm1``
model. It is a CSTR on the *liquid* phase with a gas headspace:

    dC/dt = f_reaction(C) + (Q_in / V_liq) * (C_in - C) * liquid_mask

The reaction term ``model.dCdt`` already contains the ADM1 biochemistry, the
gas–liquid transfer and the overpressure-driven biogas outflow, and the
state-derived charge-balance pH. The dilution term applies only to the liquid
states; the three gas-headspace states (``S_gas_*``) are not diluted by the
feed (``liquid_mask`` is 0 there) — they exchange with the liquid and leave as
biogas purely through the reaction model. The liquid effluent leaves at the
feed flow (constant liquid volume).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.coupling import CouplingAware
from aquakin.plant.streams import Stream, mixed_feed, mixed_scalars, total_flow

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


# ADM1 gas-headspace states (BSM2 form): not part of the liquid throughput.
ADM1_GAS_SPECIES: tuple[str, ...] = ("S_gas_h2", "S_gas_ch4", "S_gas_co2")


@dataclass
class ADM1DigesterUnit(CouplingAware):
    """Continuously-fed ADM1 digester (BSM2 mesophilic, 35 °C).

    Parameters
    ----------
    name : str
    model : CompiledModel
        The ``adm1`` model (26 liquid + 3 gas states).
    volume : float
        Liquid volume ``V_liq`` (m³); the dilution rate is ``Q_in / V_liq``.
    input_port_names : list[str]
        Incoming-stream ports; multiple inflows are Q-weighted-mixed.
    conditions : dict[str, float]
        Spatially-uniform condition values (e.g. ``{"T": 308.15}``). Defaults to
        the model's declared condition defaults. The digester pH is *derived*
        from the state (charge balance) and must not be supplied here.
    gas_species : tuple[str, ...]
        Headspace states excluded from feed dilution.
    output_port : str
        Liquid-effluent output port.
    """

    name: str
    model: CompiledModel
    volume: float
    input_port_names: list[str] = field(default_factory=lambda: ["inlet"])
    conditions: dict[str, float] = field(default_factory=dict)
    gas_species: tuple[str, ...] = ADM1_GAS_SPECIES
    output_port: str = "effluent"

    # The digester is heated and held at a fixed operating temperature, so a
    # HeatBalanceTemperature model does NOT give it a dynamic temperature state
    # (matching the BSM2 protocol: "except the digester, fixed at 35 degC"). A
    # plain class attribute, not a dataclass field, so it never enters the state.
    temperature_fixed = True

    def __post_init__(self) -> None:
        # Fill in any unspecified required conditions from the model defaults.
        defaults = {
            name: float(arr[0]) for name, arr in self.model.default_conditions().fields.items()
        }
        conds = {**defaults, **self.conditions}
        missing = set(self.model.conditions_required) - set(conds)
        if missing:
            raise ValueError(
                f"ADM1DigesterUnit '{self.name}' is missing condition values "
                f"for: {sorted(missing)}."
            )
        self._condition_arrays = {
            name: jnp.asarray([float(conds[name])]) for name in self.model.conditions_required
        }
        # liquid_mask: 1.0 on liquid states, 0.0 on the gas-headspace states.
        mask = jnp.ones((self.model.n_species,))
        for sp in self.gas_species:
            if sp in self.model.species_index:
                mask = mask.at[self.model.species_index[sp]].set(0.0)
        self._liquid_mask = mask
        # The gas-transfer stoichiometry scales the headspace gain by the liquid
        # volume V_liq (a unit of liquid lost to transfer raises the headspace
        # concentration by V_liq/V_gas). The model ships V_liq at the BSM2
        # default; slave it to this unit's actual liquid volume so the gas
        # transfer is correct for any digester size. None if the model has no
        # V_liq parameter (then the model's own default ratio is used).
        self._v_liq_idx = self.model.param_index.get("V_liq")

    @property
    def state_size(self) -> int:
        return self.model.n_species

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        return self.model.default_concentrations()

    def coupling_pattern(self):
        """Structural Jacobian sparsity (issue #388).

        ``self`` is the ADM1 kinetics + gas-transfer + state-derived-pH coupling
        from the rate AST. ``inlet`` is the dilution, which couples each *liquid*
        species' derivative to its own inlet concentration -- the gas-headspace
        states are not fed (``liquid_mask`` is 0 there), so their inlet rows are
        empty.
        """
        import numpy as np

        from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
        from aquakin.plant.coupling import CouplingPattern

        return CouplingPattern(
            self_pattern=structural_sparsity_pattern(self.model),
            inlet_pattern=np.diag(np.asarray(self._liquid_mask) > 0.0),
        )

    def operating_pH(self, state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """The digester's instantaneous, state-derived pH.

        Read from the charge-balance speciation the model already solves each
        step. The ASM<->ADM interfaces use it because BSM2 evaluates
        their inorganic-carbon charge balance at the digester pH. Falls back to
        the static ``pH`` condition if the model declares no speciation.
        """
        fn = self.model.derived_condition_fn
        if fn is None:
            return jnp.reshape(self._condition_arrays["pH"], ())
        return fn(state, params, self._condition_arrays, 0)["pH"]

    def _state_volume_vector(self, params: jnp.ndarray):
        """Per-species holdup volume (m³): the liquid volume for every state,
        except the three gas-headspace states, which live in the headspace
        volume ``V_gas``.

        This is the digester's declaration of *where each state lives* — the
        single source of truth used by both the results-level mass-balance
        inventory and its reaction-production integral, so neither has to know
        the ADM1 gas-headspace layout.

        Parameters
        ----------
        params : jnp.ndarray
            This unit's parameter vector (to read ``V_gas``).

        Returns
        -------
        numpy.ndarray
            Per-species volume vector, shape ``(n_species,)``.
        """
        import numpy as np

        V = float(self.volume)
        vol = np.full(self.model.n_species, V)
        idx = self.model.param_index.get("V_gas")
        v_gas = float(params[idx]) if idx is not None else V
        for sp in self.gas_species:
            j = self.model.species_index.get(sp)
            if j is not None:
                vol[j] = v_gas
        return vol

    def component_inventory(self, state, content, params):
        """Canonical-component inventory held in the digester (``{component:
        grams}``).

        Each liquid state is held at the liquid volume and each gas-headspace
        state at ``V_gas`` (via :meth:`_state_volume_vector`), so the inventory
        is ``Σ_species (C·volume)·content`` per component. Implements the shared
        stateful-unit inventory contract consumed by
        :func:`aquakin.plant.balance.mass_balance`.

        Parameters
        ----------
        state : array
            The digester's flat state vector, shape ``(n_species,)``.
        content : dict of str to ndarray
            ``{component: (n_species,) canonical content}`` for this model.
        params : jnp.ndarray
            This unit's parameter vector (to read ``V_gas``).

        Returns
        -------
        dict of str to float
            ``{component: grams}`` held in the digester.
        """
        import numpy as np

        sv = np.asarray(state)
        vol_vec = self._state_volume_vector(params)
        return {comp: float(np.dot(sv * vol_vec, vec)) for comp, vec in content.items()}

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        # Constant liquid volume: effluent flow equals the total inflow. The
        # effluent carries the flow-weighted inlet side-channel scalars (the
        # temperature is a heat balance, matching every other multi-inlet unit via
        # the one shared combiner).
        Q_total = total_flow(inputs[name].Q for name in self.input_port_names)
        scalars_out = mixed_scalars(inputs, self.input_port_names)
        return {self.output_port: Stream(Q=Q_total, C=state, model=self.model, scalars=scalars_out)}

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        Q_total = total_flow(input_flows[name] for name in self.input_port_names)
        return {self.output_port: Q_total}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        # Mix inflows (Q-weighted) into one feed composition.
        Q_total, C_in = mixed_feed(inputs, self.input_port_names)

        # Slave the model's V_liq parameter to this unit's liquid volume so the
        # gas-transfer headspace-gain ratio V_liq/V_gas matches the actual
        # geometry (the model default is BSM2's 3400 m³).
        if self._v_liq_idx is not None:
            params = params.at[self._v_liq_idx].set(float(self.volume))
        reaction = self.model.dCdt(state, params, self._condition_arrays, 0)
        dilution = (Q_total / self.volume) * (C_in - state) * self._liquid_mask
        return reaction + dilution
