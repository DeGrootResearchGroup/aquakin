"""Results-level mass-balance closure check on a solved plant.

The stoichiometry-level checks in :mod:`aquakin.utils.balance` answer "is each
reaction balanced?". This answers the question an engineer actually asks of a
*result*: **does what went in equal what came out, plus what accumulated, plus
what left as gas, over my simulation window?** A closing balance is the first
evidence a plant result is trustworthy.

For each component (COD / N / P) the balance accounts, over ``[t0, t1]``:

- **inflow** -- the component carried in by every influent stream.
- **outflow** -- the component carried out by every terminal (boundary) material
  stream (final effluent, wasted sludge, disposal cake).
- **gas** -- the component leaving the bulk liquid as gas or via an electron
  acceptor that is not a tracked state: oxygen transferred in by aeration
  (removing COD), the digester biogas (CH₄ COD), and denitrification
  (nitrate-N reduced to N₂ gas, which also oxidises COD at the model's COD/N
  ratio). Computed from the aeration mass-transfer term, the digester
  headspace, and a reaction-production integral over the activated-sludge
  reactors -- *independently* of the in/out/accumulation bookkeeping, so the
  residual below is a genuine check.
- **accumulation** -- the change in the component's inventory held in every unit
  (reactor / clarifier / digester liquid + headspace / storage tank / settler
  sludge blanket) between ``t0`` and ``t1``.

The **imbalance** ``inflow − outflow − gas − accumulation`` is zero for a closed
balance. Everything is reported on one canonical gram basis (g COD / g N / g P),
so inventories and fluxes sum across models of different units (the ASM water
line in g/m³, the ADM digester in kg/m³ and kmol/m³) via
:func:`aquakin.canonical_content`.

The gas integrals are evaluated from the saved trajectory, so they are exact at
steady state (constant rates) and otherwise accurate to the ``t_eval`` sampling;
the reaction integral evaluates each unit's kinetics at the conditions its
``rhs`` used in the solve (its operating temperature).

One exception does not shrink with the sampling: a unit whose inventory only
approximates a component (:class:`~aquakin.plant.units.ApproximateInventoryUnit`;
the Takács settler with ``composition_mode="lumped_tss"`` values its blanket's N
and P at the model-default solids composition) leaves a closure error in that
component whenever its inventory changes -- about 2e-3 relative in N on BSM1
over days 1-3 from the warm start -- and none at steady state. The affected
components carry the unit's explanation in :attr:`ComponentBalance.notes`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aquakin.plant.units import (
    ApproximateInventoryUnit,
    ComponentInventoryUnit,
    GasExportUnit,
    LiquidVolumeUnit,
)
from aquakin.utils.composition import canonical_content

# CH4 / H2 oxygen demand (g COD per g gas): CH4 + 2 O2 -> CO2 + 2 H2O = 64/16.
_COD_PER_CH4 = 4.0


@dataclass
class ComponentBalance:
    """The closure of one conserved component over the simulation window.

    All terms are in canonical grams of the component (g COD, g N, g P) summed
    over the window ``[t0, t1]``. A closed balance has
    ``imbalance = inflow − outflow − gas − accumulation ≈ 0``.

    Attributes
    ----------
    component : str
        ``"COD"``, ``"N"`` or ``"P"``.
    inflow, outflow : float
        Component carried in by the influents / out by the terminal material
        streams (g).
    gas : float
        Component that left as gas / via an untracked electron acceptor (g).
    accumulation : float
        Change in the component's plant inventory, ``inventory(t1) − inventory(t0)`` (g).
    imbalance : float
        ``inflow − outflow − gas − accumulation`` (g); zero when closed.
    notes : list of str
        Why this closure is approximate beyond the ``t_eval`` sampling, one note
        per unit whose inventory of this component is an estimate (a
        ``lumped_tss`` Takács settler for N and P). Empty when every inventory
        is exact.
    """

    component: str
    inflow: float
    outflow: float
    gas: float
    accumulation: float
    imbalance: float
    notes: list[str] = field(default_factory=list)

    @property
    def relative_imbalance(self) -> float:
        """``imbalance`` as a fraction of the throughput (max of in / out / gas).

        A scale-free closure error: ~1e-3 or below is a well-closed balance.
        """
        scale = max(abs(self.inflow), abs(self.outflow), abs(self.gas), 1e-30)
        return self.imbalance / scale


@dataclass
class MassBalance:
    """Per-component closure of a plant over a simulation window.

    Returned by :meth:`aquakin.plant.Plant.mass_balance`. Index it by component
    name (``mb["COD"]`` -> :class:`ComponentBalance`) and call :meth:`closed` for
    a pass/fail, or :meth:`summary` for a printable table.
    """

    components: dict[str, ComponentBalance]
    window: tuple[float, float]  # (t0, t1) in the solution's time unit
    influent_ports: list[str] = field(default_factory=list)
    effluent_ports: list[str] = field(default_factory=list)
    gas_detail: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, component: str) -> ComponentBalance:
        return self.components[component]

    def closed(self, rtol: float = 1.0e-2) -> bool:
        """True when every component's ``relative_imbalance`` is within ``rtol``."""
        return all(abs(c.relative_imbalance) <= rtol for c in self.components.values())

    def summary(self) -> str:
        """A printable per-component table (canonical g over the window)."""
        t0, t1 = self.window
        lines = [
            f"Mass balance over t = [{t0:g}, {t1:g}] (canonical g of component over the window):",
            f"  {'comp':4s} {'in':>13s} {'out':>13s} {'gas':>13s} "
            f"{'accum':>13s} {'imbalance':>13s} {'rel':>10s}",
        ]
        for name, c in self.components.items():
            lines.append(
                f"  {name:4s} {c.inflow:13.4g} {c.outflow:13.4g} {c.gas:13.4g} "
                f"{c.accumulation:13.4g} {c.imbalance:13.4g} "
                f"{c.relative_imbalance:10.2e}"
            )
        lines.append(f"  closed (rtol=1e-2): {self.closed()}")
        for name, c in self.components.items():
            lines.extend(f"  note ({name}): {note}" for note in c.notes)
        return "\n".join(lines)


# --- per-unit inventory ------------------------------------------------------


def _unit_inventory(plant, unit_name, state_vec, content_by_model, params):
    """Component inventory held in one unit (a ``{component: grams}`` dict).

    Dispatches on the unit's own inventory contract, never on its private state
    layout:

    - a unit that declares ``component_inventory(state, content, params)`` (the
      layered Takács settler summing its blanket over layers, the ADM1 digester
      weighting its three gas-headspace states by ``V_gas``) returns its own
      inventory;
    - a unit holding a single well-mixed liquid volume declares that volume via
      ``liquid_volume(state)`` (StorageTank / MBR / SBR / the primary clarifier,
      whose states are ``[C..., one-or-more scalars]``), so the inventory is
      ``V·Σ C·content`` over the concentration head block;
    - a plain concentration-vector unit at its fixed ``volume`` (CSTR) holds
      ``volume·Σ C·content``;
    - anything else -- a stateless unit, or a non-concentration state such as an
      attached-growth biofilm -- holds nothing.

    A unit with a novel state representation participates by implementing the
    ``component_inventory`` contract, not by editing this helper.
    """
    unit = plant.units[unit_name]
    net = getattr(unit, "model", None)
    if net is None or state_vec.size == 0:
        return {}
    content = content_by_model[net.name]  # {component: (n_species,) array}

    if isinstance(unit, ComponentInventoryUnit):
        return unit.component_inventory(
            state_vec, content, plant._params_for_unit(unit_name, params)
        )

    sv = np.asarray(state_vec)
    if isinstance(unit, LiquidVolumeUnit):
        V = float(unit.liquid_volume(state_vec))
        C = sv[: net.n_species]
        return {comp: V * float(np.dot(C, vec)) for comp, vec in content.items()}

    if sv.size != net.n_species:
        return {}  # non-concentration state; skip
    volume = float(getattr(unit, "volume", 0.0))
    if volume <= 0.0:
        return {}
    return {comp: volume * float(np.dot(sv, vec)) for comp, vec in content.items()}


def _inventory_notes(plant, comps):
    """``{component: [note, ...]}``: why each component's closure is approximate,
    from every unit declaring :class:`ApproximateInventoryUnit` (in plant order)."""
    notes = {q: [] for q in comps}
    for unit in plant.units.values():
        if isinstance(unit, ApproximateInventoryUnit):
            for q, note in unit.inventory_caveats().items():
                if q in notes:
                    notes[q].append(note)
    return notes


def _flux(Q, C, content_vec):
    """Component flux of a stream over time: ``Q·(C·content)``, shape ``(n_t,)``
    (canonical g/d)."""
    return np.asarray(Q) * (np.asarray(C) @ content_vec)


def _content_by_model(plant, comps, params):
    """Canonical per-species content vectors for every model in the plant.

    Returns ``{model_name: {component: (n_species,) array}}``, keeping only the
    components the model actually carries (a model with no P contributes nothing
    to the P balance).

    Lab-COD convention for reporting: nitrate / N₂ carry no COD, so a reported
    COD is the organic oxygen demand (an analyst's COD), not a total electron
    demand. The closure is self-consistent under either convention. Each model's
    composition fractions are read from the *run* parameters (a unit of that
    model), so a calibrated / BSM-specific i_XB flows through.
    """
    models = {}  # name -> CompiledModel
    for u in plant.units.values():
        net = getattr(u, "model", None)
        if net is not None:
            models[net.name] = net
    for s in plant.influents.values():
        models[s.model.name] = s.model
    net_params = {}
    for uname, u in plant.units.items():
        net = getattr(u, "model", None)
        if net is not None and net.name not in net_params:
            net_params[net.name] = plant._params_for_unit(uname, params)

    def _content(net, q):
        try:
            return canonical_content(
                net, q, electron_acceptor_cod=False, params=net_params.get(net.name)
            )
        except KeyError:
            # No shipped table: use the model's own declared species
            # composition (a balance closes under any linear content table, as
            # long as inventory, flows, reactions and gas share it).
            declared = net.composition()
            if not declared:
                raise
            return canonical_content(net, q, composition=declared)

    return {name: {q: _content(net, q) for q in comps} for name, net in models.items()}


def _inflow_terms(plant, t, comps, in_names, content_by_model):
    """Component carried in by every influent series over the window (g)."""
    inflow = dict.fromkeys(comps, 0.0)
    for name in in_names:
        series = plant.influents[name]
        net = series.model
        cvec = content_by_model[net.name]
        Q = np.asarray([float(series.at(tt).Q) for tt in t])
        C = np.asarray([np.asarray(series.at(tt).C) for tt in t])
        for q in comps:
            inflow[q] += float(np.trapezoid(_flux(Q, C, cvec[q]), t))
    return inflow


def _dosing_inflow(plant, solution, t, comps, content_by_model, params):
    """Reagent mass injected by dosing units over the window (g).

    A DosingUnit adds its reagent's mass to the through-stream from outside the
    plant boundary, so it is a component source -- counted as inflow (the
    through-stream's own mass already enters via its upstream influent).
    """
    inflow = dict.fromkeys(comps, 0.0)
    for uname, u in plant.units.items():
        comp_vec = getattr(getattr(u, "reagent", None), "composition", None)
        if comp_vec is None:
            continue
        cvec = content_by_model[u.model.name]
        comp_vec = np.asarray(comp_vec)
        if u.flow is not None:
            Q_dose = np.full(len(t), float(u.flow))
        else:
            sig = u.required_signals[0]
            Q_dose = np.asarray(
                [
                    float(plant.signals_at(tt, solution.state[i], params)[sig] * u.gain)
                    for i, tt in enumerate(t)
                ]
            )
        C_dose = np.broadcast_to(comp_vec, (len(t), comp_vec.shape[0]))
        for q in comps:
            inflow[q] += float(np.trapezoid(_flux(Q_dose, C_dose, cvec[q]), t))
    return inflow


def _outflow_terms(plant, solution, t, comps, effluent_ports, content_by_model, params):
    """Component carried out by every terminal (boundary) material stream (g)."""
    outflow = dict.fromkeys(comps, 0.0)
    if effluent_ports:
        from aquakin.plant.bsm.evaluation import _reconstruct

        recon = _reconstruct(plant, solution, params, effluent_ports)
        for ep in effluent_ports:
            Q, C = recon[ep]
            unit = plant._parse_endpoint(ep, role="source")[0]
            cvec = content_by_model[plant.units[unit].model.name]
            for q in comps:
                outflow[q] += float(np.trapezoid(_flux(np.asarray(Q), np.asarray(C), cvec[q]), t))
    return outflow


def _gas_export_terms(plant, solution, comps, content_by_model, params):
    """Each component vented through the units' own gas phases over the window
    (g): the integral of every :class:`~aquakin.plant.units.GasExportUnit`'s
    ``gas_export`` dotted with its model's component content."""
    exported = dict.fromkeys(comps, 0.0)
    t = np.asarray(solution.t_native)
    for unit_name, (start, size) in plant._state_layout.items():
        unit = plant.units[unit_name]
        if not isinstance(unit, GasExportUnit):
            continue
        unit_params = plant._params_for_unit(unit_name, params)
        rows = np.asarray(
            [unit.gas_export(state[start : start + size], unit_params) for state in solution.state]
        )
        content = content_by_model[unit.model.name]
        for q in comps:
            exported[q] += float(np.trapezoid(rows @ content[q], t))
    return exported


def _accumulation_terms(plant, solution, comps, content_by_model, params):
    """Change in each component's plant inventory, ``inventory(t1) − inventory(t0)`` (g)."""
    accumulation = dict.fromkeys(comps, 0.0)
    layout = plant._state_layout
    for unit_name, (start, size) in layout.items():
        inv0 = _unit_inventory(
            plant, unit_name, solution.state[0][start : start + size], content_by_model, params
        )
        inv1 = _unit_inventory(
            plant, unit_name, solution.state[-1][start : start + size], content_by_model, params
        )
        for q in comps:
            accumulation[q] += inv1.get(q, 0.0) - inv0.get(q, 0.0)
    return accumulation


def _gas_terms(plant, solution, comps, content_by_model, params):
    """Component leaving as gas / via an untracked electron acceptor (g).

    By the integrated plant RHS identity, summed over all units the internal
    streams cancel, leaving:  ΔInventory = (boundary in − out) + R + aeration,
    where R is the reaction-production integral over the reactive units and
    aeration is the non-reaction O2 source. So the component leaving as gas is
    gas = −(R + aeration): for COD, O2 transferred in (aeration removes COD)
    minus R_COD (the reactions' net COD production -- negative, since
    denitrification oxidises COD and the digester gas-outflow exports biogas);
    for N, −R_N (denitrification N₂; nitrification and the digester conserve N).

    Returns ``(gas, gas_detail)``.
    """
    gas = dict.fromkeys(comps, 0.0)
    gas_detail = {}

    o2_transfer, R, stripped = _reaction_and_aeration_gas(
        plant, solution, params, content_by_model, comps
    )
    if "COD" in comps:
        gas["COD"] += o2_transfer - R.get("COD", 0.0)
        gas_detail["aeration_O2"] = o2_transfer
        gas_detail["reaction_COD"] = -R.get("COD", 0.0)
    if "N" in comps:
        gas["N"] += -R.get("N", 0.0)
        gas_detail["denitrification_N2"] = -R.get("N", 0.0)
    # Gases co-stripped by aeration (Aeration(stripped=...)): the transfer is
    # into the liquid, so the gas leaving is its negative.
    for q, val in stripped.items():
        if val != 0.0:
            gas[q] += -val
            gas_detail[f"stripped_{q}"] = -val
    # Material vented through a unit's own gas phase (a digester headspace that
    # belongs to the unit rather than the model), outside any reaction term.
    for q, val in _gas_export_terms(plant, solution, comps, content_by_model, params).items():
        if val != 0.0:
            gas[q] += val
            gas_detail[f"gas_export_{q}"] = val
    biogas = _biogas_cod(plant, solution, params)  # informational only
    if biogas is not None:
        gas_detail["biogas_COD"] = biogas
    return gas, gas_detail


def mass_balance(
    plant,
    solution,
    *,
    components=("COD", "N", "P"),
    influent_ports: list | None = None,
    effluent_ports: list | None = None,
    params=None,
) -> MassBalance:
    """Results-level mass-balance closure for a solved plant. See
    :meth:`aquakin.plant.Plant.mass_balance`."""
    import jax.numpy as jnp

    params = plant.default_parameters() if params is None else jnp.asarray(params)
    plant._build_state_layout()
    plant._build_parameter_layout()
    # The window is reported in the solution's own unit (``solution.t``), but the
    # flux integrals run over native time: every flow is per native time unit
    # (e.g. m3/d for an ASM plant) and the influent / control signals are
    # sampled at native times, so the integrated grams are independent of any
    # ``solve(time_unit=...)`` override.
    window = (float(solution.t[0]), float(solution.t[-1]))
    t = np.asarray(solution.t_native)
    comps = list(components)
    content_by_model = _content_by_model(plant, comps, params)

    if effluent_ports is None:
        effluent_ports = list(plant.check().dangling_outputs)
    in_names = list(plant.influents) if influent_ports is None else list(influent_ports)

    inflow = _inflow_terms(plant, t, comps, in_names, content_by_model)
    dosing = _dosing_inflow(plant, solution, t, comps, content_by_model, params)
    for q in comps:
        inflow[q] += dosing[q]
    outflow = _outflow_terms(plant, solution, t, comps, effluent_ports, content_by_model, params)
    accumulation = _accumulation_terms(plant, solution, comps, content_by_model, params)
    gas, gas_detail = _gas_terms(plant, solution, comps, content_by_model, params)

    notes = _inventory_notes(plant, comps)
    out = {}
    for q in comps:
        imb = inflow[q] - outflow[q] - gas[q] - accumulation[q]
        out[q] = ComponentBalance(
            component=q,
            inflow=inflow[q],
            outflow=outflow[q],
            gas=gas[q],
            accumulation=accumulation[q],
            imbalance=imb,
            notes=notes[q],
        )
    return MassBalance(
        components=out,
        window=window,
        influent_ports=in_names,
        effluent_ports=list(effluent_ports),
        gas_detail=gas_detail,
    )


def _reaction_terms_at(plant, t, state_full, params_full, reactive, maps=(None, None)):
    """The reaction (chemistry) term ``dC/dt`` of each reactive unit at one
    saved instant, ``{unit: (n_species,)}``.

    Each is the same :meth:`~aquakin.plant.kinetic.KineticUnit._reaction` the
    unit's ``rhs`` calls (the canonical ``model.dCdt``, positivity limiter
    included, per the plant's time unit), at the conditions that ``rhs`` used:
    :meth:`~aquakin.plant.kinetic.KineticUnit._reaction_conditions` given the
    inputs and signals the plant's RHS assembly hands the unit at ``(t,
    state_full)``. So a reactor's kinetics run at its flow-weighted inlet
    temperature, or its heat-balance temperature state, as in the solve. For an
    ADM1 digester the term also runs the gas-liquid transfer and overpressure
    gas outflow, so the biogas export is included.

    Parameters
    ----------
    plant : Plant
        The plant, with its state and parameter layouts built.
    t : float or jnp.ndarray
        Time (the plant's native unit).
    state_full : jnp.ndarray
        Flat plant state at ``t``, shape ``(total_state_size,)``.
    params_full : jnp.ndarray
        Full plant parameter vector.
    reactive : list[str]
        The units whose reaction term to return.
    maps : tuple, optional
        The cached ``(recycle_map, flow_map)`` (``RecycleResolver.cached_maps``),
        or ``(None, None)`` to probe them.

    Returns
    -------
    dict[str, jnp.ndarray]
        ``{unit: dC/dt}`` over the unit's species, per the plant's time unit.
    """
    return _unit_terms_at(plant, t, state_full, params_full, reactive, _reaction_term, maps)


def _reaction_term(unit, t, state, inputs, signals, params):
    """A unit's reaction term (:class:`~aquakin.plant.units.ReactionBalanceUnit`)."""
    return unit.reaction_term(state, inputs, params, signals)


def _oxygen_transfer_term(unit, t, state, inputs, signals, params):
    """An aerated unit's ``(oxygen, stripped)`` transfer
    (:class:`~aquakin.plant.units.OxygenTransferUnit`), as its ``rhs`` adds it."""
    return unit.oxygen_transfer(t, state, inputs, params, signals)


def _unit_terms_at(plant, t, state_full, params_full, names, term, maps=(None, None)):
    """``term(unit, t, state, inputs, signals, params)`` of each named unit at one
    saved instant, ``{unit: term}``, fed the inputs and signals the plant's RHS
    assembly hands the unit at ``(t, state_full)``."""
    import jax.numpy as jnp

    state_full = jnp.asarray(state_full)
    states = plant._split_state(state_full)
    recycle_map, flow_map = maps
    t = jnp.asarray(t)
    inputs, signals, _ = plant._unit_inputs_and_signals(
        t, states, params_full, recycle_map=recycle_map, flow_map=flow_map
    )
    out = {}
    for name in names:
        out[name] = term(
            plant.units[name],
            t,
            states[name],
            inputs[name],
            signals[name],
            plant._params_for_unit(name, params_full),
        )
    return out


def unit_terms(plant, solution, params, names, term):
    """:func:`_unit_terms_at` over every saved time, batched by ``jax.vmap`` as
    the stream reconstruction is.

    Shared by the mass balance's reaction and aeration integrals and the SBR's
    :func:`~aquakin.plant.sbr.storage_report`, so each evaluates its unit with
    the inputs and signals the solve's right-hand side gave it.

    Parameters
    ----------
    plant : Plant
        The solved plant.
    solution : PlantSolution
        Its solution.
    params : jnp.ndarray
        The plant parameter vector the solve used.
    names : list of str
        The units to evaluate.
    term : callable
        ``term(unit, t, state, inputs, signals, unit_params)``.

    Returns
    -------
    dict
        ``{unit: term over the saved times}``, each array with a leading
        ``n_t`` axis (a tuple term stacks per entry).
    """
    import jax
    import jax.numpy as jnp

    if not names:
        return {}
    params_full = plant._coerce_params(params)
    ts = jnp.asarray(solution.t_native)
    states_flat = jnp.asarray(solution.state)
    maps = plant._recycle.cached_maps(ts[0], plant._split_state(states_flat[0]), params_full)
    return jax.vmap(
        lambda t_i, row: _unit_terms_at(plant, t_i, row, params_full, names, term, maps)
    )(ts, states_flat)


def oxygen_transfer_history(plant, solution, params, names):
    """Each named unit's aeration transfer over the saved trajectory, as its
    ``rhs`` added it in the solve.

    Parameters
    ----------
    plant : Plant
        The solved plant.
    solution : PlantSolution
        The saved trajectory.
    params : jnp.ndarray or None
        The plant parameters the solve used (``None`` for the defaults).
    names : list[str]
        Units declaring :class:`~aquakin.plant.units.OxygenTransferUnit`.

    Returns
    -------
    dict[str, tuple[jnp.ndarray, jnp.ndarray]]
        ``{unit: (oxygen, stripped)}``, each shape ``(n_t, n_species)``: the
        transfer rate per unit volume (per the plant's time unit) on the aerated
        oxygen species only, and on the co-stripped gases only. It includes the
        aeration temperature correction and the closed-loop kLa, since it is the
        term the solve integrated.
    """
    if params is None:
        params = plant.default_parameters()
    return unit_terms(plant, solution, params, names, _oxygen_transfer_term)


def _reaction_and_aeration_gas(plant, solution, params, content_by_model, comps):
    """Integrate, over the saved trajectory, the aeration oxygen transfer (g O2/d
    == g COD/d removed) and the reaction-production of each component summed over
    every reactive unit (the activated-sludge reactors, the SBR and the ADM1
    digester).

    The units taking part are those declaring
    :class:`~aquakin.plant.units.ReactionBalanceUnit`, and the aerated ones those
    declaring :class:`~aquakin.plant.units.OxygenTransferUnit`; each supplies
    its own reaction term, reaction volumes and aeration transfer, so a
    variable-volume or time-phased unit (the SBR) is weighted by its volume and
    aerated at its phase at each saved time.

    Returns ``(o2_transfer, R, stripped)`` where ``R[component]`` is the window
    integral of ``Σ_units Σ_species (dC/dt)·content·volume`` (canonical g) over
    each unit's ``reaction_gas_components``. For a component conserved among
    tracked species ``R`` is zero; where it is not (denitrification reducing
    nitrate to N₂, the digester exporting biogas) ``R`` is the negative of the
    gas that left, so ``gas = −R`` (plus the aeration oxygen for COD).
    ``stripped[component]`` is the integral of the co-stripped gases' transfer
    dotted with their content (negative: a loss).
    """
    import jax.numpy as jnp

    from aquakin.plant.units import OxygenTransferUnit, ReactionBalanceUnit

    t = np.asarray(solution.t_native)  # native: the rates are per native time unit
    layout = plant._state_layout

    reactive = [n for n in plant._unit_order if isinstance(plant.units[n], ReactionBalanceUnit)]
    aerated = [n for n in reactive if isinstance(plant.units[n], OxygenTransferUnit)]
    rqs = [q for q in comps if q in ("COD", "N")]
    unit_params = {n: plant._params_for_unit(n, params) for n in reactive}
    gas_comps = {
        n: [q for q in rqs if q in plant.units[n].reaction_gas_components] for n in reactive
    }
    reactions = unit_terms(plant, solution, params, reactive, _reaction_term)
    transfers = oxygen_transfer_history(plant, solution, params, aerated)
    weights = {}  # unit -> (volumes, volumes as jnp, {component: content x volume})
    o2_rows, R_rows, strip_rows = [], {q: [] for q in rqs}, {q: [] for q in rqs}
    for i in range(t.shape[0]):
        state_i = solution.state[i]
        o2 = 0.0
        R = dict.fromkeys(rqs, 0.0)
        S = dict.fromkeys(rqs, 0.0)
        for name in reactive:
            unit = plant.units[name]
            start, size = layout[name]
            unit_state = state_i[start : start + size]
            vols = np.asarray(unit.reaction_volumes(unit_state, unit_params[name]))
            if name not in weights or not np.array_equal(vols, weights[name][0]):
                # Rebuilt only when the volumes change (a variable-volume unit),
                # so a fixed-volume unit builds its weights once.
                cvec = content_by_model[unit.model.name]
                weights[name] = (
                    vols,
                    jnp.asarray(vols),
                    {q: jnp.asarray(cvec[q] * vols) for q in rqs},
                )
            _, vols_j, content = weights[name]
            react = reactions[name][i]
            for q in gas_comps[name]:
                R[q] += float(jnp.dot(react, content[q]))
            if name in aerated:
                oxygen, stripped = transfers[name][0][i], transfers[name][1][i]
                # Aeration O2 source: transfer per volume times the volume.
                o2 += float(jnp.dot(oxygen, vols_j))
                # Co-stripped gases (N2O, NO, ...) leaving to the air: their
                # component content is a gas loss (transfer is negative).
                for q in rqs:
                    S[q] += float(jnp.dot(stripped, content[q]))
        o2_rows.append(o2)
        for q in rqs:
            R_rows[q].append(R[q])
            strip_rows[q].append(S[q])

    o2_transfer = float(np.trapezoid(np.asarray(o2_rows), t))
    R = {q: float(np.trapezoid(np.asarray(R_rows[q]), t)) for q in rqs}
    stripped = {q: float(np.trapezoid(np.asarray(strip_rows[q]), t)) for q in rqs}
    return o2_transfer, R, stripped


def _biogas_cod(plant, solution, params):
    """Digester biogas COD exported over the window (g COD), or ``None`` if the
    plant has no ADM1 digester. CH₄ at 4 g COD/g (H₂ is negligible)."""
    from aquakin.plant.bsm.evaluation import digester_gas
    from aquakin.plant.errors import NoDigesterError

    try:
        gas = digester_gas(plant, solution, params)
    except NoDigesterError:
        return None
    t = np.asarray(solution.t_native)  # ch4 is per native time unit (kg/d)
    ch4_g_per_d = np.asarray(gas.ch4) * 1000.0 * _COD_PER_CH4  # kg/d -> g COD/d
    return float(np.trapezoid(ch4_g_per_d, t))
