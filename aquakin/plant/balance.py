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
so inventories and fluxes sum across networks of different units (the ASM water
line in g/m³, the ADM digester in kg/m³ and kmol/m³) via
:func:`aquakin.canonical_content`.

The gas integrals are evaluated from the saved trajectory, so they are exact at
steady state (constant rates) and otherwise accurate to the ``t_eval`` sampling;
the activated-sludge reaction integral uses each reactor's operating
temperature condition, exact when the influent temperature is constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

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
    """

    component: str
    inflow: float
    outflow: float
    gas: float
    accumulation: float
    imbalance: float

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
    window: tuple[float, float]
    influent_ports: list[str] = field(default_factory=list)
    effluent_ports: list[str] = field(default_factory=list)
    gas_detail: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, component: str) -> ComponentBalance:
        return self.components[component]

    def closed(self, rtol: float = 1.0e-2) -> bool:
        """True when every component's ``relative_imbalance`` is within ``rtol``."""
        return all(abs(c.relative_imbalance) <= rtol
                   for c in self.components.values())

    def summary(self) -> str:
        """A printable per-component table (canonical g over the window)."""
        t0, t1 = self.window
        lines = [f"Mass balance over t = [{t0:g}, {t1:g}] "
                 f"(canonical g of component over the window):",
                 f"  {'comp':4s} {'in':>13s} {'out':>13s} {'gas':>13s} "
                 f"{'accum':>13s} {'imbalance':>13s} {'rel':>10s}"]
        for name, c in self.components.items():
            lines.append(
                f"  {name:4s} {c.inflow:13.4g} {c.outflow:13.4g} {c.gas:13.4g} "
                f"{c.accumulation:13.4g} {c.imbalance:13.4g} "
                f"{c.relative_imbalance:10.2e}")
        lines.append(f"  closed (rtol=1e-2): {self.closed()}")
        return "\n".join(lines)


# --- per-unit inventory ------------------------------------------------------

def _unit_inventory(plant, unit_name, state_vec, content_by_network, params):
    """Component inventory held in one unit (a ``{component: grams}`` dict).

    Handles every shipped unit type: a concentration-vector unit (CSTR /
    primary clarifier / digester) holds ``volume × Σ C·content``; the digester
    weights its three gas-headspace states by the headspace volume ``V_gas``; a
    sequencing batch reactor carries ``[C..., V, <settling state>]`` (volume at
    ``n_species``, the trailing settling state massless); a membrane bioreactor
    carries ``[C..., R_f]`` at its fixed ``volume`` (the trailing fouling
    resistance is massless); a storage tank carries its liquid volume as the last
    state entry; the layered Takács settler sums its particulate blanket over the
    layers; stateless units hold nothing.
    """
    unit = plant.units[unit_name]
    net = getattr(unit, "network", None)
    if net is None or state_vec.size == 0:
        return {}
    content = content_by_network[net.name]      # {component: (n_species,) array}

    # Layered Takács settler: the particulate head block is (n_layers, n_part);
    # inventory is the blanket summed over layers at the per-layer volume. With
    # soluble_holdup the soluble tail block (n_layers, n_sol) adds its own
    # convective-only inventory (the liquid holdup).
    if hasattr(unit, "_part_indices") and hasattr(unit, "n_layers"):
        layer_vol = float(unit.area) * float(unit.height) / int(unit.n_layers)
        sv = np.asarray(state_vec)
        pb = int(unit._part_block_size)
        prof = sv[:pb].reshape(int(unit.n_layers), unit._n_part)
        out = {}
        for comp, vec in content.items():
            part_content = np.asarray([vec[i] for i in unit._part_indices])
            out[comp] = layer_vol * float(np.sum(prof * part_content[None, :]))
        if getattr(unit, "soluble_holdup", False):
            sol = sv[pb:].reshape(int(unit.n_layers), unit._n_sol)
            for comp, vec in content.items():
                sol_content = np.asarray([vec[i] for i in unit._soluble_indices])
                out[comp] += layer_vol * float(np.sum(sol * sol_content[None, :]))
        return out

    sv = np.asarray(state_vec)
    # Units with a single well-mixed liquid volume (StorageTank, MBR, SBR --
    # whose states are all [C..., one-or-more scalars]) declare that volume
    # explicitly via ``liquid_volume(state)``, so the inventory is ``V*C`` with the
    # concentration head block. This replaces the previous fragile hasattr/state-
    # size dispatch (whose MBR-before-storage ordering existed only because both
    # are [C.., scalar]); a future such unit just implements the contract instead
    # of risking silent misclassification.
    if hasattr(unit, "liquid_volume"):
        V = float(unit.liquid_volume(state_vec))
        C = sv[:net.n_species]
        return {comp: V * float(np.dot(C, vec)) for comp, vec in content.items()}

    if sv.size != net.n_species:
        return {}                                # non-concentration state; skip
    volume = float(getattr(unit, "volume", 0.0))
    if volume <= 0.0:
        return {}

    # Digester: liquid states at V_liq (= unit.volume), the gas headspace at V_gas.
    vol_vec = np.full(net.n_species, volume)
    if "S_gas_ch4" in net.species_index:
        p_unit = plant._params_for_unit(unit_name, params)
        v_gas = (float(p_unit[net.param_index["V_gas"]])
                 if "V_gas" in net.param_index else volume)
        for sp in ("S_gas_h2", "S_gas_ch4", "S_gas_co2"):
            if sp in net.species_index:
                vol_vec[net.species_index[sp]] = v_gas
    return {comp: float(np.dot(sv * vol_vec, vec)) for comp, vec in content.items()}


def _flux(Q, C, content_vec):
    """Component flux of a stream over time: ``Q·(C·content)``, shape ``(n_t,)``
    (canonical g/d)."""
    return np.asarray(Q) * (np.asarray(C) @ content_vec)


def mass_balance(
    plant,
    solution,
    *,
    components=("COD", "N", "P"),
    influent_ports: Optional[list] = None,
    effluent_ports: Optional[list] = None,
    params=None,
) -> MassBalance:
    """Results-level mass-balance closure for a solved plant. See
    :meth:`aquakin.plant.Plant.mass_balance`."""
    import jax.numpy as jnp

    params = (plant.default_parameters() if params is None
              else jnp.asarray(params))
    plant._build_state_layout()
    plant._build_parameter_layout()
    t = np.asarray(solution.t)
    window = (float(t[0]), float(t[-1]))

    # Canonical content vectors per network, keeping only components the network
    # actually carries (a network with no P contributes nothing to the P balance).
    networks = {}                                # name -> CompiledNetwork
    for u in plant.units.values():
        net = getattr(u, "network", None)
        if net is not None:
            networks[net.name] = net
    for s in plant.influents.values():
        networks[s.network.name] = s.network
    # Lab-COD convention for reporting: nitrate / N₂ carry no COD, so a reported
    # COD is the organic oxygen demand (an analyst's COD), not a total electron
    # demand. The closure is self-consistent under either convention. Each
    # network's composition fractions are read from the *run* parameters (a unit
    # of that network), so a calibrated / BSM-specific i_XB flows through.
    net_params = {}
    for uname, u in plant.units.items():
        net = getattr(u, "network", None)
        if net is not None and net.name not in net_params:
            net_params[net.name] = plant._params_for_unit(uname, params)
    content_by_network = {
        name: {q: canonical_content(net, q, electron_acceptor_cod=False,
                                    params=net_params.get(name))
               for q in components}
        for name, net in networks.items()
    }
    comps = list(components)

    # --- boundary ports ------------------------------------------------------
    if effluent_ports is None:
        effluent_ports = list(plant.check().dangling_outputs)
    in_names = (list(plant.influents) if influent_ports is None
                else list(influent_ports))

    # --- inflow (influent series) -------------------------------------------
    inflow = {q: 0.0 for q in comps}
    for name in in_names:
        series = plant.influents[name]
        net = series.network
        cvec = content_by_network[net.name]
        Q = np.asarray([float(series.at(tt).Q) for tt in t])
        C = np.asarray([np.asarray(series.at(tt).C) for tt in t])
        for q in comps:
            inflow[q] += float(np.trapezoid(_flux(Q, C, cvec[q]), t))

    # --- inflow (reagent mass injected by dosing units) ---------------------
    # A DosingUnit adds its reagent's mass to the through-stream from outside the
    # plant boundary, so it is a component source -- counted as inflow (the
    # through-stream's own mass already enters via its upstream influent).
    for uname, u in plant.units.items():
        comp_vec = getattr(getattr(u, "reagent", None), "composition", None)
        if comp_vec is None:
            continue
        cvec = content_by_network[u.network.name]
        comp_vec = np.asarray(comp_vec)
        if u.flow is not None:
            Q_dose = np.full(len(t), float(u.flow))
        else:
            sig = u.required_signals[0]
            Q_dose = np.asarray([
                float(plant.signals_at(tt, solution.state[i], params)[sig]
                      * u.gain)
                for i, tt in enumerate(t)])
        C_dose = np.broadcast_to(comp_vec, (len(t), comp_vec.shape[0]))
        for q in comps:
            inflow[q] += float(np.trapezoid(_flux(Q_dose, C_dose, cvec[q]), t))

    # --- outflow (terminal material streams) --------------------------------
    outflow = {q: 0.0 for q in comps}
    if effluent_ports:
        from aquakin.plant.bsm.evaluation import _reconstruct
        recon = _reconstruct(plant, solution, params, effluent_ports)
        for ep in effluent_ports:
            Q, C = recon[ep]
            unit = plant._parse_endpoint(ep, role="source")[0]
            cvec = content_by_network[plant.units[unit].network.name]
            for q in comps:
                outflow[q] += float(np.trapezoid(
                    _flux(np.asarray(Q), np.asarray(C), cvec[q]), t))

    # --- accumulation (inventory change t1 - t0) ----------------------------
    accumulation = {q: 0.0 for q in comps}
    layout = plant._state_layout
    for unit_name, (start, size) in layout.items():
        inv0 = _unit_inventory(plant, unit_name, solution.state[0][start:start + size],
                               content_by_network, params)
        inv1 = _unit_inventory(plant, unit_name, solution.state[-1][start:start + size],
                               content_by_network, params)
        for q in comps:
            accumulation[q] += inv1.get(q, 0.0) - inv0.get(q, 0.0)

    # --- gas: O2 in by aeration, everything else out by reaction -------------
    # By the integrated plant RHS identity, summed over all units the internal
    # streams cancel, leaving:  ΔInventory = (boundary in − out) + R + aeration,
    # where R is the reaction-production integral over the reactive units and
    # aeration is the non-reaction O2 source. So the component leaving as gas is
    # gas = −(R + aeration): for COD, O2 transferred in (aeration removes COD)
    # minus R_COD (the reactions' net COD production -- negative, since
    # denitrification oxidises COD and the digester gas-outflow exports biogas);
    # for N, −R_N (denitrification N₂; nitrification and the digester conserve N).
    gas = {q: 0.0 for q in comps}
    gas_detail = {}

    o2_transfer, R = _reaction_and_aeration_gas(
        plant, solution, params, content_by_network, comps)
    if "COD" in comps:
        gas["COD"] += o2_transfer - R.get("COD", 0.0)
        gas_detail["aeration_O2"] = o2_transfer
        gas_detail["reaction_COD"] = -R.get("COD", 0.0)
    if "N" in comps:
        gas["N"] += -R.get("N", 0.0)
        gas_detail["denitrification_N2"] = -R.get("N", 0.0)
    biogas = _biogas_cod(plant, solution, params)        # informational only
    if biogas is not None:
        gas_detail["biogas_COD"] = biogas

    out = {}
    for q in comps:
        imb = inflow[q] - outflow[q] - gas[q] - accumulation[q]
        out[q] = ComponentBalance(
            component=q, inflow=inflow[q], outflow=outflow[q], gas=gas[q],
            accumulation=accumulation[q], imbalance=imb)
    return MassBalance(components=out, window=window,
                       influent_ports=in_names, effluent_ports=list(effluent_ports),
                       gas_detail=gas_detail)


def _reaction_volume(plant, unit_name, params):
    """Per-species volume vector (m³) for a reactive unit's reaction term: the
    liquid volume for every state, except an ADM1 digester's three gas-headspace
    states, which live in the headspace volume ``V_gas``."""
    unit = plant.units[unit_name]
    net = unit.network
    V = float(unit.volume)
    vol = np.full(net.n_species, V)
    if "S_gas_ch4" in net.species_index:
        p_unit = plant._params_for_unit(unit_name, params)
        v_gas = (float(p_unit[net.param_index["V_gas"]])
                 if "V_gas" in net.param_index else V)
        for sp in ("S_gas_h2", "S_gas_ch4", "S_gas_co2"):
            if sp in net.species_index:
                vol[net.species_index[sp]] = v_gas
    return vol


def _reaction_term(plant, unit_name, C, params):
    """The reaction (chemistry) term ``dC/dt`` of a reactive unit, reproducing
    exactly what its ``rhs`` evaluates: an aerated CSTR's ``stoichᵀ·rates``, or an
    ADM1 digester's ``network.dCdt`` (which also runs the gas-liquid transfer and
    overpressure gas outflow, so the biogas export is included)."""
    unit = plant.units[unit_name]
    net = unit.network
    p_unit = plant._params_for_unit(unit_name, params)
    if hasattr(unit, "_liquid_mask"):                # ADM1 digester
        if getattr(unit, "_v_liq_idx", None) is not None:
            p_unit = p_unit.at[unit._v_liq_idx].set(float(unit.volume))
        return net.dCdt(C, p_unit, unit._condition_arrays, 0)
    stoich = net.compute_stoich(p_unit)              # aerated/anoxic CSTR
    rates = net.rates(C, p_unit, unit._condition_arrays, 0)
    return stoich.T @ rates


def _reaction_and_aeration_gas(plant, solution, params, content_by_network, comps):
    """Integrate, over the saved trajectory, the aeration oxygen transfer (g O2/d
    == g COD/d removed) and the reaction-production of each component summed over
    every reactive unit (the activated-sludge reactors and the ADM1 digester).

    Returns ``(o2_transfer, R)`` where ``R[component]`` is the window integral of
    ``Σ_units Σ_species (dC/dt)·content·volume`` (canonical g). For a
    component conserved among tracked species ``R`` is zero; where it is not
    (denitrification reducing nitrate to N₂, the digester exporting biogas) ``R``
    is the negative of the gas that left, so ``gas = −R`` (plus the aeration
    oxygen for COD).
    """
    import jax.numpy as jnp

    t = np.asarray(solution.t)
    layout = plant._state_layout

    reactive = [n for n in plant._unit_order
                if hasattr(plant.units[n], "aeration")
                or hasattr(plant.units[n], "_liquid_mask")]
    aerated = [n for n in reactive if hasattr(plant.units[n], "aeration")]
    need_signals = any(plant.units[n]._controlled_kla for n in aerated)
    rqs = [q for q in comps if q in ("COD", "N")]
    vols = {n: _reaction_volume(plant, n, params) for n in reactive}
    content = {n: {q: jnp.asarray(content_by_network[plant.units[n].network.name][q]
                                  * vols[n]) for q in rqs} for n in reactive}

    # Which components each reactive unit can export to an untracked gas /
    # acceptor: an aerated/anoxic reactor reduces nitrate to N₂ (N) and oxidises
    # COD with it (COD); an ADM1 digester exports biogas (COD via CH₄/H₂) but has
    # no nitrogen gas phase, so it must NOT contribute to the N gas term -- if its
    # reactions do not conserve N, that surfaces as a balance imbalance rather
    # than being silently absorbed.
    gas_comps = {n: (set(rqs) if hasattr(plant.units[n], "aeration")
                     else {q for q in rqs if q != "N"})
                 for n in reactive}
    o2_rows, R_rows = [], {q: [] for q in rqs}
    for i in range(t.shape[0]):
        state_i = solution.state[i]
        sig = (plant.signals_at(t[i], state_i, params) if need_signals else {})
        o2 = 0.0
        R = {q: 0.0 for q in rqs}
        for name in reactive:
            unit = plant.units[name]
            start, size = layout[name]
            # Use the species part only: a unit may carry trailing non-species
            # state (an MBR's fouling resistance), which the per-species reaction
            # and aeration terms must not see.
            n_sp = unit.network.n_species
            C = state_i[start:start + n_sp]
            react = _reaction_term(plant, name, C, params)
            for q in gas_comps[name]:
                R[q] += float(jnp.dot(react, content[name][q]))
            if name in aerated:                      # aeration O2 source (only SO)
                kla = unit._kla_vec
                ctrl = unit._controlled_kla.get("SO")
                if ctrl is not None and sig:
                    kla = kla.at[unit.network.species_index["SO"]].set(
                        sig[ctrl[0]] * ctrl[1])
                o2 += float(jnp.sum(kla * (unit._sat_vec - C)) * float(unit.volume))
        o2_rows.append(o2)
        for q in rqs:
            R_rows[q].append(R[q])

    o2_transfer = float(np.trapezoid(np.asarray(o2_rows), t))
    R = {q: float(np.trapezoid(np.asarray(R_rows[q]), t)) for q in rqs}
    return o2_transfer, R


def _biogas_cod(plant, solution, params):
    """Digester biogas COD exported over the window (g COD), or ``None`` if the
    plant has no ADM1 digester. CH₄ at 4 g COD/g (H₂ is negligible)."""
    from aquakin.plant.bsm.evaluation import digester_gas

    try:
        gas = digester_gas(plant, solution, params)
    except ValueError:
        return None
    t = np.asarray(solution.t)
    ch4_g_per_d = np.asarray(gas.ch4) * 1000.0 * _COD_PER_CH4   # kg/d -> g COD/d
    return float(np.trapezoid(ch4_g_per_d, t))
