"""Reference warm-start states for the BSM plants.

**Why BSM2 needs a warm start.** The BSM2 anaerobic digester has a ~19-day
hydraulic retention time, so from a cold start (network defaults) the plant's
slowest mode takes weeks of simulated time to settle, and the cold-start
transient -- a near-empty activated-sludge basin filling against the recycle
loops -- is stiff enough to crawl or hit the integrator step ceiling. Seeding
the five activated-sludge reactors with a healthy biomass close to the steady
state removes that transient: the AS line is already alive, so only the digester
has to settle and it does so smoothly.

This module ships the reference reactor compositions so every script stops
copy-pasting the same biomass dictionary, and exposes :func:`bsm1_warm_start` /
:func:`bsm2_warm_start` that build a ready-to-use flat ``y0`` for a plant: the
activated-sludge reactors seeded with the reference biomass, every other unit
(digester, clarifiers, recycle units) left at its own default.

The BSM2 composition is the validated reference reactor state (the BSM2 plant
reproduces the published open-loop steady state to within a few percent from
it); the BSM1 composition is approximately aquakin's BSM1 open-loop steady
state. Both are *seeds* -- the solve relaxes them to the exact steady state --
so the precise values only affect how fast the plant settles, not where.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork
    from aquakin.plant.plant import Plant


# Reference activated-sludge reactor composition (ASM1 species, g/m³).
#
# BSM2: the reference reactor state -- the warm biomass the BSM2 plant settles
# from to reproduce the validated open-loop steady state.
BSM2_WARM_REACTOR_COMPOSITION = {
    "SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
    "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
    "SND": 0.7, "XND": 3.0, "SALK": 5.0,
}

# BSM1: approximately aquakin's BSM1 open-loop steady-state reactor composition
# (a representative mid-cascade tank). A warm-start seed; the solve relaxes it.
BSM1_WARM_REACTOR_COMPOSITION = {
    "SI": 30.0, "SS": 3.0, "XI": 1143.0, "XS": 70.0, "XB_H": 1768.0,
    "XB_A": 1809.0, "XP": 1802.0, "SO": 1.0, "SNO": 6.5, "SNH": 1.0,
    "SND": 0.85, "XND": 5.0, "SALK": 4.7,
}


def _as_reactor_names(plant: "Plant") -> list:
    """The activated-sludge reactor CSTRs (volume + aeration), in plant order.
    The digester and other volumed units have no ``aeration`` field."""
    reactors = [name for name in plant._unit_order
                if hasattr(plant.units[name], "aeration")
                and hasattr(plant.units[name], "volume")]
    if not reactors:
        raise ValueError(
            "No activated-sludge reactors found in this plant; cannot build a "
            "warm start. (Expected CSTR reactor units with aeration.)")
    return reactors


def _warm_y0(plant, composition, asm1_network):
    """Build a flat plant ``y0`` seeding the AS reactors with ``composition``."""
    reactors = _as_reactor_names(plant)
    asm1 = (asm1_network if asm1_network is not None
            else plant.units[reactors[0]].network)
    warm = asm1.concentrations(composition)
    return plant.initial_state(overrides={name: warm for name in reactors})


def bsm2_warm_start(
    plant: "Plant",
    asm1_network: Optional["CompiledNetwork"] = None,
) -> jnp.ndarray:
    """A warm-start initial state ``y0`` for a BSM2 plant.

    Seeds the five activated-sludge reactors with the reference biomass
    (:data:`BSM2_WARM_REACTOR_COMPOSITION`) and leaves every other unit at its
    default, returning the flat plant state to pass as ``plant.solve(y0=...)``.
    **Recommended for every BSM2 run** -- the digester's ~19-day retention makes
    a cold start slow and stiff (see the module docstring).

    Parameters
    ----------
    plant : Plant
        A BSM2 plant from :func:`aquakin.plant.bsm.build_bsm2`, with its
        influent already added.
    asm1_network : CompiledNetwork, optional
        The water-line ASM1 network (for the composition vector). Defaults to
        the network of the plant's first AS reactor.

    Returns
    -------
    jnp.ndarray
        The flat warm-start plant state, shape ``(total_state_size,)``.

    Examples
    --------
    >>> plant = build_bsm2(asm1, adm1)                       # doctest: +SKIP
    >>> plant.add_influent("feed", influent)                # doctest: +SKIP
    >>> y0 = bsm2_warm_start(plant)                          # doctest: +SKIP
    >>> sol = plant.solve(t_span=(0.0, 200.0), params=params, y0=y0)  # doctest: +SKIP
    """
    return _warm_y0(plant, BSM2_WARM_REACTOR_COMPOSITION, asm1_network)


def bsm1_warm_start(
    plant: "Plant",
    asm1_network: Optional["CompiledNetwork"] = None,
) -> jnp.ndarray:
    """A warm-start initial state ``y0`` for a BSM1 plant.

    Seeds the five activated-sludge reactors with a healthy biomass
    (:data:`BSM1_WARM_REACTOR_COMPOSITION`) and leaves the clarifier at its
    default. BSM1 has no slow digester, so it settles from a cold start too;
    the warm start simply shortens the transient.

    Parameters
    ----------
    plant : Plant
        A BSM1 plant from :func:`aquakin.plant.bsm.build_bsm1`, with its
        influent already added.
    asm1_network : CompiledNetwork, optional
        The ASM1 network. Defaults to the network of the plant's first reactor.

    Returns
    -------
    jnp.ndarray
        The flat warm-start plant state, shape ``(total_state_size,)``.
    """
    return _warm_y0(plant, BSM1_WARM_REACTOR_COMPOSITION, asm1_network)
