"""Settling models for batch clarification (the SBR settle phase).

A :class:`SettlingModel` is a pluggable strategy that answers one question for a
well-mixed batch reactor whose contents are allowed to settle: *what is the
particulate concentration of the supernatant drawn off during decant?* The bulk
reactor state stays the well-mixed average (the biology is well-mixed); a
settling model adds only the optional internal state needed to describe how the
clarity of the supernatant evolves while the tank settles, and reports a
per-species multiplier the decant draw is scaled by (1 for solubles, < 1 for
the particulates that have settled out of the top of the tank).

Mass is conserved by the consumer (:class:`~aquakin.plant.sbr.SBRUnit`): the
decant removes ``Q_out * C_decant`` and the rest stays in the tank, so a model
that clarifies the draw automatically concentrates the retained solids -- no
matter which model is plugged in. Two models ship:

* :class:`InterfaceSettling` -- a one-state interface/blanket model: a clarified
  fraction grows at a settling velocity while the tank settles and relaxes back
  to fully mixed otherwise. Lightweight, captures the SBR's defining behaviour
  (a clarified effluent that improves with settle time).
* :class:`LayeredSettling` -- a vertical stack of layers (a Takács-style profile)
  resolving the particulate distribution, so the supernatant clarity emerges
  from the settling dynamics rather than a single velocity. Heavier state, more
  faithful.

Both implement the same interface, so the SBR's settle phase is configured by
passing one or the other; new models slot in by implementing :class:`SettlingModel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


class SettlingModel(ABC):
    """Strategy interface for the SBR settle/decant clarification.

    A model owns an optional block of *settling* state (appended to the SBR
    state after the bulk concentrations and volume) and answers, each RHS call,
    how that state evolves and what multiplier the decant draw is scaled by per
    species. It is *bound* to a network once (``bind``) so it can resolve which
    species are particulate; the SBR calls ``bind`` at construction.
    """

    #: set by ``bind`` -- the (n_species,) 0/1 mask of particulate species.
    _particulate_mask: "jnp.ndarray | None" = None

    def bind(self, network: "CompiledNetwork", particulate_species) -> None:
        """Resolve the particulate-species mask against ``network`` (called once
        by the SBR at construction). ``particulate_species`` is the list of
        species names that settle."""
        idx = [network.species_index[s] for s in particulate_species]
        mask = jnp.zeros((network.n_species,)).at[jnp.asarray(idx, dtype=int)].set(1.0) \
            if idx else jnp.zeros((network.n_species,))
        self._particulate_mask = mask
        self._n_species = network.n_species

    @abstractmethod
    def extra_state_size(self) -> int:
        """Number of internal settling-state variables (0 for a stateless model)."""

    @abstractmethod
    def initial_extra_state(self) -> jnp.ndarray:
        """Initial settling state, shape ``(extra_state_size(),)``."""

    @abstractmethod
    def extra_rhs(self, C: jnp.ndarray, V: jnp.ndarray, extra: jnp.ndarray,
                  settling_active: jnp.ndarray) -> jnp.ndarray:
        """``d(extra)/dt`` of the settling state.

        ``settling_active`` is a 0/1 scalar (1 during the settle phase). A
        stateless model returns ``zeros((0,))``.
        """

    @abstractmethod
    def decant_multiplier(self, C: jnp.ndarray, V: jnp.ndarray,
                          extra: jnp.ndarray) -> jnp.ndarray:
        """Per-species multiplier (shape ``(n_species,)``) the decant draw is
        scaled by: ``C_decant = multiplier * C``. 1 for solubles; ``<= 1`` for
        particulates that have settled below the decant level."""


@dataclass
class InterfaceSettling(SettlingModel):
    """One-state interface model: a clarified fraction of the supernatant.

    A single state ``c in [0, 1]`` tracks how clarified the top of the tank is
    (0 = fully mixed, 1 = fully clarified supernatant). While the tank settles it
    grows at the rate the sludge-water interface descends, ``v_settle / depth``
    (depth ``= V / area``); otherwise it relaxes back to 0 at ``remix_rate`` (the
    contents re-mix once aeration/feed resumes). The decant draws particulates at
    ``(1 - c)`` of the bulk -- a longer settle gives a cleaner effluent. Solubles
    are unaffected (they do not settle).

    Parameters
    ----------
    v_settle : float
        Zone-settling (interface) velocity, length/time (e.g. m/d).
    area : float
        Tank cross-sectional area, length^2, converting volume to depth.
    remix_rate : float
        First-order rate (1/time) at which clarity decays back to 0 when not
        settling. Large enough that the tank is well mixed within a fill/react
        phase; default 1e3 (effectively instant on a daily cycle).
    """

    v_settle: float
    area: float
    remix_rate: float = 1.0e3
    _particulate_mask: "jnp.ndarray | None" = field(default=None, repr=False)

    def extra_state_size(self) -> int:
        return 1

    def initial_extra_state(self) -> jnp.ndarray:
        return jnp.zeros((1,))  # start fully mixed

    def extra_rhs(self, C, V, extra, settling_active) -> jnp.ndarray:
        c = extra[0]
        depth = jnp.maximum(V / self.area, 1e-9)
        grow = self.v_settle / depth                  # interface descent -> clarity
        # Settling grows c toward 1; otherwise it relaxes to 0 (re-mixing).
        dc = settling_active * grow * (1.0 - c) \
            - (1.0 - settling_active) * self.remix_rate * c
        return jnp.reshape(dc, (1,))

    def decant_multiplier(self, C, V, extra) -> jnp.ndarray:
        c = jnp.clip(extra[0], 0.0, 1.0)
        # Particulates drawn at (1-c) of bulk; solubles at 1.0.
        return 1.0 - c * self._particulate_mask


@dataclass
class LayeredSettling(SettlingModel):
    """Vertical-layer settling profile (a Takács-style representation).

    Resolves the particulate distribution into ``n_layers`` equal-volume layers
    (layer 0 = top supernatant, layer ``n_layers-1`` = bottom). The internal
    state is the per-layer particulate-concentration *ratio* to the bulk average
    (so it is volume-independent and starts uniform at 1.0). While the tank
    settles, particulate ratio moves downward by a first-order interlayer flux
    proportional to the settling velocity, depleting the top layers and enriching
    the bottom; otherwise the profile relaxes back to uniform (re-mixing). The
    decant draws the **top layer**, so the supernatant clarity emerges from the
    profile rather than a single number. Conserves the cross-layer average ratio
    (the bulk average, carried by the SBR's own state, is untouched by the
    redistribution).

    Parameters
    ----------
    n_layers : int
        Number of vertical layers (>= 2).
    v_settle : float
        Settling velocity, length/time.
    area : float
        Tank cross-sectional area, length^2.
    remix_rate : float
        Re-mixing relaxation rate (1/time) when not settling. Default 1e3.
    """

    n_layers: int = 4
    v_settle: float = 100.0
    area: float = 1.0
    remix_rate: float = 1.0e3
    _particulate_mask: "jnp.ndarray | None" = field(default=None, repr=False)

    def __post_init__(self):
        if self.n_layers < 2:
            raise ValueError("LayeredSettling needs n_layers >= 2.")

    def extra_state_size(self) -> int:
        return self.n_layers

    def initial_extra_state(self) -> jnp.ndarray:
        return jnp.ones((self.n_layers,))  # uniform profile (ratio 1 everywhere)

    def extra_rhs(self, C, V, extra, settling_active) -> jnp.ndarray:
        # Per-layer particulate ratio r (mean 1). Settling moves mass from layer
        # k to k+1 at a velocity-set rate; the downward flux out of the bottom is
        # reflected (no mass leaves the profile -- the average ratio is conserved).
        r = extra
        depth = jnp.maximum(V / self.area, 1e-9)
        layer_depth = depth / self.n_layers
        k = self.v_settle / jnp.maximum(layer_depth, 1e-9)   # interlayer rate
        # Downward flux f_k = k * r_k out of each layer except the bottom.
        flux = k * r
        flux = flux.at[-1].set(0.0)                          # bottom is a sink/floor
        # dr_k = (flux into k from above) - (flux out of k).
        inflow = jnp.concatenate([jnp.zeros((1,)), flux[:-1]])
        d_settle = inflow - flux
        # Re-mixing relaxes the profile to uniform (mean of r).
        mean_r = jnp.mean(r)
        d_mix = self.remix_rate * (mean_r - r)
        dr = settling_active * d_settle + (1.0 - settling_active) * d_mix
        return dr

    def decant_multiplier(self, C, V, extra) -> jnp.ndarray:
        # Top layer's particulate ratio (clamped to >= 0) scales the bulk draw.
        top = jnp.clip(extra[0], 0.0, None)
        return 1.0 - (1.0 - top) * self._particulate_mask
