"""State translators: convert a stream from one kinetic model to another.

For single-model plants like BSM1, every translator is the
:class:`IdentityTranslator`. The interface exists so that BSM2-style
plants — which use ASM1 in the activated-sludge tanks and ADM1 in the
anaerobic digester — can plug an ASM↔ADM mapping into the framework
without touching plant assembly code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


@runtime_checkable
class StateTranslator(Protocol):
    """Maps a concentration vector from one model's species ordering to
    another's. Must be AD-clean (used inside the plant RHS).

    Attributes
    ----------
    source_model : CompiledModel
        The kinetic model whose species ordering the input concentration
        vector follows.
    target_model : CompiledModel
        The kinetic model the output concentration vector is expressed
        in.
    """

    source_model: "CompiledModel"
    target_model: "CompiledModel"

    def translate(self, C_source: jnp.ndarray, digester_pH=None) -> jnp.ndarray:
        """Map ``C_source`` to the target model's species ordering.

        ``digester_pH`` optionally supplies the digester's instantaneous,
        state-derived pH for a translator whose mapping has a pH-dependent
        (charge-balance) term; a translator without one ignores it. The plant
        supplies it when the translator declares ``needs_dest_pH``.
        """
        ...


def translator_coupling_pattern(translator, n_states: int = 96, seed: int = 0):
    """Structural coupling of a translator: which target species each source
    species can influence, as a boolean ``(n_target, n_source)`` matrix.

    This is what lets a translator participate in the plant's colored-Jacobian
    sparsity pattern (issue #388): a cross-model edge (ASM<->ADM) introduces
    couplings that live in the *translator*, not in either model, and that are
    regime-dependent (an interface's greedy nitrogen-budget allocation switches
    branches with the influent), so a numerical probe at one operating point
    misses the branches that activate at another.

    The pattern is derived by **forward-AD of ``translate`` unioned over many
    diverse source states** (each component scaled over a wide multiplicative
    range), so every min/max allocation branch is exercised and the result is a
    structural superset, not a single-state snapshot. The map is small and cheap,
    so a far denser sample than a stiff plant solve could afford is used. A
    translator may **override** this by defining its own ``coupling_pattern()``
    method -- e.g. a declarative translator built from per-species expressions
    could emit its pattern exactly from that declaration. This is the extension
    point that lets a user add a custom cross-model translator and have it work
    with the colored solver automatically.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    override = getattr(translator, "coupling_pattern", None)
    if callable(override):
        return np.asarray(override(), dtype=bool)

    src = translator.source_model
    n_src = src.n_species
    n_tgt = translator.target_model.n_species
    base = np.maximum(np.abs(np.asarray(src.default_concentrations())), 1e-3)
    # fixed representative pH for the (value-independent) coupling structure
    fj = jax.jit(lambda c: jax.jacfwd(lambda x: translator.translate(x, 7.0))(c))
    rng = np.random.default_rng(seed)
    P = np.zeros((n_tgt, n_src), dtype=bool)
    for _ in range(n_states):
        c = jnp.asarray(base * 10.0 ** rng.uniform(-2.0, 2.0, size=n_src))
        P |= np.abs(np.asarray(fj(c))) > 0.0
    return P


class IdentityTranslator:
    """Pass-through translator for when source and target models are the
    same — the only kind of translator BSM1 needs.

    The plant inserts one of these automatically on any connection whose
    source and target units share a model reference, so users don't
    normally need to instantiate it directly.
    """

    def __init__(self, model: "CompiledModel") -> None:
        self.source_model = model
        self.target_model = model

    def translate(self, C_source: jnp.ndarray, digester_pH=None) -> jnp.ndarray:
        return C_source

    def coupling_pattern(self):
        """The identity map couples each species only to itself."""
        import numpy as np

        return np.eye(self.source_model.n_species, dtype=bool)
