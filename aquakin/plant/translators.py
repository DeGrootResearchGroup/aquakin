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

    source_model: CompiledModel
    target_model: CompiledModel

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

    def __init__(self, model: CompiledModel) -> None:
        self.source_model = model
        self.target_model = model

    def translate(self, C_source: jnp.ndarray, digester_pH=None) -> jnp.ndarray:
        return C_source

    def coupling_pattern(self):
        """The identity map couples each species only to itself."""
        import numpy as np

        return np.eye(self.source_model.n_species, dtype=bool)


class UnitScalingTranslator:
    """Translator between two models with the same species in different units.

    Each species is multiplied by the factor from the unit the source model
    declares for it to the unit the target model declares (``kgCOD/m3`` to
    ``g_COD/m3`` is 1000). :meth:`Plant.connect` and :meth:`Plant.add_influent`
    insert one automatically when the two ends' models list the same species in
    the same order but declare some of them in different units; build one
    yourself only to pass it as ``translator=``.

    Parameters
    ----------
    source_model, target_model : CompiledModel
        Models with identical species lists.

    Raises
    ------
    ValueError
        If the species lists differ, or a species' two units cannot be
        compared: one measures another quantity (moles against grams, g N
        against g COD), or one is unreadable while the other differs from it.

    Examples
    --------
    >>> units = {**asm1.species_units, "XS": "kgCOD/m3"}
    >>> asm1_kg = dataclasses.replace(asm1, species_units=units)
    >>> C_g = UnitScalingTranslator(asm1_kg, asm1).translate(C_kg)  # XS x 1000
    """

    def __init__(self, source_model: CompiledModel, target_model: CompiledModel) -> None:
        self.source_model = source_model
        self.target_model = target_model
        self._scale = unit_scaling(source_model, target_model)

    def translate(self, C_source: jnp.ndarray, digester_pH=None) -> jnp.ndarray:
        """``C_source`` in the target model's units."""
        if self._scale is None:
            return C_source
        return C_source * self._scale

    def coupling_pattern(self):
        """Each species maps only to itself."""
        import numpy as np

        return np.eye(self.source_model.n_species, dtype=bool)


def unit_scaling(source_model: CompiledModel, target_model: CompiledModel):
    """Per-species factors from ``source_model``'s units to ``target_model``'s.

    Returns ``None`` when every factor is exactly 1, else a ``(n_species,)``
    array. Raises ``ValueError`` when the species lists differ or a pair of
    units cannot be compared (see :class:`UnitScalingTranslator`).
    """
    from aquakin.utils.units import conversion_factor

    species = list(source_model.species)
    if species != list(target_model.species):
        raise ValueError(
            f"models '{source_model.name}' and '{target_model.name}' do not list the "
            f"same species in the same order, so their units cannot be matched "
            f"species by species."
        )
    factors = []
    for sp in species:
        src = source_model.species_units.get(sp, "")
        dst = target_model.species_units.get(sp, "")
        if src.strip() == dst.strip():
            factors.append(1.0)
            continue
        try:
            factor = conversion_factor(src, dst)
        except ValueError as exc:
            raise ValueError(f"species '{sp}': {exc}") from None
        if factor is None:
            raise ValueError(
                f"species '{sp}' is in '{src}' in model '{source_model.name}' and "
                f"'{dst}' in model '{target_model.name}', and one of the two cannot "
                f"be read, so the factor between them is unknown."
            )
        factors.append(float(factor))
    if all(f == 1.0 for f in factors):
        return None
    return jnp.asarray(factors)
