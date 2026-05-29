"""Compile-time context used by ASTNode.compile()."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompileContext:
    """
    Index maps used during AST compilation.

    Attributes
    ----------
    species_index : dict[str, int]
        Maps species name to its position in the concentration vector ``C``.
    param_index : dict[str, int]
        Maps fully-namespaced parameter name (e.g. ``"O3_Br_direct.k1"``) to
        its position in the flat ``params`` vector.
    condition_fields : frozenset[str]
        Set of valid condition field names. Used to validate
        ``ConditionNode`` references at parse/compile time.
    reaction_name : str
        The reaction that owns the expression currently being compiled. Used
        for namespacing local parameter names.
    """

    species_index: dict[str, int]
    param_index: dict[str, int]
    condition_fields: frozenset[str]
    reaction_name: str = ""
