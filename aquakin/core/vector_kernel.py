"""Vectorized rate kernel.

Builds, from a model's per-reaction rate ASTs, a single callable that returns
the ``(n_reactions,)`` rate vector by **interning every distinct subexpression**
(node type + operand positions) and evaluating **all instances of each
primitive in one batched elementwise operation**, in topological order.

This is global common-subexpression elimination plus vectorization by node type.
The scalar path stacks one nested closure tree per reaction, so the traced
jaxpr holds ``O(reactions x ops-per-reaction)`` scalar primitives (each leaf a
``slice`` + ``squeeze``); XLA fuses them at runtime but its optimization passes
scale with that op count, so the *compile* is dominated by it -- amplified in
the reverse-mode adjoint, where the RHS jaxpr is differentiated ~80x per step.
The kernel collapses the traced op count to ``~O(node-types x depth)``,
independent of the reaction count, which is the entire payoff (compile / trace
time; runtime is unchanged -- XLA already fuses either form).

The result is **bit-identical** to the scalar path: each interned instance is a
lane of a batched op that performs the identical scalar arithmetic the
per-reaction closure would, and IEEE elementwise ops are deterministic per lane.
The interning dedups identical subexpressions (CSE), which is also bit-identical
because the scalar path recomputes them to the same bits.

If a future AST node type is not handled here, :func:`build_vectorized_rates`
raises :class:`UnsupportedNode`; the caller falls back to the scalar path for
the whole model, so the kernel is a safe, transparent overlay.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.core.nodes import (
    ASTNode,
    ConditionNode,
    ConstantNode,
    ParamNode,
    PowerNode,
    SpeciesNode,
)


class UnsupportedNode(Exception):
    """Raised when an AST node type has no vectorized kernel.

    The caller catches this and falls back to the scalar per-reaction path, so
    a newly added node type degrades gracefully (slower compile) rather than
    breaking.
    """


# Raw 1-D gather (``out[k] = src[idx[k]]``) via ``lax.gather`` directly, which
# skips the negative-index normalization (``where(i<0, i+n, i)`` -> lt/add/
# select_n) that ``src[idx]`` / ``src.at[idx].get`` insert. Our pool indices are
# built non-negative and in-bounds, so the normalization is pure overhead -- and
# with ~one gather per operand edge it otherwise dominates the kernel jaxpr.
_GATHER_DN = jax.lax.GatherDimensionNumbers(
    offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
)
_PROMISE_IN_BOUNDS = jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS


def _gather(src, idx):
    """``src[idx]`` for a 1-D ``src`` and 1-D non-negative in-bounds ``idx``."""
    return jax.lax.gather(
        src,
        idx[:, None],
        dimension_numbers=_GATHER_DN,
        slice_sizes=(1,),
        mode=_PROMISE_IN_BOUNDS,
        indices_are_sorted=False,
        unique_indices=False,
    )


# --- batched per-kind kernels ------------------------------------------------
#
# Each kernel takes a tuple of operand-value arrays (already gathered from the
# pool, one array per operand slot, all the same length = the number of
# instances of this kind at this depth) and returns the array of results. The
# kernels are the AST nodes' own ``op`` staticmethods -- the *same* functions
# the scalar per-reaction closures call in ``core/nodes.py`` -- so the two paths
# are identical by construction rather than by hand-aligned copies. The table is
# built by walking the node hierarchy, so a node added with a ``KIND`` gets its
# kernel here automatically (and one added *without* one raises ``UnsupportedNode``
# at build time, falling back to the scalar path).


def _k_powc(o, exp_array):
    """``base ** constant_exponent`` with the exponent a **static** array.

    A constant exponent must stay static (not a traced pool value): a traced
    exponent activates the generic ``pow`` JVP's ``base**exp * log(base)`` term,
    which is ``0 * log(0) = NaN`` at ``base == 0`` even though the forward value
    and the true derivative are finite (the term is multiplied by the
    exponent's zero tangent). Keeping the exponent static -- as the scalar
    ``PowerNode`` does with its captured constant -- prunes that term, so the
    derivative matches the scalar path (finite). The per-instance exponents are
    carried as one static array so different constant powers still batch. The
    arithmetic is ``PowerNode.op`` itself, so it cannot drift from the ``pow``
    path.
    """
    return PowerNode.op((o[0], exp_array))


def _collect_kernels() -> dict[str, Callable]:
    """Map each operator node's ``KIND`` to its batched ``op`` by walking the AST
    node hierarchy, so the kernel table can never drift from the node set."""
    kernels: dict[str, Callable] = {}
    stack = list(ASTNode.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        kind = cls.__dict__.get("KIND")
        if kind is not None:
            kernels[kind] = cls.op
    return kernels


_KERNELS: dict[str, Callable] = _collect_kernels()
# ``powc`` is a kernel-internal variant of ``PowerNode`` (a *constant* exponent
# kept static for AD finiteness -- see _k_powc); it has no node of its own.
_KERNELS["powc"] = _k_powc


# --- interning compiler ------------------------------------------------------


def _resolve_param(name: str, reaction_name: str, param_index: dict[str, int]) -> int:
    """Resolve a (possibly reaction-local) parameter name to its flat index.

    Mirrors :meth:`ParamNode.compile`: reaction-local (``<reaction>.<name>``)
    first, then model-level.
    """
    if reaction_name:
        local = f"{reaction_name}.{name}"
        if local in param_index:
            return param_index[local]
    if name in param_index:
        return param_index[name]
    raise KeyError(
        f"Parameter '{name}' (reaction '{reaction_name}') not found in the "
        f"parameter index while building the vectorized rate kernel."
    )


@dataclass
class _Interner:
    """Builds the deduplicated subexpression table for the whole model."""

    species_index: dict[str, int]
    param_index: dict[str, int]

    def __post_init__(self):
        # id_map: dedup key -> pool id.  records: list indexed by id, each a
        # (kind, operand_ids tuple, literal) triple.  literal carries the leaf
        # payload (species/param index, condition field, or constant value).
        self.id_map: dict = {}
        self.kinds: list[str] = []
        self.operands: list[tuple] = []
        self.literals: list = []

    def _add(self, key, kind, operand_ids, literal):
        cached = self.id_map.get(key)
        if cached is not None:
            return cached
        idx = len(self.kinds)
        self.id_map[key] = idx
        self.kinds.append(kind)
        self.operands.append(tuple(operand_ids))
        self.literals.append(literal)
        return idx

    def intern(self, node, reaction_name: str) -> int:
        """Intern ``node`` (and its subtree); return its pool id."""
        # Leaves carry a literal payload (an index / field name / constant),
        # not a kernel; they map to the pool's leaf blocks.
        if isinstance(node, ConstantNode):
            v = float(node.value)
            return self._add(("const", v), "const", (), v)
        if isinstance(node, SpeciesNode):
            if node.name not in self.species_index:
                raise KeyError(f"Species '{node.name}' not declared.")
            i = self.species_index[node.name]
            return self._add(("species", i), "species", (), i)
        if isinstance(node, ParamNode):
            i = _resolve_param(node.name, reaction_name, self.param_index)
            return self._add(("param", i), "param", (), i)
        if isinstance(node, ConditionNode):
            return self._add(("cond", node.field_name), "cond", (), node.field_name)

        # Power with a *constant* exponent -> powc: the exponent stays static
        # (see _k_powc) so its JVP matches the scalar PowerNode and stays finite
        # at base 0. A non-constant exponent falls through to the generic path
        # below (kind "pow").
        if isinstance(node, PowerNode) and isinstance(node.right, ConstantNode):
            base = self.intern(node.left, reaction_name)
            exp = float(node.right.value)
            return self._add(("powc", (base,), exp), "powc", (base,), exp)

        # Every other operator node derives its kernel key and operand order
        # straight from the node type: operands are its interned AST children
        # (field order) followed by any condition fields it declares (pH / T),
        # matching the operand order the node's ``op`` expects. So a new node
        # type needs no edit here -- ``KIND`` + ``op`` on the class is enough.
        kind = type(node).KIND
        if kind is not None:
            operands = [self.intern(c, reaction_name) for c in node.children()]
            for field_name in type(node).EXTRA_CONDITIONS:
                operands.append(self._add(("cond", field_name), "cond", (), field_name))
            return self._op(kind, tuple(operands))

        raise UnsupportedNode(type(node).__name__)

    def _op(self, kind: str, operand_ids: tuple) -> int:
        key = (kind, operand_ids)
        return self._add(key, kind, operand_ids, None)


@dataclass
class VectorizedRates:
    """Compiled vectorized rate evaluator.

    Call it with the canonical rate signature ``(C, params, condition_arrays,
    loc_idx) -> (n_reactions,)``.

    The pool ``P`` is built **append-only by concatenation** (1 jaxpr op each),
    never by scatter (~5 ops each, incl. index fixup): leaves form the first
    block, then each ``(depth, kind)`` step appends its batch. Pool positions
    therefore never shift, so each step gathers its operands with static
    indices into the already-built prefix of ``P``.
    """

    # Leaf blocks (static numpy index arrays into C / params; constant values;
    # condition field names). Each forms one contiguous block of the pool, in
    # this order: species, params, constants, conditions.
    species_src: np.ndarray
    param_src: np.ndarray
    const_vals: jnp.ndarray
    cond_fields: tuple[str, ...]
    # Op steps in append (ascending-depth) order: (kernel, operand_pos_arrays,
    # aux), where each operand_pos array indexes into the pool prefix existing
    # when the step runs, and ``aux`` is a static per-instance array (the
    # constant exponents for a ``powc`` step) or ``None``.
    steps: list[tuple]
    root_pos: np.ndarray
    n_reactions: int

    def __call__(self, C, params, condition_arrays, loc_idx):
        blocks = []
        if self.species_src.size:
            blocks.append(_gather(C, self.species_src))
        if self.param_src.size:
            blocks.append(_gather(params, self.param_src))
        if self.const_vals.size:
            blocks.append(self.const_vals.astype(C.dtype))
        if self.cond_fields:
            blocks.append(
                jnp.stack([condition_arrays[f][loc_idx] for f in self.cond_fields]).astype(C.dtype)
            )
        P = jnp.concatenate(blocks) if len(blocks) > 1 else blocks[0]
        for kernel, operand_arrays, aux in self.steps:
            operands = tuple(_gather(P, a) for a in operand_arrays)
            out = kernel(operands) if aux is None else kernel(operands, aux)
            P = jnp.concatenate([P, out])
        return _gather(P, self.root_pos)


def build_vectorized_rates(
    rate_asts,
    reaction_names,
    species_index: dict[str, int],
    param_index: dict[str, int],
) -> VectorizedRates:
    """Build a :class:`VectorizedRates` from the per-reaction ASTs.

    Raises
    ------
    UnsupportedNode
        If any AST contains a node type without a vectorized kernel. The caller
        should fall back to the scalar path.
    """
    interner = _Interner(species_index, param_index)
    root_ids = [
        interner.intern(ast, name) for ast, name in zip(rate_asts, reaction_names, strict=True)
    ]

    n_ids = len(interner.kinds)
    kinds = interner.kinds
    operands = interner.operands
    literals = interner.literals

    # Topological depth of each interned id (leaves = 0).
    depth = [0] * n_ids
    for i in range(n_ids):
        ops = operands[i]
        if ops:
            depth[i] = 1 + max(depth[o] for o in ops)

    # ``pos`` maps an intern id to its position in the append-ordered pool:
    # leaf blocks first (species, params, consts, conds), then each (depth,
    # kind) step block in ascending-depth order. The pool is concatenated in
    # exactly this order at runtime, so positions never shift.
    pos = [-1] * n_ids
    cursor = 0

    def _assign(ids):
        nonlocal cursor
        for i in ids:
            pos[i] = cursor
            cursor += 1

    species_ids = [i for i in range(n_ids) if kinds[i] == "species"]
    param_ids = [i for i in range(n_ids) if kinds[i] == "param"]
    const_ids = [i for i in range(n_ids) if kinds[i] == "const"]
    cond_ids = [i for i in range(n_ids) if kinds[i] == "cond"]
    _assign(species_ids)
    _assign(param_ids)
    _assign(const_ids)
    _assign(cond_ids)

    species_src = np.asarray([literals[i] for i in species_ids], dtype=np.int32)
    param_src = np.asarray([literals[i] for i in param_ids], dtype=np.int32)
    const_vals = jnp.asarray([literals[i] for i in const_ids], dtype=jnp.float64)
    cond_fields = tuple(literals[i] for i in cond_ids)

    # Op steps grouped by (depth, kind), ascending depth. Each step's output
    # ids are assigned the next contiguous pool positions; its operand arrays
    # are remapped through ``pos`` (operands always have lower depth, so they
    # are already in the pool prefix).
    max_depth = max(depth) if depth else 0
    grouped = []
    for d in range(1, max_depth + 1):
        by_kind: dict[str, list[int]] = {}
        for i in range(n_ids):
            if depth[i] == d and operands[i]:
                by_kind.setdefault(kinds[i], []).append(i)
        for kind, ids in by_kind.items():
            _assign(ids)
            grouped.append((kind, ids))

    steps = []
    for kind, ids in grouped:
        kernel = _KERNELS[kind]
        arity = len(operands[ids[0]])
        operand_arrays = tuple(
            np.asarray([pos[operands[i][slot]] for i in ids], dtype=np.int32)
            for slot in range(arity)
        )
        # ``powc`` carries its per-instance constant exponents as a static array
        # (kept off the traced pool so the pow JVP stays finite at base 0).
        aux = np.asarray([literals[i] for i in ids], dtype=np.float64) if kind == "powc" else None
        steps.append((kernel, operand_arrays, aux))

    return VectorizedRates(
        species_src=species_src,
        param_src=param_src,
        const_vals=const_vals,
        cond_fields=cond_fields,
        steps=steps,
        root_pos=np.asarray([pos[r] for r in root_ids], dtype=np.int32),
        n_reactions=len(rate_asts),
    )
