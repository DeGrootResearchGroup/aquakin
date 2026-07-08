"""AST node types for rate expressions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from aquakin.core.context import CompileContext
from aquakin.core.hints import did_you_mean

RateCallable = Callable[
    [jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray], int | jnp.ndarray], jnp.ndarray
]

GAS_CONSTANT = 8.314462618  # J / (mol K)

_LN10 = float(jnp.log(10.0))

# ADM1 lower-pH Hill inhibition: the Hill exponent is n = _PH_INHIBIT_HILL_SLOPE
# / (pH_UL - pH_LL), so the inhibition spans ~the [pH_LL, pH_UL] window
# (Batstone et al. 2002; Rosen & Jeppsson 2006).
_PH_INHIBIT_HILL_SLOPE = 3.0

# Floor on the (pH_UL - pH_LL) window so a degenerate or inverted window does not
# make the Hill exponent (and the rate) non-finite. A real window is ~1-2 pH
# units, far above this floor, so it is identity for any sane input; at the
# zero-width limit the inhibition becomes an (infinitely) steep but finite step
# at the midpoint, the physical limit of a vanishing window.
_PH_INHIBIT_MIN_WIDTH = 1.0e-6


def _safe_ratio(num, denom):
    """``num / denom`` that returns 0 (not NaN) where ``denom == 0``, with a
    finite gradient there.

    The Monod saturation / inhibition terms are all ``num / denom`` with a
    denominator that is zero only when the limiting quantity is fully depleted
    (``K + X`` with ``K = X = 0``, a saturation constant calibrated to its zero
    bound beside a depleted species). The physically safe limit is 0 (no
    substrate -> no rate). A bare ``num / denom`` is ``0/0 = NaN`` there, and the
    naive ``where(denom != 0, num/denom, 0)`` still back-propagates a NaN through
    the unused branch -- so the *denominator* is guarded too (the double-where):
    the live branch never divides by zero and the masked branch uses a finite
    stand-in, so both the value and its gradient stay clean. Identity for any
    nonzero denominator (the only change is exactly at the singularity).
    """
    nonzero = denom != 0.0
    safe = jnp.where(nonzero, denom, 1.0)
    return jnp.where(nonzero, num / safe, 0.0)


class ASTNode(ABC):
    """
    Base class for rate-expression AST nodes.

    Each subclass implements :meth:`compile`, which returns a closure with the
    canonical rate-callable signature ``(C, params, condition_arrays, loc_idx)
    -> scalar`` using only JAX operations.

    Operator nodes (everything but the four leaves) additionally declare the
    single source of truth for their arithmetic: a ``KIND`` string (the kernel
    key) and an ``op(operands)`` staticmethod evaluated by both the scalar
    closure here and the batched kernel in :mod:`aquakin.core.vector_kernel`.
    See :class:`_OperatorNode`.
    """

    # Operator nodes override these; leaves keep the defaults (KIND ``None``
    # means "not an operator" -- the kernel builder and interner skip it).
    KIND: ClassVar[str | None] = None
    #: Condition fields this node reads directly (beyond its AST children),
    #: appended as trailing operands -- e.g. the ``T`` an Arrhenius term needs
    #: or the ``pH`` a pH switch reads. Drives both the required-field check and
    #: :meth:`condition_names`.
    EXTRA_CONDITIONS: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def compile(self, ctx: CompileContext) -> RateCallable:
        """Return a JAX-compatible callable for this node."""

    def children(self) -> tuple[ASTNode, ...]:
        """Direct child AST nodes, in field order.

        Generic over every concrete node (all are frozen dataclasses): returns
        each dataclass-field value that is itself an :class:`ASTNode`. Leaf
        nodes (no AST-valued fields) return ``()``. Generic AST traversals
        drive off this, so a new node type's children can never be silently
        skipped by a hand-enumerated walk.
        """
        return tuple(v for f in fields(self) if isinstance(v := getattr(self, f.name), ASTNode))

    def map_children(self, fn: Callable[[ASTNode], ASTNode]) -> ASTNode:
        """Return a copy with each direct child replaced by ``fn(child)``.

        Leaf nodes (no AST children) return ``self``. Reconstructs the
        frozen-dataclass node via :func:`dataclasses.replace`, and returns
        ``self`` unchanged when no child actually changed (so unaffected
        subtrees keep their identity).
        """
        replacements: dict[str, ASTNode] = {}
        changed = False
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, ASTNode):
                nv = fn(v)
                replacements[f.name] = nv
                changed = changed or nv is not v
        if not changed:
            return self
        return replace(self, **replacements)

    def species(self) -> set[str]:
        """Names of species referenced anywhere in this subtree.

        Generic: the union over :meth:`children`. Leaf nodes (no children) get
        the empty set; ``SpeciesNode`` overrides this to add its own name.
        Driving the three accessors off ``children()`` means a new node type is
        traversed automatically -- it cannot silently under-report.
        """
        return set().union(*(c.species() for c in self.children()))

    def param_names(self) -> set[str]:
        """Local (un-namespaced) rate-constant names in this subtree (union over
        :meth:`children`; ``ParamNode`` adds its own name)."""
        return set().union(*(c.param_names() for c in self.children()))

    def condition_names(self) -> set[str]:
        """Condition field names in this subtree (union over :meth:`children`,
        plus this node's own :attr:`EXTRA_CONDITIONS`; ``ConditionNode`` adds
        its own field). The temperature/pH nodes get ``T`` / ``pH`` for free
        via ``EXTRA_CONDITIONS``, so no per-node override is needed."""
        names = set(self.EXTRA_CONDITIONS)
        for c in self.children():
            names |= c.condition_names()
        return names


# --- Leaf nodes ---------------------------------------------------------


@dataclass(frozen=True)
class ConstantNode(ASTNode):
    """Literal numeric constant."""

    value: float

    def compile(self, ctx: CompileContext) -> RateCallable:
        v = jnp.asarray(self.value)

        def _eval(C, params, condition_arrays, loc_idx):
            return v

        return _eval


@dataclass(frozen=True)
class SpeciesNode(ASTNode):
    """Looks up a species concentration by index from ``C``."""

    name: str

    def compile(self, ctx: CompileContext) -> RateCallable:
        if self.name not in ctx.species_index:
            raise KeyError(
                f"Species '{self.name}' is referenced in a rate expression but "
                f"not declared.{did_you_mean(self.name, ctx.species_index)}"
            )
        idx = ctx.species_index[self.name]

        def _eval(C, params, condition_arrays, loc_idx):
            return C[idx]

        return _eval

    def species(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class ParamNode(ASTNode):
    """Looks up a rate constant by name from ``params``.

    Resolution order: reaction-local (``<reaction>.<name>``) first, then
    model-level (bare ``<name>``).
    """

    name: str  # local (un-namespaced) name

    def compile(self, ctx: CompileContext) -> RateCallable:
        if ctx.reaction_name:
            local_key = f"{ctx.reaction_name}.{self.name}"
            if local_key in ctx.param_index:
                idx = ctx.param_index[local_key]

                def _eval_local(C, params, condition_arrays, loc_idx):
                    return params[idx]

                return _eval_local
        if self.name in ctx.param_index:
            idx = ctx.param_index[self.name]

            def _eval_global(C, params, condition_arrays, loc_idx):
                return params[idx]

            return _eval_global
        raise KeyError(
            f"Parameter '{self.name}' is referenced in a rate expression but "
            f"not declared (neither as a reaction-local parameter on "
            f"'{ctx.reaction_name}' nor as a model-level parameter)."
            f"{did_you_mean(self.name, ctx.param_index)}"
        )

    def param_names(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class ConditionNode(ASTNode):
    """Indexes into ``condition_arrays[field_name][loc_idx]``."""

    field_name: str

    def compile(self, ctx: CompileContext) -> RateCallable:
        if self.field_name not in ctx.condition_fields:
            raise KeyError(
                f"Condition field '{self.field_name}' is referenced in a rate "
                f"expression but not declared."
                f"{did_you_mean(self.field_name, ctx.condition_fields)}"
            )
        name = self.field_name

        def _eval(C, params, condition_arrays, loc_idx):
            return condition_arrays[name][loc_idx]

        return _eval

    def condition_names(self) -> set[str]:
        return {self.field_name}


# --- Operator nodes -----------------------------------------------------


class _OperatorNode(ASTNode):
    """Base for every non-leaf node -- the arithmetic lives in one place.

    A subclass declares ``KIND`` (its kernel key), an ``op(operands)``
    staticmethod (the elementwise arithmetic), and optionally
    ``EXTRA_CONDITIONS`` (condition fields it reads beyond its AST children).
    :meth:`compile` here is generic: it evaluates each operand -- the node's AST
    children in field order, then its ``EXTRA_CONDITIONS`` -- and applies
    ``op``. The **same** ``op`` is what :mod:`aquakin.core.vector_kernel`
    evaluates in batch, so the scalar and vectorized paths are identical by
    construction (bit-identical arithmetic, not two hand-aligned copies).

    ``op`` takes a single tuple of operand values -- scalars in this scalar
    closure, arrays in the batched kernel -- and returns the node's value. Every
    op is written with plain JAX elementwise operations, which are agnostic to
    that distinction.
    """

    #: ``op(operands: tuple) -> value``. Operand order is the node's AST-child
    #: fields followed by its ``EXTRA_CONDITIONS``.
    op: ClassVar[Callable[[tuple], Any]]

    def compile(self, ctx: CompileContext) -> RateCallable:
        for field_name in self.EXTRA_CONDITIONS:
            if field_name not in ctx.condition_fields:
                raise KeyError(
                    f"{type(self).__name__} requires a condition field named '{field_name}'."
                )
        child_fns = tuple(c.compile(ctx) for c in self.children())
        extra = self.EXTRA_CONDITIONS
        op = type(self).op

        def _eval(C, params, condition_arrays, loc_idx):
            operands = tuple(f(C, params, condition_arrays, loc_idx) for f in child_fns)
            operands += tuple(condition_arrays[name][loc_idx] for name in extra)
            return op(operands)

        return _eval


@dataclass(frozen=True)
class _BinaryNode(_OperatorNode):
    """Shared field layout for binary arithmetic AST nodes."""

    left: ASTNode
    right: ASTNode


class AddNode(_BinaryNode):
    KIND = "add"
    op = staticmethod(lambda o: o[0] + o[1])


class SubtractNode(_BinaryNode):
    KIND = "sub"
    op = staticmethod(lambda o: o[0] - o[1])


class MultiplyNode(_BinaryNode):
    KIND = "mul"
    op = staticmethod(lambda o: o[0] * o[1])


class DivideNode(_BinaryNode):
    KIND = "div"
    op = staticmethod(lambda o: o[0] / o[1])


class PowerNode(_BinaryNode):
    KIND = "pow"
    op = staticmethod(lambda o: o[0] ** o[1])


@dataclass(frozen=True)
class NegateNode(_OperatorNode):
    """Unary minus."""

    operand: ASTNode

    KIND = "neg"
    op = staticmethod(lambda o: -o[0])


# --- Domain-specific function nodes ------------------------------------


@dataclass(frozen=True)
class ArrheniusNode(_OperatorNode):
    """
    Temperature-dependent rate factor: ``A * exp(-Ea / (R * T))``.

    Requires a condition field named ``T`` (in Kelvin).
    """

    A: ASTNode
    Ea: ASTNode

    KIND = "arrhenius"
    FUNCTION_NAME = "arrhenius"
    EXTRA_CONDITIONS = ("T",)

    @staticmethod
    def op(o):
        A, Ea, T = o
        return A * jnp.exp(-Ea / (GAS_CONSTANT * T))


@dataclass(frozen=True)
class MonodNode(_OperatorNode):
    """Saturation Monod term: ``X / (K + X)``.

    Standard in microbial kinetics: substrate-limited growth fraction.
    Equivalent to writing ``[X] / (K + [X])`` inline, but more compact.
    """

    X: ASTNode
    K: ASTNode

    KIND = "monod"
    FUNCTION_NAME = "monod"

    @staticmethod
    def op(o):
        x, k = o
        return _safe_ratio(x, k + x)


@dataclass(frozen=True)
class MonodInhibitionNode(_OperatorNode):
    """Inhibition Monod term: ``K / (K + X)``.

    Equal to ``1 - monod(X, K)``. Used as an aerobic-off / anoxic-on
    switch in ASM-family models.
    """

    X: ASTNode
    K: ASTNode

    KIND = "monod_inh"
    FUNCTION_NAME = "monod_inh"

    @staticmethod
    def op(o):
        x, k = o
        return _safe_ratio(k, k + x)


@dataclass(frozen=True)
class MonodRatioNode(_OperatorNode):
    """Surface-ratio Monod term: ``(A/B) / (K + A/B)``.

    The kinetic form used in ASM1/2/3 hydrolysis where the rate-limiting
    quantity is the substrate-to-biomass ratio (``XS/XH``) rather than
    bulk substrate concentration. Written mathematically equivalently as
    ``A / (K*B + A)`` to avoid the ``B=0`` singularity at startup; the
    remaining ``A = B = 0`` point (full depletion) evaluates to 0 via
    :func:`_safe_ratio` rather than ``NaN``.
    """

    A: ASTNode
    B: ASTNode
    K: ASTNode

    KIND = "monod_ratio"
    FUNCTION_NAME = "monod_ratio"

    @staticmethod
    def op(o):
        a, b, k = o
        return _safe_ratio(a, k * b + a)


@dataclass(frozen=True)
class MonodInhibitionRatioNode(_OperatorNode):
    """Inhibition ratio Monod term: ``K / (K + A/B)``.

    The inhibition counterpart of :class:`MonodRatioNode`. Appears in
    ASM-family bio-P models as a saturation gate on the storage-to-biomass
    ratio (e.g. PHA/PAO). Stable form: ``K*B / (K*B + A)``.
    """

    A: ASTNode
    B: ASTNode
    K: ASTNode

    KIND = "monod_inh_ratio"
    FUNCTION_NAME = "monod_inh_ratio"

    @staticmethod
    def op(o):
        a, b, k = o
        kb = k * b
        return _safe_ratio(kb, kb + a)


@dataclass(frozen=True)
class SafeDivideNode(_OperatorNode):
    """Division ``num / denom`` that returns 0 where ``denom == 0``.

    The ``safe_div(num, denom)`` rate function. Use it for a ratio whose
    denominator can legitimately reach exactly zero -- e.g. a substrate-
    competition fraction ``[A] / ([A] + [B])`` where both substrates can deplete
    to 0 -- so the rate takes its physical limit (0, with a finite gradient) at
    that point instead of ``inf`` / ``NaN``, without padding the denominator with
    a dimensionless epsilon. Built on the same double-where guard
    (:func:`_safe_ratio`) as the Monod nodes; identity for any nonzero
    denominator.
    """

    num: ASTNode
    denom: ASTNode

    KIND = "safediv"
    FUNCTION_NAME = "safe_div"

    @staticmethod
    def op(o):
        return _safe_ratio(o[0], o[1])


@dataclass(frozen=True)
class MaxNode(_OperatorNode):
    """Elementwise maximum ``max(a, b)`` -- the ``max(a, b)`` rate function.

    Used to one-sidedly clip a quantity, e.g. ``max(0, P_gas - P_atm)`` so an
    overpressure-driven flux only ever leaves (never reverses) when the driving
    difference goes negative. AD-safe (``jnp.maximum`` carries the subgradient at
    the kink); identity away from it.
    """

    a: ASTNode
    b: ASTNode

    KIND = "max"
    FUNCTION_NAME = "max"

    @staticmethod
    def op(o):
        return jnp.maximum(o[0], o[1])


@dataclass(frozen=True)
class pHSwitchNode(_OperatorNode):
    """
    Acid/base speciation fraction: ``1 / (1 + 10^(pH - pKa))``.

    Returns the fraction of the conjugate-acid form. The deprotonated form is
    ``1 - pH_switch(pKa)``. Requires a condition field named ``pH``.

    Implemented as a sigmoid for numerical stability at extreme pH:
    ``sigmoid(-ln(10) * (pH - pKa))``.
    """

    pKa: ASTNode

    KIND = "phswitch"
    FUNCTION_NAME = "pH_switch"
    EXTRA_CONDITIONS = ("pH",)

    @staticmethod
    def op(o):
        pKa, pH = o
        return jax.nn.sigmoid(-_LN10 * (pH - pKa))


@dataclass(frozen=True)
class pHInhibitNode(_OperatorNode):
    """
    ADM1 lower-pH (Hill) inhibition factor

        I_pH = pHLim^n / (S_H^n + pHLim^n),
        pHLim = 10^(-(pH_UL + pH_LL)/2),  n = 3/(pH_UL - pH_LL),  S_H = 10^(-pH)

    (Batstone et al. 2002; Rosen & Jeppsson 2006). Implemented in the equivalent,
    numerically stable sigmoid form
    ``I_pH = sigmoid(ln(10) * n * (pH - (pH_UL + pH_LL)/2))`` -- 1 at high pH (no
    inhibition), 0 at low pH (full inhibition). Requires a condition field named
    ``pH``. Arguments are the lower and upper pH limits (parameters or constants).
    """

    pH_LL: ASTNode
    pH_UL: ASTNode

    KIND = "phinhibit"
    FUNCTION_NAME = "pH_inhibit"
    EXTRA_CONDITIONS = ("pH",)

    @staticmethod
    def op(o):
        ll, ul, pH = o
        # Floor the window width so an equal/inverted (pH_LL, pH_UL) -- which a
        # calibration can drive into -- gives a finite (steep) factor instead
        # of a division by zero (NaN); identity for any real window.
        width = jnp.maximum(ul - ll, _PH_INHIBIT_MIN_WIDTH)
        n = _PH_INHIBIT_HILL_SLOPE / width
        return jax.nn.sigmoid(_LN10 * n * (pH - 0.5 * (ul + ll)))
