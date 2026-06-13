"""AST node types for rate expressions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp

from aquakin.core.context import CompileContext

RateCallable = Callable[[jnp.ndarray, jnp.ndarray, dict, jnp.ndarray], jnp.ndarray]

GAS_CONSTANT = 8.314462618  # J / (mol K)

# ADM1 lower-pH Hill inhibition: the Hill exponent is n = _PH_INHIBIT_HILL_SLOPE
# / (pH_UL - pH_LL), so the inhibition spans ~the [pH_LL, pH_UL] window
# (Batstone et al. 2002; Rosen & Jeppsson 2006).
_PH_INHIBIT_HILL_SLOPE = 3.0


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
    """

    @abstractmethod
    def compile(self, ctx: CompileContext) -> RateCallable:
        """Return a JAX-compatible callable for this node."""

    def children(self) -> tuple["ASTNode", ...]:
        """Direct child AST nodes, in field order.

        Generic over every concrete node (all are frozen dataclasses): returns
        each dataclass-field value that is itself an :class:`ASTNode`. Leaf
        nodes (no AST-valued fields) return ``()``. Generic AST traversals
        drive off this, so a new node type's children can never be silently
        skipped by a hand-enumerated walk.
        """
        return tuple(
            v for f in fields(self)
            if isinstance(v := getattr(self, f.name), ASTNode)
        )

    def map_children(self, fn: "Callable[[ASTNode], ASTNode]") -> "ASTNode":
        """Return a copy with each direct child replaced by ``fn(child)``.

        Leaf nodes (no AST children) return ``self``. Reconstructs the
        frozen-dataclass node via :func:`dataclasses.replace`, and returns
        ``self`` unchanged when no child actually changed (so unaffected
        subtrees keep their identity).
        """
        replacements: dict[str, "ASTNode"] = {}
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
        """Condition field names in this subtree (union over :meth:`children`;
        ``ConditionNode`` adds its own, and the temperature/pH nodes add the
        field they require)."""
        return set().union(*(c.condition_names() for c in self.children()))


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
                f"not declared. Declared species: {sorted(ctx.species_index)}"
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
    network-level (bare ``<name>``).
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
            f"'{ctx.reaction_name}' nor as a network-level parameter). "
            f"Declared parameters: {sorted(ctx.param_index)}"
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
                f"expression but not declared. Declared conditions: "
                f"{sorted(ctx.condition_fields)}"
            )
        name = self.field_name

        def _eval(C, params, condition_arrays, loc_idx):
            return condition_arrays[name][loc_idx]

        return _eval

    def condition_names(self) -> set[str]:
        return {self.field_name}


# --- Binary operations --------------------------------------------------


@dataclass(frozen=True)
class _BinaryNode(ASTNode):
    """Shared implementation for binary arithmetic AST nodes."""

    left: ASTNode
    right: ASTNode

    # Subclasses override _op with a JAX-compatible scalar op (l, r) -> scalar.
    _op: ClassVar[Callable[[Any, Any], Any]]

    def compile(self, ctx: CompileContext) -> RateCallable:
        lf = self.left.compile(ctx)
        rf = self.right.compile(ctx)
        op = type(self)._op

        def _eval(C, params, condition_arrays, loc_idx):
            return op(
                lf(C, params, condition_arrays, loc_idx),
                rf(C, params, condition_arrays, loc_idx),
            )

        return _eval


class AddNode(_BinaryNode):
    _op = staticmethod(lambda l, r: l + r)


class SubtractNode(_BinaryNode):
    _op = staticmethod(lambda l, r: l - r)


class MultiplyNode(_BinaryNode):
    _op = staticmethod(lambda l, r: l * r)


class DivideNode(_BinaryNode):
    _op = staticmethod(lambda l, r: l / r)


class PowerNode(_BinaryNode):
    _op = staticmethod(lambda l, r: l ** r)


@dataclass(frozen=True)
class NegateNode(ASTNode):
    """Unary minus."""

    operand: ASTNode

    def compile(self, ctx: CompileContext) -> RateCallable:
        f = self.operand.compile(ctx)

        def _eval(C, params, condition_arrays, loc_idx):
            return -f(C, params, condition_arrays, loc_idx)

        return _eval


# --- Domain-specific function nodes ------------------------------------


@dataclass(frozen=True)
class ArrheniusNode(ASTNode):
    """
    Temperature-dependent rate factor: ``A * exp(-Ea / (R * T))``.

    Requires a condition field named ``T`` (in Kelvin).
    """

    A: ASTNode
    Ea: ASTNode

    def compile(self, ctx: CompileContext) -> RateCallable:
        if "T" not in ctx.condition_fields:
            raise KeyError(
                "arrhenius(...) requires a condition field named 'T' (Kelvin)."
            )
        af = self.A.compile(ctx)
        ef = self.Ea.compile(ctx)

        def _eval(C, params, condition_arrays, loc_idx):
            T = condition_arrays["T"][loc_idx]
            return af(C, params, condition_arrays, loc_idx) * jnp.exp(
                -ef(C, params, condition_arrays, loc_idx) / (GAS_CONSTANT * T)
            )

        return _eval

    def condition_names(self) -> set[str]:
        return {"T"} | super().condition_names()


@dataclass(frozen=True)
class MonodNode(ASTNode):
    """Saturation Monod term: ``X / (K + X)``.

    Standard in microbial kinetics: substrate-limited growth fraction.
    Equivalent to writing ``[X] / (K + [X])`` inline, but more compact.
    """

    X: ASTNode
    K: ASTNode

    def compile(self, ctx: CompileContext) -> RateCallable:
        xf = self.X.compile(ctx)
        kf = self.K.compile(ctx)

        def _eval(C, params, condition_arrays, loc_idx):
            x = xf(C, params, condition_arrays, loc_idx)
            k = kf(C, params, condition_arrays, loc_idx)
            return _safe_ratio(x, k + x)

        return _eval


@dataclass(frozen=True)
class MonodInhibitionNode(ASTNode):
    """Inhibition Monod term: ``K / (K + X)``.

    Equal to ``1 - monod(X, K)``. Used as an aerobic-off / anoxic-on
    switch in ASM-family models.
    """

    X: ASTNode
    K: ASTNode

    def compile(self, ctx: CompileContext) -> RateCallable:
        xf = self.X.compile(ctx)
        kf = self.K.compile(ctx)

        def _eval(C, params, condition_arrays, loc_idx):
            x = xf(C, params, condition_arrays, loc_idx)
            k = kf(C, params, condition_arrays, loc_idx)
            return _safe_ratio(k, k + x)

        return _eval


@dataclass(frozen=True)
class MonodRatioNode(ASTNode):
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

    def compile(self, ctx: CompileContext) -> RateCallable:
        af = self.A.compile(ctx)
        bf = self.B.compile(ctx)
        kf = self.K.compile(ctx)

        def _eval(C, params, condition_arrays, loc_idx):
            a = af(C, params, condition_arrays, loc_idx)
            b = bf(C, params, condition_arrays, loc_idx)
            k = kf(C, params, condition_arrays, loc_idx)
            return _safe_ratio(a, k * b + a)

        return _eval


@dataclass(frozen=True)
class MonodInhibitionRatioNode(ASTNode):
    """Inhibition ratio Monod term: ``K / (K + A/B)``.

    The inhibition counterpart of :class:`MonodRatioNode`. Appears in
    ASM-family bio-P models as a saturation gate on the storage-to-biomass
    ratio (e.g. PHA/PAO). Stable form: ``K*B / (K*B + A)``.
    """

    A: ASTNode
    B: ASTNode
    K: ASTNode

    def compile(self, ctx: CompileContext) -> RateCallable:
        af = self.A.compile(ctx)
        bf = self.B.compile(ctx)
        kf = self.K.compile(ctx)

        def _eval(C, params, condition_arrays, loc_idx):
            a = af(C, params, condition_arrays, loc_idx)
            b = bf(C, params, condition_arrays, loc_idx)
            k = kf(C, params, condition_arrays, loc_idx)
            kb = k * b
            return _safe_ratio(kb, kb + a)

        return _eval


@dataclass(frozen=True)
class pHSwitchNode(ASTNode):
    """
    Acid/base speciation fraction: ``1 / (1 + 10^(pH - pKa))``.

    Returns the fraction of the conjugate-acid form. The deprotonated form is
    ``1 - pH_switch(pKa)``. Requires a condition field named ``pH``.

    Implemented as a sigmoid for numerical stability at extreme pH:
    ``sigmoid(-ln(10) * (pH - pKa))``.
    """

    pKa: ASTNode

    def compile(self, ctx: CompileContext) -> RateCallable:
        if "pH" not in ctx.condition_fields:
            raise KeyError(
                "pH_switch(...) requires a condition field named 'pH'."
            )
        kf = self.pKa.compile(ctx)
        ln10 = float(jnp.log(10.0))

        def _eval(C, params, condition_arrays, loc_idx):
            pH = condition_arrays["pH"][loc_idx]
            pKa = kf(C, params, condition_arrays, loc_idx)
            return jax.nn.sigmoid(-ln10 * (pH - pKa))

        return _eval

    def condition_names(self) -> set[str]:
        return {"pH"} | super().condition_names()


@dataclass(frozen=True)
class pHInhibitNode(ASTNode):
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

    def compile(self, ctx: CompileContext) -> RateCallable:
        if "pH" not in ctx.condition_fields:
            raise KeyError("pH_inhibit(...) requires a condition field named 'pH'.")
        ll_f = self.pH_LL.compile(ctx)
        ul_f = self.pH_UL.compile(ctx)
        ln10 = float(jnp.log(10.0))

        def _eval(C, params, condition_arrays, loc_idx):
            pH = condition_arrays["pH"][loc_idx]
            ll = ll_f(C, params, condition_arrays, loc_idx)
            ul = ul_f(C, params, condition_arrays, loc_idx)
            n = _PH_INHIBIT_HILL_SLOPE / (ul - ll)
            return jax.nn.sigmoid(ln10 * n * (pH - 0.5 * (ul + ll)))

        return _eval

    def condition_names(self) -> set[str]:
        return {"pH"} | super().condition_names()
