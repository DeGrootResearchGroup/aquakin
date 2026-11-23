"""Registering a rate function from outside the package.

The contract has two halves, and the second is the one worth guarding:

1. A registered function is **not second class** -- it parses, evaluates,
   reaches the batched kernel bit-identically, differentiates and renders,
   on the same terms as a built-in.
2. Registration is the **only** way in. Discovery walks live subclasses, so
   without a restriction a class that was merely *defined* could take a
   built-in's name on the next rebuild, depending on traversal order, and
   bypass the collision check entirely. That is a silent change to what every
   model file using that name means.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from aquakin.core.context import CompileContext
from aquakin.core.nodes import ASTNode, MonodNode, RateFunction
from aquakin.core.parser import ParseError, parse_rate_expression
from aquakin.core.registry import (
    register_rate_function,
    registered_rate_functions,
    unregister_rate_function,
)
from aquakin.utils.latex import to_latex
from aquakin.utils.units import DIMENSIONLESS


@pytest.fixture(autouse=True)
def _no_leaks():
    """Remove anything a test registered, so functions cannot leak between
    tests (or into the rest of the suite) through the module-level tables."""
    before = set(registered_rate_functions())
    yield
    for name in set(registered_rate_functions()) - before:
        unregister_rate_function(name)


def _hill_cls():
    """A Hill saturation term: a plausible thing to want and not a built-in."""

    @dataclass(frozen=True)
    class HillNode(RateFunction):
        X: ASTNode
        K: ASTNode
        n: ASTNode

        KIND = "test_hill"
        FUNCTION_NAME = "hill"

        @staticmethod
        def op(o):
            x, k, n = o
            return x**n / (k**n + x**n)

        @staticmethod
        def DIMENSION(args):
            return DIMENSIONLESS

    return HillNode


def _ctx():
    return CompileContext(
        species_index={"A": 0},
        param_index={"r.K": 0},
        condition_fields=frozenset(),
        reaction_name="r",
    )


def _eval(node, a=2.0, k=1.5):
    fn = node.compile(_ctx())
    return float(fn(jnp.asarray([a]), jnp.asarray([k]), {}, 0))


# --- the function works like a built-in -------------------------------


def test_registered_function_parses_and_evaluates():
    register_rate_function(_hill_cls())
    node = parse_rate_expression("hill([A], K, 2.0)")
    # 2^2 / (1.5^2 + 2^2) = 4 / 6.25
    assert _eval(node) == pytest.approx(4.0 / 6.25)


def test_registration_returns_the_class_so_it_works_as_a_decorator():
    cls = _hill_cls()
    assert register_rate_function(cls) is cls


def test_registered_function_reaches_the_vectorized_kernel_bit_identically():
    """The whole point: a registered function is batched like a built-in, and
    byte-for-byte equal to its own scalar closure because both call one ``op``."""
    from aquakin.core.vector_kernel import build_vectorized_rates

    register_rate_function(_hill_cls())
    node = parse_rate_expression("hill([A], K, 2.0)")
    si, pi = {"A": 0}, {"r.K": 0}
    kernel = build_vectorized_rates([node], ["r"], si, pi)
    scalar = node.compile(_ctx())

    rng = jnp.asarray([0.0, 0.3, 1.0, 2.5, 9.0])
    for a in rng:
        for k in rng:
            C, p = jnp.asarray([a]), jnp.asarray([k])
            kv, sv = kernel(C, p, {}, 0)[0], scalar(C, p, {}, 0)
            # NaN-aware: this Hill term is 0/0 at a fully depleted state, which
            # is the author's choice (it declares no guard -- that is what
            # ``_safe_ratio`` does for the built-ins). The contract under test
            # is that the two paths agree *exactly*, NaN included, not that the
            # function is well behaved.
            same = bool(kv == sv) or (bool(jnp.isnan(kv)) and bool(jnp.isnan(sv)))
            assert same, f"kernel {kv} != scalar {sv} at a={a}, k={k}"


def test_grad_flows_through_a_registered_function():
    register_rate_function(_hill_cls())
    fn = parse_rate_expression("hill([A], K, 2.0)").compile(_ctx())

    def g(a):
        return fn(jnp.asarray([a]), jnp.asarray([1.5]), {}, 0)

    grad = jax.grad(g)(2.0)
    assert jnp.isfinite(grad) and float(grad) != 0.0


def test_registered_function_renders_as_latex_instead_of_raising():
    register_rate_function(_hill_cls())
    out = to_latex(parse_rate_expression("hill([A], K, 2.0)"))
    assert "hill" in out
    assert r"\mathrm" in out


def test_registered_function_can_declare_its_dimension():
    """The optional units hook: a Hill term is a dimensionless fraction, so a
    rate built from it still checks out."""
    from aquakin.core.parser import parse_rate_expression as parse
    from aquakin.utils.units import check_rate_units, parse_units

    register_rate_function(_hill_cls())
    species = {"X": parse_units("g_COD/m3")}
    params = {"R.k": parse_units("1/d"), "R.K": parse_units("g_COD/m3")}
    # k[1/d] * hill(...)[-] * X[g_COD/m3] -> a concentration rate
    assert check_rate_units(parse("k * hill([X], K, 2.0) * [X]"), "R", species, params, {}) == []


def test_a_function_without_a_dimension_hook_is_skipped_not_guessed():
    """No hook means "unknown", which skips the check rather than inventing a
    dimension -- the safe default."""
    from aquakin.utils.units import check_rate_units, parse_units

    @dataclass(frozen=True)
    class OpaqueNode(RateFunction):
        X: ASTNode

        KIND = "test_opaque"
        FUNCTION_NAME = "opaque"

        @staticmethod
        def op(o):
            return o[0]

    register_rate_function(OpaqueNode)
    species = {"X": parse_units("g_COD/m3")}
    params = {"R.k": parse_units("1/d")}
    # Unknown dimension propagates, so no finding is reported either way.
    findings = check_rate_units(
        parse_rate_expression("k * opaque([X])"), "R", species, params, {}, check_root=False
    )
    assert findings == []


# --- registration is the only way in ----------------------------------


def test_defining_a_node_cannot_shadow_a_builtin():
    """The hole this restriction closes: a class that is defined and never
    registered must not be able to take a built-in's name on a rebuild."""
    from aquakin.core import parser as P

    @dataclass(frozen=True)
    class EvilNode(MonodNode):
        KIND = "test_evil"
        FUNCTION_NAME = "monod"

        @staticmethod
        def op(o):
            return o[0] * 0.0

    P.refresh_function_registry()
    assert P._FUNCTIONS["monod"][0] is MonodNode
    assert type(parse_rate_expression("monod([A], K)")) is MonodNode


def test_registering_over_a_builtin_name_is_refused():
    @dataclass(frozen=True)
    class Clash(RateFunction):
        X: ASTNode
        K: ASTNode

        KIND = "test_clash"
        FUNCTION_NAME = "monod"

        @staticmethod
        def op(o):
            return o[0]

    with pytest.raises(ValueError, match="built-in rate function"):
        register_rate_function(Clash)


def test_registering_over_a_builtin_kind_is_refused():
    @dataclass(frozen=True)
    class KindClash(RateFunction):
        X: ASTNode

        KIND = "monod"  # the kernel keys on this
        FUNCTION_NAME = "not_monod"

        @staticmethod
        def op(o):
            return o[0]

    with pytest.raises(ValueError, match="KIND"):
        register_rate_function(KindClash)


def test_unregistering_removes_it_from_the_grammar():
    register_rate_function(_hill_cls())
    assert "hill" in registered_rate_functions()
    unregister_rate_function("hill")
    assert "hill" not in registered_rate_functions()
    with pytest.raises(ParseError, match="Unknown function 'hill'"):
        parse_rate_expression("hill([A], K, 2.0)")


def test_unregistering_an_unknown_name_is_a_no_op():
    unregister_rate_function("never_registered")  # must not raise


def test_registered_functions_mapping_is_a_copy():
    register_rate_function(_hill_cls())
    registered_rate_functions()["hill"] = None
    assert registered_rate_functions()["hill"] is not None


def test_unknown_function_error_names_the_remedy():
    with pytest.raises(ParseError, match="register_rate_function"):
        parse_rate_expression("not_a_function([A])")


# --- the validations --------------------------------------------------


def test_rejects_a_non_node_class():
    with pytest.raises(TypeError, match="RateFunction"):
        register_rate_function(dict)


def test_rejects_an_unfrozen_dataclass():
    @dataclass  # not frozen
    class Thawed(RateFunction):
        X: ASTNode

        KIND = "test_thawed"
        FUNCTION_NAME = "thawed"

        @staticmethod
        def op(o):
            return o[0]

    with pytest.raises(TypeError, match="frozen"):
        register_rate_function(Thawed)


@pytest.mark.parametrize("missing", ["FUNCTION_NAME", "KIND"])
def test_rejects_a_missing_name_or_kind(missing):
    attrs = {"KIND": "test_m", "FUNCTION_NAME": "m", "op": staticmethod(lambda o: o[0])}
    del attrs[missing]
    cls = dataclass(frozen=True)(
        type("Partial", (RateFunction,), {"__annotations__": {"X": ASTNode}, **attrs})
    )
    with pytest.raises(TypeError, match=missing):
        register_rate_function(cls)


def test_rejects_a_zero_argument_function():
    @dataclass(frozen=True)
    class Nullary(RateFunction):
        KIND = "test_nullary"
        FUNCTION_NAME = "nullary"

        @staticmethod
        def op(o):
            return 1.0

    with pytest.raises(ValueError, match="at least one"):
        register_rate_function(Nullary)


def test_rejects_a_field_with_a_default():
    """A default promises an optionality the grammar does not have: the parser
    requires exactly one parsed argument per field, so the default can never be
    used and ``mixed([X])`` is an arity error rather than scale=2.0."""

    @dataclass(frozen=True)
    class Mixed(RateFunction):
        X: ASTNode
        scale: ASTNode = None

        KIND = "test_mixed"
        FUNCTION_NAME = "mixed"

        @staticmethod
        def op(o):
            return o[0]

    with pytest.raises(ValueError, match="default"):
        register_rate_function(Mixed)


def test_rejects_an_op_with_the_wrong_operand_contract():
    """``op`` takes ONE tuple; a two-positional-argument op is the common
    authoring slip and is caught at registration, not at solve time."""

    @dataclass(frozen=True)
    class BadOp(RateFunction):
        X: ASTNode
        K: ASTNode

        KIND = "test_badop"
        FUNCTION_NAME = "badop"

        @staticmethod
        def op(x, k):  # wrong: should be op(operands)
            return x + k

    with pytest.raises(ValueError, match="op"):
        register_rate_function(BadOp)


def test_re_registering_the_same_class_is_idempotent():
    cls = _hill_cls()
    register_rate_function(cls)
    register_rate_function(cls)
    assert registered_rate_functions()["hill"] is cls


def test_rejects_something_that_is_not_a_class():
    """An *instance* slipped in where the class was meant -- the easy slip when
    a decorator is written with parentheses it does not take."""
    with pytest.raises(TypeError, match="RateFunction"):
        register_rate_function(object())


def test_rejects_a_plain_class_that_is_not_a_dataclass():
    class NotADataclass(RateFunction):
        KIND = "test_plain"
        FUNCTION_NAME = "plain"

        @staticmethod
        def op(o):
            return o[0]

    with pytest.raises(TypeError, match="dataclass"):
        register_rate_function(NotADataclass)


def test_rejects_a_node_with_no_op():
    """``RateFunction`` only *annotates* ``op``, so forgetting it leaves no
    attribute at all rather than an inherited stub."""

    @dataclass(frozen=True)
    class NoOp(RateFunction):
        X: ASTNode

        KIND = "test_noop"
        FUNCTION_NAME = "noop"

    with pytest.raises(TypeError, match="op"):
        register_rate_function(NoOp)


def test_rejects_a_node_whose_children_do_not_cover_its_fields():
    """The kernel's operand order is ``children()``, so a node that hides a
    field from it would silently mis-call ``op``. Reachable by overriding
    ``children()`` -- which is why the guard is not dead code."""

    @dataclass(frozen=True)
    class HalfBlind(RateFunction):
        X: ASTNode
        K: ASTNode

        KIND = "test_halfblind"
        FUNCTION_NAME = "halfblind"

        def children(self):
            return (self.X,)  # drops K

        @staticmethod
        def op(o):
            return o[0]

    with pytest.raises(ValueError, match="AST child"):
        register_rate_function(HalfBlind)


# --- the public base ---------------------------------------------------


def test_a_rate_function_on_the_public_base_registers_parses_and_differentiates():
    """The documented way in uses only public names: subclass
    ``aquakin.RateFunction`` and pass it to ``aquakin.register_rate_function``.
    The gradient is checked against the closed form, not just for finiteness."""
    import aquakin

    @aquakin.register_rate_function
    @dataclass(frozen=True)
    class Hill2(aquakin.RateFunction):
        X: ASTNode
        K: ASTNode

        KIND = "test_public_hill2"
        FUNCTION_NAME = "hill2"

        @staticmethod
        def op(o):
            x, k = o
            return x**2 / (k**2 + x**2)

    fn = parse_rate_expression("hill2([A], K)").compile(_ctx())
    a, k = 2.0, 1.5
    assert _eval(parse_rate_expression("hill2([A], K)"), a, k) == pytest.approx(
        a**2 / (k**2 + a**2)
    )
    grad = jax.grad(lambda x: fn(jnp.asarray([x]), jnp.asarray([k]), {}, 0))(a)
    # d/dx x^2 / (k^2 + x^2) = 2 x k^2 / (k^2 + x^2)^2
    assert float(grad) == pytest.approx(2 * a * k**2 / (k**2 + a**2) ** 2, rel=1e-12)


def test_the_public_base_is_the_one_every_operator_node_derives_from():
    """One class, not an alias: the built-ins and a user's function share it."""
    import aquakin
    from aquakin.core import nodes

    assert "RateFunction" in aquakin.__all__
    assert aquakin.RateFunction is nodes.RateFunction
    assert issubclass(MonodNode, aquakin.RateFunction)
    assert not hasattr(nodes, "_OperatorNode")


def test_ast_node_is_exported_as_the_type_a_parsed_expression_has():
    """``parse_rate_expression`` and a rate function's arguments are typed
    ``aquakin.ASTNode``, the same class the parser builds, so it is public."""
    import aquakin
    from aquakin.core import nodes

    assert "ASTNode" in aquakin.__all__
    assert aquakin.ASTNode is nodes.ASTNode
    assert issubclass(aquakin.RateFunction, aquakin.ASTNode)
    parsed = aquakin.parse_rate_expression("monod([S], K) * k")
    assert isinstance(parsed, aquakin.ASTNode)
    assert isinstance(parsed.left, aquakin.ASTNode) and isinstance(parsed.right, aquakin.ASTNode)


def test_rejects_a_node_that_does_not_derive_from_the_public_base():
    """A leaf-style ``ASTNode`` subclass has the dataclass shape but no ``op``
    machinery, so it is refused by type, and the message names the base."""

    @dataclass(frozen=True)
    class Bare(ASTNode):
        X: ASTNode

        KIND = "test_bare"
        FUNCTION_NAME = "bare"

        def compile(self, ctx):
            return self.X.compile(ctx)

        @staticmethod
        def op(o):
            return o[0]

    with pytest.raises(TypeError, match=r"aquakin\.RateFunction"):
        register_rate_function(Bare)


def test_the_guarded_kernels_named_for_custom_ops_keep_the_gradient_finite():
    """A custom ``op`` gets no guards, so the docs point at the public guarded
    kernels. Written with ``safe_pow``, a half-order term has a finite (zero)
    derivative at a depleted species where plain ``**`` gives ``inf``."""
    import aquakin
    from aquakin.core.kernels import safe_log, safe_pow, safe_sqrt

    @aquakin.register_rate_function
    @dataclass(frozen=True)
    class HalfOrder(aquakin.RateFunction):
        X: ASTNode

        KIND = "test_half_order"
        FUNCTION_NAME = "half_order"

        @staticmethod
        def op(o):
            return safe_pow(o[0], 0.5)

    fn = parse_rate_expression("half_order([A])").compile(_ctx())
    grad = jax.grad(lambda x: fn(jnp.asarray([x]), jnp.asarray([1.0]), {}, 0))(0.0)
    assert float(grad) == 0.0
    assert float(jax.grad(lambda x: x**0.5)(0.0)) == float("inf")  # the unguarded spelling
    assert float(jax.grad(safe_sqrt)(0.0)) == 0.0
    assert jnp.isfinite(jax.grad(safe_log)(0.0))
