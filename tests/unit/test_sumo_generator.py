"""Operator-precedence emission in the SUMO -> aquakin generator.

The generator (``scripts/sumo_to_aquakin.py``) turns a SUMO stoichiometry AST
into an aquakin rate-expression string. Subtraction and division are
left-associative and non-commutative, so a right operand at equal precedence
must be parenthesised: ``a / (b * c)`` is not ``a / b * c``. A missing
parenthesisation here silently mis-emitted ASM2d's anoxic-growth nitrate
coefficient ``(1-YH)/(iNO3_N2*YH)`` as ``(1-YH)/iNO3_N2*YH`` (i.e.
``(1-YH)*YH/iNO3_N2``), breaking COD continuity (#210). This pins the fix.
"""
import importlib.util
from pathlib import Path

import pytest

_GEN = Path(__file__).resolve().parents[2] / "scripts" / "sumo_to_aquakin.py"


@pytest.fixture(scope="module")
def emit():
    spec = importlib.util.spec_from_file_location("sumo_to_aquakin", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return lambda ast: mod.emit(ast, set(), set())


def _v(name):
    return {"var": name}


def _op(o, *args):
    return {"op": o, "args": list(args)}


@pytest.mark.parametrize("ast_factory, expected", [
    # the regression: a / (b * c) -- product in the denominator
    (lambda: _op("div", _op("sub", {"const": 1}, _v("YH")),
                 _op("mul", _v("iNO3_N2"), _v("YH"))),
     "(1 - YH) / (iNO3_N2 * YH)"),
    # a / (b / c) -- nested division in the denominator
    (lambda: _op("div", _v("a"), _op("div", _v("b"), _v("c"))), "a / (b / c)"),
    # a - (b - c) -- subtraction is also left-associative
    (lambda: _op("sub", _v("a"), _op("sub", _v("b"), _v("c"))), "a - (b - c)"),
    # (a - b) / c -- a numerator sum/difference still parenthesises
    (lambda: _op("div", _op("sub", _v("a"), _v("b")), _v("c")), "(a - b) / c"),
    # legitimate (a / b) * c -- must NOT gain parentheses (the common
    # substrate-per-yield x content factor, e.g. (-1/YH) * iN_SF)
    (lambda: _op("mul", _op("div", {"const": -1}, _v("YH")), _v("iN_SF")),
     "(-1) / YH * iN_SF"),
    # a * b * c -- multiplication is associative, no parentheses
    (lambda: _op("mul", _v("a"), _op("mul", _v("b"), _v("c"))), "a * b * c"),
])
def test_emit_respects_left_associativity(emit, ast_factory, expected):
    assert emit(ast_factory()) == expected
