"""Registering a rate function from outside the package.

The built-in rate functions (``monod``, ``arrhenius``, ``exp``, ``sqrt``, ...)
are node classes in :mod:`aquakin.core.nodes`, discovered by walking the node
hierarchy. :func:`register_rate_function` lets a *user* add one the same way,
without editing aquakin -- the counterpart, for rate laws, of the ``Unit``
protocol that already lets a user add a plant unit operation.

A registered function is not second class. The parser's grammar, the batched
kernel's operand order and the scalar closure are all derived from the node
type itself, so a registered function parses, evaluates, vectorizes and
differentiates on exactly the same terms as a built-in -- and its batched
kernel is bit-identical to its own scalar closure, because both call the one
``op`` the class defines.

    >>> from dataclasses import dataclass
    >>> import jax.numpy as jnp
    >>> import aquakin
    >>> from aquakin.core.nodes import ASTNode, _OperatorNode
    >>>
    >>> @dataclass(frozen=True)
    ... class HillNode(_OperatorNode):
    ...     X: ASTNode
    ...     K: ASTNode
    ...     n: ASTNode
    ...     KIND = "hill"
    ...     FUNCTION_NAME = "hill"
    ...     @staticmethod
    ...     def op(o):
    ...         x, k, n = o
    ...         return x**n / (k**n + x**n)
    >>>
    >>> aquakin.register_rate_function(HillNode)   # doctest: +SKIP
    >>> # ``hill(X, K, n)`` is now callable in any rate expression.

**Register before loading.** A model compiles its rate expressions when it is
built, and :func:`aquakin.load_model` caches the result, so registration has to
happen before the model that uses the function is loaded. A model file naming
an unregistered function fails at load with a parse error listing what is
available.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass

from aquakin.core import parser as _parser
from aquakin.core import vector_kernel as _vector_kernel
from aquakin.core.nodes import ConstantNode, _OperatorNode

__all__ = [
    "register_rate_function",
    "registered_rate_functions",
    "unregister_rate_function",
]

#: Function names registered from outside the package, in registration order.
#: The authoritative copy lives here; ``parser._EXTRA_FUNCTIONS`` and
#: ``vector_kernel._EXTRA_KERNELS`` are the projections those modules read.
_REGISTERED: dict[str, type] = {}


def _builtin_names() -> tuple[set[str], set[str]]:
    """The built-in ``FUNCTION_NAME`` and ``KIND`` sets.

    Taken as "everything in the live tables that did not come from here", so
    re-registering the same class is idempotent rather than a self-collision.
    """
    names = set(_parser._FUNCTIONS) - set(_REGISTERED)
    kinds = set(_vector_kernel._KERNELS) - {cls.KIND for cls in _REGISTERED.values()}
    return names, kinds


def _refresh() -> None:
    """Push the registered set into the parser and kernel tables."""
    _parser._EXTRA_FUNCTIONS.clear()
    _parser._EXTRA_FUNCTIONS.update(_REGISTERED)
    _vector_kernel._EXTRA_KERNELS.clear()
    _vector_kernel._EXTRA_KERNELS.update({cls.KIND: cls.op for cls in _REGISTERED.values()})
    _parser.refresh_function_registry()
    _vector_kernel.refresh_kernels()


def register_rate_function(node_cls: type) -> type:
    """Make ``node_cls`` callable by name in rate expressions.

    Parameters
    ----------
    node_cls : type
        A frozen dataclass subclassing ``aquakin.core.nodes._OperatorNode``,
        declaring:

        - ``FUNCTION_NAME`` -- the call spelling in a rate expression;
        - ``KIND`` -- the batched kernel's key for this node type;
        - ``op(operands)`` -- a staticmethod taking one tuple of operand values
          and returning the node's value, written with ``jax.numpy`` so it is
          differentiable. Operand order is the node's dataclass fields followed
          by any ``EXTRA_CONDITIONS`` it declares.

        Every dataclass field must hold an AST child; the parser constructs the
        node by passing one parsed argument per field, in order, so the field
        count is the function's arity.

    Returns
    -------
    type
        ``node_cls`` unchanged, so this also works as a decorator.

    Raises
    ------
    TypeError
        If ``node_cls`` is not a frozen dataclass subclassing ``_OperatorNode``,
        or is missing ``FUNCTION_NAME`` / ``KIND`` / ``op``.
    ValueError
        If it takes no arguments, if a field does not hold an AST child, or if
        its name or kind collides with a built-in. A collision is refused
        rather than resolved: silently shadowing ``monod`` would change the
        meaning of every model file that uses it.

    Notes
    -----
    Registering the same class twice is a no-op. Registering a *different*
    class under a name already registered replaces it -- which is what you want
    while iterating in a notebook, and is why collisions are only checked
    against the built-ins.
    """
    if not isinstance(node_cls, type) or not issubclass(node_cls, _OperatorNode):
        raise TypeError(
            f"register_rate_function expects a subclass of _OperatorNode; got {node_cls!r}. "
            "A rate function has to be a node class so the parser, the batched kernel and "
            "the dimensional check can all read its shape off the type."
        )
    if not is_dataclass(node_cls) or not node_cls.__dataclass_params__.frozen:
        raise TypeError(
            f"{node_cls.__name__} must be a frozen dataclass "
            "(@dataclass(frozen=True)): nodes are interned and hashed."
        )
    for attr in ("FUNCTION_NAME", "KIND"):
        value = getattr(node_cls, attr, None)
        if not isinstance(value, str) or not value:
            raise TypeError(f"{node_cls.__name__} must declare a non-empty string {attr}.")
    if not callable(getattr(node_cls, "op", None)):
        raise TypeError(f"{node_cls.__name__} must declare a static op(operands) method.")

    flds = fields(node_cls)
    if not flds:
        raise ValueError(
            f"{node_cls.__name__} takes no arguments; a rate function needs at least one."
        )

    name, kind = node_cls.FUNCTION_NAME, node_cls.KIND
    builtin_names, builtin_kinds = _builtin_names()
    if name in builtin_names:
        raise ValueError(
            f"'{name}' is a built-in rate function; choose another FUNCTION_NAME. "
            "Shadowing a built-in would change what existing model files mean."
        )
    if kind in builtin_kinds:
        raise ValueError(
            f"KIND '{kind}' is already used by a built-in node; choose another. "
            "The batched kernel keys on KIND, so a clash would mis-dispatch."
        )

    # A default on a field promises an optionality the grammar does not have:
    # the parser requires exactly one argument per field and passes a parsed
    # AST node for each, so a default can never be used, and a caller who
    # writes ``f(x)`` for a two-field node gets an arity error rather than the
    # default they expected.
    defaulted = [
        f.name for f in flds if f.default is not MISSING or f.default_factory is not MISSING
    ]
    if defaulted:
        raise ValueError(
            f"{node_cls.__name__} gives a default to {', '.join(defaulted)}, but every field "
            "of a rate-function node is a required argument: the parser passes one parsed "
            "expression per field, so a default is never used."
        )

    # Probe instance: the kernel's operand order is ``children()``, so this
    # pins that every field really does carry an AST child through to it.
    probe = node_cls(*[ConstantNode(1.0) for _ in flds])
    if len(probe.children()) != len(flds):
        raise ValueError(
            f"{node_cls.__name__} has {len(flds)} field(s) but only "
            f"{len(probe.children())} hold an AST child. Every field of a rate-function "
            "node is one of its arguments."
        )
    # Smoke-check the operand contract before anything can compile against it.
    n_operands = len(flds) + len(node_cls.EXTRA_CONDITIONS)
    try:
        node_cls.op(tuple(1.0 for _ in range(n_operands)))
    except Exception as exc:
        raise ValueError(
            f"{node_cls.__name__}.op failed on {n_operands} operand(s): {exc!r}. "
            "op takes ONE tuple -- its fields in order, then its EXTRA_CONDITIONS."
        ) from exc

    _REGISTERED[name] = node_cls
    _refresh()
    return node_cls


def unregister_rate_function(name: str) -> None:
    """Remove a previously registered rate function.

    For iterating in a notebook (re-defining a class under the same name
    replaces it, but dropping one needs this) and for keeping tests from
    leaking a function into each other. Unknown names are ignored, so it is
    safe in a teardown.

    A model already compiled against the function keeps working: it holds the
    compiled rate callables, not the name.
    """
    if _REGISTERED.pop(name, None) is not None:
        _refresh()


def registered_rate_functions() -> dict[str, type]:
    """The rate functions registered from outside the package, by name."""
    return dict(_REGISTERED)
