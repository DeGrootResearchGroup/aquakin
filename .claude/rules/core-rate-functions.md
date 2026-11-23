---
paths:
  - "aquakin/core/nodes.py"
  - "aquakin/core/registry.py"
  - "aquakin/core/parser.py"
  - "aquakin/core/vector_kernel.py"
---

# Rules — rate functions and the vectorized kernel (`aquakin/core/`)

How a rate function is defined once (`RateFunction.op`) and derived everywhere
else: user registration, node authoring, the parser's function registry and the
batched rate kernel. Split out of `.claude/rules/core.md`, which holds the AST
node list and the expression languages.

### Rate functions registered from outside the package

`aquakin.register_rate_function(node_cls)` (`core/registry.py`) lets a user add
a rate function without editing aquakin — the rate-law counterpart of the
`Unit` protocol. A registered node is **not** second class: the parser's
grammar, the batched kernel's operand order and the scalar closure are all
derived from the node type, so it parses, vectorizes (bit-identically to its
own scalar path) and differentiates exactly as a built-in does.

**Discovery is restricted to `core/nodes.py`, and this is load-bearing.**
`parser._build_function_registry` and `vector_kernel._collect_kernels` walk
live subclasses, so before this restriction a class that was merely *defined*
and never registered could take a built-in's name on the next rebuild —
silently, and depending on traversal order — bypassing the collision check
registration does. (Demonstrated: a subclass of `MonodNode` declaring
`FUNCTION_NAME = "monod"` captured `monod` on a refresh.) So:

- built-ins = nodes whose `__module__` is `core.nodes` (`_NODES_MODULE`);
- everything else enters only through `parser._EXTRA_FUNCTIONS` and
  `vector_kernel._EXTRA_KERNELS`, which **only** `core/registry.py` writes;
- `registry._refresh()` projects its `_REGISTERED` map into both and calls
  `parser.refresh_function_registry()` / `vector_kernel.refresh_kernels()`.

**The base is public and is one class.** A user's function subclasses
`aquakin.RateFunction` (defined in `core/nodes.py`, re-exported in
`aquakin.__all__`), the same class every built-in operator node derives from,
and `register_rate_function` refuses anything not derived from it. Do not add a
private twin or an alias: the documented extension point must not depend on a
private name. A custom `op` gets none of the built-in guards, so the guarded
numerics a user may need (`safe_pow`, `safe_sqrt`, `safe_log` in
`core/kernels.py`) are public, named in `RateFunction`'s docstring and in
`docs/model_format.md`; renaming one is an API change.

Registration refuses a `FUNCTION_NAME` or `KIND` that collides with a built-in,
a non-frozen dataclass, a zero-argument node, a field with a default (the
parser requires one parsed argument per field, so a default can never be used)
and an `op` that does not take a single operand tuple. Keep the validations
there rather than discovering breakage at solve time. `to_latex` has a generic
`\mathrm{name}(args)` fallback for any `RateFunction` carrying a
`FUNCTION_NAME`, so a registered function renders — and so a future built-in
added without a LaTeX branch renders plainly instead of crashing export. A node
may optionally declare `DIMENSION(arg_dims)` for `check_units`; without it the
dimension is unknown and the check is skipped, which is the safe default.
Covered by `tests/unit/test_rate_function_registry.py` and
`tests/integration/test_registered_rate_function.py`.

New domain-specific node types are added here as needed. An **operator** node
(everything but the four leaves) subclasses `RateFunction` and declares its
arithmetic **once**, as a single `op(operands)` staticmethod plus a `KIND`
string:

```python
@dataclass(frozen=True)
class MonodNode(RateFunction):
    X: ASTNode
    K: ASTNode
    KIND = "monod"              # kernel key
    FUNCTION_NAME = "monod"     # optional: the call spelling in a rate expression
    # EXTRA_CONDITIONS = ("T",) # optional: condition fields read beyond the AST children

    @staticmethod
    def op(o):                  # operands = AST children (field order) + EXTRA_CONDITIONS
        x, k = o
        return _safe_ratio(x, k + x)
```

`RateFunction.compile` is generic (evaluates the operands and applies `op`), so
there is no per-node `compile`. That one `op` is the single source of truth:
**everything else derives from the node class** —

- the **scalar** rate closure (`RateFunction.compile`) calls `op`;
- the **vectorized kernel** table (`vector_kernel._KERNELS`) is built by walking
  the node hierarchy and maps each `KIND` to the node's own `op` — so the two
  paths run the *identical* function (bit-identicality is structural, not two
  hand-aligned copies), and a node added *without* a `KIND` simply raises
  `UnsupportedNode` → scalar fallback;
- the **interner** dispatch (`vector_kernel._Interner.intern`) reads `KIND`,
  `children()` and `EXTRA_CONDITIONS` off the node type — no per-node branch;
- the **parser** built-in registry (`parser._FUNCTIONS`) is discovered from each
  node's `FUNCTION_NAME` (with argument names read off its dataclass fields).

So adding a domain-function node is a *single* edit — define the class — rather
than coordinated edits across `nodes.py`, `parser._FUNCTIONS`,
`vector_kernel._KERNELS` and the interner. `PowerNode` keeps one special case: a
*constant* exponent interns to the `powc` kernel (a kernel-internal variant that
holds the exponent static so its JVP stays finite at base 0); the `powc` kernel
is `PowerNode.op` itself, so even it cannot drift.

A **leaf** node (`ConstantNode`, `SpeciesNode`, `ParamNode`, `ConditionNode`)
carries a literal payload rather than a kernel, so it keeps its own `compile` and
is handled explicitly in the interner (mapping to a pool leaf-block).

### Vectorized rate kernel (`core/vector_kernel.py`)

Evaluating the rates as `jnp.stack([f(C, params, ...) for f in rate_callables])`
traces **one nested closure tree per reaction**, so the jaxpr holds
`O(reactions x ops-per-reaction)` scalar primitives — each leaf a `slice` +
`squeeze`. XLA fuses them so the *runtime* is fine, but its optimization passes
scale with that op count, so the **compile** is dominated by it — and amplified
in the reverse-mode adjoint, where the RHS jaxpr is differentiated ~80x per
step (the multi-minute discrete-adjoint compiles). Compile, not run, is the cost
of a stiff solve (`.claude/rules/integration.md`, "Compiled-solve caching"), so
this *compile-time* lever is what hurts the test suite, where each test compiles a
distinct configuration once.

`build_vectorized_rates` builds, from the per-reaction rate ASTs, a single
callable returning the `(n_reactions,)` rate vector by **interning every
distinct subexpression** (node type + operand positions) and evaluating **all
instances of each primitive in one batched elementwise op**, in topological
order — global common-subexpression elimination plus vectorization by node type.
The pool is built **append-only by concatenation** (1 jaxpr op each, vs ~5 for a
scatter incl. index fixup), and operands are read with a **raw `lax.gather`**
(skips the negative-index normalization `slice`/`squeeze`/`select_n` that
`P[idx]` inserts). The traced op count collapses to `~O(node-types x depth)`,
independent of the reaction count.

It is **bit-identical** to the scalar path: each interned instance is a lane of
a batched op doing the identical scalar arithmetic, and IEEE elementwise ops are
deterministic per lane; the interning dedups identical subexpressions (also
bit-identical, since the scalar path recomputes them to the same bits). So it is
*not* the issue's masked-`jnp.prod` assembly (which would change the product
reduction order); the order-preserving batched ops need no revalidation rebasing.
Built once in `CompiledModel.__post_init__` (`_rate_kernel`) and dispatched
from `CompiledModel.rates` after the clip / derived-condition / temperature
preprocessing (which is unchanged); a future AST node type with no batched
kernel raises `UnsupportedNode` and `rates` falls back to the scalar stack, so
the kernel is a safe, transparent overlay. Do not remove it to simplify
`rates`: the rate jaxpr dominates a differentiated stiff-solve compile, more so
on the larger models (measurements in `.claude/notes/rate-kernel-log.md`); the
runtime is unchanged either way. Regression-guarded
(bit-identicality on randomized states, op-count reduction, AD parity, and the
unsupported-node fallback) in `tests/unit/test_vector_kernel.py`.
